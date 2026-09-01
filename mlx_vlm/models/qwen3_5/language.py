from functools import lru_cache, partial
from typing import Any, List, Optional

import mlx.core as mx
import mlx.nn as nn

from ..activations import swiglu
from ..base import (
    LanguageModelOutput,
    create_attention_mask,
    scaled_dot_product_attention,
)
from ..cache import ArraysCache, KVCache
from ..rope_utils import MRoPERotaryEmbedding
from ..rope_utils import apply_multimodal_rotary_pos_emb as _apply_mrope
from .config import ModelConfig, TextConfig
from .gated_delta import (
    gated_delta_accept_states,
    gated_delta_state_update,
    gated_delta_update,
)
from .speculative_verifier import Qwen3_5ExactSpeculativeVerifier

_EXACT_SPECULATIVE_VERIFIER = Qwen3_5ExactSpeculativeVerifier()


class Qwen3_5RotaryEmbedding(MRoPERotaryEmbedding):
    def __init__(
        self,
        dim,
        max_position_embeddings=2048,
        base=10000,
        mrope_section=[11, 11, 0],
    ):
        super().__init__(
            dim,
            max_position_embeddings=max_position_embeddings,
            base=base,
            mrope_section=mrope_section,
            style="interleaved",
        )
        mx.eval(self.inv_freq)


def apply_multimodal_rotary_pos_emb(q, k, cos, sin, unqueeze_dim=1):
    return _apply_mrope(q, k, cos, sin, style="interleaved", unsqueeze_dim=unqueeze_dim)


class Qwen3_5RMSNormGated(nn.Module):
    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = mx.ones(hidden_size)

    def __call__(
        self, hidden_states: mx.array, gate: mx.array | None = None
    ) -> mx.array:
        x = mx.fast.rms_norm(hidden_states, self.weight, self.eps)
        if gate is not None:
            return _precise_swiglu(hidden_states, gate, x)
        return x.astype(hidden_states.dtype)


@partial(mx.compile, shapeless=True)
def _precise_swiglu(h, gate, x):
    gate = nn.silu(gate.astype(mx.float32))
    x = x.astype(mx.float32)
    return (gate * x).astype(h.dtype)


@partial(mx.compile, shapeless=True)
def _qwen3_5_decode_depthwise_conv(conv_input: mx.array, weight: mx.array):
    out = mx.sum(conv_input.astype(mx.float32) * weight[None, :, :], axis=1)
    return out.astype(conv_input.dtype)[:, None, :]


def _decode_quantized_linears_fused(linears, x: mx.array):
    if (
        x.ndim != 3
        or x.shape[1] != 1
        or len(linears) != 4
        or not all(isinstance(linear, nn.QuantizedLinear) for linear in linears)
    ):
        return None

    first = linears[0]
    if not all(
        linear.bits == first.bits
        and linear.group_size == first.group_size
        and linear.mode == first.mode
        and linear.biases is not None
        and linear.scales.dtype == x.dtype
        and linear.biases.dtype == x.dtype
        and "bias" not in linear
        for linear in linears
    ):
        return None

    cache_key = tuple(
        (id(linear.weight), id(linear.scales), id(linear.biases)) for linear in linears
    )
    cached = getattr(first, "_qwen3_5_fused_decode_linears", None)
    if cached is None or cached[0] != cache_key:
        weights = mx.concatenate([linear.weight for linear in linears], axis=0)
        scales = mx.concatenate([linear.scales for linear in linears], axis=0)
        biases = mx.concatenate([linear.biases for linear in linears], axis=0)
        split_indices = []
        offset = 0
        for linear in linears[:-1]:
            offset += linear.weight.shape[0]
            split_indices.append(offset)
        mx.eval(weights, scales, biases)
        cached = (cache_key, weights, scales, biases, split_indices)
        first._qwen3_5_fused_decode_linears = cached

    _, weights, scales, biases, split_indices = cached
    output = mx.quantized_matmul(
        x,
        weights,
        scales=scales,
        biases=biases,
        transpose=True,
        group_size=first.group_size,
        bits=first.bits,
        mode=first.mode,
    )
    return tuple(mx.split(output, split_indices, axis=-1))


def _extract_row_cache(cache_entry, row: int):
    if isinstance(cache_entry, ArraysCache):
        row_cache = ArraysCache(size=len(cache_entry.cache))
        row_cache.cache = [
            None if cached is None else cached[row : row + 1]
            for cached in cache_entry.cache
        ]
        lengths = getattr(cache_entry, "lengths", None)
        if lengths is not None:
            row_cache.lengths = lengths[row : row + 1]
        return row_cache

    if hasattr(cache_entry, "extract") and not cache_entry.empty():
        return cache_entry.extract(row)

    if hasattr(cache_entry, "left_padding"):
        row_cache = KVCache()
        return row_cache

    return cache_entry


def _is_single_row_batch_cache(cache_entry) -> bool:
    left_padding = getattr(cache_entry, "left_padding", None)
    # Quantized batch caches must update in place; the singleton shortcut
    # rebuilds only plain BatchKVCache state after the row-wise forward.
    return (
        isinstance(left_padding, mx.array)
        and left_padding.ndim > 0
        and left_padding.size == 1
        and not hasattr(cache_entry, "bits")
    )


def _pad_row_time(x: mx.array, pad: int, target_length: int) -> mx.array:
    if pad <= 0:
        return x
    if x.shape[1] >= target_length:
        return x
    return mx.concatenate(
        [
            mx.zeros((x.shape[0], pad, *x.shape[2:]), dtype=x.dtype),
            x,
        ],
        axis=1,
    )


def _restore_batch_padding_metadata(cache_entry, offsets, steps: int):
    if offsets is None:
        return cache_entry
    if not (
        hasattr(cache_entry, "offset")
        and hasattr(cache_entry, "left_padding")
        and hasattr(cache_entry, "_idx")
    ):
        return cache_entry
    cache_entry.offset = offsets + steps
    cache_entry.left_padding = cache_entry._idx - cache_entry.offset
    return cache_entry


def _qwen3_5_left_padding_info(cache):
    left_padding = getattr(cache, "left_padding", None)
    if not (
        isinstance(left_padding, mx.array)
        and left_padding.ndim > 0
        and left_padding.size > 0
    ):
        return None

    cached = getattr(cache, "_qwen3_5_left_padding_info", None)
    if cached is None or cached[0] is not left_padding:
        pads = tuple(int(p) for p in left_padding.tolist())
        cached = (left_padding, pads, max(pads) if pads else 0)
        cache._qwen3_5_left_padding_info = cached
    return cached[1], cached[2]


def _qwen3_5_set_left_padding_info(cache, pads):
    left_padding = getattr(cache, "left_padding", None)
    if not isinstance(left_padding, mx.array):
        return
    pads = tuple(int(p) for p in pads)
    cache._qwen3_5_left_padding_info = (
        left_padding,
        pads,
        max(pads) if pads else 0,
    )


def _qwen3_5_advance_left_padding_info(cache, steps: int):
    cached = getattr(cache, "_qwen3_5_left_padding_info", None)
    if cached is None:
        return
    _left_padding, pads, _max_pad = cached
    _qwen3_5_set_left_padding_info(cache, (p - steps for p in pads))


def _qwen3_5_lengths_info(cache):
    lengths = getattr(cache, "lengths", None)
    if not (isinstance(lengths, mx.array) and lengths.ndim > 0 and lengths.size > 0):
        return None
    cached = getattr(cache, "_qwen3_5_lengths_info", None)
    if cached is None or cached[0] is not lengths:
        values = tuple(int(v) for v in lengths.tolist())
        cached = (lengths, min(values) if values else 0)
        cache._qwen3_5_lengths_info = cached
    return cached[1]


def _qwen3_5_advance_lengths_info(cache, steps: int):
    lengths = getattr(cache, "lengths", None)
    cached = getattr(cache, "_qwen3_5_lengths_info", None)
    if cached is None or not isinstance(lengths, mx.array):
        return
    _lengths, min_value = cached
    cache._qwen3_5_lengths_info = (lengths, min_value - steps)


def _create_qwen3_5_ssm_mask(h: mx.array, cache):
    if not (cache and hasattr(cache, "make_mask")):
        return None

    lengths = getattr(cache, "lengths", None)
    left_padding = getattr(cache, "left_padding", None)
    if isinstance(left_padding, mx.array):
        batch_size = int(left_padding.shape[0]) if left_padding.ndim > 0 else 1
        if (
            lengths is None
            and getattr(cache, "_qwen3_5_ssm_no_mask_batch_size", None) == batch_size
        ):
            return None
        left_padding_info = _qwen3_5_left_padding_info(cache)
        max_left_padding = left_padding_info[1] if left_padding_info else 0
        if max_left_padding <= 0:
            if lengths is None:
                cache._qwen3_5_ssm_no_mask_batch_size = batch_size
            return None
        if hasattr(cache, "_qwen3_5_ssm_no_mask_batch_size"):
            delattr(cache, "_qwen3_5_ssm_no_mask_batch_size")

    lengths_min = _qwen3_5_lengths_info(cache)
    if lengths_min is not None and lengths_min >= h.shape[1]:
        return None

    return cache.make_mask(h.shape[1])


def _create_qwen3_5_attention_mask(h: mx.array, cache):
    if cache is None:
        return create_attention_mask(h, cache)

    if hasattr(cache, "_qwen3_5_decode_left_padding"):
        delattr(cache, "_qwen3_5_decode_left_padding")

    left_padding = getattr(cache, "left_padding", None)
    if h.shape[1] == 1 and isinstance(left_padding, mx.array) and left_padding.ndim > 0:
        padding_cache = getattr(cache, "_qwen3_5_left_padding_cache", None)
        if padding_cache is None or padding_cache[0] is not left_padding:
            left_padding_info = _qwen3_5_left_padding_info(cache)
            pads = list(left_padding_info[0]) if left_padding_info else []
            padding_cache = (left_padding, pads, max(pads) if pads else 0)
            cache._qwen3_5_left_padding_cache = padding_cache
        pads = padding_cache[1]
        if padding_cache[2] <= 0:
            return None
        cache._qwen3_5_decode_left_padding = pads
        return "left_padded_decode"
    return create_attention_mask(h, cache)


def _set_qwen3_5_decode_left_padding(caches, layers, pads):
    if caches is None:
        return
    for layer, cache_entry in zip(layers, caches):
        if layer.is_linear or cache_entry is None:
            continue
        if pads is None:
            if hasattr(cache_entry, "_qwen3_5_decode_left_padding"):
                delattr(cache_entry, "_qwen3_5_decode_left_padding")
        else:
            cache_entry._qwen3_5_decode_left_padding = pads


_QWEN3_5_RAGGED_SDPA_ONE_PASS_SOURCE = r"""
    uint q_batch_head_idx = threadgroup_position_in_grid.y;
    uint simd_gid = simdgroup_index_in_threadgroup;
    uint simd_lid = thread_index_in_simdgroup;

    constexpr int BN = 32;
    constexpr int BD = 32;
    constexpr int qk_per_thread = D_SIZE / BD;
    constexpr int v_per_thread = V_SIZE / BD;

    typedef float U;
    thread U q[qk_per_thread];
    thread U k[qk_per_thread];
    thread U o[v_per_thread];
    threadgroup U outputs[BN * BD];
    threadgroup U max_scores[BN];
    threadgroup U sum_exp_scores[BN];

    int K_SIZE = int(k_size[0]);
    int batch_idx = int(q_batch_head_idx) / NUM_Q_HEADS;
    int q_head_idx = int(q_batch_head_idx) - batch_idx * NUM_Q_HEADS;
    int kv_head_idx = q_head_idx / GQA_FACTOR;
    int pad = int(pads[batch_idx]);
    int N = K_SIZE - pad;

    const device T* qptr =
        queries + int(q_batch_head_idx) * D_SIZE + int(simd_lid) * qk_per_thread;
    const device T* kptr =
        keys + (batch_idx * NUM_KV_HEADS + kv_head_idx) * K_SIZE * D_SIZE +
        (pad + int(simd_gid)) * D_SIZE + int(simd_lid) * qk_per_thread;
    const device T* vptr =
        values + (batch_idx * NUM_KV_HEADS + kv_head_idx) * K_SIZE * V_SIZE +
        (pad + int(simd_gid)) * V_SIZE + int(simd_lid) * v_per_thread;
    device T* optr =
        out + int(q_batch_head_idx) * V_SIZE + int(simd_gid) * v_per_thread;

    U s = U(scale[0]);
    for (int i = 0; i < qk_per_thread; i++) {
        q[i] = s * qptr[i];
    }
    for (int i = 0; i < v_per_thread; i++) {
        o[i] = 0;
    }

    U max_score = -3.4028234663852886e38f;
    U sum_exp_score = 0;

    for (int i = int(simd_gid); i < N; i += BN) {
        for (int j = 0; j < qk_per_thread; j++) {
            k[j] = kptr[j];
        }

        U score = 0;
        for (int j = 0; j < qk_per_thread; j++) {
            score += q[j] * k[j];
        }
        score = simd_sum(score);

        U new_max = max(max_score, score);
        U factor = fast::exp(max_score - new_max);
        U exp_score = fast::exp(score - new_max);

        max_score = new_max;
        sum_exp_score = sum_exp_score * factor + exp_score;

        for (int j = 0; j < v_per_thread; j++) {
            o[j] = o[j] * factor + exp_score * vptr[j];
        }

        kptr += BN * D_SIZE;
        vptr += BN * V_SIZE;
    }

    if (simd_lid == 0) {
        max_scores[simd_gid] = max_score;
        sum_exp_scores[simd_gid] = sum_exp_score;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    max_score = max_scores[simd_lid];
    U new_max = simd_max(max_score);
    U factor = fast::exp(max_score - new_max);
    sum_exp_score = simd_sum(sum_exp_scores[simd_lid] * factor);

    for (int i = 0; i < v_per_thread; i++) {
        outputs[simd_lid * BD + simd_gid] = o[i];
        threadgroup_barrier(mem_flags::mem_threadgroup);
        o[i] = simd_sum(outputs[simd_gid * BD + simd_lid] * factor);
        o[i] = sum_exp_score == 0 ? o[i] : (o[i] / sum_exp_score);
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (simd_lid == 0) {
        for (int i = 0; i < v_per_thread; i++) {
            optr[i] = static_cast<T>(o[i]);
        }
    }
"""


_QWEN3_5_RAGGED_SDPA_TWO_PASS_1_SOURCE = r"""
    uint simd_lid = thread_index_in_simdgroup;
    uint kv_head_idx = threadgroup_position_in_grid.x;
    uint batch_idx = threadgroup_position_in_grid.y;
    uint block_idx = threadgroup_position_in_grid.z;
    uint gqa_idx = thread_position_in_threadgroup.y;

    constexpr int BD = 32;
    constexpr int qk_per_thread = D_SIZE / BD;
    constexpr int v_per_thread = V_SIZE / BD;

    typedef float U;
    thread U q[qk_per_thread];
    thread U o[v_per_thread] = {0};

    int K_SIZE = int(k_size[0]);
    int q_head_idx = int(GQA_FACTOR * kv_head_idx + gqa_idx);
    int q_batch_head_idx = int(batch_idx) * NUM_Q_HEADS + q_head_idx;
    int pad = int(pads[batch_idx]);
    int N = K_SIZE - pad;

    const device T* qptr =
        queries + q_batch_head_idx * D_SIZE + int(simd_lid) * qk_per_thread;
    const device T* kptr =
        keys + (int(batch_idx) * NUM_KV_HEADS + int(kv_head_idx)) *
                   K_SIZE * D_SIZE +
        (pad + int(block_idx)) * D_SIZE + int(simd_lid) * qk_per_thread;
    const device T* vptr =
        values + (int(batch_idx) * NUM_KV_HEADS + int(kv_head_idx)) *
                     K_SIZE * V_SIZE +
        (pad + int(block_idx)) * V_SIZE + int(simd_lid) * v_per_thread;
    device T* optr =
        partials + q_batch_head_idx * BLOCKS * V_SIZE +
        int(block_idx) * V_SIZE + int(simd_lid) * v_per_thread;
    device float* sump =
        sums + q_batch_head_idx * BLOCKS + int(block_idx);
    device float* maxp =
        maxs + q_batch_head_idx * BLOCKS + int(block_idx);

    U s = U(scale[0]);
    for (int i = 0; i < qk_per_thread; i++) {
        q[i] = s * qptr[i];
    }

    U max_score = -3.4028234663852886e38f;
    U sum_exp_score = 0;

    for (int i = int(block_idx); i < N; i += BLOCKS) {
        U score = 0;
        for (int j = 0; j < qk_per_thread; j++) {
            score += q[j] * kptr[j];
        }
        score = simd_sum(score);

        U new_max = max(max_score, score);
        U factor = fast::exp(max_score - new_max);
        U exp_score = fast::exp(score - new_max);

        max_score = new_max;
        sum_exp_score = sum_exp_score * factor + exp_score;

        for (int j = 0; j < v_per_thread; j++) {
            o[j] = o[j] * factor + exp_score * vptr[j];
        }

        kptr += BLOCKS * D_SIZE;
        vptr += BLOCKS * V_SIZE;
    }

    if (simd_lid == 0) {
        sump[0] = sum_exp_score;
        maxp[0] = max_score;
    }
    for (int i = 0; i < v_per_thread; i++) {
        optr[i] = static_cast<T>(o[i]);
    }
"""


_QWEN3_5_RAGGED_SDPA_TWO_PASS_2_SOURCE = r"""
    uint q_batch_head_idx = threadgroup_position_in_grid.y;
    uint simd_gid = simdgroup_index_in_threadgroup;
    uint simd_lid = thread_index_in_simdgroup;

    constexpr int BN = 32;
    constexpr int BD = 32;
    constexpr int elem_per_thread = D_SIZE / BD;

    typedef float U;
    thread U o[elem_per_thread] = {0};
    threadgroup U outputs[BN * BD];

    const device T* part =
        partials + int(q_batch_head_idx) * BLOCKS * D_SIZE +
        int(simd_gid) * D_SIZE + int(simd_lid) * elem_per_thread;
    const device float* sump = sums + int(q_batch_head_idx) * BLOCKS;
    const device float* maxp = maxs + int(q_batch_head_idx) * BLOCKS;
    device T* optr =
        out + int(q_batch_head_idx) * D_SIZE + int(simd_gid) * elem_per_thread;

    U sum_exp_score = 0.0;
    U max_score = -3.4028234663852886e38f;

    for (int b = 0; b < BLOCKS / BN; ++b) {
        max_score = max(max_score, maxp[int(simd_lid) + BN * b]);
    }
    max_score = simd_max(max_score);

    for (int b = 0; b < BLOCKS / BN; ++b) {
        U factor = fast::exp(maxp[int(simd_lid) + BN * b] - max_score);
        sum_exp_score += factor * sump[int(simd_lid) + BN * b];
    }
    sum_exp_score = simd_sum(sum_exp_score);

    for (int b = 0; b < BLOCKS / BN; ++b) {
        U factor = fast::exp(maxp[int(simd_gid)] - max_score);
        for (int i = 0; i < elem_per_thread; i++) {
            o[i] += factor * static_cast<U>(part[i]);
        }
        maxp += BN;
        sump += BN;
        part += BN * D_SIZE;
    }

    for (int i = 0; i < elem_per_thread; i++) {
        outputs[simd_lid * BD + simd_gid] = o[i];
        threadgroup_barrier(mem_flags::mem_threadgroup);
        o[i] = simd_sum(outputs[simd_gid * BD + simd_lid]);
        o[i] = sum_exp_score == 0 ? o[i] : (o[i] / sum_exp_score);
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (simd_lid == 0) {
        for (int i = 0; i < elem_per_thread; i++) {
            optr[i] = static_cast<T>(o[i]);
        }
    }
"""


@lru_cache(maxsize=1)
def _qwen3_5_device_arch_suffix() -> str:
    info = mx.device_info() if hasattr(mx, "device_info") else mx.metal.device_info()
    return str(info.get("architecture", ""))[-1:]


def _qwen3_5_sdpa_vector_blocks(seq_len: int, gqa_factor: int) -> int:
    devc = _qwen3_5_device_arch_suffix()
    n_simds = gqa_factor
    if devc == "s":
        blocks = 64
        if seq_len > 1024 and n_simds > 4:
            if seq_len <= 8192:
                blocks = 128
            elif seq_len <= 32768:
                blocks = 256
            elif seq_len <= 65536:
                blocks = 512
            else:
                blocks = 1024
        return blocks
    if devc == "d":
        blocks = 128
        if n_simds <= 2 and seq_len > 8192:
            blocks = 256
        elif n_simds >= 6:
            if 16384 <= seq_len < 65536:
                blocks = 512
            elif seq_len >= 65536:
                blocks = 1024
        return blocks
    if n_simds >= 4:
        return 64
    return 32


def _qwen3_5_sdpa_vector_plan(seq_len: int, q_heads: int, kv_heads: int):
    devc = _qwen3_5_device_arch_suffix()
    if (devc in {"d", "s"} and seq_len >= 1024) or (
        kv_heads < q_heads and seq_len >= 4096
    ):
        return ("two_pass", _qwen3_5_sdpa_vector_blocks(seq_len, q_heads // kv_heads))
    return ("one_pass", 0)


@lru_cache(maxsize=None)
def _qwen3_5_ragged_sdpa_one_pass_kernel(dtype, d_size, v_size):
    dtype_name = {mx.bfloat16: "bf16", mx.float16: "fp16"}.get(dtype, "unk")
    return mx.fast.metal_kernel(
        name=f"qwen3_5_ragged_sdpa_1p_{dtype_name}_d{d_size}_v{v_size}",
        input_names=["queries", "keys", "values", "pads", "scale", "k_size"],
        output_names=["out"],
        header="#include <metal_simdgroup>\nusing namespace metal;\n",
        source=_QWEN3_5_RAGGED_SDPA_ONE_PASS_SOURCE,
    )


@lru_cache(maxsize=None)
def _qwen3_5_ragged_sdpa_two_pass_1_kernel(dtype, d_size, v_size, blocks):
    dtype_name = {mx.bfloat16: "bf16", mx.float16: "fp16"}.get(dtype, "unk")
    return mx.fast.metal_kernel(
        name=(
            f"qwen3_5_ragged_sdpa_2p1_{dtype_name}_" f"d{d_size}_v{v_size}_b{blocks}"
        ),
        input_names=["queries", "keys", "values", "pads", "scale", "k_size"],
        output_names=["partials", "sums", "maxs"],
        header="#include <metal_simdgroup>\nusing namespace metal;\n",
        source=_QWEN3_5_RAGGED_SDPA_TWO_PASS_1_SOURCE,
    )


@lru_cache(maxsize=None)
def _qwen3_5_ragged_sdpa_two_pass_2_kernel(dtype, v_size, blocks):
    dtype_name = {mx.bfloat16: "bf16", mx.float16: "fp16"}.get(dtype, "unk")
    return mx.fast.metal_kernel(
        name=f"qwen3_5_ragged_sdpa_2p2_{dtype_name}_v{v_size}_b{blocks}",
        input_names=["partials", "sums", "maxs"],
        output_names=["out"],
        header="#include <metal_simdgroup>\nusing namespace metal;\n",
        source=_QWEN3_5_RAGGED_SDPA_TWO_PASS_2_SOURCE,
    )


@lru_cache(maxsize=128)
def _qwen3_5_cached_i32_array(values):
    return mx.array(values, dtype=mx.int32)


@lru_cache(maxsize=128)
def _qwen3_5_cached_sdpa_scalars(scale: float, k_size: int):
    return (
        mx.array([scale], dtype=mx.float32),
        mx.array([k_size], dtype=mx.int32),
    )


# Threads per threadgroup the one-pass kernel and the second pass of the
# two-pass plan are written for. Both assume exactly 32 SIMD groups of 32 lanes
# (BN == BD == 32): the block reduction and the threadgroup transpose index off
# simd_gid over [0, 32), so this cannot simply be lowered without reworking them.
_QWEN3_5_SDPA_THREADS = 1024

# Metal caps threads-per-threadgroup per compiled *pipeline*, not per device --
# a kernel holding more registers live gets a lower ceiling than the device
# maximum. At D_SIZE == 256 the two-pass reduction keeps elem_per_thread == 8
# floats per thread, and on applegpu_g14d (M2 Ultra) that compiles to a ceiling
# of 896 threads, so requesting 1024 raises:
#
#   ValueError: Thread group size (1024) is greater than the maximum allowed
#   threads per threadgroup (896).
#
# D_SIZE is what moves it: 64/96/128 launch fine on the same GPU, and the same
# 256 launches fine on applegpu_g16s (M4 Max). Nothing in the shapes predicts
# which parts are affected, and the error is raised lazily
# when the command buffer is built rather than when the kernel is called, so it
# cannot be caught around the call site. Probe each kernel signature once with a
# forced eval, cache the verdict, and route around whichever launches this GPU
# rejects: two-pass degrades to one-pass, one-pass degrades to the caller's
# portable per-pad-group fallback. No-op wherever 1024 threads are legal.
_QWEN3_5_SDPA_LAUNCHABLE = {}


def _qwen3_5_try_launch(key, launch):
    """Run ``launch()``, or return None if this GPU rejects that launch.

    ``key`` must capture everything that determines the compiled pipeline (the
    kernel and its template arguments), so the verdict is cached per pipeline
    rather than per call.
    """
    verdict = _QWEN3_5_SDPA_LAUNCHABLE.get(key)
    if verdict is False:
        return None
    try:
        outputs = launch()
        if verdict is None:
            # First use of this pipeline: force it now so an illegal
            # threadgroup surfaces here instead of at the caller's next eval.
            mx.eval(outputs)
            _QWEN3_5_SDPA_LAUNCHABLE[key] = True
        return outputs
    except ValueError:
        _QWEN3_5_SDPA_LAUNCHABLE[key] = False
        return None


def _qwen3_5_ragged_decode_attention(
    queries: mx.array,
    keys: mx.array,
    values: mx.array,
    pads: List[int],
    scale: float,
) -> Optional[mx.array]:
    # Metal-only fast path; on other backends (e.g. CUDA) return None so the
    # caller falls back to portable per-pad-group scaled_dot_product_attention.
    if not mx.metal.is_available():
        return None
    if (
        queries.ndim != 4
        or keys.ndim != 4
        or values.ndim != 4
        or queries.shape[2] != 1
        or queries.dtype not in (mx.bfloat16, mx.float16)
        or keys.dtype != queries.dtype
        or values.dtype != queries.dtype
    ):
        return None

    batch, q_heads, _, d_size = queries.shape
    pads = tuple(int(p) for p in pads)
    if len(pads) != batch or any(p < 0 for p in pads):
        return None
    kv_heads = keys.shape[1]
    k_size = keys.shape[2]
    v_size = values.shape[-1]
    if (
        q_heads % kv_heads != 0
        or d_size != v_size
        or d_size not in (64, 96, 128, 256)
        or any(p >= k_size for p in pads)
    ):
        return None

    plans = [_qwen3_5_sdpa_vector_plan(k_size - pad, q_heads, kv_heads) for pad in pads]
    if len(set(plans)) != 1:
        return None
    mode, blocks = plans[0]

    queries = mx.contiguous(queries)
    keys = mx.contiguous(keys)
    values = mx.contiguous(values)
    pads_array = _qwen3_5_cached_i32_array(pads)
    scale_array, k_size_array = _qwen3_5_cached_sdpa_scalars(float(scale), int(k_size))
    template = [
        ("T", queries.dtype),
        ("D_SIZE", int(d_size)),
        ("V_SIZE", int(v_size)),
        ("NUM_Q_HEADS", int(q_heads)),
        ("NUM_KV_HEADS", int(kv_heads)),
        ("GQA_FACTOR", int(q_heads // kv_heads)),
    ]

    signature = (queries.dtype, int(d_size), int(v_size), int(q_heads), int(kv_heads))

    if mode == "two_pass":
        kernel_1 = _qwen3_5_ragged_sdpa_two_pass_1_kernel(
            queries.dtype, d_size, v_size, blocks
        )
        pass_1 = _qwen3_5_try_launch(
            ("two_pass_1", *signature, int(blocks)),
            lambda: kernel_1(
                inputs=[queries, keys, values, pads_array, scale_array, k_size_array],
                template=[*template, ("BLOCKS", int(blocks))],
                grid=(32 * kv_heads, (q_heads // kv_heads) * batch, blocks),
                threadgroup=(32, q_heads // kv_heads, 1),
                output_shapes=[
                    (batch, q_heads, 1, blocks, v_size),
                    (batch, q_heads, 1, blocks),
                    (batch, q_heads, 1, blocks),
                ],
                output_dtypes=[queries.dtype, mx.float32, mx.float32],
            ),
        )
        if pass_1 is not None:
            partials, sums, maxs = pass_1
            kernel_2 = _qwen3_5_ragged_sdpa_two_pass_2_kernel(
                queries.dtype, v_size, blocks
            )
            pass_2 = _qwen3_5_try_launch(
                ("two_pass_2", queries.dtype, int(v_size), int(blocks)),
                lambda: kernel_2(
                    inputs=[partials, sums, maxs],
                    template=[
                        ("T", queries.dtype),
                        ("D_SIZE", int(v_size)),
                        ("BLOCKS", int(blocks)),
                    ],
                    grid=(_QWEN3_5_SDPA_THREADS, batch * q_heads, 1),
                    threadgroup=(_QWEN3_5_SDPA_THREADS, 1, 1),
                    output_shapes=[(batch, q_heads, 1, v_size)],
                    output_dtypes=[queries.dtype],
                ),
            )
            if pass_2 is not None:
                return pass_2[0]
        # This GPU will not launch the two-pass kernels at this size. The
        # one-pass kernel computes the same attention in a single dispatch, so
        # prefer it over dropping out of the fast path entirely.

    kernel = _qwen3_5_ragged_sdpa_one_pass_kernel(queries.dtype, d_size, v_size)
    one_pass = _qwen3_5_try_launch(
        ("one_pass", *signature),
        lambda: kernel(
            inputs=[queries, keys, values, pads_array, scale_array, k_size_array],
            template=template,
            grid=(_QWEN3_5_SDPA_THREADS, batch * q_heads, 1),
            threadgroup=(_QWEN3_5_SDPA_THREADS, 1, 1),
            output_shapes=[(batch, q_heads, 1, v_size)],
            output_dtypes=[queries.dtype],
        ),
    )
    return None if one_pass is None else one_pass[0]


def _qwen3_5_left_padded_attention(
    queries: mx.array,
    keys: mx.array,
    values: mx.array,
    *,
    cache,
    scale: float,
    mask: Optional[mx.array],
) -> Optional[mx.array]:
    if hasattr(cache, "bits") or queries.ndim != 4 or keys.ndim != 4:
        return None

    pads = getattr(cache, "_qwen3_5_decode_left_padding", None)
    if pads is None:
        left_padding_info = _qwen3_5_left_padding_info(cache)
        if left_padding_info is None or left_padding_info[1] <= 0:
            return None
        pads = list(left_padding_info[0])
    if max(pads) <= 0:
        return None

    output = _qwen3_5_ragged_decode_attention(queries, keys, values, pads, scale)
    if output is not None:
        return output

    row_outputs = {}
    for pad in sorted(set(pads)):
        rows = [i for i, row_pad in enumerate(pads) if row_pad == pad]
        row_idx = mx.array(rows, dtype=mx.int32)
        group_queries = mx.take(queries, row_idx, axis=0)
        group_keys = mx.take(keys, row_idx, axis=0)[:, :, pad:, :]
        group_values = mx.take(values, row_idx, axis=0)[:, :, pad:, :]

        if group_queries.shape[2] > 1:
            prefix_len = group_keys.shape[-2] - group_queries.shape[2]
            group_output = mx.concatenate(
                [
                    scaled_dot_product_attention(
                        group_queries[:, :, i : i + 1, :],
                        group_keys[:, :, : prefix_len + i + 1, :],
                        group_values[:, :, : prefix_len + i + 1, :],
                        cache=None,
                        scale=scale,
                        mask=None,
                    )
                    for i in range(group_queries.shape[2])
                ],
                axis=2,
            )
        else:
            group_output = scaled_dot_product_attention(
                group_queries,
                group_keys,
                group_values,
                cache=None,
                scale=scale,
                mask=None,
            )

        for j, row in enumerate(rows):
            row_outputs[row] = group_output[j : j + 1]

    return mx.concatenate([row_outputs[i] for i in range(queries.shape[0])], axis=0)


class Qwen3_5Attention(nn.Module):
    def __init__(self, args: TextConfig):
        super().__init__()
        self.num_key_value_heads = args.num_key_value_heads
        self.num_attention_heads = args.num_attention_heads
        self.head_dim = args.head_dim
        self.scale = self.head_dim**-0.5

        self.q_proj = nn.Linear(
            args.hidden_size,
            self.num_attention_heads * self.head_dim * 2,
            bias=args.attention_bias,
        )
        self.k_proj = nn.Linear(
            args.hidden_size,
            self.num_key_value_heads * self.head_dim,
            bias=args.attention_bias,
        )
        self.v_proj = nn.Linear(
            args.hidden_size,
            self.num_key_value_heads * self.head_dim,
            bias=args.attention_bias,
        )
        self.o_proj = nn.Linear(
            self.num_attention_heads * self.head_dim,
            args.hidden_size,
            bias=args.attention_bias,
        )

        self.q_norm = nn.RMSNorm(self.head_dim, eps=args.rms_norm_eps)
        self.k_norm = nn.RMSNorm(self.head_dim, eps=args.rms_norm_eps)

        self.rotary_emb = Qwen3_5RotaryEmbedding(
            int(self.head_dim * args.rope_parameters["partial_rotary_factor"]),
            max_position_embeddings=args.max_position_embeddings,
            base=args.rope_parameters["rope_theta"],
            mrope_section=args.rope_parameters["mrope_section"],
        )

    def __call__(
        self,
        x: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional[Any] = None,
        position_ids: Optional[mx.array] = None,
        position_embeddings: Optional[tuple[mx.array, mx.array]] = None,
    ) -> mx.array:
        B, L, D = x.shape
        q_proj_output, keys, values = (
            self.q_proj(x),
            self.k_proj(x),
            self.v_proj(x),
        )
        queries, keys, values, gate, mask = self._prepare_projected_qkv(
            q_proj_output,
            keys,
            values,
            cache,
            position_ids,
            position_embeddings,
            mask,
        )

        left_padded_decode = (
            mask == "left_padded_decode" if isinstance(mask, str) else False
        )
        if left_padded_decode:
            mask = None
            output = _qwen3_5_left_padded_attention(
                queries, keys, values, cache=cache, scale=self.scale, mask=mask
            )
        else:
            output = None

        if output is None:
            output = scaled_dot_product_attention(
                queries, keys, values, cache=cache, scale=self.scale, mask=mask
            )
        output = output.transpose(0, 2, 1, 3).reshape(B, L, -1)

        return self.o_proj(output * mx.sigmoid(gate))

    def _prepare_projected_qkv(
        self,
        q_proj_output,
        keys,
        values,
        cache,
        position_ids,
        position_embeddings,
        mask,
    ):
        B, L, _ = q_proj_output.shape
        queries, gate = mx.split(
            q_proj_output.reshape(B, L, self.num_attention_heads, -1), 2, axis=-1
        )
        gate = gate.reshape(B, L, -1)

        queries = self.q_norm(queries).transpose(0, 2, 1, 3)
        keys = self.k_norm(keys.reshape(B, L, self.num_key_value_heads, -1)).transpose(
            0, 2, 1, 3
        )
        values = values.reshape(B, L, self.num_key_value_heads, -1).transpose(
            0, 2, 1, 3
        )

        kv_seq_len = keys.shape[-2]

        if position_ids is None:
            cache_offset = cache.offset
            if isinstance(cache_offset, mx.array) and cache_offset.ndim > 0:
                offsets = mx.maximum(cache_offset[:B], 0)
                kv_seq_len = kv_seq_len + offsets + 1
                position_ids = offsets[:, None] + mx.arange(L)[None, :]
                position_ids = mx.expand_dims(position_ids, axis=0)
                position_ids = mx.tile(position_ids, (3, 1, 1))
            else:
                if isinstance(cache_offset, mx.array):
                    cache_offset = int(cache_offset.item())
                kv_seq_len += cache_offset + 1
                position_ids = mx.arange(cache_offset, cache_offset + L)
                position_ids = mx.expand_dims(position_ids, axis=0)
                position_ids = mx.tile(position_ids, (3, 1, 1))
        else:
            kv_seq_len += cache.offset + 1 if cache is not None else 0

        if position_embeddings is None:
            queries, keys = self.rotary_emb.apply_rotary(
                queries,
                keys,
                position_ids,
                unsqueeze_dim=1,
            )
        else:
            cos, sin = position_embeddings
            queries, keys = apply_multimodal_rotary_pos_emb(queries, keys, cos, sin)

        if mask is not None and isinstance(mask, mx.array):
            if (
                cache is not None
                and hasattr(cache, "_idx")
                and hasattr(cache, "left_padding")
            ):
                kv_seq_len = int(cache._idx) + L
            elif isinstance(kv_seq_len, mx.array):
                kv_seq_len = kv_seq_len.max().item()
            mask = mask[..., : int(kv_seq_len)]

        if cache is not None:
            keys, values = cache.update_and_fetch(keys, values)
        return queries, keys, values, gate, mask


class Qwen3_5MLP(nn.Module):
    def __init__(self, dim, hidden_dim):
        super().__init__()
        self.gate_proj = nn.Linear(dim, hidden_dim, bias=False)
        self.down_proj = nn.Linear(hidden_dim, dim, bias=False)
        self.up_proj = nn.Linear(dim, hidden_dim, bias=False)

    def __call__(self, x) -> mx.array:
        return self.down_proj(swiglu(self.gate_proj(x), self.up_proj(x)))


class Qwen3_5GatedDeltaNet(nn.Module):
    def __init__(self, config: TextConfig):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.num_v_heads = config.linear_num_value_heads
        self.num_k_heads = config.linear_num_key_heads
        self.head_k_dim = config.linear_key_head_dim
        self.head_v_dim = config.linear_value_head_dim
        self.key_dim = self.head_k_dim * self.num_k_heads
        self.value_dim = self.head_v_dim * self.num_v_heads
        if self.num_v_heads % self.num_k_heads != 0:
            raise ValueError(
                f"num_v_heads ({self.num_v_heads}) must be divisible by num_k_heads ({self.num_k_heads})"
            )

        self.conv_kernel_size = config.linear_conv_kernel_dim
        self.layer_norm_epsilon = config.rms_norm_eps

        self.conv_dim = self.key_dim * 2 + self.value_dim
        self.conv1d = nn.Conv1d(
            in_channels=self.conv_dim,
            out_channels=self.conv_dim,
            bias=False,
            kernel_size=self.conv_kernel_size,
            groups=self.conv_dim,
            padding=0,
        )

        self.in_proj_qkv = nn.Linear(
            self.hidden_size, self.key_dim * 2 + self.value_dim, bias=False
        )
        self.in_proj_z = nn.Linear(self.hidden_size, self.value_dim, bias=False)
        self.in_proj_b = nn.Linear(self.hidden_size, self.num_v_heads, bias=False)
        self.in_proj_a = nn.Linear(self.hidden_size, self.num_v_heads, bias=False)

        self.dt_bias = mx.ones(self.num_v_heads)

        A = mx.random.uniform(low=0, high=16, shape=(self.num_v_heads,))
        self.A_log = mx.log(A)

        self.norm = Qwen3_5RMSNormGated(self.head_v_dim, eps=self.layer_norm_epsilon)

        self.out_proj = nn.Linear(self.value_dim, self.hidden_size, bias=False)

    def _causal_conv1d_decode(self, conv_input: mx.array) -> mx.array:
        cached = getattr(self, "_qwen3_5_decode_conv_weight", None)
        cache_key = id(self.conv1d.weight)
        if cached is None or cached[0] != cache_key:
            weight = self.conv1d.weight[:, :, 0].T.astype(mx.float32)
            mx.eval(weight)
            cached = (cache_key, weight)
            self._qwen3_5_decode_conv_weight = cached

        weight = cached[1]
        return _qwen3_5_decode_depthwise_conv(conv_input, weight)

    def _normalize_qk(self, q: mx.array, k: mx.array):
        inv_scale = k.shape[-1] ** -0.5
        q = (inv_scale**2) * mx.fast.rms_norm(q, None, 1e-6)
        k = inv_scale * mx.fast.rms_norm(k, None, 1e-6)
        return q, k

    def __call__(
        self,
        inputs: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional[Any] = None,
    ) -> mx.array:
        B, S, _ = inputs.shape
        mixed_qkv, z, b, a = (
            self.in_proj_qkv(inputs),
            self.in_proj_z(inputs),
            self.in_proj_b(inputs),
            self.in_proj_a(inputs),
        )

        z = z.reshape(B, S, -1, self.head_v_dim)

        if cache is not None and cache[0] is not None:
            conv_state = cache[0]
            if conv_state.shape[0] != B:
                conv_state = mx.zeros(
                    (B, self.conv_kernel_size - 1, self.conv_dim),
                    dtype=inputs.dtype,
                )
        else:
            conv_state = mx.zeros(
                (B, self.conv_kernel_size - 1, self.conv_dim),
                dtype=inputs.dtype,
            )

        if mask is not None:
            if mask.shape[0] != B:
                mask = None
            else:
                mixed_qkv = mx.where(mask[..., None], mixed_qkv, 0)
        conv_input = mx.concatenate([conv_state, mixed_qkv], axis=1)
        if cache is not None:
            n_keep = self.conv_kernel_size - 1
            if getattr(cache, "lengths", None) is not None:
                ends = mx.clip(cache.lengths, 0, S)
                positions = (ends[:, None] + mx.arange(n_keep))[..., None]
                cache[0] = mx.take_along_axis(conv_input, positions, axis=1)
            else:
                cache[0] = mx.contiguous(conv_input[:, -n_keep:, :])
        if (
            S == 1
            and conv_input.shape[1] == self.conv_kernel_size
            and self.conv1d.weight.dtype in (mx.bfloat16, mx.float16)
        ):
            conv_out = nn.silu(self._causal_conv1d_decode(conv_input))
        else:
            conv_out = nn.silu(self.conv1d(conv_input))

        q, k, v = [
            t.reshape(B, S, h, d)
            for t, h, d in zip(
                mx.split(conv_out, [self.key_dim, 2 * self.key_dim], -1),
                [self.num_k_heads, self.num_k_heads, self.num_v_heads],
                [self.head_k_dim, self.head_k_dim, self.head_v_dim],
            )
        ]

        state = cache[1] if cache else None
        if state is not None and state.shape[0] != B:
            state = None
        q, k = self._normalize_qk(q, k)

        out, state = gated_delta_update(
            q,
            k,
            v,
            a,
            b,
            self.A_log,
            self.dt_bias,
            state,
            mask,
            use_kernel=not self.training,
        )

        if cache is not None:
            cache[1] = state
            if hasattr(cache, "advance"):
                cache.advance(S)
                _qwen3_5_advance_left_padding_info(cache, S)
                _qwen3_5_advance_lengths_info(cache, S)

        out = self.norm(out, z)
        return self.out_proj(out.reshape(B, S, -1))


class Qwen3_5DecoderLayer(nn.Module):
    def __init__(self, args: TextConfig, layer_idx: int):
        super().__init__()
        self.is_linear = (layer_idx + 1) % args.full_attention_interval != 0
        if self.is_linear:
            self.linear_attn = Qwen3_5GatedDeltaNet(args)
        else:
            self.self_attn = Qwen3_5Attention(args)

        self.input_layernorm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)
        self.post_attention_layernorm = nn.RMSNorm(
            args.hidden_size, eps=args.rms_norm_eps
        )
        self.mlp = Qwen3_5MLP(args.hidden_size, args.intermediate_size)

    def __call__(
        self,
        x: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional[Any] = None,
        position_ids: Optional[mx.array] = None,
        position_embeddings: Optional[tuple[mx.array, mx.array]] = None,
    ) -> mx.array:
        if self.is_linear:
            r = self.linear_attn(
                self.input_layernorm(x),
                mask,
                cache,
            )
        else:
            r = self.self_attn(
                self.input_layernorm(x),
                mask=mask,
                cache=cache,
                position_ids=position_ids,
                position_embeddings=position_embeddings,
            )
        h = x + r
        return h + self.mlp(self.post_attention_layernorm(h))


class Qwen3_5Model(nn.Module):
    def __init__(self, args: TextConfig):
        super().__init__()
        self.args = args
        self.embed_tokens = nn.Embedding(args.vocab_size, args.hidden_size)
        self.layers = [
            Qwen3_5DecoderLayer(args=args, layer_idx=i)
            for i in range(args.num_hidden_layers)
        ]
        self.norm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)
        self.ssm_idx = 0
        self.fa_idx = args.full_attention_interval - 1

    def __call__(
        self,
        inputs: mx.array,
        inputs_embeds: Optional[mx.array] = None,
        mask: Optional[mx.array] = None,
        cache=None,
        position_ids: Optional[mx.array] = None,
        capture_layer_ids: Optional[List[int]] = None,
        hidden_sink: Optional[list] = None,
    ):
        if inputs_embeds is None:
            h = self.embed_tokens(inputs)
        else:
            h = inputs_embeds

        if cache is None:
            cache = [None] * len(self.layers)

        fa_cache = cache[self.fa_idx]
        if (
            h.shape[0] == 1
            and hidden_sink is None
            and fa_cache is not None
            and _is_single_row_batch_cache(fa_cache)
        ):
            row_cache = []
            for cache_entry in cache:
                if cache_entry is None:
                    row_cache.append(None)
                elif _is_single_row_batch_cache(cache_entry):
                    row_cache.append(_extract_row_cache(cache_entry, 0))
                else:
                    row_cache.append(cache_entry)

            row_out = self(
                inputs,
                inputs_embeds=h,
                cache=row_cache,
                position_ids=position_ids,
            )
            for i, cache_entry in enumerate(row_cache):
                if cache[i] is None or cache_entry is None:
                    continue
                if hasattr(cache[i].__class__, "merge"):
                    cache[i] = cache[i].__class__.merge([cache_entry])
            return row_out

        if (
            h.shape[0] > 1
            and h.shape[1] > 1
            and hidden_sink is None
            and fa_cache is not None
            and hasattr(fa_cache, "extract")
            and hasattr(fa_cache.__class__, "merge")
            and isinstance(getattr(fa_cache, "offset", None), mx.array)
            and fa_cache.offset.ndim > 0
        ):
            query_left_padding = mx.minimum(mx.maximum(-fa_cache.offset, 0), h.shape[1])
            cache_left_padding = getattr(fa_cache, "left_padding", None)
            has_left_padding = (
                isinstance(cache_left_padding, mx.array)
                and cache_left_padding.ndim > 0
                and int(cache_left_padding.max().item()) > 0
            )
            if has_left_padding or int(query_left_padding.max().item()) > 0:
                row_outputs = []
                row_caches = [[] for _ in cache]
                batch_offsets = []
                for cache_entry in cache:
                    offsets = getattr(cache_entry, "offset", None)
                    if (
                        isinstance(offsets, mx.array)
                        and offsets.ndim > 0
                        and offsets.size >= h.shape[0]
                    ):
                        batch_offsets.append(offsets[: h.shape[0]])
                    else:
                        batch_offsets.append(None)
                for row, pad in enumerate(query_left_padding.tolist()):
                    pad = min(max(int(pad), 0), h.shape[1])
                    current_cache = []
                    for cache_entry in cache:
                        if cache_entry is None:
                            current_cache.append(None)
                        else:
                            current_cache.append(_extract_row_cache(cache_entry, row))
                    if pad == h.shape[1]:
                        row_outputs.append(mx.zeros_like(h[row : row + 1]))
                        for i, cache_entry in enumerate(current_cache):
                            row_caches[i].append(cache_entry)
                        continue
                    row_inputs = inputs[row : row + 1, pad:]
                    row_embeds = h[row : row + 1, pad:]
                    row_position_ids = None
                    if position_ids is not None:
                        if position_ids.ndim == 2:
                            row_position_ids = position_ids[row : row + 1, pad:]
                        else:
                            row_position_ids = position_ids[:, row : row + 1, pad:]

                    row_out = self(
                        row_inputs,
                        inputs_embeds=row_embeds,
                        cache=current_cache,
                        position_ids=row_position_ids,
                    )
                    if pad > 0:
                        row_out = _pad_row_time(row_out, pad, h.shape[1])
                    row_outputs.append(row_out)
                    for i, cache_entry in enumerate(current_cache):
                        row_caches[i].append(cache_entry)

                for i, entries in enumerate(row_caches):
                    if cache[i] is None:
                        continue
                    if hasattr(cache[i].__class__, "merge"):
                        cache[i] = _restore_batch_padding_metadata(
                            cache[i].__class__.merge(entries),
                            batch_offsets[i],
                            h.shape[1],
                        )
                return mx.concatenate(row_outputs, axis=0)

        fa_mask = _create_qwen3_5_attention_mask(h, cache[self.fa_idx])
        ssm_mask = _create_qwen3_5_ssm_mask(h, cache[self.ssm_idx])
        decode_left_padding = (
            getattr(cache[self.fa_idx], "_qwen3_5_decode_left_padding", None)
            if isinstance(fa_mask, str) and fa_mask == "left_padded_decode"
            else None
        )
        _set_qwen3_5_decode_left_padding(cache, self.layers, decode_left_padding)

        position_embeddings = None
        if position_ids is not None:
            for layer in self.layers:
                if not layer.is_linear:
                    if not layer.self_attn.rotary_emb.fused_apply:
                        position_embeddings = layer.self_attn.rotary_emb(
                            h, position_ids
                        )
                    break

        capture_set = set(capture_layer_ids) if capture_layer_ids else set()
        for i, (layer, c) in enumerate(zip(self.layers, cache)):
            mask = ssm_mask if layer.is_linear else fa_mask
            h = layer(
                h,
                mask=mask,
                cache=c,
                position_ids=position_ids,
                position_embeddings=position_embeddings,
            )
            if hidden_sink is not None and i in capture_set:
                hidden_sink.append(h)

        return self.norm(h)


class LanguageModel(nn.Module):
    requires_uniform_batch_acceptance = True

    def __init__(self, args: TextConfig, config: ModelConfig = None):
        super().__init__()
        self.args = args
        self.config = config
        self.model_type = args.model_type
        self.model = Qwen3_5Model(args)
        self._position_ids = None
        self._rope_deltas = None

        if not args.tie_word_embeddings:
            self.lm_head = nn.Linear(args.hidden_size, args.vocab_size, bias=False)

    def chunked_prefill_policy(
        self,
        *,
        input_ids=None,
        inputs_embeds=None,
        prompt_cache=None,
        draft_model=None,
        draft_kind=None,
        prefill_kwargs=None,
    ) -> bool:
        del input_ids, inputs_embeds, prompt_cache
        prefill_kwargs = prefill_kwargs or {}
        if draft_model is None:
            return True
        if draft_kind == "mtp":
            return bool(prefill_kwargs.get("return_hidden", False)) and bool(
                prefill_kwargs.get("return_shared_kv", False)
            )
        if draft_kind in ("dflash", "eagle3"):
            return prefill_kwargs.get("capture_layer_ids") is not None
        return draft_kind is None

    def rollback_speculative_cache(
        self,
        caches: List[Any],
        gdn_states: List,
        accepted,
        block_size: int,
    ) -> int:
        if isinstance(accepted, int):
            accepted_list = [int(accepted)]
        elif isinstance(accepted, mx.array):
            accepted_list = [int(x) for x in accepted.reshape(-1).tolist()]
        else:
            accepted_list = [int(x) for x in accepted]

        max_a = max(accepted_list)
        n = max_a + 1
        trim = block_size - n
        is_batch = len(accepted_list) > 1
        valid_ends_list = [a + 1 for a in accepted_list]
        accepted_mx = None
        valid_ends_mx = None

        def accepted_array():
            nonlocal accepted_mx
            if accepted_mx is None:
                accepted_mx = mx.array(accepted_list, dtype=mx.int32)
            return accepted_mx

        def valid_ends_array():
            nonlocal valid_ends_mx
            if valid_ends_mx is None:
                valid_ends_mx = mx.array(valid_ends_list, dtype=mx.int32)
            return valid_ends_mx

        def _is_ssm_cache(c):
            return not c.is_trimmable() and not hasattr(c, "zero_row_tail")

        ssm_caches = []
        for c in caches:
            if c is None:
                continue
            if _is_ssm_cache(c):
                ssm_caches.append(c)
                continue
            if c.is_trimmable() and trim > 0:
                c.trim(trim)
            right_trimmed = False
            if is_batch and max_a > 0:
                extra_trim_list = [max_a - a for a in accepted_list]
                if any(extra_trim_list):
                    prepare = getattr(c, "prepare", None)
                    finalize = getattr(c, "finalize", None)
                    if c.keys is not None and callable(prepare) and callable(finalize):
                        prepare(right_padding=extra_trim_list)
                        finalize()
                        right_trimmed = True
            if (
                is_batch
                and not right_trimmed
                and hasattr(c, "_idx")
                and c.keys is not None
                and max_a > 0
            ):
                kv_len = c._idx
                verify_start = kv_len - n
                if any(verify_start + ve < kv_len for ve in valid_ends_list):
                    raise RuntimeError(
                        "Qwen3.5 batched speculative rollback requires uniform "
                        f"per-row acceptance; got ragged accepts {accepted_list}. "
                        "Zeroing a rejected row's KV tail leaves phantom keys "
                        "attended (issue #1962); set "
                        "requires_uniform_batch_acceptance on the drafter or target "
                        "so accepts are clamped before rollback."
                    )

        if not ssm_caches:
            return max_a

        if all(len(s) > 11 and s[11] is not None for s in gdn_states):
            a0 = accepted_list[0] if not is_batch else None
            if is_batch:
                intermediate_parts = []
                conv_input_parts = []
                live_state_parts = []
                live_conv_parts = []
                layer_batch_sizes = []
                kernel_sizes = []

                for j, c in enumerate(ssm_caches):
                    (
                        _q,
                        _k,
                        _v,
                        _a,
                        _b,
                        _A_log,
                        _dt_bias,
                        _init_state,
                        _mask,
                        conv_input,
                        K,
                        intermediate_states,
                        *_,
                    ) = gdn_states[j]
                    rows = intermediate_states.shape[0]
                    layer_batch_sizes.append(rows)
                    kernel_sizes.append(int(K))
                    intermediate_parts.append(intermediate_states)
                    conv_input_parts.append(conv_input)

                    live_state = c[1]
                    if live_state is None:
                        live_state = mx.zeros(
                            (
                                rows,
                                intermediate_states.shape[2],
                                intermediate_states.shape[3],
                                intermediate_states.shape[4],
                            ),
                            dtype=intermediate_states.dtype,
                        )
                    live_state_parts.append(live_state)

                    live_conv = c[0]
                    if live_conv is None:
                        live_conv = mx.zeros(
                            (rows, int(K) - 1, conv_input.shape[-1]),
                            dtype=conv_input.dtype,
                        )
                    live_conv_parts.append(live_conv)

                if len(set(kernel_sizes)) != 1:
                    raise ValueError("Qwen GDN layers must share conv kernel size.")

                accepted_mx = accepted_array()
                accepted_bat = mx.concatenate([accepted_mx for _ in ssm_caches], axis=0)
                state_bat, conv_bat = gated_delta_accept_states(
                    mx.concatenate(intermediate_parts, axis=0),
                    mx.concatenate(conv_input_parts, axis=0),
                    mx.concatenate(live_state_parts, axis=0),
                    mx.concatenate(live_conv_parts, axis=0),
                    accepted_bat,
                    kernel_sizes[0],
                    use_kernel=True,
                )

                offset = 0
                for c, rows in zip(ssm_caches, layer_batch_sizes):
                    c[1] = state_bat[offset : offset + rows]
                    c[0] = conv_bat[offset : offset + rows]
                    offset += rows
            else:
                for j, c in enumerate(ssm_caches):
                    (
                        _q,
                        _k,
                        _v,
                        _a,
                        _b,
                        _A_log,
                        _dt_bias,
                        _init_state,
                        _mask,
                        conv_input,
                        K,
                        intermediate_states,
                        *_,
                    ) = gdn_states[j]
                    if a0 < intermediate_states.shape[1]:
                        c[1] = intermediate_states[:, a0]
                        c[0] = conv_input[:, a0 + 1 : a0 + K]
            return max_a

        # Batch all SSM rollbacks into a single state-only gated delta kernel.
        # Rollback does not need the layer output, so this avoids the verifier
        # replay's q projection and y materialization.
        N = len(ssm_caches)

        k_list, v_list, a_list, b_list = [], [], [], []
        A_log_list, dt_bias_list, state_list = [], [], []
        steps_list = []
        mask_parts = []
        layer_batch_sizes = []
        conv_data = []
        for j in range(N):
            (
                _q,
                k,
                v,
                a,
                b,
                A_log,
                dt_bias,
                init_state,
                mask,
                conv_input,
                K,
                *_,
            ) = gdn_states[j]
            k = k[:, :n]
            v = v[:, :n]
            a = a[:, :n]
            b = b[:, :n]
            batch_rows = k.shape[0]
            k_list.append(k)
            v_list.append(v)
            a_list.append(a)
            b_list.append(b)
            if is_batch:
                steps_list.append(valid_ends_array())
            else:
                steps_list.append(mx.full((batch_rows,), n, dtype=mx.int32))
            A_log_list.append(
                mx.broadcast_to(A_log[None, None, :], (batch_rows, 1, A_log.shape[0]))
            )
            dt_bias_list.append(
                mx.broadcast_to(
                    dt_bias[None, None, :], (batch_rows, 1, dt_bias.shape[0])
                )
            )
            if init_state is None:
                init_state = mx.zeros(
                    (batch_rows, v.shape[-2], v.shape[-1], k.shape[-1]),
                    dtype=mx.float32,
                )
            state_list.append(init_state)
            layer_batch_sizes.append(batch_rows)
            conv_data.append((conv_input, K))
            mask_parts.append(None if mask is None else mask[:, :n])

        # Stack along batch dim: (N, n, H, D) — one kernel launch for all layers.
        k_bat = mx.concatenate(k_list, axis=0)
        v_bat = mx.concatenate(v_list, axis=0)
        a_bat = mx.concatenate(a_list, axis=0)
        b_bat = mx.concatenate(b_list, axis=0)
        A_log_bat = mx.concatenate(A_log_list, axis=0)  # (N, 1, Hv)
        dt_bias_bat = mx.concatenate(dt_bias_list, axis=0)  # (N, 1, Hv)
        state_bat = mx.concatenate(state_list, axis=0)  # (N, Hv, Dv, Dk)
        steps_bat = mx.concatenate(steps_list, axis=0)

        replay_mask = None
        if any(mask is not None for mask in mask_parts):
            replay_mask = mx.concatenate(
                [
                    (mask if mask is not None else mx.ones((rows, n), dtype=mx.bool_))
                    for mask, rows in zip(mask_parts, layer_batch_sizes)
                ],
                axis=0,
            )

        states_out = gated_delta_state_update(
            k_bat,
            v_bat,
            a_bat,
            b_bat,
            A_log_bat,
            dt_bias_bat,
            state_bat,
            steps_bat,
            replay_mask,
            use_kernel=True,
        )

        # Scatter results back to individual caches.
        a0 = accepted_list[0] if not is_batch else None
        state_offset = 0
        for j, c in enumerate(ssm_caches):
            batch_rows = layer_batch_sizes[j]
            c[1] = states_out[state_offset : state_offset + batch_rows]
            state_offset += batch_rows
            conv_input, K = conv_data[j]
            if is_batch:
                slices = [
                    conv_input[
                        bi : bi + 1,
                        accepted_list[bi] + 1 : accepted_list[bi] + K,
                    ]
                    for bi in range(len(accepted_list))
                ]
                c[0] = mx.concatenate(slices, axis=0)
            else:
                c[0] = conv_input[:, a0 + 1 : a0 + K]
        return max_a

    def get_rope_index(
        self,
        input_ids: mx.array,
        image_grid_thw: Optional[mx.array] = None,
        video_grid_thw: Optional[mx.array] = None,
        attention_mask: Optional[mx.array] = None,
    ):
        batch_size, seq_length = input_ids.shape
        position_ids = mx.arange(seq_length, dtype=mx.int32)
        position_ids = mx.broadcast_to(position_ids[None, :], (batch_size, seq_length))
        spatial_merge_size = self.config.vision_config.spatial_merge_size
        image_token_id = self.config.image_token_id
        video_token_id = self.config.video_token_id
        vision_start_token_id = self.config.vision_start_token_id
        mrope_position_deltas = []
        if input_ids is not None and (
            image_grid_thw is not None or video_grid_thw is not None
        ):
            total_input_ids = input_ids
            if attention_mask is None:
                attention_mask = mx.ones_like(input_ids)
            position_ids = mx.ones(
                (3, input_ids.shape[0], input_ids.shape[1]), dtype=input_ids.dtype
            )
            image_index, video_index = 0, 0
            for i, input_ids in enumerate(total_input_ids):
                row_mask = attention_mask[i].tolist()
                input_tokens = [
                    token
                    for token, keep in zip(input_ids.tolist(), row_mask)
                    if keep == 1
                ]
                image_nums, video_nums = 0, 0
                vision_tokens = [
                    input_tokens[idx + 1]
                    for idx, token in enumerate(input_tokens[:-1])
                    if token == vision_start_token_id
                ]
                image_nums = sum(token == image_token_id for token in vision_tokens)
                video_nums = sum(token == video_token_id for token in vision_tokens)
                llm_pos_ids_list: list = []
                st = 0
                remain_images, remain_videos = image_nums, video_nums
                for _ in range(image_nums + video_nums):
                    if image_token_id in input_tokens and remain_images > 0:
                        ed_image = input_tokens.index(image_token_id, st)
                    else:
                        ed_image = len(input_tokens) + 1
                    if video_token_id in input_tokens and remain_videos > 0:
                        ed_video = input_tokens.index(video_token_id, st)
                    else:
                        ed_video = len(input_tokens) + 1
                    if ed_image < ed_video:
                        t, h, w = (
                            image_grid_thw[image_index][0],
                            image_grid_thw[image_index][1],
                            image_grid_thw[image_index][2],
                        )
                        image_index += 1
                        remain_images -= 1
                        ed = ed_image
                    else:
                        t, h, w = (
                            video_grid_thw[video_index][0],
                            video_grid_thw[video_index][1],
                            video_grid_thw[video_index][2],
                        )
                        video_index += 1
                        remain_videos -= 1
                        ed = ed_video
                    llm_grid_t, llm_grid_h, llm_grid_w = (
                        t.item(),
                        h.item() // spatial_merge_size,
                        w.item() // spatial_merge_size,
                    )
                    text_len = ed - st
                    st_idx = (
                        llm_pos_ids_list[-1].max() + 1
                        if len(llm_pos_ids_list) > 0
                        else 0
                    )
                    index = mx.arange(text_len).reshape(1, text_len)
                    index = mx.broadcast_to(index, (3, text_len))
                    index = index + st_idx
                    llm_pos_ids_list.append(index)
                    t_index = mx.arange(llm_grid_t).reshape(llm_grid_t, 1)
                    t_index = mx.broadcast_to(
                        t_index, (llm_grid_t, llm_grid_h * llm_grid_w)
                    )
                    t_index = t_index.flatten()

                    h_index = mx.arange(llm_grid_h).reshape(1, llm_grid_h, 1)
                    h_index = mx.broadcast_to(
                        h_index, (llm_grid_t, llm_grid_h, llm_grid_w)
                    )
                    h_index = h_index.flatten()

                    w_index = mx.arange(llm_grid_w).reshape(1, 1, llm_grid_w)
                    w_index = mx.broadcast_to(
                        w_index, (llm_grid_t, llm_grid_h, llm_grid_w)
                    )
                    w_index = w_index.flatten()

                    llm_pos_ids_list.append(
                        mx.stack([t_index, h_index, w_index]) + text_len + st_idx
                    )
                    st = ed + llm_grid_t * llm_grid_h * llm_grid_w
                if st < len(input_tokens):
                    st_idx = (
                        llm_pos_ids_list[-1].max() + 1
                        if len(llm_pos_ids_list) > 0
                        else 0
                    )
                    text_len = len(input_tokens) - st

                    t_index = mx.arange(text_len).reshape(1, text_len)
                    t_index = mx.broadcast_to(t_index, (3, text_len))

                    llm_pos_ids_list.append(t_index + st_idx)

                if not llm_pos_ids_list:
                    mrope_position_deltas.append(0)
                    continue

                llm_positions = mx.concatenate(llm_pos_ids_list, axis=1).reshape(3, -1)
                compact_max_position = llm_positions.max()
                padded_positions = [[1] * total_input_ids.shape[1] for _ in range(3)]
                compact_positions = llm_positions.tolist()
                compact_idx = 0
                for col, keep in enumerate(row_mask):
                    if keep == 1:
                        for dim in range(3):
                            padded_positions[dim][col] = compact_positions[dim][
                                compact_idx
                            ]
                        compact_idx += 1
                llm_positions = mx.array(padded_positions, dtype=position_ids.dtype)
                mask = mx.array(row_mask, dtype=mx.bool_)
                expanded_mask = mx.expand_dims(mask, axis=0)
                expanded_mask = mx.broadcast_to(expanded_mask, (3, 1, mask.shape[0]))
                expanded_positions = mx.expand_dims(llm_positions, axis=1)
                new_positions = mx.where(
                    expanded_mask, expanded_positions, position_ids[:, i : i + 1, :]
                )
                updated_position_ids = mx.concatenate(
                    [
                        position_ids[:, :i, :],
                        new_positions,
                        position_ids[:, i + 1 :, :],
                    ],
                    axis=1,
                )
                position_ids = updated_position_ids
                mrope_position_deltas.append(
                    compact_max_position + 1 - len(input_tokens)
                )
            mrope_position_deltas = mx.array(mrope_position_deltas).reshape(-1, 1)
            return position_ids, mrope_position_deltas
        else:
            if attention_mask is not None:
                position_ids = mx.cumsum(attention_mask.astype(mx.int64), axis=-1) - 1
                position_ids = mx.where(
                    attention_mask == 0, mx.ones_like(position_ids), position_ids
                )
                max_position_ids = position_ids.max(axis=-1, keepdims=True)
                mrope_position_deltas = max_position_ids + 1 - attention_mask.shape[-1]
            else:
                position_ids = mx.arange(input_ids.shape[1]).reshape(1, -1)
                position_ids = mx.broadcast_to(
                    position_ids, (input_ids.shape[0], input_ids.shape[1])
                )
                mrope_position_deltas = mx.zeros(
                    [input_ids.shape[0], 1],
                    dtype=input_ids.dtype,
                )
            return position_ids, mrope_position_deltas

    def __call__(
        self,
        inputs: mx.array,
        inputs_embeds: Optional[mx.array] = None,
        mask: Optional[mx.array] = None,
        cache=None,
        **kwargs,
    ):
        position_ids = kwargs.pop("position_ids", None)
        pixel_values = kwargs.pop("pixel_values", None)
        image_grid_thw = kwargs.pop("image_grid_thw", None)
        video_grid_thw = kwargs.pop("video_grid_thw", None)
        attention_mask = kwargs.pop("attention_mask", None)
        capture_layer_ids = kwargs.pop("capture_layer_ids", None)
        speculative_verify = bool(kwargs.pop("speculative_verify", False))
        return_hidden = kwargs.pop("return_hidden", False)
        return_shared_kv = kwargs.pop("return_shared_kv", False)
        skip_logits = kwargs.pop("skip_logits", False)
        rope_deltas_kw = kwargs.pop("rope_deltas", None)
        if (
            mask is None
            and attention_mask is not None
            and attention_mask.shape[-1] == inputs.shape[-1]
        ):
            mask = attention_mask
        if pixel_values is not None:
            self._rope_deltas = None
            self._position_ids = None

        if rope_deltas_kw is not None:
            self._rope_deltas = rope_deltas_kw

        cache_offset = 0
        cache_offsets = None  # per-element offsets for batched caches
        c0 = None
        if cache and cache[self.model.fa_idx] is not None:
            c0 = cache[self.model.fa_idx]
            cache_offset = c0._idx if hasattr(c0, "_idx") else c0.offset
            if (
                isinstance(c0.offset, mx.array)
                and c0.offset.ndim > 0
                and c0.offset.size > 1
            ):
                cache_offsets = mx.maximum(c0.offset, 0)

        if position_ids is not None and cache_offsets is None:
            seq_length = inputs.shape[-1]
            if position_ids.shape[-1] > seq_length:
                position_ids = position_ids[
                    ..., cache_offset : cache_offset + seq_length
                ]

        if (
            mask is None
            and c0 is not None
            and cache_offsets is None
            and cache_offset == 0
        ):
            left_padding = getattr(c0, "left_padding", None)
            if (
                isinstance(left_padding, mx.array)
                and left_padding.ndim > 0
                and left_padding.size >= inputs.shape[0]
            ):
                positions = mx.arange(inputs.shape[-1])[None, :]
                mask = positions >= left_padding[: inputs.shape[0], None]

        # Check if mask shape matches input shape (for chunked prefill compatibility)
        rope_mask = mask
        if mask is not None and mask.shape[-1] != inputs.shape[-1]:
            rope_mask = None

        if position_ids is None and (rope_mask is None or rope_mask.ndim == 2):
            batch_size, seq_length = inputs.shape

            if (
                (
                    cache is not None
                    and cache[self.model.fa_idx] is not None
                    and (cache_offsets is None and cache_offset == 0)
                )
                or self._rope_deltas is None
                or cache is None
            ):
                if self._position_ids is not None:
                    if (
                        self._position_ids.ndim == 3
                        and self._position_ids.shape[1] == batch_size
                        and self._position_ids.shape[-1] >= cache_offset + seq_length
                    ):
                        position_ids = self._position_ids[
                            :, :, cache_offset : cache_offset + seq_length
                        ]
                    elif (
                        self._position_ids.ndim == 2
                        and self._position_ids.shape[0] == batch_size
                        and self._position_ids.shape[-1] >= cache_offset + seq_length
                    ):
                        position_ids = self._position_ids[
                            :, cache_offset : cache_offset + seq_length
                        ]
                    else:
                        position_ids, rope_deltas = self.get_rope_index(
                            inputs, image_grid_thw, video_grid_thw, rope_mask
                        )
                        self._rope_deltas = rope_deltas
                        self._position_ids = position_ids
                else:
                    position_ids, rope_deltas = self.get_rope_index(
                        inputs, image_grid_thw, video_grid_thw, rope_mask
                    )
                    if image_grid_thw is None and video_grid_thw is None:
                        rope_deltas = mx.zeros((batch_size, 1), dtype=rope_deltas.dtype)
                    self._rope_deltas = rope_deltas
                    self._position_ids = position_ids
            else:
                rope_deltas_src = (
                    rope_deltas_kw if rope_deltas_kw is not None else self._rope_deltas
                )
                if cache_offsets is not None and cache_offsets.size >= batch_size:
                    offsets = cache_offsets[:batch_size]
                    rope_deltas = rope_deltas_src
                    if rope_deltas.shape[0] > batch_size:
                        rope_deltas = rope_deltas[:batch_size]
                    delta = (offsets + rope_deltas.squeeze(-1))[:, None]
                else:
                    delta = mx.array(
                        cache_offset + rope_deltas_src if cache is not None else 0
                    )
                    if delta.ndim == 0:
                        delta = mx.expand_dims(delta, axis=0)
                    if delta.shape[0] < batch_size:
                        delta = mx.tile(delta, (batch_size, 1))
                    else:
                        delta = delta[:batch_size]

                position_ids = mx.arange(seq_length).reshape(1, -1)
                position_ids = mx.broadcast_to(position_ids, (batch_size, seq_length))
                position_ids = mx.add(position_ids, delta)
                if (
                    rope_deltas_kw is not None
                    or self._position_ids is not None
                    and self._position_ids.ndim == 3
                ):
                    position_ids = position_ids[None, ...]
                    position_ids = mx.broadcast_to(
                        position_ids, (3, batch_size, seq_length)
                    )

        if speculative_verify:
            return _EXACT_SPECULATIVE_VERIFIER(
                self,
                inputs,
                cache=cache,
                inputs_embeds=inputs_embeds,
                position_ids=position_ids,
                capture_layer_ids=capture_layer_ids,
                return_hidden=return_hidden,
                return_shared_kv=return_shared_kv,
                skip_logits=skip_logits,
            )

        batch_invariant_decode = getattr(self, "_batch_invariant_decode", None)
        supports_batch_invariant_decode = getattr(
            self, "_supports_batch_invariant_decode", None
        )
        if (
            cache is not None
            and inputs.ndim == 2
            and inputs.shape[1] == 1
            and callable(batch_invariant_decode)
            and (
                supports_batch_invariant_decode is None
                or supports_batch_invariant_decode()
            )
        ):
            return batch_invariant_decode(
                inputs,
                cache=cache,
                inputs_embeds=inputs_embeds,
                position_ids=position_ids,
                skip_logits=skip_logits,
            )

        hidden_sink: Optional[List[mx.array]] = (
            [] if capture_layer_ids is not None else None
        )

        out = self.model(
            inputs,
            cache=cache,
            inputs_embeds=inputs_embeds,
            position_ids=position_ids,
            capture_layer_ids=capture_layer_ids,
            hidden_sink=hidden_sink,
        )
        if return_hidden:
            if hidden_sink is None:
                hidden_sink = []
            hidden_sink.append(out)

        if skip_logits:
            logits = None
        elif self.args.tie_word_embeddings:
            logits = self.model.embed_tokens.as_linear(out)
        else:
            logits = self.lm_head(out)
        return LanguageModelOutput(
            logits=logits,
            hidden_states=hidden_sink,
            gdn_states=None,
            shared_kv_states={} if return_shared_kv else None,
        )

    def speculative_logits_from_hidden(self, hidden: mx.array) -> mx.array:
        if self.args.tie_word_embeddings:
            return self.model.embed_tokens.as_linear(hidden)
        out = _EXACT_SPECULATIVE_VERIFIER.quantized_linear(self.lm_head, hidden)
        if out is not None:
            return out
        return self.lm_head(hidden)

    def speculative_argmax_from_hidden(self, hidden: mx.array) -> Optional[mx.array]:
        if not self.args.tie_word_embeddings:
            out = _EXACT_SPECULATIVE_VERIFIER.quantized_argmax(self.lm_head, hidden)
            if out is not None:
                return out
        logits = self.speculative_logits_from_hidden(hidden)
        return mx.argmax(logits, axis=-1)

    def supports_fused_greedy_logits_processors(self, logits_processors) -> bool:
        return (
            len(logits_processors) <= 4
            and all(
                processors
                and len(processors) == 1
                and callable(getattr(processors[0], "prepare_next_token_mask", None))
                for processors in logits_processors
            )
            and not self.args.tie_word_embeddings
            and _EXACT_SPECULATIVE_VERIFIER.can_quantized_head(self.lm_head)
            and "bias" not in self.lm_head
        )

    def fused_greedy_decode(
        self,
        inputs: mx.array,
        cache=None,
        logits_processors=None,
        **kwargs,
    ):
        if (
            self.args.tie_word_embeddings
            or not _EXACT_SPECULATIVE_VERIFIER.can_quantized_head(self.lm_head)
            or "bias" in self.lm_head
        ):
            return None

        token_mask = None
        if logits_processors:
            if not self.supports_fused_greedy_logits_processors(logits_processors):
                return None
            token_mask = mx.concatenate(
                [
                    processors[0].prepare_next_token_mask(token)
                    for processors, token in zip(
                        logits_processors, inputs[:, -1].tolist()
                    )
                ],
                axis=0,
            )
            token_mask = _EXACT_SPECULATIVE_VERIFIER.pad_token_mask(
                token_mask, self.lm_head.weight.shape[0]
            )

        output = self(
            inputs,
            cache=cache,
            return_hidden=True,
            skip_logits=True,
            **kwargs,
        )
        hidden = output.hidden_states[-1]
        sampled = _EXACT_SPECULATIVE_VERIFIER.quantized_argmax(
            self.lm_head, hidden, token_mask=token_mask
        )
        if sampled is not None:
            return sampled
        if token_mask is not None:
            raise RuntimeError("masked fused greedy decode became unsupported")
        return mx.argmax(self.speculative_logits_from_hidden(hidden), axis=-1)

    def speculative_verify_logits(self, inputs: mx.array, cache, sampler):
        out = self(
            inputs,
            cache=cache,
            capture_layer_ids=[],
            speculative_verify=True,
            return_hidden=True,
            return_shared_kv=True,
        )
        return (
            out.hidden_states[-1],
            out.shared_kv_states,
            out.gdn_states,
            sampler(out.logits),
        )

    def speculative_verify_hidden(self, inputs: mx.array, cache):
        out = self(
            inputs,
            cache=cache,
            capture_layer_ids=[],
            speculative_verify=True,
            return_hidden=True,
            return_shared_kv=True,
            skip_logits=True,
        )
        return out.hidden_states[-1], out.shared_kv_states, out.gdn_states

    @property
    def layers(self):
        return self.model.layers

    def make_cache(self):
        return [ArraysCache(size=2) if l.is_linear else KVCache() for l in self.layers]

    @property
    def head_dim(self):
        return self.args.hidden_size // self.args.num_attention_heads

    @property
    def n_kv_heads(self):
        return self.args.num_key_value_heads

    @property
    def quant_predicate(self):

        if getattr(self.args, "num_experts", 0) <= 0:
            return None

        def predicate(path, _):
            if path.endswith("mlp.gate") or path.endswith("shared_expert_gate"):
                return {"group_size": 64, "bits": 8}
            return True

        return predicate

    @property
    def cast_predicate(self):
        def predicate(path: str):
            if path.endswith("A_log"):
                return False
            return True

        return predicate
