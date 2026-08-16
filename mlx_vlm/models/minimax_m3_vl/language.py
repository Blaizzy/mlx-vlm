from functools import lru_cache, partial
from typing import Any, List, Optional

import mlx.core as mx
import mlx.nn as nn
from mlx.nn.layers.distributed import shard_inplace, shard_linear, sum_gradients

from ..base import (
    LanguageModelOutput,
    create_attention_mask,
    scaled_dot_product_attention,
)
from ..cache import BatchKVCache, KVCache, dynamic_roll
from ..rope_utils import initialize_rope
from ..switch_layers import SwitchGLU, SwitchLinear
from .config import ModelConfig, TextConfig


def _is_bool_mask(mask: mx.array) -> bool:
    return mask.dtype == mx.bool_


def _is_integer_mask(mask: mx.array) -> bool:
    return mask.dtype in {
        mx.int8,
        mx.int16,
        mx.int32,
        mx.int64,
        mx.uint8,
        mx.uint16,
        mx.uint32,
        mx.uint64,
    }


def _is_empty_kv_state(state) -> bool:
    return state is None or (
        isinstance(state, (tuple, list))
        and len(state) >= 2
        and state[0] is None
        and state[1] is None
    )


def _can_skip_uniform_batch_decode_mask(cache, seq_length: int) -> bool:
    if seq_length != 1 or cache is None:
        return False

    if isinstance(cache, (list, tuple)):
        return any(
            _can_skip_uniform_batch_decode_mask(entry, seq_length)
            for entry in cache
            if entry is not None
        )

    nested = getattr(cache, "caches", None)
    if nested is not None:
        return _can_skip_uniform_batch_decode_mask(nested, seq_length)

    can_skip_decode_mask = getattr(cache, "can_skip_decode_mask", None)
    if callable(can_skip_decode_mask):
        return bool(can_skip_decode_mask(seq_length))

    kv_cache = getattr(cache, "kv_cache", None)
    if kv_cache is not None and kv_cache is not cache:
        return _can_skip_uniform_batch_decode_mask(kv_cache, seq_length)

    return False


def _switch_gather_sort(x: mx.array, indices: mx.array):
    *_, num_selected = indices.shape
    indices = indices.flatten()
    order = mx.argsort(indices)
    inv_order = mx.argsort(order)
    return x.flatten(0, -3)[order // num_selected], indices[order], inv_order


def _switch_scatter_unsort(x: mx.array, inv_order: mx.array, shape=None):
    x = x[inv_order]
    if shape is not None:
        x = mx.unflatten(x, 0, shape)
    return x


@mx.compile
def _minimax_moe_select(
    gates: mx.array,
    correction_bias: mx.array,
    k: int,
    routed_scaling_factor: float,
    scoring_func: str,
):
    if scoring_func == "sigmoid":
        scores = mx.sigmoid(gates)
    else:
        scores = mx.softmax(gates, axis=-1, precise=True)

    biased_scores = scores + correction_bias
    inds = mx.argpartition(-biased_scores, kth=k - 1, axis=-1)[..., :k]
    weights = mx.take_along_axis(scores, inds, axis=-1)
    weights = weights / (mx.sum(weights, axis=-1, keepdims=True) + 1e-20)
    return inds, weights * routed_scaling_factor


@mx.compile
def _build_sparse_causal_mask_compiled(
    idx_queries: mx.array,
    idx_keys: mx.array,
    q_positions: mx.array,
    scale: float,
    block_size: int,
    sparse_topk_blocks: int,
    sparse_init_blocks: int,
    sparse_local_blocks: int,
):
    B, H_idx, L, _ = idx_queries.shape
    total_len = idx_keys.shape[2]
    neg = mx.array(-float("inf"), dtype=mx.float32)

    scores = mx.matmul(
        idx_queries.astype(mx.float32),
        idx_keys.astype(mx.float32).swapaxes(-1, -2),
    )
    scores = scores * scale

    qpos = q_positions
    kpos = mx.arange(total_len)
    causal = kpos[None, None, :] <= qpos[:, :, None]
    scores = mx.where(causal[:, None], scores, neg)

    num_blocks = (total_len + block_size - 1) // block_size
    pad = num_blocks * block_size - total_len
    if pad:
        pad_values = mx.full(
            (*scores.shape[:-1], pad), -float("inf"), dtype=scores.dtype
        )
        scores = mx.concatenate([scores, pad_values], axis=-1)

    blocks = mx.arange(num_blocks)
    cur_block = qpos // block_size
    causal_block = blocks[None, None, :] <= cur_block[:, :, None]
    valid_blocks = causal_block[:, None]

    scores = scores.reshape(B, H_idx, L, num_blocks, block_size)
    block_scores = mx.max(scores, axis=-1)
    block_scores = mx.max(block_scores, axis=1)
    block_scores = mx.where(block_scores == block_scores, block_scores, neg)

    init_blocks = blocks[None, None, :] < sparse_init_blocks
    if sparse_local_blocks > 0:
        local_start = mx.maximum(cur_block - sparse_local_blocks + 1, 0)
        local_blocks = (blocks[None, None, :] >= local_start[:, :, None]) & (
            blocks[None, None, :] <= cur_block[:, :, None]
        )
    else:
        local_blocks = mx.zeros((B, L, num_blocks), dtype=mx.bool_)

    valid_blocks = mx.broadcast_to(valid_blocks[:, 0], block_scores.shape)
    selected_scores = mx.where(valid_blocks, block_scores, neg)
    forced_init_blocks = (init_blocks & causal_block) & valid_blocks
    forced_local_blocks = (local_blocks & causal_block) & valid_blocks
    selected_scores = mx.where(
        forced_init_blocks,
        mx.array(1e30, dtype=selected_scores.dtype),
        selected_scores,
    )
    selected_scores = mx.where(
        forced_local_blocks,
        mx.array(1e29, dtype=selected_scores.dtype),
        selected_scores,
    )

    topk_idx = mx.argpartition(-selected_scores, kth=sparse_topk_blocks - 1, axis=-1)[
        ..., :sparse_topk_blocks
    ]
    topk_valid = mx.take_along_axis(valid_blocks, topk_idx, axis=-1)

    block_selected = mx.any(topk_idx[..., None] == blocks, axis=-2)
    block_selected = block_selected & valid_blocks

    key_blocks = (kpos // block_size).astype(mx.int32)
    key_blocks = mx.broadcast_to(key_blocks[None, None, :], (B, L, total_len))
    key_selected = mx.take_along_axis(block_selected, key_blocks, axis=-1)
    key_selected = key_selected & causal

    sparse_mask = key_selected[:, None]
    return sparse_mask, topk_idx[:, None], topk_valid[:, None]


@mx.compile
def _select_sparse_block_indices_compiled(
    idx_queries: mx.array,
    idx_keys: mx.array,
    q_positions: mx.array,
    scale: float,
    block_size: int,
    sparse_topk_blocks: int,
    sparse_init_blocks: int,
    sparse_local_blocks: int,
):
    B, H_idx, L, _ = idx_queries.shape
    total_len = idx_keys.shape[2]
    neg = mx.array(-float("inf"), dtype=mx.float32)

    scores = mx.matmul(
        idx_queries.astype(mx.float32),
        idx_keys.astype(mx.float32).swapaxes(-1, -2),
    )
    scores = scores * scale

    qpos = q_positions
    kpos = mx.arange(total_len)
    causal = kpos[None, None, :] <= qpos[:, :, None]
    scores = mx.where(causal[:, None], scores, neg)

    num_blocks = (total_len + block_size - 1) // block_size
    pad = num_blocks * block_size - total_len
    pad_values = mx.full((*scores.shape[:-1], pad), -float("inf"), dtype=scores.dtype)
    scores = mx.concatenate([scores, pad_values], axis=-1)

    blocks = mx.arange(num_blocks)
    cur_block = qpos // block_size
    causal_block = blocks[None, None, :] <= cur_block[:, :, None]
    valid_blocks = causal_block[:, None]

    scores = scores.reshape(B, H_idx, L, num_blocks, block_size)
    block_scores = mx.max(scores, axis=-1)
    block_scores = mx.max(block_scores, axis=1)
    selected_scores = mx.where(block_scores == block_scores, block_scores, neg)
    valid_blocks = mx.broadcast_to(valid_blocks[:, 0], selected_scores.shape)
    selected_scores = mx.where(valid_blocks, selected_scores, neg)

    if sparse_init_blocks > 0:
        init_blocks = blocks[None, None, :] < sparse_init_blocks
        selected_scores = mx.where(
            (init_blocks & causal_block) & valid_blocks,
            mx.array(1e30, dtype=selected_scores.dtype),
            selected_scores,
        )

    if sparse_local_blocks > 0:
        local_start = mx.maximum(cur_block - sparse_local_blocks + 1, 0)
        local_blocks = (blocks[None, None, :] >= local_start[:, :, None]) & (
            blocks[None, None, :] <= cur_block[:, :, None]
        )
        selected_scores = mx.where(
            (local_blocks & causal_block) & valid_blocks,
            mx.array(1e29, dtype=selected_scores.dtype),
            selected_scores,
        )

    topk_idx = mx.argpartition(-selected_scores, kth=sparse_topk_blocks - 1, axis=-1)[
        ..., :sparse_topk_blocks
    ]
    topk_valid = mx.take_along_axis(valid_blocks, topk_idx, axis=-1)
    invalid = mx.full(topk_idx.shape, num_blocks, dtype=topk_idx.dtype)
    block_indices = mx.where(topk_valid, topk_idx, invalid)
    order = mx.argsort(block_indices, axis=-1)
    block_indices = mx.take_along_axis(block_indices, order, axis=-1)
    return mx.where(block_indices == num_blocks, mx.array(-1), block_indices)


_MINIMAX_M3_SPARSE_PREFILL_ONE_PASS_SOURCE = r"""
    uint row_idx = threadgroup_position_in_grid.y;
    uint simd_gid = simdgroup_index_in_threadgroup;
    uint simd_lid = thread_index_in_simdgroup;

    constexpr int BN = 32;
    constexpr int BD = 32;
    constexpr int qk_per_thread = D_SIZE / BD;
    constexpr int v_per_thread = D_SIZE / BD;

    typedef float U;
    thread U q[qk_per_thread];
    thread U o[v_per_thread];
    threadgroup U outputs[BN * BD];
    threadgroup U max_scores[BN];
    threadgroup U sum_exp_scores[BN];

    int key_length = int(k_size[0]);
    int query_idx = int(row_idx % Q_LEN);
    int batch_head_idx = int(row_idx / Q_LEN);
    int batch_idx = batch_head_idx / NUM_Q_HEADS;
    int q_head_idx = batch_head_idx - batch_idx * NUM_Q_HEADS;
    int kv_head_idx = q_head_idx / GQA_FACTOR;

    const device T* qptr =
        queries + ((batch_head_idx * Q_LEN + query_idx) * D_SIZE) +
        int(simd_lid) * qk_per_thread;
    device T* optr =
        out + ((batch_head_idx * Q_LEN + query_idx) * D_SIZE) +
        int(simd_gid) * v_per_thread;

    U s = U(scale[0]);
    for (int i = 0; i < qk_per_thread; i++) {
        q[i] = s * static_cast<U>(qptr[i]);
    }
    for (int i = 0; i < v_per_thread; i++) {
        o[i] = 0;
    }

    int qpos = int(q_positions[batch_idx * Q_LEN + query_idx]);
    int blocks_offset = (batch_idx * Q_LEN + query_idx) * TOPK_BLOCKS;

    U max_score = -3.4028234663852886e38f;
    U sum_exp_score = 0;

    for (int selected_idx = int(simd_gid); selected_idx < SELECTED_LENGTH;
         selected_idx += BN) {
        int block_slot = selected_idx / BLOCK_SIZE;
        int block_offset = selected_idx - block_slot * BLOCK_SIZE;
        int block_idx = int(block_indices[blocks_offset + block_slot]);
        int key_pos = block_idx * BLOCK_SIZE + block_offset;
        bool valid = block_idx >= 0 && key_pos < key_length && key_pos <= qpos;

        U score = -3.4028234663852886e38f;
        if (valid) {
            const device T* kptr =
                keys + (((batch_idx * NUM_KV_HEADS + kv_head_idx) * key_length +
                         key_pos) *
                        D_SIZE) +
                int(simd_lid) * qk_per_thread;
            score = 0;
            for (int j = 0; j < qk_per_thread; j++) {
                score += q[j] * static_cast<U>(kptr[j]);
            }
            score = simd_sum(score);
        }

        U new_max = max(max_score, score);
        U factor = fast::exp(max_score - new_max);
        U exp_score = valid ? fast::exp(score - new_max) : U(0);

        max_score = new_max;
        sum_exp_score = sum_exp_score * factor + exp_score;

        if (valid) {
            const device T* vptr =
                values + (((batch_idx * NUM_KV_HEADS + kv_head_idx) * key_length +
                           key_pos) *
                          D_SIZE) +
                int(simd_lid) * v_per_thread;
            for (int j = 0; j < v_per_thread; j++) {
                o[j] = o[j] * factor + exp_score * static_cast<U>(vptr[j]);
            }
        } else {
            for (int j = 0; j < v_per_thread; j++) {
                o[j] = o[j] * factor;
            }
        }
    }

    if (simd_lid == 0) {
        max_scores[simd_gid] = max_score;
        sum_exp_scores[simd_gid] = sum_exp_score;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    max_score = max_scores[simd_lid];
    U new_max = simd_max(max_score);
    U factor = fast::exp(max_score - new_max);
    U total_sum = simd_sum(sum_exp_scores[simd_lid] * factor);

    for (int i = 0; i < v_per_thread; i++) {
        outputs[simd_lid * BD + simd_gid] = o[i];
        threadgroup_barrier(mem_flags::mem_threadgroup);
        o[i] = simd_sum(outputs[simd_gid * BD + simd_lid] * factor);
        o[i] = total_sum == 0 ? U(0) : (o[i] / total_sum);
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (simd_lid == 0) {
        for (int i = 0; i < v_per_thread; i++) {
            optr[i] = static_cast<T>(o[i]);
        }
    }
"""


@lru_cache(maxsize=None)
def _minimax_m3_sparse_prefill_one_pass_kernel(
    dtype,
    d_size: int,
    selected_length: int,
    block_size: int,
    topk_blocks: int,
    q_heads: int,
    kv_heads: int,
):
    dtype_name = {mx.bfloat16: "bf16", mx.float16: "fp16"}.get(dtype, "unk")
    return mx.fast.metal_kernel(
        name=(
            "minimax_m3_sparse_prefill_1p_"
            f"{dtype_name}_d{d_size}_s{selected_length}_b{block_size}_"
            f"k{topk_blocks}_qh{q_heads}_kh{kv_heads}"
        ),
        input_names=[
            "queries",
            "keys",
            "values",
            "block_indices",
            "q_positions",
            "scale",
            "k_size",
        ],
        output_names=["out"],
        header="#include <metal_simdgroup>\nusing namespace metal;\n",
        source=_MINIMAX_M3_SPARSE_PREFILL_ONE_PASS_SOURCE,
    )


@lru_cache(maxsize=128)
def _minimax_m3_sparse_prefill_scalars(scale: float, key_length: int):
    return (
        mx.array([scale], dtype=mx.float32),
        mx.array([key_length], dtype=mx.int32),
    )


def _minimax_m3_sparse_prefill_attention(
    queries: mx.array,
    keys: mx.array,
    values: mx.array,
    block_indices: mx.array,
    q_positions: mx.array,
    scale: float,
    block_size: int,
) -> Optional[mx.array]:
    if (
        queries.ndim != 4
        or keys.ndim != 4
        or values.ndim != 4
        or block_indices.ndim != 3
        or q_positions.ndim != 2
        or queries.dtype not in (mx.bfloat16, mx.float16)
        or keys.dtype != queries.dtype
        or values.dtype != queries.dtype
        or mx.default_device() != mx.gpu
        or not mx.metal.is_available()
    ):
        return None

    B, q_heads, query_length, d_size = queries.shape
    kv_heads = keys.shape[1]
    key_length = keys.shape[2]
    topk_blocks = block_indices.shape[-1]
    selected_length = topk_blocks * block_size
    if (
        B != block_indices.shape[0]
        or B != q_positions.shape[0]
        or query_length != block_indices.shape[1]
        or query_length != q_positions.shape[1]
        or values.shape != keys.shape
        or q_heads % kv_heads != 0
        or d_size != values.shape[-1]
        or d_size % 32 != 0
        or selected_length >= key_length
        or key_length < selected_length * 7
    ):
        return None

    queries = mx.contiguous(queries)
    keys = mx.contiguous(keys)
    values = mx.contiguous(values)
    block_indices = mx.contiguous(block_indices.astype(mx.int32))
    q_positions = mx.contiguous(q_positions.astype(mx.int32))
    scale_array, key_length_array = _minimax_m3_sparse_prefill_scalars(
        float(scale), int(key_length)
    )
    kernel = _minimax_m3_sparse_prefill_one_pass_kernel(
        queries.dtype,
        int(d_size),
        int(selected_length),
        int(block_size),
        int(topk_blocks),
        int(q_heads),
        int(kv_heads),
    )
    return kernel(
        inputs=[
            queries,
            keys,
            values,
            block_indices,
            q_positions,
            scale_array,
            key_length_array,
        ],
        template=[
            ("T", queries.dtype),
            ("D_SIZE", int(d_size)),
            ("Q_LEN", int(query_length)),
            ("NUM_Q_HEADS", int(q_heads)),
            ("NUM_KV_HEADS", int(kv_heads)),
            ("GQA_FACTOR", int(q_heads // kv_heads)),
            ("BLOCK_SIZE", int(block_size)),
            ("TOPK_BLOCKS", int(topk_blocks)),
            ("SELECTED_LENGTH", int(selected_length)),
        ],
        grid=(1024, B * q_heads * query_length, 1),
        threadgroup=(1024, 1, 1),
        output_shapes=[queries.shape],
        output_dtypes=[queries.dtype],
    )[0]


@partial(mx.compile, shapeless=True)
def _swiglu_oai(
    x_linear,
    x_glu,
    alpha: float = 1.702,
    limit: float = 7.0,
    beta: float = 1.0,
):
    x_glu = mx.clip(x_glu, a_min=None, a_max=limit)
    x_linear = mx.clip(x_linear, a_min=-limit, a_max=limit)
    return x_glu * mx.sigmoid(alpha * x_glu) * (x_linear + beta)


class MiniMaxSwiGLUOAI(nn.Module):
    def __init__(
        self,
        alpha: float = 1.702,
        limit: float = 7.0,
        beta: float = 1.0,
    ):
        super().__init__()
        self.alpha = alpha
        self.limit = limit
        self.beta = beta

    def __call__(self, x, gate):
        return _swiglu_oai(x, gate, self.alpha, self.limit, self.beta)


class MiniMaxRMSNorm(nn.Module):
    def __init__(self, dims: int, eps: float = 1e-6, gemma: bool = True):
        super().__init__()
        self.weight = mx.zeros((dims,)) if gemma else mx.ones((dims,))
        self.eps = eps
        self.gemma = gemma

    def __call__(self, x):
        weight = self.weight + 1 if self.gemma else self.weight
        return mx.fast.rms_norm(x, weight, self.eps)


class MiniMaxM3KVCache:
    step = KVCache.step

    def __init__(self):
        self.kv_cache = KVCache()
        self.index_keys = None
        self.index_offset = 0

    @property
    def offset(self):
        return self.kv_cache.offset

    @offset.setter
    def offset(self, value):
        self.kv_cache.offset = int(value)
        self.index_offset = int(value)

    def update_and_fetch(self, keys: mx.array, values: mx.array):
        return self.kv_cache.update_and_fetch(keys, values)

    def to_batch(self, left_padding):
        batch_cache = MiniMaxM3BatchKVCache(left_padding)
        left_padding = mx.array(left_padding)

        if self.kv_cache.empty() and self.index_keys is None:
            return batch_cache

        if left_padding.size != 1:
            raise ValueError(
                "Cannot convert a warm MiniMax M3 cache to a multi-row batch "
                f"cache with left padding {left_padding.tolist()}."
            )

        pad = int(left_padding.item())
        keys, values = self.kv_cache.state
        if pad:
            keys = mx.pad(keys, [(0, 0), (0, 0), (pad, 0), (0, 0)])
            values = mx.pad(values, [(0, 0), (0, 0), (pad, 0), (0, 0)])
        batch_cache.kv_cache.state = (
            keys,
            values,
            mx.array([self.offset], dtype=mx.int32),
            left_padding.astype(mx.int32),
        )

        if self.index_keys is not None:
            index_keys = self.index_keys[..., : self.index_offset, :]
            if pad:
                index_keys = mx.pad(index_keys, [(0, 0), (0, 0), (pad, 0), (0, 0)])
            batch_cache.index_keys = index_keys
            batch_cache.index_offset = index_keys.shape[2]

        return batch_cache

    def update_index_and_fetch(self, keys: mx.array):
        prev = self.index_offset
        incoming = keys.shape[2]
        if self.index_keys is None or (prev + incoming) > self.index_keys.shape[2]:
            B, n_heads, _, head_dim = keys.shape
            n_steps = (self.step + incoming - 1) // self.step
            new_shape = (B, n_heads, n_steps * self.step, head_dim)
            new_keys = mx.zeros(new_shape, keys.dtype)
            if self.index_keys is not None:
                if prev % self.step != 0:
                    self.index_keys = self.index_keys[..., :prev, :]
                self.index_keys = mx.concatenate([self.index_keys, new_keys], axis=2)
            else:
                self.index_keys = new_keys

        self.index_offset += incoming
        self.index_keys[..., prev : self.index_offset, :] = keys
        return self.index_keys[..., : self.index_offset, :]

    def make_mask(self, *args, **kwargs):
        return self.kv_cache.make_mask(*args, **kwargs)

    def size(self):
        return self.kv_cache.size()

    def empty(self):
        return self.kv_cache.empty()

    def is_trimmable(self):
        return True

    def trim(self, n):
        trimmed = self.kv_cache.trim(n)
        self.index_offset = max(0, self.index_offset - trimmed)
        return trimmed

    @classmethod
    def merge(cls, caches, prefix_lens=None):
        return MiniMaxM3BatchKVCache.merge(caches, prefix_lens=prefix_lens)

    @property
    def state(self):
        kv_state = None if self.kv_cache.empty() else self.kv_cache.state
        index_state = (
            None
            if self.index_keys is None
            else self.index_keys[..., : self.index_offset, :]
        )
        return kv_state, index_state

    @state.setter
    def state(self, value):
        kv_state, index_state = value
        self.kv_cache = KVCache()
        if not _is_empty_kv_state(kv_state):
            self.kv_cache.state = kv_state
        self.index_keys = index_state
        self.index_offset = 0 if index_state is None else index_state.shape[2]

    @property
    def meta_state(self):
        return str(self.index_offset)

    @meta_state.setter
    def meta_state(self, value):
        self.index_offset = int(value) if value else 0

    @property
    def nbytes(self):
        index_nbytes = 0 if self.index_keys is None else self.index_keys.nbytes
        return self.kv_cache.nbytes + index_nbytes


class MiniMaxM3BatchKVCache:
    step = BatchKVCache.step

    def __init__(self, left_padding):
        self.kv_cache = BatchKVCache(left_padding)
        self.index_keys = None
        self.index_offset = 0
        self._can_skip_decode_mask = self._all_zero_padding(left_padding)

    @staticmethod
    def _all_zero_padding(padding) -> bool:
        if isinstance(padding, mx.array):
            return False
        try:
            return all(int(value) == 0 for value in padding)
        except TypeError:
            return False

    @property
    def offset(self):
        return self.kv_cache.offset

    @property
    def left_padding(self):
        return self.kv_cache.left_padding

    @property
    def _idx(self):
        return self.kv_cache._idx

    def update_and_fetch(self, keys: mx.array, values: mx.array):
        return self.kv_cache.update_and_fetch(keys, values)

    def update_index_and_fetch(self, keys: mx.array):
        prev = self.index_offset
        incoming = keys.shape[2]
        if self.index_keys is None or (prev + incoming) > self.index_keys.shape[2]:
            B, n_heads, _, head_dim = keys.shape
            n_steps = (self.step + incoming - 1) // self.step
            new_shape = (B, n_heads, n_steps * self.step, head_dim)
            new_keys = mx.zeros(new_shape, keys.dtype)
            if self.index_keys is not None:
                if prev % self.step != 0:
                    self.index_keys = self.index_keys[..., :prev, :]
                self.index_keys = mx.concatenate([self.index_keys, new_keys], axis=2)
            else:
                self.index_keys = new_keys

        self.index_offset += incoming
        self.index_keys[..., prev : self.index_offset, :] = keys
        return self.index_keys[..., : self.index_offset, :]

    def prepare(self, **kwargs):
        left_padding = kwargs.get("left_padding")
        right_padding = kwargs.get("right_padding")
        if (left_padding is not None and not self._all_zero_padding(left_padding)) or (
            right_padding is not None and not self._all_zero_padding(right_padding)
        ):
            self._can_skip_decode_mask = False
        self.kv_cache.prepare(**kwargs)

    def finalize(self):
        right_padding = getattr(self.kv_cache, "_right_padding", None)
        if right_padding is not None:
            self._can_skip_decode_mask = False
        self.kv_cache.finalize()
        if right_padding is not None and self.index_keys is not None:
            self.index_keys = dynamic_roll(
                self.index_keys, right_padding[:, None], axis=2
            )

    def can_skip_decode_mask(self, N: int) -> bool:
        return (
            N == 1
            and self._can_skip_decode_mask
            and getattr(self.kv_cache, "_right_padding", None) is None
        )

    def make_mask(self, *args, **kwargs):
        if args and self.can_skip_decode_mask(int(args[0])):
            return None
        return self.kv_cache.make_mask(*args, **kwargs)

    def filter(self, batch_indices):
        min_left_pad = self.left_padding[batch_indices].min().item()
        self.kv_cache.filter(batch_indices)
        if self.index_keys is not None:
            self.index_keys = self.index_keys[batch_indices]
            if min_left_pad > 0:
                self.index_keys = self.index_keys[..., min_left_pad:, :]
                self.index_offset -= min_left_pad

    def extend(self, other):
        if not isinstance(other, MiniMaxM3BatchKVCache):
            raise TypeError(f"Cannot extend MiniMaxM3BatchKVCache with {type(other)}")
        can_skip_decode_mask = (
            self._can_skip_decode_mask
            and other._can_skip_decode_mask
            and self.kv_cache._idx == other.kv_cache._idx
        )
        self.kv_cache.extend(other.kv_cache)
        self._can_skip_decode_mask = can_skip_decode_mask
        self._extend_index_keys(other)

    def _extend_index_keys(self, other):
        if self.index_keys is None and other.index_keys is None:
            return

        max_idx = max(self.index_offset, other.index_offset)
        self_len = 0 if self.index_keys is None else self.index_keys.shape[2]
        other_len = 0 if other.index_keys is None else other.index_keys.shape[2]
        max_size = max(self_len, other_len)
        ref = self.index_keys if self.index_keys is not None else other.index_keys
        _, heads, _, dims = ref.shape

        def pad(cache):
            keys = cache.index_keys
            if keys is None:
                batch = cache.offset.shape[0]
                keys = mx.zeros((batch, heads, 0, dims), dtype=ref.dtype)
            left = max_idx - cache.index_offset
            right = max_size - keys.shape[2] - left
            if right < 0:
                keys = keys[..., :right, :]
                right = 0
            if left != 0 or right != 0:
                keys = mx.pad(keys, [(0, 0), (0, 0), (left, right), (0, 0)])
            return keys

        self.index_keys = mx.concatenate([pad(self), pad(other)], axis=0)
        self.index_offset = max_idx

    def extract(self, idx):
        cache = MiniMaxM3KVCache()
        cache.kv_cache = self.kv_cache.extract(idx)
        padding = self.left_padding[idx].item()
        if self.index_keys is not None:
            cache.index_keys = mx.contiguous(
                self.index_keys[idx : idx + 1, :, padding : self.index_offset]
            )
            cache.index_offset = cache.index_keys.shape[2]
        return cache

    @classmethod
    def merge(cls, caches, prefix_lens=None):
        caches = list(caches)
        out = cls([0] * len(caches))
        if not caches:
            return out

        out.kv_cache = BatchKVCache.merge([cache.kv_cache for cache in caches])
        lengths = [cache.kv_cache.size() for cache in caches]
        out._can_skip_decode_mask = len(set(lengths)) == 1 and all(
            getattr(cache, "_can_skip_decode_mask", True) for cache in caches
        )
        index_states = [
            (
                None
                if cache.index_keys is None
                or int(getattr(cache, "index_offset", 0) or 0) <= 0
                else cache.index_keys[..., : int(cache.index_offset), :]
            )
            for cache in caches
        ]
        sample = next((state for state in index_states if state is not None), None)
        if sample is None:
            return out

        max_idx = max(
            [int(getattr(out.kv_cache, "_idx", 0) or 0)]
            + [int(v) for v in (prefix_lens or [])]
            + [int(getattr(cache, "index_offset", 0) or 0) for cache in caches]
        )
        _, heads, _, dims = sample.shape
        rows = []
        for cache, state in zip(caches, index_states):
            index_len = int(getattr(cache, "index_offset", 0) or 0)
            if state is None:
                state = mx.zeros((1, heads, 0, dims), dtype=sample.dtype)
                index_len = 0
            left = max_idx - index_len
            right = max_idx - state.shape[2] - left
            if right < 0:
                state = state[..., :right, :]
                right = 0
            if left != 0 or right != 0:
                state = mx.pad(state, [(0, 0), (0, 0), (left, right), (0, 0)])
            rows.append(state)

        out.index_keys = mx.concatenate(rows, axis=0)
        out.index_offset = max_idx
        return out

    def size(self):
        return self.kv_cache.size()

    def empty(self):
        return self.kv_cache.empty()

    def is_trimmable(self):
        return self.kv_cache.is_trimmable()

    def trim(self, n):
        trimmed = self.kv_cache.trim(n)
        self.index_offset = max(0, self.index_offset - trimmed)
        return trimmed

    @property
    def state(self):
        kv_state = (
            (None, None, self.kv_cache.offset, self.kv_cache.left_padding)
            if self.kv_cache.empty()
            else self.kv_cache.state
        )
        index_state = (
            None
            if self.index_keys is None
            else self.index_keys[..., : self.index_offset, :]
        )
        return kv_state, index_state

    @state.setter
    def state(self, value):
        kv_state, index_state = value
        if _is_empty_kv_state(kv_state):
            left_padding = [0] if kv_state is None or len(kv_state) < 4 else kv_state[3]
            self.kv_cache = BatchKVCache(left_padding)
            self._can_skip_decode_mask = self._all_zero_padding(left_padding)
            if kv_state is not None and len(kv_state) >= 3 and kv_state[2] is not None:
                self.kv_cache.offset = kv_state[2]
        else:
            self.kv_cache.state = kv_state
            self._can_skip_decode_mask = False
        self.index_keys = index_state
        self.index_offset = 0 if index_state is None else index_state.shape[2]

    @property
    def meta_state(self):
        return str(self.index_offset)

    @meta_state.setter
    def meta_state(self, value):
        self.index_offset = int(value) if value else 0

    @property
    def nbytes(self):
        index_nbytes = 0 if self.index_keys is None else self.index_keys.nbytes
        return self.kv_cache.nbytes + index_nbytes


class MiniMaxMLP(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        alpha: float = 1.702,
        limit: float = 7.0,
        beta: float = 1.0,
        bias: bool = False,
    ):
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=bias)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=bias)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=bias)
        self.act_fn = MiniMaxSwiGLUOAI(alpha, limit, beta)

    def __call__(self, x):
        return self.down_proj(self.act_fn(self.up_proj(x), self.gate_proj(x)))


class MiniMaxPackedSwitchGLU(nn.Module):
    def __init__(
        self,
        input_dims: int,
        hidden_dims: int,
        num_experts: int,
        activation: MiniMaxSwiGLUOAI,
        bias: bool = False,
    ):
        super().__init__()
        self.gate_up_proj = SwitchLinear(
            input_dims, 2 * hidden_dims, num_experts, bias=bias
        )
        self.down_proj = SwitchLinear(hidden_dims, input_dims, num_experts, bias=bias)
        self.activation = activation

    def __call__(self, x: mx.array, indices: mx.array) -> mx.array:
        x = mx.expand_dims(x, (-2, -3))

        do_sort = indices.size >= 6
        idx = indices
        inv_order = None
        if do_sort:
            x, idx, inv_order = _switch_gather_sort(x, indices)
        if self.training:
            idx = mx.stop_gradient(idx)

        gate_up = self.gate_up_proj(x, idx, sorted_indices=do_sort)
        gate, up = mx.split(gate_up, 2, axis=-1)
        x = self.down_proj(
            self.activation(up, gate),
            idx,
            sorted_indices=do_sort,
        )

        if do_sort:
            x = _switch_scatter_unsort(x, inv_order, indices.shape)

        return x.squeeze(-2)


class MiniMaxAttention(nn.Module):
    def __init__(self, args: TextConfig, layer_idx: int):
        super().__init__()
        self.hidden_dim = args.hidden_size
        self.num_attention_heads = args.num_attention_heads
        self.num_key_value_heads = args.num_key_value_heads
        self.head_dim = args.head_dim or args.hidden_size // args.num_attention_heads
        self.scale = self.head_dim**-0.5
        self.use_qk_norm = args.use_qk_norm

        self.q_proj = nn.Linear(
            args.hidden_size, self.num_attention_heads * self.head_dim, bias=False
        )
        self.k_proj = nn.Linear(
            args.hidden_size, self.num_key_value_heads * self.head_dim, bias=False
        )
        self.v_proj = nn.Linear(
            args.hidden_size, self.num_key_value_heads * self.head_dim, bias=False
        )
        self.o_proj = nn.Linear(
            self.num_attention_heads * self.head_dim, args.hidden_size, bias=False
        )

        if self.use_qk_norm:
            self.q_norm = MiniMaxRMSNorm(
                self.head_dim, eps=args.rms_norm_eps, gemma=args.use_gemma_norm
            )
            self.k_norm = MiniMaxRMSNorm(
                self.head_dim, eps=args.rms_norm_eps, gemma=args.use_gemma_norm
            )

        self.has_sparse_index = args.has_sparse_index(layer_idx)
        if self.has_sparse_index:
            sparse_config = args.sparse_attention_config
            self.sparse_block_size = sparse_config.get("sparse_block_size", 128)
            self.sparse_topk_blocks = sparse_config.get("sparse_topk_blocks", 16)
            self.sparse_init_blocks = sparse_config.get("sparse_init_block", 0)
            self.sparse_local_blocks = sparse_config.get("sparse_local_block", 1)
            self.sparse_score_type = sparse_config.get("sparse_score_type", "max")
            self.index_dim = sparse_config.get("sparse_index_dim", self.head_dim)
            self.index_heads = sparse_config.get("sparse_num_index_heads", 4)
            self.index_q_proj = nn.Linear(
                args.hidden_size, self.index_heads * self.index_dim, bias=False
            )
            self.index_k_proj = nn.Linear(args.hidden_size, self.index_dim, bias=False)
            self.index_q_norm = MiniMaxRMSNorm(
                self.index_dim, eps=args.rms_norm_eps, gemma=args.use_gemma_norm
            )
            self.index_k_norm = MiniMaxRMSNorm(
                self.index_dim, eps=args.rms_norm_eps, gemma=args.use_gemma_norm
            )
            self.indexer = MiniMaxM3Indexer(self)
        else:
            self.indexer = None

        self.rope = initialize_rope(
            args.rotary_dim,
            args.rope_theta,
            traditional=False,
            scaling_config=args.rope_scaling,
            max_position_embeddings=args.max_position_embeddings,
        )

    @staticmethod
    def _sparse_query_positions(
        B: int,
        L: int,
        q_start: int,
        q_positions: Optional[mx.array] = None,
    ):
        if q_positions is None:
            positions = mx.arange(q_start, q_start + L)
            return mx.broadcast_to(positions[None, :], (B, L))

        positions = q_positions
        if positions.ndim == 0:
            positions = positions.reshape(1, 1)
        elif positions.ndim == 1:
            if positions.shape[0] == B and L == 1:
                positions = positions[:, None]
            else:
                positions = positions[None, :]
        elif positions.ndim == 3:
            positions = positions[0]

        if positions.shape[-1] != L:
            positions = positions[..., -L:]
        if positions.shape[0] == 1 and B != 1:
            positions = mx.broadcast_to(positions, (B, L))
        elif positions.shape[0] != B:
            positions = positions.reshape(B, L)
        return positions

    def _build_sparse_mask(
        self,
        idx_queries: mx.array,
        idx_keys: mx.array,
        q_start: int,
        mask: Optional[mx.array] = None,
        return_block_indices: bool = False,
        build_token_mask: bool = True,
        q_positions: Optional[mx.array] = None,
    ):
        B, H_idx, L, _ = idx_queries.shape
        total_len = idx_keys.shape[2]
        qpos = self._sparse_query_positions(B, L, q_start, q_positions)
        if (
            build_token_mask
            and (mask is None or isinstance(mask, str))
            and self.sparse_score_type == "max"
            and (total_len + self.sparse_block_size - 1) // self.sparse_block_size
            >= self.sparse_topk_blocks
        ):
            if self.num_attention_heads % H_idx != 0:
                raise ValueError(
                    "MiniMax M3 sparse index heads must divide attention heads: "
                    f"{H_idx} index heads, {self.num_attention_heads} attention heads."
                )
            sparse_mask, topk_idx, topk_valid = _build_sparse_causal_mask_compiled(
                idx_queries,
                idx_keys,
                qpos,
                self.scale,
                self.sparse_block_size,
                self.sparse_topk_blocks,
                self.sparse_init_blocks,
                self.sparse_local_blocks,
            )
            if return_block_indices:
                return sparse_mask, topk_idx, topk_valid
            return sparse_mask

        scale = self.scale
        neg = mx.array(-float("inf"), dtype=mx.float32)

        scores = mx.matmul(
            idx_queries.astype(mx.float32),
            idx_keys.astype(mx.float32).swapaxes(-1, -2),
        )
        scores = scores * scale

        kpos = mx.arange(total_len)
        causal = kpos[None, None, :] <= qpos[:, :, None]
        scores = mx.where(causal[:, None], scores, neg)
        valid = self._selection_valid_mask(mask, B, H_idx, L, total_len)
        if valid is not None:
            scores = mx.where(valid, scores, neg)

        block_size = self.sparse_block_size
        num_blocks = (total_len + block_size - 1) // block_size
        pad = num_blocks * block_size - total_len
        if pad:
            pad_values = mx.full(
                (*scores.shape[:-1], pad), -float("inf"), dtype=scores.dtype
            )
            scores = mx.concatenate([scores, pad_values], axis=-1)
            if valid is not None:
                valid_pad = mx.zeros((*valid.shape[:-1], pad), dtype=mx.bool_)
                valid = mx.concatenate([valid, valid_pad], axis=-1)

        blocks = mx.arange(num_blocks)
        cur_block = qpos // block_size
        causal_block = blocks[None, None, :] <= cur_block[:, :, None]

        scores = scores.reshape(B, H_idx, L, num_blocks, block_size)
        if valid is not None:
            valid_blocks = mx.any(
                valid.reshape(B, valid.shape[1], L, num_blocks, block_size),
                axis=-1,
            )
        else:
            valid_blocks = causal_block[:, None]

        if self.sparse_score_type == "lse":
            block_scores = mx.logsumexp(scores, axis=-1)
        else:
            block_scores = mx.max(scores, axis=-1)
        block_scores = mx.max(block_scores, axis=1)
        block_scores = mx.where(block_scores == block_scores, block_scores, neg)
        if valid_blocks.shape[1] == 1:
            valid_blocks = valid_blocks[:, 0]
        else:
            valid_blocks = mx.any(valid_blocks, axis=1)
        valid_blocks = valid_blocks & causal_block

        init_blocks = blocks[None, None, :] < self.sparse_init_blocks
        if self.sparse_local_blocks > 0:
            local_start = mx.maximum(cur_block - self.sparse_local_blocks + 1, 0)
            local_blocks = (blocks[None, None, :] >= local_start[:, :, None]) & (
                blocks[None, None, :] <= cur_block[:, :, None]
            )
        else:
            local_blocks = mx.zeros((B, L, num_blocks), dtype=mx.bool_)

        selected_scores = mx.where(valid_blocks, block_scores, neg)
        forced_init_blocks = (init_blocks & causal_block) & valid_blocks
        forced_local_blocks = (local_blocks & causal_block) & valid_blocks
        selected_scores = mx.where(
            forced_init_blocks,
            mx.array(1e30, dtype=selected_scores.dtype),
            selected_scores,
        )
        selected_scores = mx.where(
            forced_local_blocks,
            mx.array(1e29, dtype=selected_scores.dtype),
            selected_scores,
        )

        k_eff = min(self.sparse_topk_blocks, num_blocks)
        topk_idx = mx.argpartition(-selected_scores, kth=k_eff - 1, axis=-1)[
            ..., :k_eff
        ]
        topk_valid = mx.take_along_axis(valid_blocks, topk_idx, axis=-1)
        if not build_token_mask:
            return None, topk_idx[:, None], topk_valid[:, None]

        block_selected = mx.any(topk_idx[..., None] == blocks, axis=-2)
        block_selected = block_selected & valid_blocks

        key_blocks = (kpos // block_size).astype(mx.int32)
        key_blocks = mx.broadcast_to(key_blocks[None, None, :], (B, L, total_len))
        key_selected = mx.take_along_axis(block_selected, key_blocks, axis=-1)
        key_selected = key_selected & causal

        sparse_mask = key_selected[:, None]
        sparse_mask = self._merge_sparse_mask(sparse_mask, mask)
        if return_block_indices:
            return sparse_mask, topk_idx[:, None], topk_valid[:, None]
        return sparse_mask

    @staticmethod
    def _expand_2d_mask(mask: mx.array, B: int, L: int, total_len: int):
        if mask.shape[-1] != total_len:
            mask = mask[..., :total_len]

        if mask.shape[0] == B:
            return mask[:, None, None, :]
        if mask.shape[0] == L:
            return mask[None, None, :, :]
        return mask[None, None, :, :]

    @staticmethod
    def _normalize_attention_mask(
        mask: Optional[mx.array],
        B: int,
        L: int,
        total_len: int,
        *,
        causal: bool = False,
    ):
        if mask is None or isinstance(mask, str):
            return mask
        if _is_integer_mask(mask):
            mask = mask.astype(mx.bool_)
        if mask.ndim == 2:
            mask = MiniMaxAttention._expand_2d_mask(mask, B, L, total_len)
        elif mask.ndim == 3:
            if mask.shape[-1] != total_len:
                mask = mask[..., :total_len]
            if mask.shape[0] == B:
                mask = mask[:, None, :, :]
            else:
                mask = mask[None, :, :, :]
        elif mask.shape[-1] != total_len:
            mask = mask[..., :total_len]
        if causal:
            mask = MiniMaxAttention._merge_causal_mask(mask, L, total_len)
        return mask

    @staticmethod
    def _merge_causal_mask(mask: mx.array, L: int, total_len: int):
        qpos = mx.arange(total_len - L, total_len)
        kpos = mx.arange(total_len)
        causal_mask = kpos[None, :] <= qpos[:, None]
        causal_mask = causal_mask[None, None, :, :]
        if _is_bool_mask(mask):
            return mask & causal_mask
        causal_bias = mx.where(
            causal_mask,
            mx.array(0.0, dtype=mask.dtype),
            mx.array(-float("inf"), dtype=mask.dtype),
        )
        return mask + causal_bias

    @staticmethod
    def _selection_valid_mask(
        mask: Optional[mx.array], B: int, H_idx: int, L: int, total_len: int
    ):
        valid = MiniMaxAttention._normalize_attention_mask(mask, B, L, total_len)
        if valid is None or isinstance(valid, str):
            return None

        if not _is_bool_mask(valid):
            valid = valid.astype(mx.float32) > mx.array(-1e20, dtype=mx.float32)

        if valid.shape[1] != 1 and valid.shape[1] != H_idx:
            valid = mx.any(valid, axis=1, keepdims=True)

        heads = H_idx if valid.shape[1] == H_idx else 1
        valid = mx.broadcast_to(valid, (B, heads, L, total_len))
        return valid

    @staticmethod
    def _merge_sparse_mask(sparse_mask: mx.array, mask: Optional[mx.array]):
        B, _, L, total_len = sparse_mask.shape
        mask = MiniMaxAttention._normalize_attention_mask(mask, B, L, total_len)
        if mask is None or isinstance(mask, str):
            return sparse_mask
        if _is_bool_mask(mask):
            return sparse_mask & mask
        sparse_bias = mx.where(
            sparse_mask,
            mx.array(0.0, dtype=mask.dtype),
            mx.array(-float("inf"), dtype=mask.dtype),
        )
        return sparse_bias + mask

    def _sparse_prefill_attention(
        self,
        queries: mx.array,
        keys: mx.array,
        values: mx.array,
        block_indices: mx.array,
        mask: Optional[mx.array],
        q_positions: mx.array,
    ):
        if (
            queries.shape[0] != 1
            or queries.shape[2] <= 1
            or (mask is not None and not isinstance(mask, str))
        ):
            return None
        return _minimax_m3_sparse_prefill_attention(
            queries,
            keys,
            values,
            block_indices,
            q_positions,
            self.scale,
            self.sparse_block_size,
        )

    def _sparse_decode_attention(
        self,
        queries: mx.array,
        keys: mx.array,
        values: mx.array,
        block_indices: mx.array,
        mask: Optional[mx.array],
        q_positions: mx.array,
    ):
        if (
            queries.shape[2] != 1
            or block_indices.shape[1] != 1
            or q_positions.shape[1] != 1
            or keys.shape != values.shape
        ):
            return None

        B, _, _, _ = queries.shape
        key_length = keys.shape[2]
        selected_length = block_indices.shape[-1] * self.sparse_block_size
        has_explicit_mask = mask is not None and not isinstance(mask, str)
        # Dense SDPA is still faster before these measured crossover points.
        min_sparse_length = selected_length * (16 if has_explicit_mask else 64)
        if selected_length >= key_length or key_length < min_sparse_length:
            return None

        if not has_explicit_mask:
            output = _minimax_m3_sparse_prefill_attention(
                queries,
                keys,
                values,
                block_indices,
                q_positions,
                self.scale,
                self.sparse_block_size,
            )
            if output is not None:
                return output

        block_indices = block_indices.astype(mx.int32)
        offsets = mx.arange(self.sparse_block_size, dtype=mx.int32)
        token_indices = block_indices[..., None] * self.sparse_block_size + offsets
        valid = (
            (block_indices[..., None] >= 0)
            & (token_indices < key_length)
            & (token_indices <= q_positions[..., None, None].astype(mx.int32))
        )

        token_indices = token_indices.reshape(B, 1, selected_length)
        valid = valid.reshape(B, 1, selected_length)
        safe_indices = mx.where(valid, token_indices, mx.zeros_like(token_indices))

        gather_indices = mx.broadcast_to(
            safe_indices[:, None, 0, :, None],
            (B, keys.shape[1], selected_length, keys.shape[3]),
        )
        compact_keys = mx.take_along_axis(keys, gather_indices, axis=2)
        compact_values = mx.take_along_axis(values, gather_indices, axis=2)

        compact_mask = valid[:, None]
        if mask is not None and not isinstance(mask, str):
            mask = self._normalize_attention_mask(mask, B, 1, key_length)
            mask_indices = mx.broadcast_to(
                safe_indices[:, None, :, :],
                (*mask.shape[:-1], selected_length),
            )
            gathered_mask = mx.take_along_axis(mask, mask_indices, axis=-1)
            if _is_bool_mask(gathered_mask):
                compact_mask = compact_mask & gathered_mask
            else:
                sparse_bias = mx.where(
                    compact_mask,
                    mx.array(0.0, dtype=gathered_mask.dtype),
                    mx.array(-float("inf"), dtype=gathered_mask.dtype),
                )
                compact_mask = sparse_bias + gathered_mask

        return scaled_dot_product_attention(
            queries,
            compact_keys,
            compact_values,
            cache=None,
            scale=self.scale,
            mask=compact_mask,
        )

    def __call__(
        self,
        x: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional[KVCache] = None,
        position_ids: Optional[mx.array] = None,
    ) -> mx.array:
        B, L, _ = x.shape
        offset = 0 if cache is None else cache.offset
        rope_offset = offset
        if position_ids is not None:
            if position_ids.ndim == 2:
                rope_offset = position_ids[:, 0]
            else:
                rope_offset = position_ids[0] if position_ids.ndim > 0 else position_ids
        sparse_q_start = 0 if cache is None else getattr(cache, "index_offset", offset)
        sparse_q_positions = None
        if position_ids is not None:
            sparse_q_positions = self._sparse_query_positions(B, L, 0, position_ids)
            left_padding = getattr(cache, "left_padding", None)
            if isinstance(left_padding, mx.array) and left_padding.ndim > 0:
                sparse_q_positions = sparse_q_positions + left_padding[:B, None].astype(
                    sparse_q_positions.dtype
                )
        use_sparse_mask = self.has_sparse_index and (
            cache is None or hasattr(cache, "update_index_and_fetch")
        )

        queries = self.q_proj(x)
        keys = self.k_proj(x)
        values = self.v_proj(x)
        queries = queries.reshape(B, L, self.num_attention_heads, self.head_dim)
        keys = keys.reshape(B, L, self.num_key_value_heads, self.head_dim)
        values = values.reshape(B, L, self.num_key_value_heads, self.head_dim)

        if self.use_qk_norm:
            queries = self.q_norm(queries)
            keys = self.k_norm(keys)

        queries = queries.transpose(0, 2, 1, 3)
        keys = keys.transpose(0, 2, 1, 3)
        values = values.transpose(0, 2, 1, 3)

        if cache is not None:
            queries = self.rope(queries, offset=rope_offset)
            keys = self.rope(keys, offset=rope_offset)
            keys, values = cache.update_and_fetch(keys, values)
        else:
            queries = self.rope(queries, offset=rope_offset)
            keys = self.rope(keys, offset=rope_offset)

        if use_sparse_mask:
            mask = self._normalize_attention_mask(
                mask, B, L, keys.shape[2], causal=sparse_q_positions is None
            )
            block_indices, _, q_positions = self.indexer(
                x,
                self.rope,
                rope_offset,
                cache,
                int(sparse_q_start),
                mask,
                sparse_q_positions,
            )
            if block_indices is not None:
                output = self._sparse_prefill_attention(
                    queries,
                    keys,
                    values,
                    block_indices,
                    mask,
                    q_positions,
                )
                if output is not None:
                    output = output.transpose(0, 2, 1, 3).reshape(B, L, -1)
                    return self.o_proj(output)
                output = self._sparse_decode_attention(
                    queries,
                    keys,
                    values,
                    block_indices,
                    mask,
                    q_positions,
                )
                if output is not None:
                    output = output.transpose(0, 2, 1, 3).reshape(B, L, -1)
                    return self.o_proj(output)
                mask = self.indexer.build_block_mask(
                    block_indices,
                    mask,
                    keys.shape[2],
                    queries.dtype,
                    q_positions,
                )
        else:
            mask = self._normalize_attention_mask(
                mask, B, L, keys.shape[2], causal=True
            )

        output = scaled_dot_product_attention(
            queries, keys, values, cache=cache, scale=self.scale, mask=mask
        )
        output = output.transpose(0, 2, 1, 3).reshape(B, L, -1)
        return self.o_proj(output)


class MiniMaxM3Indexer:
    """MiniMax M3 sparse block selector.

    The projection modules live on MiniMaxAttention to preserve existing MLX
    checkpoint keys. This helper keeps the DeepSeek-V4/HF shape: attention asks
    the indexer for selected key blocks, then expands those blocks into a mask.
    """

    def __init__(self, attention):
        self.attention = attention
        self.block_size = attention.sparse_block_size
        self.topk_blocks = attention.sparse_topk_blocks
        self.init_blocks = attention.sparse_init_blocks
        self.local_blocks = attention.sparse_local_blocks
        self.score_type = attention.sparse_score_type

    def __call__(
        self,
        x: mx.array,
        rope,
        rope_offset,
        cache,
        q_start: int,
        mask: Optional[mx.array],
        q_positions: Optional[mx.array],
    ):
        attention = self.attention
        B, L, _ = x.shape
        idx_queries = attention.index_q_proj(x)
        idx_keys = attention.index_k_proj(x)
        idx_queries = idx_queries.reshape(
            B, L, attention.index_heads, attention.index_dim
        )
        idx_keys = idx_keys.reshape(B, L, 1, attention.index_dim)
        idx_queries = attention.index_q_norm(idx_queries).transpose(0, 2, 1, 3)
        idx_keys = attention.index_k_norm(idx_keys).transpose(0, 2, 1, 3)
        idx_queries = rope(idx_queries, offset=rope_offset)
        idx_keys = rope(idx_keys, offset=rope_offset)

        if cache is not None and hasattr(cache, "update_index_and_fetch"):
            idx_keys = cache.update_index_and_fetch(idx_keys)

        q_positions = attention._sparse_query_positions(B, L, q_start, q_positions)
        total_len = idx_keys.shape[2]
        if total_len <= self.block_size * self.topk_blocks:
            return None, total_len, q_positions

        block_indices = self.select_blocks(
            idx_queries,
            idx_keys,
            q_start,
            mask,
            q_positions,
        )
        return block_indices, total_len, q_positions

    def select_blocks(
        self,
        idx_queries: mx.array,
        idx_keys: mx.array,
        q_start: int,
        mask: Optional[mx.array] = None,
        q_positions: Optional[mx.array] = None,
    ):
        attention = self.attention
        B, H_idx, L, _ = idx_queries.shape
        total_len = idx_keys.shape[2]
        qpos = attention._sparse_query_positions(B, L, q_start, q_positions)
        num_blocks = (total_len + self.block_size - 1) // self.block_size
        if (
            (mask is None or isinstance(mask, str))
            and self.score_type == "max"
            and num_blocks >= self.topk_blocks
        ):
            if attention.num_attention_heads % H_idx != 0:
                raise ValueError(
                    "MiniMax M3 sparse index heads must divide attention heads: "
                    f"{H_idx} index heads, {attention.num_attention_heads} "
                    "attention heads."
                )
            return _select_sparse_block_indices_compiled(
                idx_queries,
                idx_keys,
                qpos,
                attention.scale,
                self.block_size,
                self.topk_blocks,
                self.init_blocks,
                self.local_blocks,
            )

        scale = attention.scale
        neg = mx.array(-float("inf"), dtype=mx.float32)
        scores = mx.matmul(
            idx_queries.astype(mx.float32),
            idx_keys.astype(mx.float32).swapaxes(-1, -2),
        )
        scores = scores * scale

        kpos = mx.arange(total_len)
        causal = kpos[None, None, :] <= qpos[:, :, None]
        scores = mx.where(causal[:, None], scores, neg)
        valid = attention._selection_valid_mask(mask, B, H_idx, L, total_len)
        if valid is not None:
            scores = mx.where(valid, scores, neg)

        pad = num_blocks * self.block_size - total_len
        if pad:
            pad_values = mx.full(
                (*scores.shape[:-1], pad), -float("inf"), dtype=scores.dtype
            )
            scores = mx.concatenate([scores, pad_values], axis=-1)
            if valid is not None:
                valid_pad = mx.zeros((*valid.shape[:-1], pad), dtype=mx.bool_)
                valid = mx.concatenate([valid, valid_pad], axis=-1)

        blocks = mx.arange(num_blocks)
        cur_block = qpos // self.block_size
        causal_block = blocks[None, None, :] <= cur_block[:, :, None]
        scores = scores.reshape(B, H_idx, L, num_blocks, self.block_size)
        if valid is not None:
            valid_blocks = mx.any(
                valid.reshape(B, valid.shape[1], L, num_blocks, self.block_size),
                axis=-1,
            )
        else:
            valid_blocks = causal_block[:, None]

        if self.score_type == "lse":
            block_scores = mx.logsumexp(scores, axis=-1)
        else:
            block_scores = mx.max(scores, axis=-1)
        block_scores = mx.max(block_scores, axis=1)
        block_scores = mx.where(block_scores == block_scores, block_scores, neg)
        if valid_blocks.shape[1] == 1:
            valid_blocks = valid_blocks[:, 0]
        else:
            valid_blocks = mx.any(valid_blocks, axis=1)
        valid_blocks = valid_blocks & causal_block

        selected_scores = mx.where(valid_blocks, block_scores, neg)
        if self.init_blocks > 0:
            init_blocks = (blocks[None, None, :] < self.init_blocks) & valid_blocks
            selected_scores = mx.where(
                init_blocks,
                mx.array(1e30, dtype=selected_scores.dtype),
                selected_scores,
            )
        if self.local_blocks > 0:
            local_start = mx.maximum(cur_block - self.local_blocks + 1, 0)
            local_blocks = (blocks[None, None, :] >= local_start[:, :, None]) & (
                blocks[None, None, :] <= cur_block[:, :, None]
            )
            selected_scores = mx.where(
                local_blocks & valid_blocks,
                mx.array(1e29, dtype=selected_scores.dtype),
                selected_scores,
            )

        topk = min(self.topk_blocks, num_blocks)
        topk_idx = mx.argpartition(-selected_scores, kth=topk - 1, axis=-1)[..., :topk]
        topk_valid = mx.take_along_axis(valid_blocks, topk_idx, axis=-1)
        return self._pack_block_indices(topk_idx, topk_valid, num_blocks)

    @staticmethod
    def _pack_block_indices(
        topk_idx: mx.array,
        topk_valid: mx.array,
        num_blocks: int,
    ):
        invalid = mx.full(topk_idx.shape, num_blocks, dtype=topk_idx.dtype)
        block_indices = mx.where(topk_valid, topk_idx, invalid)
        order = mx.argsort(block_indices, axis=-1)
        block_indices = mx.take_along_axis(block_indices, order, axis=-1)
        return mx.where(block_indices == num_blocks, mx.array(-1), block_indices)

    def build_block_mask(
        self,
        block_indices: mx.array,
        mask: Optional[mx.array],
        key_length: int,
        dtype,
        q_positions: mx.array,
    ):
        del dtype
        B, L, _ = block_indices.shape
        num_blocks = (key_length + self.block_size - 1) // self.block_size
        blocks = mx.arange(num_blocks, dtype=block_indices.dtype)
        block_keep = mx.any(block_indices[..., None] == blocks, axis=-2)

        kpos = mx.arange(key_length)
        key_blocks = (kpos // self.block_size).astype(block_indices.dtype)
        key_blocks = mx.broadcast_to(key_blocks[None, None, :], (B, L, key_length))
        key_keep = mx.take_along_axis(block_keep, key_blocks, axis=-1)
        causal = kpos[None, None, :] <= q_positions[:, :, None]
        sparse_mask = (key_keep & causal)[:, None]
        return self.attention._merge_sparse_mask(sparse_mask, mask)


class MiniMaxSparseMoeBlock(nn.Module):
    def __init__(self, args: TextConfig):
        super().__init__()
        self.num_experts_per_tok = args.num_experts_per_tok
        self.routed_scaling_factor = args.routed_scaling_factor
        self.scoring_func = args.scoring_func
        self.shared_expert_index = args.num_local_experts
        self.pack_shared_expert = (
            (
                args.n_shared_experts == 1
                and args.shared_intermediate_size == args.intermediate_size
            )
            if getattr(args, "pack_shared_expert", None) is None
            else args.pack_shared_expert
        )

        self.gate = nn.Linear(args.hidden_size, args.num_local_experts, bias=False)
        activation = MiniMaxSwiGLUOAI(
            args.swiglu_alpha,
            args.swiglu_limit,
            args.swiglu_beta,
        )
        if self.pack_shared_expert:
            self.switch_mlp = MiniMaxPackedSwitchGLU(
                args.hidden_size,
                args.intermediate_size,
                args.num_local_experts + 1,
                activation=activation,
            )
        else:
            self.switch_mlp = SwitchGLU(
                args.hidden_size,
                args.intermediate_size,
                args.num_local_experts,
                activation=activation,
            )
        self.shared_experts = (
            MiniMaxMLP(
                args.hidden_size,
                args.shared_intermediate_size,
                args.swiglu_alpha,
                args.swiglu_limit,
                args.swiglu_beta,
                bias=False,
            )
            if args.n_shared_experts and not self.pack_shared_expert
            else None
        )
        self.e_score_correction_bias = (
            mx.zeros((args.num_local_experts,)) if args.use_routing_bias else None
        )
        self.sharding_group = None

    def __call__(self, x: mx.array) -> mx.array:
        if self.sharding_group is not None:
            x = sum_gradients(self.sharding_group)(x)

        gates = self.gate(x.astype(mx.float32))
        if self.e_score_correction_bias is not None:
            inds, scores = _minimax_moe_select(
                gates,
                self.e_score_correction_bias,
                self.num_experts_per_tok,
                self.routed_scaling_factor,
                self.scoring_func,
            )
            scores = scores.astype(x.dtype)
        else:
            if self.scoring_func == "sigmoid":
                scores = mx.sigmoid(gates)
            else:
                scores = mx.softmax(gates, axis=-1, precise=True)

            k = self.num_experts_per_tok
            inds = mx.argpartition(-scores, kth=k - 1, axis=-1)[..., :k]
            scores = mx.take_along_axis(scores, inds, axis=-1)
            scores = scores / (mx.sum(scores, axis=-1, keepdims=True) + 1e-20)
            scores = (scores * self.routed_scaling_factor).astype(x.dtype)
        if self.pack_shared_expert:
            shared_inds = mx.full(
                (*inds.shape[:-1], 1), self.shared_expert_index, dtype=inds.dtype
            )
            shared_scores = mx.ones((*scores.shape[:-1], 1), dtype=scores.dtype)
            inds = mx.concatenate([inds, shared_inds], axis=-1)
            scores = mx.concatenate([scores, shared_scores], axis=-1)

        y = self.switch_mlp(x, inds)
        y = (y * scores[..., None]).sum(axis=-2)

        if self.sharding_group is not None:
            y = mx.distributed.all_sum(y, group=self.sharding_group)

        if self.shared_experts is not None:
            y = y + self.shared_experts(x)
        return y


class MiniMaxDecoderLayer(nn.Module):
    def __init__(self, args: TextConfig, layer_idx: int):
        super().__init__()
        self.self_attn = MiniMaxAttention(args, layer_idx)
        self.input_layernorm = MiniMaxRMSNorm(
            args.hidden_size, eps=args.rms_norm_eps, gemma=args.use_gemma_norm
        )
        self.post_attention_layernorm = MiniMaxRMSNorm(
            args.hidden_size, eps=args.rms_norm_eps, gemma=args.use_gemma_norm
        )
        self.is_moe_layer = args.is_moe_layer(layer_idx)
        if self.is_moe_layer:
            self.block_sparse_moe = MiniMaxSparseMoeBlock(args)
        else:
            self.mlp = MiniMaxMLP(
                args.hidden_size,
                args.dense_intermediate_size,
                args.swiglu_alpha,
                args.swiglu_limit,
                args.swiglu_beta,
                bias=False,
            )

    def __call__(
        self,
        x: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional[KVCache] = None,
        position_ids: Optional[mx.array] = None,
    ) -> mx.array:
        h = x + self.self_attn(
            self.input_layernorm(x), mask, cache, position_ids=position_ids
        )
        mlp = self.block_sparse_moe if self.is_moe_layer else self.mlp
        return h + mlp(self.post_attention_layernorm(h))


class MiniMaxM3Model(nn.Module):
    def __init__(self, args: TextConfig):
        super().__init__()
        self.args = args
        self.embed_tokens = nn.Embedding(args.vocab_size, args.hidden_size)
        self.layers = [
            MiniMaxDecoderLayer(args=args, layer_idx=layer_idx)
            for layer_idx in range(args.num_hidden_layers)
        ]
        self.norm = MiniMaxRMSNorm(
            args.hidden_size, eps=args.rms_norm_eps, gemma=args.use_gemma_norm
        )

    def __call__(
        self,
        inputs: mx.array,
        inputs_embeds: Optional[mx.array] = None,
        mask: Optional[mx.array] = None,
        cache=None,
        capture_layer_ids: Optional[List[int]] = None,
        hidden_sink: Optional[list] = None,
        position_ids: Optional[mx.array] = None,
    ) -> mx.array:
        h = self.embed_tokens(inputs) if inputs_embeds is None else inputs_embeds
        if cache is None:
            cache = [None] * len(self.layers)
        if mask is None:
            first_cache = cache[0] if cache and cache[0] is not None else cache
            if not _can_skip_uniform_batch_decode_mask(cache, h.shape[1]):
                mask = create_attention_mask(h, first_cache)

        capture_set = set(capture_layer_ids) if capture_layer_ids else set()
        for idx, (layer, c) in enumerate(zip(self.layers, cache)):
            h = layer(h, mask, c, position_ids=position_ids)
            if hidden_sink is not None and idx in capture_set:
                hidden_sink.append(h)

        if hidden_sink is not None and not capture_set:
            hidden_sink.append(h)

        return self.norm(h)


class LanguageModel(nn.Module):
    def __init__(self, args: TextConfig, config: Optional[ModelConfig] = None):
        super().__init__()
        self.args = args
        self.config = config
        self.model_type = args.model_type
        self.model = MiniMaxM3Model(args)
        self._position_ids = None
        self._rope_deltas = None
        if not args.tie_word_embeddings:
            self.lm_head = nn.Linear(args.hidden_size, args.vocab_size, bias=False)

    def get_rope_index(
        self,
        input_ids: mx.array,
        image_grid_thw: Optional[mx.array] = None,
        video_grid_thw: Optional[mx.array] = None,
        attention_mask: Optional[mx.array] = None,
    ):
        del image_grid_thw, video_grid_thw
        batch_size, seq_length = input_ids.shape
        positions = mx.arange(seq_length, dtype=mx.int32)
        positions = mx.broadcast_to(positions[None, :], (batch_size, seq_length))

        if attention_mask is not None:
            seen_token = mx.cumsum(attention_mask.astype(mx.int32), axis=-1) > 0
            left_padding = seq_length - mx.sum(seen_token.astype(mx.int32), axis=-1)
            positions = positions - left_padding.astype(positions.dtype)[:, None]

        rope_deltas = mx.zeros((batch_size, 1), dtype=positions.dtype)
        return positions, rope_deltas

    @staticmethod
    def _first_cache(cache):
        if not cache:
            return None
        for entry in cache:
            if entry is not None:
                return entry
        return None

    @staticmethod
    def _position_ids_from_cache_offsets(
        offsets: mx.array,
        seq_length: int,
    ) -> mx.array:
        offsets = mx.maximum(offsets, 0).astype(mx.int32)
        positions = mx.arange(seq_length, dtype=mx.int32)
        return offsets[:, None] + positions[None, :]

    @staticmethod
    def _scalar_cache_offset(cache_offset) -> int:
        if isinstance(cache_offset, mx.array):
            if cache_offset.ndim == 0:
                return int(cache_offset.item())
            return 0
        return int(cache_offset)

    def _slice_cached_position_ids(
        self,
        batch_size: int,
        seq_length: int,
        cache_offset: int,
    ):
        cached = self._position_ids
        if (
            cached is not None
            and cached.ndim == 2
            and cached.shape[0] == batch_size
            and cached.shape[-1] >= cache_offset + seq_length
        ):
            return cached[:, cache_offset : cache_offset + seq_length]
        return None

    @staticmethod
    def _offset_position_ids(cache_offset, batch_size: int, seq_length: int):
        if isinstance(cache_offset, mx.array):
            if cache_offset.ndim > 0:
                return LanguageModel._position_ids_from_cache_offsets(
                    cache_offset[:batch_size], seq_length
                )
            cache_offset = int(cache_offset.item())
        cache_offset = int(cache_offset)
        base = mx.arange(cache_offset, cache_offset + seq_length, dtype=mx.int32)
        return mx.broadcast_to(base[None, :], (batch_size, seq_length))

    def __call__(
        self,
        inputs: mx.array,
        inputs_embeds: Optional[mx.array] = None,
        mask: Optional[mx.array] = None,
        cache=None,
        **kwargs,
    ) -> LanguageModelOutput:
        capture_layer_ids = kwargs.pop("capture_layer_ids", None)
        return_hidden = kwargs.pop("return_hidden", False)
        return_shared_kv = kwargs.pop("return_shared_kv", False)
        skip_logits = kwargs.pop("skip_logits", False)
        position_ids = kwargs.pop("position_ids", None)
        attention_mask = kwargs.pop("attention_mask", None)
        image_grid_thw = kwargs.pop("image_grid_thw", None)
        video_grid_thw = kwargs.pop("video_grid_thw", None)
        kwargs.pop("visual_pos_masks", None)
        rope_deltas_kw = kwargs.pop("rope_deltas", None)
        if (
            mask is None
            and attention_mask is not None
            and attention_mask.shape[-1] == inputs.shape[-1]
        ):
            mask = attention_mask

        first_cache = self._first_cache(cache)
        can_skip_batch_positions = (
            position_ids is None
            and inputs.shape[0] > 1
            and mask is None
            and attention_mask is None
            and first_cache is not None
            and _can_skip_uniform_batch_decode_mask(cache, inputs.shape[-1])
        )

        if (
            position_ids is None
            and inputs.shape[0] > 1
            and not can_skip_batch_positions
        ):
            cache_offset = 0
            cache_offsets = None
            if first_cache is not None:
                cache_offset = first_cache._idx if hasattr(first_cache, "_idx") else 0
                offset = getattr(first_cache, "offset", None)
                if isinstance(offset, mx.array) and offset.ndim > 0:
                    cache_offsets = offset
                elif offset is not None and not hasattr(first_cache, "_idx"):
                    cache_offset = offset

            scalar_cache_offset = self._scalar_cache_offset(cache_offset)
            if mask is None and first_cache is not None and scalar_cache_offset == 0:
                left_padding = getattr(first_cache, "left_padding", None)
                if (
                    isinstance(left_padding, mx.array)
                    and left_padding.ndim > 0
                    and left_padding.size >= inputs.shape[0]
                ):
                    positions = mx.arange(inputs.shape[-1], dtype=mx.int32)[None, :]
                    mask = positions >= left_padding[: inputs.shape[0], None]

            rope_mask = mask
            if mask is not None and mask.shape[-1] != inputs.shape[-1]:
                rope_mask = None
            if rope_mask is not None and rope_mask.ndim != 2:
                rope_mask = None

            batch_size, seq_length = inputs.shape
            needs_rope_index = (
                cache is None
                or first_cache is None
                or self._rope_deltas is None
                or (cache_offsets is None and scalar_cache_offset == 0)
            )
            if needs_rope_index:
                position_ids = None
                if first_cache is not None:
                    position_ids = self._slice_cached_position_ids(
                        batch_size, seq_length, scalar_cache_offset
                    )
                if position_ids is None:
                    position_ids, rope_deltas = self.get_rope_index(
                        inputs,
                        image_grid_thw=image_grid_thw,
                        video_grid_thw=video_grid_thw,
                        attention_mask=rope_mask,
                    )
                    self._position_ids = position_ids
                    self._rope_deltas = rope_deltas
                elif self._rope_deltas is None:
                    self._rope_deltas = mx.zeros(
                        (batch_size, 1), dtype=position_ids.dtype
                    )
            else:
                rope_deltas = (
                    rope_deltas_kw if rope_deltas_kw is not None else self._rope_deltas
                )
                if cache_offsets is not None:
                    delta = mx.maximum(cache_offsets[:batch_size], 0).astype(mx.int32)
                    if rope_deltas is not None:
                        delta = delta + rope_deltas[:batch_size].squeeze(-1)
                    position_ids = self._position_ids_from_cache_offsets(
                        delta, seq_length
                    )
                else:
                    delta = scalar_cache_offset
                    if rope_deltas is not None:
                        delta = delta + int(rope_deltas.reshape(-1)[0].item())
                    position_ids = self._offset_position_ids(
                        delta, batch_size, seq_length
                    )
        hidden_sink = kwargs.pop(
            "hidden_sink",
            [] if capture_layer_ids is not None or return_hidden else None,
        )

        out = self.model(
            inputs,
            inputs_embeds=inputs_embeds,
            mask=mask,
            cache=cache,
            capture_layer_ids=capture_layer_ids,
            hidden_sink=hidden_sink,
            position_ids=position_ids,
        )

        logits = self.logits_from_hidden(out) if skip_logits is False else None
        return LanguageModelOutput(
            logits=logits,
            hidden_states=hidden_sink,
            gdn_states=None,
            shared_kv_states={} if return_shared_kv else None,
        )

    def logits_from_hidden(self, hidden: mx.array) -> mx.array:
        if self.args.tie_word_embeddings:
            return self.model.embed_tokens.as_linear(hidden)
        return self.lm_head(hidden)

    def speculative_logits_from_hidden(self, hidden: mx.array) -> mx.array:
        return self.logits_from_hidden(self.model.norm(hidden))

    def speculative_argmax_from_hidden(self, hidden: mx.array) -> mx.array:
        return mx.argmax(self.speculative_logits_from_hidden(hidden), axis=-1)

    def rollback_speculative_cache(
        self,
        caches: List[Any],
        gdn_states: Any,
        accepted: Any,
        block_size: int,
    ) -> int:
        del gdn_states
        if isinstance(accepted, int):
            accepted = mx.array([accepted])
        elif not isinstance(accepted, mx.array):
            accepted = mx.array(accepted)
        if accepted.ndim == 0:
            accepted = accepted.reshape(1)

        max_a = int(accepted.max().item())
        n = max_a + 1
        trim = int(block_size) - n
        is_batch = accepted.size > 1
        valid_ends = accepted + 1

        for cache in caches:
            if cache is None:
                continue

            if trim > 0 and hasattr(cache, "trim"):
                cache.trim(trim)

            if not is_batch or not hasattr(cache, "_idx"):
                continue

            kv_len = int(cache._idx)
            verify_start = kv_len - n
            if verify_start < 0:
                continue
            valid_ends_list = [int(v) for v in valid_ends.tolist()]

            kv_cache = getattr(cache, "kv_cache", cache)
            if getattr(kv_cache, "keys", None) is not None:
                for bi, valid_end in enumerate(valid_ends_list):
                    start = verify_start + valid_end
                    if start < kv_len:
                        kv_cache.keys[bi, :, start:kv_len, :] = 0
                        kv_cache.values[bi, :, start:kv_len, :] = 0

            index_keys = getattr(cache, "index_keys", None)
            if index_keys is not None:
                idx_len = int(getattr(cache, "index_offset", kv_len))
                idx_verify_start = idx_len - n
                if idx_verify_start < 0:
                    continue
                for bi, valid_end in enumerate(valid_ends_list):
                    start = idx_verify_start + valid_end
                    if start < idx_len:
                        cache.index_keys[bi, :, start:idx_len, :] = 0
        return max_a

    @property
    def layers(self):
        return self.model.layers

    @property
    def head_dim(self):
        return self.args.head_dim

    @property
    def n_kv_heads(self):
        return self.args.num_key_value_heads

    def make_cache(self):
        return [
            MiniMaxM3KVCache() if layer.self_attn.has_sparse_index else KVCache()
            for layer in self.layers
        ]

    @property
    def cast_predicate(self):
        def predicate(k):
            keep_fp32 = "e_score_correction_bias" in k or k.endswith(
                "block_sparse_moe.gate.weight"
            )
            return not keep_fp32

        return predicate

    @property
    def quant_predicate(self):
        def predicate(path, _):
            if path.endswith("block_sparse_moe.gate"):
                return {"group_size": 64, "bits": 8, "mode": "affine"}
            return True

        return predicate

    def shard(self, group: Optional[mx.distributed.Group] = None):
        group = group or mx.distributed.init()
        n = group.size()

        for layer in self.layers:
            if (
                layer.self_attn.has_sparse_index
                and layer.self_attn.index_heads % n != 0
            ):
                raise ValueError(
                    "MiniMax M3 sparse index heads must be divisible by "
                    f"the sharding group size: {layer.self_attn.index_heads} "
                    f"heads, group size {n}."
                )
            layer.self_attn.q_proj = shard_linear(
                layer.self_attn.q_proj, "all-to-sharded", group=group
            )
            layer.self_attn.k_proj = shard_linear(
                layer.self_attn.k_proj, "all-to-sharded", group=group
            )
            layer.self_attn.v_proj = shard_linear(
                layer.self_attn.v_proj, "all-to-sharded", group=group
            )
            layer.self_attn.o_proj = shard_linear(
                layer.self_attn.o_proj, "sharded-to-all", group=group
            )
            layer.self_attn.num_attention_heads //= n
            layer.self_attn.num_key_value_heads //= n
            if layer.self_attn.has_sparse_index:
                layer.self_attn.index_q_proj = shard_linear(
                    layer.self_attn.index_q_proj,
                    "all-to-sharded",
                    group=group,
                )
                layer.self_attn.index_heads //= n

            if not layer.is_moe_layer:
                continue
            if layer.block_sparse_moe.pack_shared_expert:
                shard_inplace(
                    layer.block_sparse_moe.switch_mlp.gate_up_proj,
                    "all-to-sharded",
                    group=group,
                )
            else:
                shard_inplace(
                    layer.block_sparse_moe.switch_mlp.gate_proj,
                    "all-to-sharded",
                    group=group,
                )
                shard_inplace(
                    layer.block_sparse_moe.switch_mlp.up_proj,
                    "all-to-sharded",
                    group=group,
                )
            shard_inplace(
                layer.block_sparse_moe.switch_mlp.down_proj,
                "sharded-to-all",
                group=group,
            )
            layer.block_sparse_moe.sharding_group = group
