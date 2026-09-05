from __future__ import annotations

import logging
import math
from functools import cache, lru_cache

import mlx.core as mx

from .kv_quant import AFFINE4_SCHEME
from .models.cache import create_causal_mask
from .turboquant import (
    DEFAULT_TURBOQUANT_SEED,
    BatchTurboQuantKVCache,
    TurboQuantKVCache,
    TurboQuantMSEState,
    _pack_lowbit,
    _rht_forward,
    _rht_inverse,
    _rotation_matrix,
    _slice_state,
    _state_length,
    _unpack_lowbit,
)

logger = logging.getLogger("mlx_vlm.affine4")

__all__ = [
    "AFFINE4_SCHEME",
    "Affine4Codec",
    "Affine4KVCache",
    "BatchAffine4KVCache",
]

_BITS = 4
_MIN_NATIVE_TOKENS = 256
_MAX_PADDED_ROWS = 32
_NATIVE_LAUNCHABLE = {}
_FUSED_QUANTIZE_LAUNCHABLE = {}


class Affine4Codec:
    """Signed int4 codec with deterministic orthonormal rotation."""

    bits = _BITS

    def __init__(self, dim: int, seed: int):
        if dim <= 0:
            raise ValueError(f"affine4 requires a positive head dimension, got {dim}")
        self.dim = int(dim)
        self.seed = int(seed)
        self.use_rht = (dim & (dim - 1)) == 0
        if self.use_rht:
            from .turboquant import _rht_sign_vector

            self.signs = _rht_sign_vector(dim, seed)
            self.rotation = None
            self.rotation_t = None
        else:
            self.signs = None
            self.rotation = _rotation_matrix(dim, seed)
            self.rotation_t = self.rotation.transpose()

    def _rotate_forward(self, x: mx.array) -> mx.array:
        x = x.astype(mx.float32)
        if self.use_rht:
            return _rht_forward(x, self.signs)
        return mx.matmul(x, self.rotation_t)

    def _rotate_inverse(self, x: mx.array) -> mx.array:
        if self.use_rht:
            return _rht_inverse(x, self.signs)
        return mx.matmul(x, self.rotation)

    def quantize(self, vectors: mx.array) -> TurboQuantMSEState:
        rotated = self._rotate_forward(vectors)
        scales = mx.maximum(
            mx.max(rotated, axis=-1) / 7.0,
            -mx.min(rotated, axis=-1) / 8.0,
        ).astype(mx.float16)
        codes = mx.clip(mx.round(rotated / (scales[..., None] + 1e-7)), -8, 7).astype(
            mx.int8
        )
        packed = _pack_lowbit(codes.astype(mx.int32) & 15, _BITS)
        return TurboQuantMSEState(scales, packed)

    def _codes(self, state: TurboQuantMSEState) -> mx.array:
        nibbles = _unpack_lowbit(state.indices, _BITS, self.dim).astype(mx.int16)
        return ((nibbles ^ 8) - 8).astype(mx.float32)

    def dequantize(self, state: TurboQuantMSEState) -> mx.array:
        rotated = self._codes(state) * state.norms.astype(mx.float32)[..., None]
        return self._rotate_inverse(rotated)

    def prepare_queries(self, queries: mx.array) -> mx.array:
        return self._rotate_forward(queries)

    def score_prepared(
        self, prepared_queries: mx.array, state: TurboQuantMSEState
    ) -> mx.array:
        scores = mx.einsum("bhmld,bhtd->bhmlt", prepared_queries, self._codes(state))
        return scores * state.norms.astype(mx.float32)[:, :, None, None, :]

    def score(self, queries: mx.array, state: TurboQuantMSEState) -> mx.array:
        return self.score_prepared(self.prepare_queries(queries), state)

    def weighted_sum(self, weights: mx.array, state: TurboQuantMSEState) -> mx.array:
        rotated = self._codes(state)
        weighted = mx.einsum(
            "bhmlt,bht,bhtd->bhmld",
            weights.astype(mx.float32),
            state.norms.astype(mx.float32),
            rotated,
        )
        return self._rotate_inverse(weighted)

    def weighted_sum_from_scores(
        self, scores: mx.array, state: TurboQuantMSEState
    ) -> mx.array:
        return self.weighted_sum(mx.softmax(scores, axis=-1), state)

    def weighted_sum_stats_from_scores(
        self, scores: mx.array, state: TurboQuantMSEState
    ) -> tuple[mx.array, mx.array, mx.array]:
        maximum = mx.max(scores, axis=-1)
        weights = mx.exp(scores - maximum[..., None])
        return (
            self.weighted_sum(weights, state),
            mx.sum(weights, axis=-1),
            maximum,
        )


_FUSED_QUANTIZE_SOURCE = r"""
    uint d = thread_position_in_threadgroup.x;
    uint row = threadgroup_position_in_grid.x;
    uint is_value = threadgroup_position_in_grid.y;
    uint sg = simdgroup_index_in_threadgroup;
    uint lane = thread_index_in_simdgroup;

    float input_value = is_value
        ? static_cast<float>(values[row * Dim + d])
        : static_cast<float>(keys[row * Dim + d]);
    float sign = is_value ? value_signs[d] : key_signs[d];
    threadgroup float rotated[Dim];
    rotated[d] = sign * input_value;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (int stride = 1; stride < Dim; stride *= 2) {
        int base = int(d) & ~stride;
        float low = rotated[base];
        float high = rotated[base | stride];
        float transformed = (int(d) & stride) ? low - high : low + high;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        rotated[d] = transformed;
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float value = rotated[d] * rsqrt(float(Dim));

    float sg_positive = simd_max(value);
    float sg_negative = simd_max(-value);
    threadgroup float extrema[2 * SimdGroups];
    if (lane == 0) {
        extrema[sg] = sg_positive;
        extrema[SimdGroups + sg] = sg_negative;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float positive = (sg == 0 && lane < SimdGroups) ? extrema[lane] : 0.0f;
    float negative =
        (sg == 0 && lane < SimdGroups) ? extrema[SimdGroups + lane] : 0.0f;
    positive = simd_max(positive);
    negative = simd_max(negative);
    if (sg == 0 && lane == 0)
        extrema[0] = max(positive / 7.0f, negative / 8.0f);
    threadgroup_barrier(mem_flags::mem_threadgroup);

    half quant_scale = half(extrema[0]);
    if (d == 0) {
        if (is_value) value_scales[row] = quant_scale;
        else key_scales[row] = quant_scale;
    }
    float divisor = static_cast<float>(quant_scale) + 1e-7f;
    int code = divisor > 0.0f
        ? int(rint(clamp(value / divisor, -8.0f, 7.0f)))
        : 0;
    threadgroup uint codes[Dim];
    codes[d] = uint(code) & 15u;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (d < PackedWidth) {
        uint word = 0u;
        for (int i = 0; i < 8; ++i)
            word |= codes[d * 8 + i] << (i * 4);
        if (is_value) value_packed[row * PackedWidth + d] = word;
        else key_packed[row * PackedWidth + d] = word;
    }
"""


@cache
def _fused_quantize_kernel(dim: int):
    if (
        not hasattr(mx, "metal")
        or not mx.metal.is_available()
        or dim < 32
        or dim > 512
        or dim % 8
        or (dim & (dim - 1)) != 0
    ):
        return None
    return mx.fast.metal_kernel(
        name=f"affine4_fused_kv_quantize_d{dim}",
        input_names=["keys", "values", "key_signs", "value_signs"],
        output_names=[
            "key_scales",
            "key_packed",
            "value_scales",
            "value_packed",
        ],
        source=_FUSED_QUANTIZE_SOURCE,
    )


_MPP_HEADER = r"""
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
using namespace metal;
using namespace mpp::tensor_ops;

template <int Dim, int Repeats, int QueryLength, int PaddedRows, int Warps,
          int NumKvHeads, typename ScalePtr>
METAL_FUNC void affine4_attention_impl(
    device uchar* keys,
    ScalePtr key_scales,
    device uchar* values,
    ScalePtr value_scales,
    device float* partials,
    device float* sums,
    device float* maxs,
    int tokens,
    int blocks,
    int valid_start,
    int valid_end,
    float attention_scale,
    uint tid,
    uint sg,
    uint3 group,
    threadgroup half* q_tile,
    threadgroup float* scores,
    threadgroup half* probabilities,
    threadgroup float* row_m,
    threadgroup float* row_l,
    threadgroup float* row_factor) {
    constexpr int ActiveRows = Repeats * QueryLength;
    constexpr int BK = 64;
    constexpr int OutDim = Dim / Warps;

    int bh = int(group.y) * NumKvHeads + int(group.x);
    int block = int(group.z);
    int ntiles = (tokens + BK - 1) / BK;
    int tiles_per_block = (ntiles + blocks - 1) / blocks;
    int tile_begin = block * tiles_per_block;
    int tile_end = min(ntiles, tile_begin + tiles_per_block);

    if (tile_begin >= tile_end) {
        for (int i = int(tid); i < ActiveRows * Dim; i += 32 * Warps) {
            int row = bh * ActiveRows + i / Dim;
            partials[(row * blocks + block) * Dim + i % Dim] = 0.0f;
        }
        for (int row = int(tid); row < ActiveRows; row += 32 * Warps) {
            int idx = (bh * ActiveRows + row) * blocks + block;
            sums[idx] = 0.0f;
            maxs[idx] = 0.0f;
        }
        return;
    }

    for (int row = int(tid); row < PaddedRows; row += 32 * Warps) {
        row_m[row] = -INFINITY;
        row_l[row] = 0.0f;
        row_factor[row] = 0.0f;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    using QTensor = tensor<threadgroup half, dextents<int, 2>, tensor_inline>;
    using KVTensor = tensor<device int4b_format, dextents<int, 2>, tensor_inline>;
    using PTensor = tensor<threadgroup half, dextents<int, 2>, tensor_inline>;
    QTensor q_tensor(q_tile, dextents<int, 2>{Dim, PaddedRows});
    KVTensor k_tensor(keys, dextents<int, 2>{Dim, tokens});
    KVTensor v_tensor(values, dextents<int, 2>{Dim, tokens});
    PTensor p_tensor(probabilities, dextents<int, 2>{BK, PaddedRows});

    constexpr auto av_desc = matmul2d_descriptor(
        PaddedRows, OutDim, BK, false, false, false,
        matmul2d_descriptor::mode::multiply_accumulate);
    matmul2d<av_desc, execution_simdgroup> av_op;
    auto v_first = v_tensor.template slice<OutDim, BK>(
        int(sg) * OutDim, tile_begin * BK);
    auto output_tile = av_op.template get_destination_cooperative_tensor<
        decltype(p_tensor), decltype(v_first), half>();
    for (short i = 0; i < output_tile.get_capacity(); ++i)
        output_tile[i] = 0.0f;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (int tile_idx = tile_begin; tile_idx < tile_end; ++tile_idx) {
        int token = tile_idx * BK;
        int tile_valid_start = max(token, valid_start);
        int tile_valid_end = min(min(token + BK, tokens), valid_end);
        if (tile_valid_start >= tile_valid_end)
            continue;

        if (sg == 0) {
            constexpr auto qk_desc = matmul2d_descriptor(
                PaddedRows, BK, Dim, false, true, false);
            matmul2d<qk_desc, execution_simdgroup> qk_op;
            auto q_slice = q_tensor.template slice<Dim, PaddedRows>(0, 0);
            auto k_slice = k_tensor.template slice<Dim, BK>(0, token);
            auto score_tile = qk_op.template get_destination_cooperative_tensor<
                decltype(q_slice), decltype(k_slice), float>();
            qk_op.run(q_slice, k_slice, score_tile);
            for (short i = 0; i < score_tile.get_capacity(); ++i) {
                auto coord = score_tile.get_multidimensional_index(i);
                if (score_tile.is_valid_element(i)) {
                    int column = coord[0];
                    int row = coord[1];
                    int absolute_token = token + column;
                    int position = row % QueryLength;
                    int row_valid_end = valid_end - QueryLength + position + 1;
                    scores[row * BK + column] = row < ActiveRows &&
                            absolute_token >= tile_valid_start &&
                            absolute_token < tile_valid_end &&
                            absolute_token < row_valid_end
                        ? score_tile[i] * attention_scale *
                              float(key_scales[absolute_token])
                        : -INFINITY;
                }
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (int(tid) < ActiveRows) {
            int row = int(tid);
            float tile_m = -INFINITY;
            for (int column = 0; column < BK; ++column)
                tile_m = max(tile_m, scores[row * BK + column]);
            float old_m = row_m[row];
            if (isinf(tile_m)) {
                row_factor[row] = 1.0f;
                for (int column = 0; column < BK; ++column)
                    probabilities[row * BK + column] = half(0.0f);
            } else {
                float new_m = max(old_m, tile_m);
                float factor = isinf(old_m) ? 0.0f : fast::exp(old_m - new_m);
                float l = row_l[row] * factor;
                for (int column = 0; column < BK; ++column) {
                    int absolute_token = token + column;
                    bool active = !isinf(scores[row * BK + column]);
                    float weight = active
                        ? fast::exp(scores[row * BK + column] - new_m)
                        : 0.0f;
                    l += weight;
                    probabilities[row * BK + column] = half(
                        weight * (active ? float(value_scales[absolute_token]) : 0.0f));
                }
                row_m[row] = new_m;
                row_l[row] = l;
                row_factor[row] = factor;
            }
        }
        for (int i = int(tid) + ActiveRows * BK;
             i < PaddedRows * BK; i += 32 * Warps)
            probabilities[i] = half(0.0f);
        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (short i = 0; i < output_tile.get_capacity(); ++i) {
            auto coord = output_tile.get_multidimensional_index(i);
            if (output_tile.is_valid_element(i))
                output_tile[i] *= row_factor[coord[1]];
        }
        auto v_slice = v_tensor.template slice<OutDim, BK>(
            int(sg) * OutDim, token);
        av_op.run(p_tensor, v_slice, output_tile);
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (int(tid) < ActiveRows) {
        int idx = (bh * ActiveRows + int(tid)) * blocks + block;
        sums[idx] = row_l[tid];
        maxs[idx] = row_m[tid];
    }
    for (short i = 0; i < output_tile.get_capacity(); ++i) {
        auto coord = output_tile.get_multidimensional_index(i);
        if (output_tile.is_valid_element(i) && coord[1] < ActiveRows) {
            int row = bh * ActiveRows + coord[1];
            int dimension = int(sg) * OutDim + coord[0];
            partials[(row * blocks + block) * Dim + dimension] = output_tile[i];
        }
    }
}
"""


_MPP_ATTENTION_SOURCE = r"""
    uint tid = thread_index_in_threadgroup;
    uint3 group = threadgroup_position_in_grid;
    constexpr int ActiveRows = Repeats * QueryLength;
    threadgroup half q_tile[PaddedRows * Dim];
    threadgroup float scores[PaddedRows * 64];
    threadgroup half probabilities[PaddedRows * 64];
    threadgroup float row_m[PaddedRows];
    threadgroup float row_l[PaddedRows];
    threadgroup float row_factor[PaddedRows];

    int batch = int(group.y);
    int kv_head = int(group.x);
    for (int i = int(tid); i < PaddedRows * Dim; i += 32 * Warps) {
        int row = i / Dim;
        if (row < ActiveRows) {
            int head = kv_head * Repeats + row / QueryLength;
            int position = row % QueryLength;
            q_tile[i] = half(queries[
                batch * queries_strides[0] + head * queries_strides[1] +
                position * queries_strides[2] + i % Dim]);
        } else {
            q_tile[i] = half(0.0f);
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    int tokens = int(params[0]);
    int blocks = int(params[1]);
    float attention_scale = as_type<float>(params[2]);
    int key_word_offset = batch * int(keys_strides[0]) +
        kv_head * int(keys_strides[1]);
    int value_word_offset = batch * int(values_strides[0]) +
        kv_head * int(values_strides[1]);
    int key_scale_offset = batch * int(key_scales_strides[0]) +
        kv_head * int(key_scales_strides[1]);
    int value_scale_offset = batch * int(value_scales_strides[0]) +
        kv_head * int(value_scales_strides[1]);
    affine4_attention_impl<
        Dim, Repeats, QueryLength, PaddedRows, Warps, NumKvHeads>(
        (device uchar*)(keys + key_word_offset),
        key_scales + key_scale_offset,
        (device uchar*)(values + value_word_offset),
        value_scales + value_scale_offset,
        partials,
        sums,
        maxs,
        tokens,
        blocks,
        int(valid_starts[batch]),
        int(valid_ends[batch]),
        attention_scale,
        tid,
        simdgroup_index_in_threadgroup,
        group,
        q_tile,
        scores,
        probabilities,
        row_m,
        row_l,
        row_factor);
"""


_MPP_REDUCE_SOURCE = r"""
    uint dimension = thread_index_in_threadgroup;
    uint row = threadgroup_position_in_grid.x;
    int blocks = int(params[1]);
    threadgroup float global_m;
    threadgroup float global_l;
    if (dimension == 0) {
        float maximum = -INFINITY;
        for (int block = 0; block < blocks; ++block) {
            int idx = int(row) * blocks + block;
            if (sums[idx] > 0.0f)
                maximum = max(maximum, maxs[idx]);
        }
        float normalizer = 0.0f;
        for (int block = 0; block < blocks; ++block) {
            int idx = int(row) * blocks + block;
            if (sums[idx] > 0.0f)
                normalizer += sums[idx] * fast::exp(maxs[idx] - maximum);
        }
        global_m = maximum;
        global_l = normalizer;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float value = 0.0f;
    for (int block = 0; block < blocks; ++block) {
        int idx = int(row) * blocks + block;
        if (sums[idx] > 0.0f)
            value += partials[idx * Dim + int(dimension)] *
                fast::exp(maxs[idx] - global_m);
    }
    output[int(row) * Dim + int(dimension)] = value / global_l;
"""


@cache
def _mpp_attention_kernel(
    dim: int,
    repeats: int,
    query_length: int,
    padded_rows: int,
    warps: int,
    kv_heads: int,
):
    return mx.fast.metal_kernel(
        name=(
            f"affine4_mpp_d{dim}_r{repeats}_l{query_length}_"
            f"m{padded_rows}_w{warps}_h{kv_heads}"
        ),
        input_names=[
            "queries",
            "keys",
            "key_scales",
            "values",
            "value_scales",
            "valid_starts",
            "valid_ends",
            "params",
        ],
        output_names=["partials", "sums", "maxs"],
        header=_MPP_HEADER,
        source=_MPP_ATTENTION_SOURCE,
        ensure_row_contiguous=False,
        compile_options={"math_mode": "fast"},
    )


@cache
def _mpp_reduce_kernel(dim: int):
    return mx.fast.metal_kernel(
        name=f"affine4_mpp_reduce_d{dim}",
        input_names=["partials", "sums", "maxs", "params"],
        output_names=["output"],
        header="using namespace metal;\n",
        source=_MPP_REDUCE_SOURCE,
        ensure_row_contiguous=False,
        compile_options={"math_mode": "fast"},
    )


def _float_bits(value: float) -> int:
    import struct

    return struct.unpack("I", struct.pack("f", value))[0]


@lru_cache(maxsize=1)
def _m5_mpp_available() -> bool:
    if not (hasattr(mx, "metal") and mx.metal.is_available()):
        return False
    info = mx.device_info() if hasattr(mx, "device_info") else mx.metal.device_info()
    return str(info.get("architecture", "")).startswith("applegpu_g17")


def _padded_rows(active_rows: int) -> int:
    return max(8, ((active_rows + 7) // 8) * 8)


def _attention_blocks(tokens: int, kv_heads: int, dim: int) -> int:
    token_tiles = (tokens + 63) // 64
    target_groups = 256 if dim <= 64 else 168
    return min(token_tiles, max(1, (target_groups + kv_heads - 1) // kv_heads))


def _attention_warps(dim: int) -> int:
    if dim <= 64:
        return 1
    if dim == 96:
        return 2
    return 4


def _native_attention(
    cache,
    queries: mx.array,
    keys_state: TurboQuantMSEState,
    values_state: TurboQuantMSEState,
    scale: float,
    mask,
) -> mx.array | None:
    if (
        not _m5_mpp_available()
        or queries.ndim != 4
        or queries.dtype not in (mx.float16, mx.bfloat16)
    ):
        return None
    if (
        not isinstance(cache.key_codec, Affine4Codec)
        or not isinstance(cache.value_codec, Affine4Codec)
        or not isinstance(keys_state, TurboQuantMSEState)
        or not isinstance(values_state, TurboQuantMSEState)
        or keys_state.indices.ndim != 4
        or values_state.indices.ndim != 4
        or keys_state.norms.ndim != 3
        or values_state.norms.ndim != 3
    ):
        return None
    batch, query_heads, query_length, dim = queries.shape
    kv_heads = keys_state.norms.shape[1]
    if (
        dim != cache.key_codec.dim
        or dim != cache.value_codec.dim
        or kv_heads <= 0
        or keys_state.norms.shape[:2] != (batch, kv_heads)
        or values_state.norms.shape != keys_state.norms.shape
        or keys_state.indices.shape[:3] != keys_state.norms.shape
        or values_state.indices.shape != keys_state.indices.shape
        or keys_state.indices.shape[-1] != dim // 8
        or keys_state.indices.dtype != mx.uint32
        or values_state.indices.dtype != mx.uint32
        or keys_state.norms.dtype != mx.float16
        or values_state.norms.dtype != mx.float16
        or dim % 32
        or dim > 512
        or query_heads % kv_heads
        or query_length < 1
        or query_length > 4
    ):
        return None
    repeats = query_heads // kv_heads
    if query_length > 1 and not (
        isinstance(mask, str) and mask in ("causal", "left_padded_causal")
    ):
        return None
    padded_rows = _padded_rows(repeats * query_length)
    warps = _attention_warps(dim)
    if padded_rows > _MAX_PADDED_ROWS or dim % warps:
        return None
    tokens = _state_length(keys_state)
    if (
        tokens < _MIN_NATIVE_TOKENS
        or keys_state.indices.nbytes <= 4096
        or _state_length(values_state) != tokens
    ):
        return None

    is_batch = isinstance(cache, BatchAffine4KVCache)
    if is_batch:
        if isinstance(mask, mx.array):
            return None
        if isinstance(mask, str) and mask not in (
            "causal",
            "left_padded_decode",
            "left_padded_causal",
        ):
            return None
        valid_starts = cache.left_padding.astype(mx.int32)
    else:
        if mask is not None and not (isinstance(mask, str) and mask == "causal"):
            return None
        valid_starts = mx.zeros((batch,), dtype=mx.int32)
    valid_ends = mx.full((batch,), tokens, dtype=mx.int32)

    blocks = _attention_blocks(tokens, kv_heads, dim)
    params = mx.array([tokens, blocks, _float_bits(float(scale))], dtype=mx.uint32)
    grouped = queries.reshape(batch, kv_heads, repeats, query_length, dim)
    rotated = cache.key_codec.prepare_queries(grouped).astype(mx.float16)
    rows = batch * query_heads * query_length
    signature = (dim, repeats, query_length, padded_rows, warps, kv_heads)
    if _NATIVE_LAUNCHABLE.get(signature) is False:
        return None

    try:
        pass1 = _mpp_attention_kernel(*signature)
        partials, sums, maxs = pass1(
            inputs=[
                rotated.reshape(batch, query_heads, query_length, dim),
                keys_state.indices,
                keys_state.norms,
                values_state.indices,
                values_state.norms,
                valid_starts,
                valid_ends,
                params,
            ],
            template=[
                ("Dim", dim),
                ("Repeats", repeats),
                ("QueryLength", query_length),
                ("PaddedRows", padded_rows),
                ("Warps", warps),
                ("NumKvHeads", kv_heads),
            ],
            output_shapes=[
                (rows, blocks, dim),
                (rows, blocks),
                (rows, blocks),
            ],
            output_dtypes=[mx.float32, mx.float32, mx.float32],
            grid=(kv_heads * 32, batch * warps, blocks),
            threadgroup=(32, warps, 1),
        )
        rotated_output = _mpp_reduce_kernel(dim)(
            inputs=[partials, sums, maxs, params],
            template=[("Dim", dim)],
            output_shapes=[(batch, kv_heads, repeats, query_length, dim)],
            output_dtypes=[mx.float32],
            grid=(rows * dim, 1, 1),
            threadgroup=(dim, 1, 1),
        )[0]
        output = cache.value_codec._rotate_inverse(rotated_output)
        output = output.reshape(batch, query_heads, query_length, dim).astype(
            queries.dtype
        )
        if signature not in _NATIVE_LAUNCHABLE:
            mx.eval(output)
            _NATIVE_LAUNCHABLE[signature] = True
            logger.info(
                "Affine4 native M5 attention active (d=%d, repeats=%d, rows=%d)",
                dim,
                repeats,
                query_length,
            )
        return output
    except (RuntimeError, ValueError):
        _NATIVE_LAUNCHABLE[signature] = False
        logger.warning(
            "Affine4 M5 kernel rejected geometry d=%d r=%d l=%d; using fallback",
            dim,
            repeats,
            query_length,
            exc_info=True,
        )
        return None


class Affine4KVCache(TurboQuantKVCache):
    def __init__(
        self,
        bits: float = 4.0,
        seed: int = DEFAULT_TURBOQUANT_SEED,
        key_bits: float | None = None,
        value_bits: float | None = None,
    ):
        widths = (
            bits,
            key_bits if key_bits is not None else 4.0,
            value_bits if value_bits is not None else 4.0,
        )
        if any(not math.isclose(float(width), 4.0) for width in widths):
            raise ValueError(
                f"affine4 requires exactly 4 bits for keys and values, got {widths}"
            )
        super().__init__(bits=4.0, seed=seed, key_bits=4.0, value_bits=4.0)

    def _new_batch_cache(self, left_padding):
        return BatchAffine4KVCache(left_padding, bits=4.0, seed=self.seed)

    def _ensure_codecs(self, keys: mx.array, values: mx.array):
        if self.key_codec is None:
            self.key_codec = Affine4Codec(keys.shape[-1], self.seed)
        if self.value_codec is None:
            self.value_codec = Affine4Codec(values.shape[-1], self.seed + 1)

    def _try_fused_kv_quantize(self, keys, values):
        self._ensure_codecs(keys, values)
        dim = keys.shape[-1]
        signature = (dim, keys.dtype, values.dtype)
        if _FUSED_QUANTIZE_LAUNCHABLE.get(signature) is False:
            return None, None
        if values.shape != keys.shape:
            return None, None
        try:
            kernel = _fused_quantize_kernel(dim)
        except (RuntimeError, ValueError):
            _FUSED_QUANTIZE_LAUNCHABLE[signature] = False
            return None, None
        if kernel is None:
            return None, None
        flat_keys = keys.reshape(-1, dim)
        flat_values = values.reshape(-1, dim)
        rows = flat_keys.shape[0]
        packed_width = dim // 8
        try:
            outputs = kernel(
                inputs=[
                    flat_keys,
                    flat_values,
                    self.key_codec.signs,
                    self.value_codec.signs,
                ],
                template=[
                    ("Dim", dim),
                    ("SimdGroups", (dim + 31) // 32),
                    ("PackedWidth", packed_width),
                ],
                output_shapes=[
                    (rows,),
                    (rows, packed_width),
                    (rows,),
                    (rows, packed_width),
                ],
                output_dtypes=[
                    mx.float16,
                    mx.uint32,
                    mx.float16,
                    mx.uint32,
                ],
                grid=(dim * rows, 2, 1),
                threadgroup=(dim, 1, 1),
            )
            key_scales, key_packed, value_scales, value_packed = outputs
            if signature not in _FUSED_QUANTIZE_LAUNCHABLE:
                mx.eval(*outputs)
                _FUSED_QUANTIZE_LAUNCHABLE[signature] = True
            shape = keys.shape[:-1]
            return (
                TurboQuantMSEState(
                    key_scales.reshape(shape),
                    key_packed.reshape(*shape, packed_width),
                ),
                TurboQuantMSEState(
                    value_scales.reshape(shape),
                    value_packed.reshape(*shape, packed_width),
                ),
            )
        except (RuntimeError, ValueError):
            _FUSED_QUANTIZE_LAUNCHABLE[signature] = False
            return None, None

    def decode_attention(
        self,
        queries: mx.array,
        keys_state=None,
        values_state=None,
        scale: float = 1.0,
        mask=None,
    ) -> mx.array:
        if keys_state is None or values_state is None:
            keys_state, values_state = self._attention_states()
        keys_state = self._unwrap(keys_state)
        values_state = self._unwrap(values_state)
        output = _native_attention(self, queries, keys_state, values_state, scale, mask)
        if output is not None:
            return output
        fallback_mask = mask
        materialize_mask = getattr(self, "materialize_attention_mask", None)
        if callable(materialize_mask):
            fallback_mask = materialize_mask(mask, queries.shape[-2])
        return super().decode_attention(
            queries,
            keys_state=keys_state,
            values_state=values_state,
            scale=scale,
            mask=fallback_mask,
        )

    def prefill_attention(
        self,
        queries: mx.array,
        keys_state=None,
        values_state=None,
        scale: float = 1.0,
        mask=None,
    ):
        if queries.shape[-2] <= 4 and (
            isinstance(mask, str) and mask in ("causal", "left_padded_causal")
        ):
            if keys_state is None or values_state is None:
                keys_state, values_state = self._attention_states()
            keys_state = self._unwrap(keys_state)
            values_state = self._unwrap(values_state)
            output = _native_attention(
                self, queries, keys_state, values_state, scale, mask
            )
            if output is not None:
                return output
        return super().prefill_attention(
            queries,
            keys_state=keys_state,
            values_state=values_state,
            scale=scale,
            mask=mask,
        )

    def prefix_cache_snapshot(self):
        return {
            "meta_state": self.meta_state,
            "keys": _slice_state(self.keys, self.offset),
            "values": _slice_state(self.values, self.offset),
            "key_codec": None if self.key_codec is None else self.key_codec.dim,
            "value_codec": None if self.value_codec is None else self.value_codec.dim,
        }

    def prefix_cache_restore(self, snapshot):
        meta = snapshot["meta_state"]
        self.__init__(bits=float(meta[1]), seed=int(meta[2]))
        self.meta_state = meta
        key_dim = snapshot.get("key_codec")
        value_dim = snapshot.get("value_codec")
        if key_dim is not None:
            self.key_codec = Affine4Codec(int(key_dim), self.seed)
        if value_dim is not None:
            self.value_codec = Affine4Codec(int(value_dim), self.seed + 1)
        keys = snapshot["keys"]
        values = snapshot["values"]
        self.keys = None if keys is None else TurboQuantMSEState(*keys)
        self.values = None if values is None else TurboQuantMSEState(*values)


class BatchAffine4KVCache(BatchTurboQuantKVCache, Affine4KVCache):
    def __init__(
        self,
        left_padding: list | None = None,
        bits: float = 4.0,
        seed: int = DEFAULT_TURBOQUANT_SEED,
        key_bits: float | None = None,
        value_bits: float | None = None,
    ):
        widths = (
            float(bits),
            float(key_bits) if key_bits is not None else 4.0,
            float(value_bits) if value_bits is not None else 4.0,
        )
        if any(not math.isclose(width, 4.0) for width in widths):
            raise ValueError(
                f"affine4 requires exactly 4 bits for keys and values, got {widths}"
            )
        BatchTurboQuantKVCache.__init__(
            self,
            left_padding or [],
            bits=4.0,
            seed=seed,
            key_bits=4.0,
            value_bits=4.0,
        )
        self._affine4_has_padding = any(value > 0 for value in (left_padding or []))

    def _refresh_fused_attention_eligibility(self):
        BatchTurboQuantKVCache._refresh_fused_attention_eligibility(self)
        self._affine4_has_padding = bool(mx.any(self.left_padding != 0).item())

    def _ensure_codecs(self, keys: mx.array, values: mx.array):
        Affine4KVCache._ensure_codecs(self, keys, values)

    def _try_fused_kv_quantize(self, keys, values):
        return Affine4KVCache._try_fused_kv_quantize(self, keys, values)

    def _quantize_kv(self, keys, values):
        new_keys, new_values = self._try_fused_kv_quantize(keys, values)
        if new_keys is not None:
            return new_keys, new_values
        return self.key_codec.quantize(keys), self.value_codec.quantize(values)

    def _new_single_cache(self):
        return Affine4KVCache(bits=4.0, seed=self.seed)

    def quantized_attention_applies(self):
        return True

    def materialize_attention_mask(self, mask, query_length: int):
        if isinstance(mask, str) and mask in (
            "left_padded_decode",
            "left_padded_causal",
        ):
            return create_causal_mask(
                query_length,
                offset=self._idx - query_length,
                left_padding=self.left_padding,
            )
        return mask

    def make_mask(self, n: int, return_array: bool = False, **kwargs):
        if 1 <= n <= 4 and kwargs.get("window_size") is None:
            if self._affine4_has_padding:
                return "left_padded_decode" if n == 1 else "left_padded_causal"
            if n > 1:
                return "causal"
        return super().make_mask(n, return_array=return_array, **kwargs)

    def decode_attention(self, *args, **kwargs):
        return Affine4KVCache.decode_attention(self, *args, **kwargs)

    def prefill_attention(self, *args, **kwargs):
        return Affine4KVCache.prefill_attention(self, *args, **kwargs)
