"""Fused attention over a causal window and indexed compressed KV.

Adapted from the shared indexed sparse attention kernel. Each SIMD group
reduces a subset of keys, then merges its online softmax with the other groups
and the per-head attention sink. KV is shared between keys and values (MLA).
"""

from functools import lru_cache
from typing import Optional

import mlx.core as mx

_DECODE_PREFIX = r"""
    constexpr int GROUPS = 8;
    constexpr int WIDTH = 32;
    constexpr int PER_THREAD = DIM / WIDTH;
    uint row = threadgroup_position_in_grid.y;
    uint sg = simdgroup_index_in_threadgroup;
    uint lane = thread_index_in_simdgroup;
"""

_DECODE_ACCUMULATE = r"""    int length = sizes[0];
    int window_length = sizes[1];
    int pool_length = sizes[2];
    int qi = row % length;
    int head = (row / length) % HEADS;
    int batch = row / (length * HEADS);
    int mask_batch = BATCH_MASK ? batch : 0;
    int window_end = window_length - length + qi;
    int indices_offset = (batch * length + qi) * TOPK;
    thread float q[PER_THREAD];
    thread float acc[PER_THREAD];
    threadgroup float maxima[GROUPS];
    threadgroup float sums[GROUPS];
    threadgroup float partials[GROUPS * DIM];
    for (int d = 0; d < PER_THREAD; ++d) {
        q[d] = float(queries[row * DIM + lane + d * WIDTH]) * scale[0];
        acc[d] = 0;
    }
    float maximum = -3.4028234663852886e38f;
    float denominator = 0;
    for (int entry = sg; entry < WINDOW + TOPK; entry += GROUPS) {
        bool local = entry < WINDOW;
        int pos = local ? window_end - WINDOW + 1 + entry
                        : indices[indices_offset + entry - WINDOW];
        bool valid = pos >= 0 && pos < (local ? window_length : pool_length);
        if (local && valid) {
            valid = mask[(mask_batch * length + qi) * window_length + pos];
        }
        int offset = (batch * (local ? window_length : pool_length) + pos) * DIM
                     + lane;
        float score = -3.4028234663852886e38f;
        if (valid) {
            score = 0;
            for (int d = 0; d < PER_THREAD; ++d) {
                float value = local ? float(window[offset + d * WIDTH]) : float(pool[offset + d * WIDTH]);
                score += q[d] * value;
            }
            score = simd_sum(score);
        }
        float next_maximum = max(maximum, score);
        float correction = fast::exp(maximum - next_maximum);
        float weight = valid ? fast::exp(score - next_maximum) : 0;
        denominator = denominator * correction + weight;
        maximum = next_maximum;
        for (int d = 0; d < PER_THREAD; ++d) {
            acc[d] *= correction;
            if (valid) {
                float value = local ? float(window[offset + d * WIDTH]) : float(pool[offset + d * WIDTH]);
                acc[d] += weight * value;
            }
        }
    }
"""

_DECODE_MERGE = r"""    if (lane == 0) {
        maxima[sg] = maximum;
        sums[sg] = denominator;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    maximum = lane < GROUPS ? maxima[lane] : -3.4028234663852886e38f;
    float joint_maximum = max(simd_max(maximum), sinks[head]);
    float correction = fast::exp(maximum - joint_maximum);
    float total = simd_sum(lane < GROUPS ? sums[lane] * correction : 0)
                  + fast::exp(sinks[head] - joint_maximum);
    float group_correction = fast::exp(maxima[sg] - joint_maximum);
    for (int d = 0; d < PER_THREAD; ++d) {
        partials[sg * DIM + lane + d * WIDTH] = acc[d] * group_correction;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (int d = sg * WIDTH + lane; d < DIM; d += GROUPS * WIDTH) {
        float result = 0;
        for (int g = 0; g < GROUPS; ++g) {
            result += partials[g * DIM + d];
        }
        out[row * DIM + d] = T(total == 0 ? 0 : result / total);
    }
"""

_SOURCE = _DECODE_PREFIX + _DECODE_ACCUMULATE + _DECODE_MERGE


_PREFILL_SOURCE = r"""
    // One threadgroup shares each query's selected KV across eight heads.
    constexpr int BK = 32;
    constexpr int COLS = DIM / 4;
    uint tile = threadgroup_position_in_grid.y;
    uint sg = simdgroup_index_in_threadgroup;
    uint lane = thread_index_in_simdgroup;
    int length = sizes[0], wl = sizes[1], pl = sizes[2];
    int qi = tile % length;
    int head_base = ((tile / length) % (HEADS / 8)) * 8;
    int batch = tile / (length * (HEADS / 8));
    int mask_batch = BATCH_MASK ? batch : 0;
    int end = wl - length + qi;
    int io = (batch * length + qi) * TOPK;
    short quad = lane / 4;
    short fr = (quad & 4) + ((lane / 2) % 4);
    short fc = (quad & 2) * 2 + (lane % 2) * 2;
    threadgroup float weights[8 * BK];
    threadgroup float maxima[8], sums[8], corrections[8];
    simdgroup_matrix<float, 8, 8> accumulators[COLS / 8];
    for (int t = 0; t < COLS / 8; ++t) {
        accumulators[t].thread_elements()[0] = 0;
        accumulators[t].thread_elements()[1] = 0;
    }
    if (sg == 0 && lane < 8) {
        maxima[lane] = sinks[head_base + lane];
        sums[lane] = 1;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (int start = 0; start < WINDOW + TOPK; start += BK) {
        int offsets[2];
        bool local[2], valid[2];
        for (int e = 0; e < 2; ++e) {
            int entry = start + sg * 8 + fc + e;
            local[e] = entry < WINDOW;
            int pos = local[e] ? end - WINDOW + 1 + entry
                              : (entry < WINDOW + TOPK ? indices[io + entry - WINDOW] : -1);
            valid[e] = pos >= 0 && pos < (local[e] ? wl : pl);
            if (local[e] && valid[e]) valid[e] = mask[(mask_batch * length + qi) * wl + pos];
            offsets[e] = (batch * (local[e] ? wl : pl) + pos) * DIM;
        }
        simdgroup_matrix<float, 8, 8> scores;
        scores.thread_elements()[0] = 0;
        scores.thread_elements()[1] = 0;
        for (int d = 0; d < DIM; d += 8) {
            simdgroup_matrix<float, 8, 8> a, b;
            for (int e = 0; e < 2; ++e) {
                a.thread_elements()[e] = float(queries[((batch * HEADS + head_base + fr) * length + qi) * DIM + d + fc + e]) * scale[0];
                b.thread_elements()[e] = valid[e] ? (local[e] ? float(window[offsets[e] + d + fr]) : float(pool[offsets[e] + d + fr])) : 0;
            }
            simdgroup_multiply_accumulate(scores, a, b, scores);
        }
        for (int e = 0; e < 2; ++e) {
            weights[fr * BK + sg * 8 + fc + e] = valid[e] ? scores.thread_elements()[e] : -3.4028234663852886e38f;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (int h = sg * 2; h < sg * 2 + 2; ++h) {
            float score = weights[h * BK + lane];
            float maximum = max(maxima[h], simd_max(score));
            float correction = fast::exp(maxima[h] - maximum);
            float weight = fast::exp(score - maximum);
            float total = sums[h] * correction + simd_sum(weight);
            weights[h * BK + lane] = weight;
            if (lane == 0) {
                maxima[h] = maximum;
                sums[h] = total;
                corrections[h] = correction;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (int t = 0; t < COLS / 8; ++t) {
            for (int e = 0; e < 2; ++e) accumulators[t].thread_elements()[e] *= corrections[fr];
        }
        for (int k = 0; k < BK; k += 8) {
            int entry = start + k + fr;
            bool is_local = entry < WINDOW;
            int pos = is_local ? end - WINDOW + 1 + entry
                               : (entry < WINDOW + TOPK ? indices[io + entry - WINDOW] : -1);
            bool exists = pos >= 0 && pos < (is_local ? wl : pl);
            int offset = (batch * (is_local ? wl : pl) + pos) * DIM + sg * COLS;
            simdgroup_matrix<float, 8, 8> a;
            for (int e = 0; e < 2; ++e) a.thread_elements()[e] = weights[fr * BK + k + fc + e];
            for (int t = 0; t < COLS / 8; ++t) {
                simdgroup_matrix<float, 8, 8> b;
                for (int e = 0; e < 2; ++e) {
                    b.thread_elements()[e] = exists ? (is_local ? float(window[offset + t * 8 + fc + e]) : float(pool[offset + t * 8 + fc + e])) : 0;
                }
                simdgroup_multiply_accumulate(accumulators[t], a, b, accumulators[t]);
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    for (int t = 0; t < COLS / 8; ++t) {
        for (int e = 0; e < 2; ++e) {
            out[((batch * HEADS + head_base + fr) * length + qi) * DIM + sg * COLS + t * 8 + fc + e] = T(accumulators[t].thread_elements()[e] / sums[fr]);
        }
    }
"""


@lru_cache(maxsize=2)
def _kernel(prefill=False):
    return mx.fast.metal_kernel(
        name="deepseek_v41_sparse_attention" + ("_prefill" if prefill else "_vector"),
        input_names=[
            "queries",
            "window",
            "pool",
            "indices",
            "mask",
            "sinks",
            "scale",
            "sizes",
        ],
        output_names=["out"],
        header="#include <metal_simdgroup>\n#include <metal_simdgroup_matrix>\nusing namespace metal;\n",
        source=_PREFILL_SOURCE if prefill else _SOURCE,
    )


@lru_cache(maxsize=128)
def _scalars(scale: float, length: int, window_length: int, pool_length: int):
    return (
        mx.array([scale], mx.float32),
        mx.array([length, window_length, pool_length], mx.int32),
    )


# Each of the eight SIMD groups keeps exactly the vector kernel's keys and
# accumulation order. Separate threadgroups improve decode occupancy at B>=4.
# The second pass repeats the original merge, including the attention sink.
_SPLIT_PREFIX = r"""
    constexpr int GROUPS = 8, WIDTH = 32, PER_THREAD = DIM / WIDTH;
    uint row = threadgroup_position_in_grid.y / GROUPS;
    uint sg = threadgroup_position_in_grid.y % GROUPS;
    uint lane = thread_index_in_simdgroup;
"""
_SPLIT_STORE = r"""
    if (lane == 0) {
        stats[(row * GROUPS + sg) * 2] = maximum;
        stats[(row * GROUPS + sg) * 2 + 1] = denominator;
    }
    for (int d = 0; d < PER_THREAD; ++d) {
        out[(row * GROUPS + sg) * DIM + lane + d * WIDTH] = acc[d];
    }
"""
_SPLIT_MERGE_PREFIX = r"""
    constexpr int GROUPS = 8, WIDTH = 32, PER_THREAD = DIM / WIDTH;
    uint row = threadgroup_position_in_grid.y;
    uint sg = simdgroup_index_in_threadgroup;
    uint lane = thread_index_in_simdgroup;
    uint head = row % HEADS;
    threadgroup float maxima[GROUPS], sums[GROUPS], partials[GROUPS * DIM];
    float maximum = stats[(row * GROUPS + sg) * 2];
    float denominator = stats[(row * GROUPS + sg) * 2 + 1];
    float acc[PER_THREAD];
    for (int d = 0; d < PER_THREAD; ++d) {
        acc[d] = raw[(row * GROUPS + sg) * DIM + lane + d * WIDTH];
    }
"""


@lru_cache(maxsize=1)
def _split_kernels():
    partial = mx.fast.metal_kernel(
        name="deepseek_v41_sparse_decode_partial",
        input_names=[
            "queries",
            "window",
            "pool",
            "indices",
            "mask",
            "sinks",
            "scale",
            "sizes",
        ],
        output_names=["out", "stats"],
        source=_SPLIT_PREFIX + _DECODE_ACCUMULATE + _SPLIT_STORE,
    )
    merge = mx.fast.metal_kernel(
        name="deepseek_v41_sparse_decode_merge",
        input_names=["raw", "stats", "sinks"],
        output_names=["out"],
        source=_SPLIT_MERGE_PREFIX + _DECODE_MERGE,
    )
    return partial, merge


def _split_decode(queries, window, pool, indices, mask, sinks, scale, window_size):
    batch, heads, length, dim = queries.shape
    rows = batch * heads
    partial, merge = _split_kernels()
    scale_array, sizes = _scalars(float(scale), length, window.shape[1], pool.shape[1])
    sinks = sinks.astype(mx.float32)
    raw, stats = partial(
        inputs=[
            mx.contiguous(queries),
            mx.contiguous(window),
            mx.contiguous(pool),
            mx.contiguous(indices.astype(mx.int32)),
            mx.contiguous(mask),
            sinks,
            scale_array,
            sizes,
        ],
        template=[
            ("T", queries.dtype),
            ("DIM", dim),
            ("HEADS", heads),
            ("TOPK", indices.shape[-1]),
            ("WINDOW", window_size),
            ("BATCH_MASK", mask.shape[0] != 1),
        ],
        grid=(32, rows * 8, 1),
        threadgroup=(32, 1, 1),
        output_shapes=[(rows, 8, dim), (rows, 8, 2)],
        output_dtypes=[mx.float32, mx.float32],
    )
    return merge(
        inputs=[raw, stats, sinks],
        template=[("T", queries.dtype), ("DIM", dim), ("HEADS", heads)],
        grid=(256, rows, 1),
        threadgroup=(256, 1, 1),
        output_shapes=[queries.shape],
        output_dtypes=[queries.dtype],
    )[0]


def sparse_attention(
    queries: mx.array,
    window: mx.array,
    pool: mx.array,
    indices: mx.array,
    mask: mx.array,
    sinks: mx.array,
    scale: float,
    window_size: int,
) -> Optional[mx.array]:
    """Return fused attention, or ``None`` for unsupported inputs/devices.

    Queries are [B,H,L,D], window/pool are [B,K,D], indices are [B,L,S].
    The window ends at the last query and includes up to ``window_size`` keys
    per query. Its boolean mask is [1 or B,1,L,K]; indices of -1 are ignored.
    Scores, the shared softmax (including sinks), and values accumulate in FP32.
    """
    if (
        mx.default_device() != mx.gpu
        or not mx.metal.is_available()
        or queries.ndim != 4
        or window.ndim != 3
        or pool.ndim != 3
        or indices.ndim != 3
        or queries.dtype not in (mx.bfloat16, mx.float16)
        or window.dtype != queries.dtype
        or pool.dtype != queries.dtype
        or mask is None
        or mask.dtype != mx.bool_
    ):
        return None
    batch, heads, length, dim = queries.shape
    if (
        min(batch, heads, length, dim, window_size) <= 0
        or dim % 32
        or dim > 512
        or window.shape[0] != batch
        or pool.shape[0] != batch
        or window.shape[-1] != dim
        or pool.shape[-1] != dim
        or window.shape[1] < length
        or indices.shape[:2] != (batch, length)
        or indices.dtype not in (mx.int32, mx.int64)
        or mask.shape
        not in ((1, 1, length, window.shape[1]), (batch, 1, length, window.shape[1]))
        or sinks.shape != (heads,)
    ):
        return None
    if length == 1 and batch >= 4 and dim == 512 and indices.shape[-1] >= 512:
        return _split_decode(
            queries, window, pool, indices, mask, sinks, scale, window_size
        )
    prefill = length > 1 and heads % 8 == 0 and pool.shape[1] > 0
    threads = 128 if prefill else 256
    rows = batch * heads * length // (8 if prefill else 1)
    scale_array, sizes = _scalars(float(scale), length, window.shape[1], pool.shape[1])
    return _kernel(prefill)(
        inputs=[
            mx.contiguous(queries),
            mx.contiguous(window),
            mx.contiguous(pool),
            mx.contiguous(indices.astype(mx.int32)),
            mx.contiguous(mask),
            sinks.astype(mx.float32),
            scale_array,
            sizes,
        ],
        template=[
            ("T", queries.dtype),
            ("DIM", dim),
            ("HEADS", heads),
            ("TOPK", indices.shape[-1]),
            ("WINDOW", window_size),
            ("BATCH_MASK", mask.shape[0] != 1),
        ],
        grid=(threads, rows, 1),
        threadgroup=(threads, 1, 1),
        output_shapes=[queries.shape],
        output_dtypes=[queries.dtype],
    )[0]
