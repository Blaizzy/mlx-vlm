"""Fused GPU operations for MiniMax inference, with live weights and offsets."""

from functools import lru_cache

import mlx.core as mx


@lru_cache(maxsize=1)
def _gemma_rms_kernel():
    return mx.fast.metal_kernel(
        name="minimax_m3_gemma_rms",
        input_names=["x", "weight", "eps"],
        output_names=["out"],
        source="""
    uint lid = thread_position_in_threadgroup.x;
    uint row = threadgroup_position_in_grid.x;
    uint lane = thread_index_in_simdgroup;
    uint simd = simdgroup_index_in_threadgroup;
    uint size = x_shape[x_ndim - 1];
    float sum = 0;
    for (uint start = 4 * lid; start < size; start += 4 * threads_per_threadgroup.x) {
        for (uint j = 0; j < 4; ++j) {
            float v = start + j < size ? float(x[row * size + start + j]) : 0.0f;
            sum += v * v;
        }
    }
    sum = simd_sum(sum);
    float inv;
    if (threads_per_threadgroup.x == 32) {
        inv = metal::precise::rsqrt(sum / size + eps[0]);
    } else {
        threadgroup float sums[32];
        threadgroup float inverse;
        if (simd == 0) sums[lane] = 0;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (lane == 0) sums[simd] = sum;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (simd == 0) {
            sum = simd_sum(sums[lane]);
            if (lane == 0) inverse = metal::precise::rsqrt(sum / size + eps[0]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        inv = inverse;
    }
    for (uint start = 4 * lid; start < size; start += 4 * threads_per_threadgroup.x) {
        for (uint j = 0; j < 4; ++j) {
            if (start + j < size) {
                float v = float(x[row * size + start + j]) * inv;
                float w = float(weight[start + j]) + 1.0f;
                out[row * size + start + j] = T(w * v);
            }
        }
    }
""",
    )


@mx.compile
def gemma_rms_norm(x, weight, eps):
    """Fuse the FP32 weight offset and normalization, rounding only the output."""
    threads = min(1024, 32 * ((x.shape[-1] + 127) // 128))
    return _gemma_rms_kernel()(
        inputs=[x, weight, mx.array([eps], mx.float32)],
        template=[("T", x.dtype)],
        grid=(x.size // x.shape[-1] * threads, 1, 1),
        threadgroup=(threads, 1, 1),
        output_shapes=[x.shape],
        output_dtypes=[x.dtype],
    )[0]


@lru_cache(maxsize=1)
def _decode_scores_kernel():
    return mx.fast.metal_kernel(
        name="minimax_m3_decode_block_scores",
        input_names=["queries", "keys", "positions", "scale"],
        output_names=["scores"],
        ensure_row_contiguous=False,
        source="""
    uint lane = thread_index_in_simdgroup;
    uint simd = simdgroup_index_in_threadgroup;
    uint block = threadgroup_position_in_grid.x;
    uint head = threadgroup_position_in_grid.y;
    uint batch = threadgroup_position_in_grid.z;
    uint key_len = keys_shape[2];
    uint dim = queries_shape[3];
    long pos = positions[batch * positions_strides[0]];
    long current_block = pos >= 0 ? pos / BLOCK_SIZE : -1;
    size_t qoff = batch * queries_strides[0] + head * queries_strides[1];
    float best = -INFINITY;
    bool has_nan = false;
    for (uint j = simd; j < BLOCK_SIZE; j += 4) {
        uint key = block * BLOCK_SIZE + j;
        if (key < key_len && long(key) <= pos) {
            size_t koff = batch * keys_strides[0] + key * keys_strides[2];
            float score = 0.0f;
            for (uint d = lane * 4; d < dim; d += 128) {
                for (uint i = 0; i < 4 && d + i < dim; ++i) {
                    float q = queries[qoff + (d + i) * queries_strides[3]];
                    float k = keys[koff + (d + i) * keys_strides[3]];
                    score += q * k;
                }
            }
            score = simd_sum(score) * scale[0];
            has_nan = has_nan || isnan(score);
            best = max(best, score);
        }
    }
    threadgroup float maxima[4];
    threadgroup uint nans[4];
    if (lane == 0) { maxima[simd] = best; nans[simd] = has_nan; }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (simd == 0) {
        best = simd_max(lane < 4 ? maxima[lane] : -INFINITY);
        uint nan_count = simd_sum(lane < 4 ? nans[lane] : 0);
        if (nan_count) best = -INFINITY;
        if (long(block) <= current_block) {
            if (block < INIT_BLOCKS) best = 1e30f;
            if (LOCAL_BLOCKS > 0 &&
                long(block) >= max(current_block - LOCAL_BLOCKS + 1, 0L)) {
                best = 1e29f;
            }
        }
        if (lane == 0) {
            uint blocks = (key_len + BLOCK_SIZE - 1) / BLOCK_SIZE;
            scores[(batch * queries_shape[1] + head) * blocks + block] = best;
        }
    }
""",
    )


def decode_block_scores(queries, keys, positions, scale, block_size, init, local):
    """Score one query per head without materializing FP32 keys or token scores."""
    blocks = (keys.shape[2] + block_size - 1) // block_size
    return _decode_scores_kernel()(
        inputs=[queries, keys, positions, mx.array([scale], mx.float32)],
        template=[
            ("BLOCK_SIZE", block_size),
            ("INIT_BLOCKS", init),
            ("LOCAL_BLOCKS", local),
        ],
        grid=(blocks * 128, queries.shape[1], queries.shape[0]),
        threadgroup=(128, 1, 1),
        output_shapes=[(queries.shape[0], queries.shape[1], 1, blocks)],
        output_dtypes=[mx.float32],
    )[0]


@lru_cache(maxsize=1)
def _decode_mask_kernel():
    return mx.fast.metal_kernel(
        name="minimax_m3_decode_block_mask",
        input_names=["indices", "positions", "length"],
        output_names=["mask"],
        ensure_row_contiguous=False,
        source="""
    uint key = thread_position_in_grid.x;
    uint head = thread_position_in_grid.y;
    uint batch = thread_position_in_grid.z;
    uint lane = thread_index_in_simdgroup;
    int block = int(key / BLOCK_SIZE);
    size_t off = batch * indices_strides[0] + head * indices_strides[1];
    bool keep = false;
    for (uint i = lane; i < indices_shape[3]; i += 32) {
        keep = keep || indices[off + i * indices_strides[3]] == block;
    }
    keep = simd_any(keep);
    uint size = length[0];
    if (key < size) {
        keep = keep && long(key) <= positions[batch * positions_strides[0]];
        for (uint h = 0; h < HEADS_PER_INDEX; ++h) {
            size_t row = (batch * indices_shape[1] + head) * HEADS_PER_INDEX + h;
            size_t out = row * size + key;
            mask[out] = keep;
        }
    }
""",
    )


def decode_block_mask(indices, positions, key_length, block_size, query_heads):
    """Expand selections and apply causality; each SIMD group stays in one block."""
    batch, heads, _, _ = indices.shape
    return _decode_mask_kernel()(
        inputs=[indices, positions, mx.array([key_length], mx.int32)],
        template=[
            ("BLOCK_SIZE", block_size),
            ("HEADS_PER_INDEX", query_heads // heads),
        ],
        grid=(((key_length + 127) // 128) * 128, heads, batch),
        threadgroup=(128, 1, 1),
        output_shapes=[(batch, query_heads, 1, key_length)],
        output_dtypes=[mx.bool_],
    )[0]
