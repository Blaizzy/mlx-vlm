"""Indexed sparse attention kernel for Qwen Sparse Attention (QSA)."""

from functools import lru_cache
from typing import Optional

import mlx.core as mx

_QSA_SPARSE_ATTENTION_SOURCE = r"""
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
    int query_end = int(query_ends[batch_idx * Q_LEN + query_idx]);

    const device T* qptr =
        queries + ((batch_head_idx * Q_LEN + query_idx) * D_SIZE) +
        int(simd_lid) * qk_per_thread;
    device T* optr =
        out + ((batch_head_idx * Q_LEN + query_idx) * D_SIZE) +
        int(simd_gid) * v_per_thread;

    U s = U(scale[0]);
    for (int i = 0; i < qk_per_thread; ++i) {
        q[i] = s * static_cast<U>(qptr[i]);
    }
    for (int i = 0; i < v_per_thread; ++i) {
        o[i] = 0;
    }

    int blocks_offset = (batch_idx * Q_LEN + query_idx) * TOPK_BLOCKS;
    U max_score = -3.4028234663852886e38f;
    U sum_exp_score = 0;

    // The selected blocks are sorted by token position before launch. Each
    // SIMD group owns an interleaved subset and the final reduction combines
    // their independent online-softmax states.
    for (int selected_idx = int(simd_gid); selected_idx < SELECTED_LENGTH;
         selected_idx += BN) {
        int block_slot = selected_idx / BLOCK_SIZE;
        int block_offset = selected_idx - block_slot * BLOCK_SIZE;
        int block_idx = int(block_indices[blocks_offset + block_slot]);
        int key_pos = block_idx * BLOCK_SIZE + block_offset;
        bool valid = block_idx >= 0 && key_pos < key_length && key_pos < query_end;

        U score = -3.4028234663852886e38f;
        if (valid) {
            const device T* kptr =
                keys + (((batch_idx * NUM_KV_HEADS + kv_head_idx) * key_length +
                         key_pos) * D_SIZE) +
                int(simd_lid) * qk_per_thread;
            score = 0;
            for (int j = 0; j < qk_per_thread; ++j) {
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
                           key_pos) * D_SIZE) +
                int(simd_lid) * v_per_thread;
            for (int j = 0; j < v_per_thread; ++j) {
                o[j] = o[j] * factor + exp_score * static_cast<U>(vptr[j]);
            }
        } else {
            for (int j = 0; j < v_per_thread; ++j) {
                o[j] *= factor;
            }
        }
    }

    // QSA also attends the incomplete block immediately after the selected
    // complete blocks. It contains at most BLOCK_SIZE - 1 tokens.
    int tail_start = (query_end / BLOCK_SIZE) * BLOCK_SIZE;
    int tail_pos = tail_start + int(simd_gid);
    bool valid_tail = tail_pos < query_end && tail_pos < key_length;
    if (valid_tail) {
        const device T* kptr =
            keys + (((batch_idx * NUM_KV_HEADS + kv_head_idx) * key_length +
                     tail_pos) * D_SIZE) +
            int(simd_lid) * qk_per_thread;
        U score = 0;
        for (int j = 0; j < qk_per_thread; ++j) {
            score += q[j] * static_cast<U>(kptr[j]);
        }
        score = simd_sum(score);

        U new_max = max(max_score, score);
        U factor = fast::exp(max_score - new_max);
        U exp_score = fast::exp(score - new_max);
        max_score = new_max;
        sum_exp_score = sum_exp_score * factor + exp_score;

        const device T* vptr =
            values + (((batch_idx * NUM_KV_HEADS + kv_head_idx) * key_length +
                       tail_pos) * D_SIZE) +
            int(simd_lid) * v_per_thread;
        for (int j = 0; j < v_per_thread; ++j) {
            o[j] = o[j] * factor + exp_score * static_cast<U>(vptr[j]);
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

    for (int i = 0; i < v_per_thread; ++i) {
        outputs[simd_lid * BD + simd_gid] = o[i];
        threadgroup_barrier(mem_flags::mem_threadgroup);
        o[i] = simd_sum(outputs[simd_gid * BD + simd_lid] * factor);
        o[i] = total_sum == 0 ? U(0) : (o[i] / total_sum);
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (simd_lid == 0) {
        for (int i = 0; i < v_per_thread; ++i) {
            optr[i] = static_cast<T>(o[i]);
        }
    }
"""


@lru_cache(maxsize=None)
def _qsa_sparse_attention_kernel(
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
            "qwen4_exp_qsa_sparse_attention_"
            f"{dtype_name}_d{d_size}_s{selected_length}_b{block_size}_"
            f"k{topk_blocks}_qh{q_heads}_kh{kv_heads}"
        ),
        input_names=[
            "queries",
            "keys",
            "values",
            "block_indices",
            "query_ends",
            "scale",
            "k_size",
        ],
        output_names=["out"],
        header="#include <metal_simdgroup>\nusing namespace metal;\n",
        source=_QSA_SPARSE_ATTENTION_SOURCE,
    )


@lru_cache(maxsize=128)
def _qsa_sparse_attention_scalars(scale: float, key_length: int):
    return (
        mx.array([scale], dtype=mx.float32),
        mx.array([key_length], dtype=mx.int32),
    )


def qsa_sparse_attention(
    queries: mx.array,
    keys: mx.array,
    values: mx.array,
    block_indices: mx.array,
    query_ends: mx.array,
    *,
    scale: float,
    block_size: int,
) -> Optional[mx.array]:
    """Attend directly to QSA-selected blocks without gathering or a dense mask."""

    if (
        not isinstance(queries, mx.array)
        or not isinstance(keys, mx.array)
        or not isinstance(values, mx.array)
        or queries.ndim != 4
        or keys.ndim != 4
        or values.ndim != 4
        or block_indices.ndim != 3
        or query_ends.ndim != 2
        or queries.dtype not in (mx.bfloat16, mx.float16)
        or keys.dtype != queries.dtype
        or values.dtype != queries.dtype
        or mx.default_device() != mx.gpu
        or not mx.metal.is_available()
    ):
        return None

    batch, q_heads, query_length, d_size = queries.shape
    kv_heads = keys.shape[1]
    key_length = keys.shape[2]
    topk_blocks = block_indices.shape[-1]
    selected_length = topk_blocks * block_size
    if (
        batch != block_indices.shape[0]
        or batch != query_ends.shape[0]
        or query_length != block_indices.shape[1]
        or query_length != query_ends.shape[1]
        or values.shape != keys.shape
        or q_heads % kv_heads != 0
        or d_size != values.shape[-1]
        or d_size % 32 != 0
        # The dense vector kernel is faster for singleton decode; this kernel
        # is designed to amortize its 1,024-thread launch across prefill rows.
        or query_length <= 1
        or block_size <= 0
        or selected_length >= key_length
        # Dense SDPA is faster while the selected set is a large fraction of K.
        or key_length < selected_length * 7
    ):
        return None

    queries = mx.contiguous(queries)
    keys = mx.contiguous(keys)
    values = mx.contiguous(values)
    block_indices = mx.contiguous(mx.sort(block_indices.astype(mx.int32), axis=-1))
    query_ends = mx.contiguous(query_ends.astype(mx.int32))
    scale_array, key_length_array = _qsa_sparse_attention_scalars(
        float(scale), int(key_length)
    )
    kernel = _qsa_sparse_attention_kernel(
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
            query_ends,
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
        grid=(1024, batch * q_heads * query_length, 1),
        threadgroup=(1024, 1, 1),
        output_shapes=[queries.shape],
        output_dtypes=[queries.dtype],
    )[0]
