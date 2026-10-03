from functools import lru_cache
from typing import Optional

import mlx.core as mx

_INDEXED_SPARSE_ATTENTION_SOURCE = r"""
    uint row_idx = threadgroup_position_in_grid.y;
    uint simd_gid = simdgroup_index_in_threadgroup;
    uint simd_lid = thread_index_in_simdgroup;

    constexpr int SIMD_GROUPS = 32;
    constexpr int SIMD_WIDTH = 32;
    constexpr int qk_per_thread = QK_DIM / SIMD_WIDTH;
    constexpr int v_per_thread = V_DIM / SIMD_WIDTH;

    typedef float U;
    thread U q[qk_per_thread];
    thread U o[v_per_thread];
    threadgroup U outputs[SIMD_GROUPS * SIMD_WIDTH];
    threadgroup U max_scores[SIMD_GROUPS];
    threadgroup U sum_exp_scores[SIMD_GROUPS];

    int query_length = int(q_size[0]);
    int key_length = int(k_size[0]);
    int query_idx = int(row_idx % query_length);
    int batch_head_idx = int(row_idx / query_length);
    int batch_idx = batch_head_idx / NUM_Q_HEADS;
    int q_head_idx = batch_head_idx - batch_idx * NUM_Q_HEADS;
    int kv_head_idx = q_head_idx / GQA_FACTOR;

    const device T* qptr =
        queries + ((batch_head_idx * query_length + query_idx) * QK_DIM) +
        int(simd_lid) * qk_per_thread;
    device T* optr =
        out + ((batch_head_idx * query_length + query_idx) * V_DIM) +
        int(simd_gid) * v_per_thread;

    U s = U(scale[0]);
    for (int i = 0; i < qk_per_thread; i++) {
        q[i] = s * static_cast<U>(qptr[i]);
    }
    for (int i = 0; i < v_per_thread; i++) {
        o[i] = 0;
    }

    int indices_offset = (batch_idx * query_length + query_idx) * TOPK;
    U max_score = -3.4028234663852886e38f;
    U sum_exp_score = 0;

    for (int selected_idx = int(simd_gid); selected_idx < TOPK;
         selected_idx += SIMD_GROUPS) {
        int key_pos = int(indices[indices_offset + selected_idx]);
        bool valid = key_pos >= 0 && key_pos < key_length;

        U score = -3.4028234663852886e38f;
        if (valid) {
            const device T* kptr =
                keys + (((batch_idx * NUM_KV_HEADS + kv_head_idx) * key_length +
                         key_pos) *
                        QK_DIM) +
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
                          V_DIM) +
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
        outputs[simd_lid * SIMD_WIDTH + simd_gid] = o[i];
        threadgroup_barrier(mem_flags::mem_threadgroup);
        o[i] = simd_sum(
            outputs[simd_gid * SIMD_WIDTH + simd_lid] * factor
        );
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
def _indexed_sparse_attention_kernel(
    dtype,
    qk_dim: int,
    v_dim: int,
    topk: int,
    q_heads: int,
    kv_heads: int,
):
    dtype_name = {mx.bfloat16: "bf16", mx.float16: "fp16"}.get(dtype, "unk")
    return mx.fast.metal_kernel(
        name=(
            "indexed_sparse_attention_"
            f"{dtype_name}_q{qk_dim}_v{v_dim}_k{topk}_"
            f"qh{q_heads}_kh{kv_heads}"
        ),
        input_names=[
            "queries",
            "keys",
            "values",
            "indices",
            "scale",
            "q_size",
            "k_size",
        ],
        output_names=["out"],
        header="#include <metal_simdgroup>\nusing namespace metal;\n",
        source=_INDEXED_SPARSE_ATTENTION_SOURCE,
    )


@lru_cache(maxsize=128)
def _indexed_sparse_attention_scalars(scale: float, query_length: int, key_length: int):
    return (
        mx.array([scale], dtype=mx.float32),
        mx.array([query_length], dtype=mx.int32),
        mx.array([key_length], dtype=mx.int32),
    )


def indexed_sparse_attention(
    queries: mx.array,
    keys: mx.array,
    values: mx.array,
    indices: mx.array,
    scale: float,
    *,
    min_sparse_ratio: int = 0,
) -> Optional[mx.array]:
    """Fused indexed attention for Metal, or ``None`` when unsupported.

    The kernel reads arbitrary per-query token indices directly from the KV
    cache and performs score calculation, online softmax, and value reduction
    without materializing gathered K/V tensors. Inputs use ``[B, H, L, D]``
    layout and indices use ``[B, L, K]`` with ``-1`` marking invalid entries.
    """

    if (
        queries.ndim != 4
        or keys.ndim != 4
        or values.ndim != 4
        or indices.ndim != 3
        or queries.dtype not in (mx.bfloat16, mx.float16)
        or keys.dtype != queries.dtype
        or values.dtype != queries.dtype
        or mx.default_device() != mx.gpu
        or not mx.metal.is_available()
    ):
        return None

    batch, q_heads, query_length, qk_dim = queries.shape
    key_batch, kv_heads, key_length, key_dim = keys.shape
    value_batch, value_heads, value_length, v_dim = values.shape
    topk = indices.shape[-1]
    if (
        batch != key_batch
        or batch != value_batch
        or indices.shape[:2] != (batch, query_length)
        or value_heads != kv_heads
        or value_length != key_length
        or key_dim != qk_dim
        or q_heads % kv_heads != 0
        or qk_dim % 32 != 0
        or v_dim % 32 != 0
        or topk == 0
        or key_length < topk * min_sparse_ratio
    ):
        return None

    queries = mx.contiguous(queries)
    keys = mx.contiguous(keys)
    values = mx.contiguous(values)
    indices = mx.contiguous(indices.astype(mx.int32))
    scale_array, query_length_array, key_length_array = (
        _indexed_sparse_attention_scalars(
            float(scale), int(query_length), int(key_length)
        )
    )
    kernel = _indexed_sparse_attention_kernel(
        queries.dtype,
        int(qk_dim),
        int(v_dim),
        int(topk),
        int(q_heads),
        int(kv_heads),
    )
    return kernel(
        inputs=[
            queries,
            keys,
            values,
            indices,
            scale_array,
            query_length_array,
            key_length_array,
        ],
        template=[
            ("T", queries.dtype),
            ("QK_DIM", int(qk_dim)),
            ("V_DIM", int(v_dim)),
            ("TOPK", int(topk)),
            ("NUM_Q_HEADS", int(q_heads)),
            ("NUM_KV_HEADS", int(kv_heads)),
            ("GQA_FACTOR", int(q_heads // kv_heads)),
        ],
        grid=(1024, batch * q_heads * query_length, 1),
        threadgroup=(1024, 1, 1),
        output_shapes=[(batch, q_heads, query_length, v_dim)],
        output_dtypes=[queries.dtype],
    )[0]
