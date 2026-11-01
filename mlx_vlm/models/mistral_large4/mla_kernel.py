"""Sparse latent decode attention with FP32 online softmax accumulation."""

from functools import lru_cache

import mlx.core as mx


@lru_cache(maxsize=1)
def _kernels():
    parts = mx.fast.metal_kernel(
        name="ml4_sparse_mla_parts",
        input_names=["ql", "qr", "kv", "kr", "indices", "mask", "scale"],
        output_names=["partial", "stats"],
        ensure_row_contiguous=False,
        source=r"""
        uint lane = thread_index_in_simdgroup;
        uint h = threadgroup_position_in_grid.y;
        uint b = threadgroup_position_in_grid.z / S;
        uint part = threadgroup_position_in_grid.z % S;
        constexpr int RD = (D + 31) / 32;
        constexpr int RR = (R + 31) / 32;
        float query[RD], rope[RR > 0 ? RR : 1], acc[RD];
        for (int j = 0; j < RD; ++j) {
            int d = j * 32 + lane;
            query[j] = d < D ? float(ql[b*ql_strides[0]+h*ql_strides[1]+d]) : 0;
            acc[j] = 0;
        }
        for (int j = 0; j < RR; ++j) {
            int d = j * 32 + lane;
            rope[j] = d < R ? float(qr[b*qr_strides[0]+h*qr_strides[1]+d]) : 0;
        }
        float maximum = -INFINITY, denom = 0;
        for (int p = part * C; p < min(int(part + 1) * C, K); ++p) {
            int idx = indices[b*indices_strides[0]+p*indices_strides[1]];
            if (!mask[b*mask_strides[0]+idx*mask_strides[3]]) continue;
            float values[RD], dot = 0, dot_rope = 0;
            for (int j = 0; j < RD; ++j) {
                int d = j * 32 + lane;
                values[j] = d < D ? float(kv[b*kv_strides[0]+idx*kv_strides[2]+d]) : 0;
                dot += query[j]*values[j];
            }
            for (int j = 0; j < RR; ++j) {
                int d = j*32+lane;
                if (d < R) dot_rope += rope[j]*float(kr[b*kr_strides[0]+idx*kr_strides[2]+d]);
            }
            float score = simd_sum(dot)*scale + simd_sum(dot_rope)*scale;
            float next_max = max(maximum, score);
            float old_weight = exp(maximum-next_max), weight = exp(score-next_max);
            denom = denom*old_weight+weight;
            for (int j = 0; j < RD; ++j) acc[j] = acc[j]*old_weight+values[j]*weight;
            maximum = next_max;
        }
        size_t row = (b*H+h)*S+part;
        if (lane == 0) { stats[row*2] = maximum; stats[row*2+1] = denom; }
        for (int j = 0; j < RD; ++j) {
            int d = j*32+lane;
            if (d < D) partial[row*D+d] = acc[j];
        }
        """,
    )

    merge = mx.fast.metal_kernel(
        name="ml4_sparse_mla_merge",
        input_names=["partial", "stats"],
        output_names=["out"],
        source=r"""
        uint d = thread_position_in_grid.x;
        uint row = thread_position_in_grid.y;
        if (d >= D) return;
        float maximum = -INFINITY;
        for (int s = 0; s < S; ++s) maximum = max(maximum, stats[(row*S+s)*2]);
        float total = 0, value = 0;
        for (int s = 0; s < S; ++s) {
            float sum = stats[(row*S+s)*2+1];
            float w = sum > 0 ? exp(stats[(row*S+s)*2]-maximum) : 0;
            total += w*sum;
            value += w*partial[(row*S+s)*D+d];
        }
        out[row*D+d] = T(total > 0 ? value/total : 0);
        """,
    )
    return parts, merge


def sparse_mla(ql, qr, kv, kr, indices, mask, scale, chunk=128):
    """Read sorted cache indices directly; preserve query dtype at the output.

    Queries have shape [B, H, 1, D], caches [B, 1, N, D], indices
    [B, K], and boolean mask [B, 1, 1, N]. Cache prefix views may be
    strided between batches; feature dimensions must be contiguous.
    """
    parts_kernel, merge_kernel = _kernels()
    b, h, _, d = ql.shape
    k = indices.shape[-1]
    splits = (k + chunk - 1) // chunk
    partial, stats = parts_kernel(
        inputs=[ql, qr, kv, kr, indices, mask, mx.array(scale, mx.float32)],
        template=[
            ("D", d),
            ("R", qr.shape[-1]),
            ("H", h),
            ("S", splits),
            ("K", k),
            ("C", chunk),
        ],
        grid=(32, h, b * splits),
        threadgroup=(32, 1, 1),
        output_shapes=[(b, h, splits, d), (b, h, splits, 2)],
        output_dtypes=[mx.float32, mx.float32],
    )
    return merge_kernel(
        inputs=[partial, stats],
        template=[("D", d), ("S", splits), ("T", ql.dtype)],
        grid=(d, b * h, 1),
        threadgroup=(min(d, 256), 1, 1),
        output_shapes=[ql.shape],
        output_dtypes=[ql.dtype],
    )[0]
