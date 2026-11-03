"""Independent causal latent-attention queries in one dispatch."""

from functools import lru_cache

import mlx.core as mx

from .mla_kernel import _kernels


@lru_cache(maxsize=1)
def _parts():
    return mx.fast.metal_kernel(
        name="ml4_block_sparse_mla_parts",
        input_names=["ql", "qr", "kv", "kr", "indices", "mask", "scale", "counts"],
        output_names=["partial", "stats"],
        ensure_row_contiguous=False,
        source=r"""
        uint lane = thread_index_in_simdgroup;
        uint h = threadgroup_position_in_grid.y;
        uint bt = threadgroup_position_in_grid.z / S;
        uint b = bt / L;
        uint t = bt % L;
        uint part = threadgroup_position_in_grid.z % S;
        constexpr int RD = (D + 31) / 32;
        constexpr int RR = (R + 31) / 32;
        float query[RD], rope[RR > 0 ? RR : 1], acc[RD];
        for (int j = 0; j < RD; ++j) {
            int d = j * 32 + lane;
            query[j] = d < D ? float(ql[b*ql_strides[0]+h*ql_strides[1]+t*ql_strides[2]+d]) : 0;
            acc[j] = 0;
        }
        for (int j = 0; j < RR; ++j) {
            int d = j * 32 + lane;
            rope[j] = d < R ? float(qr[b*qr_strides[0]+h*qr_strides[1]+t*qr_strides[2]+d]) : 0;
        }
        float maximum = -INFINITY, denom = 0;
        for (int p = part * C; p < min(int(part + 1) * C, int(counts[t])); ++p) {
            int idx = indices[b*indices_strides[0]+t*indices_strides[1]+p*indices_strides[2]];
            if (!mask[b*mask_strides[0]+t*mask_strides[2]+idx*mask_strides[3]]) continue;
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
        size_t row = ((b*H+h)*L+t)*S+part;
        if (lane == 0) { stats[row*2] = maximum; stats[row*2+1] = denom; }
        for (int j = 0; j < RD; ++j) {
            int d = j*32+lane;
            if (d < D) partial[row*D+d] = acc[j];
        }
        """,
    )


def block_sparse_mla(ql, qr, kv, kr, indices, mask, scale, counts, chunk=128):
    b, h, length, d = ql.shape
    splits = (indices.shape[-1] + chunk - 1) // chunk
    rope_dim = qr.shape[-1]
    if rope_dim == 0:
        # R=0 never reads rotary inputs, but Metal still needs valid bindings.
        qr, kr = ql, kv
    partial, stats = _parts()(
        inputs=[
            ql,
            qr,
            kv,
            kr,
            indices,
            mask,
            mx.array(scale, mx.float32),
            mx.array(counts, mx.int32),
        ],
        template=[
            ("D", d),
            ("R", rope_dim),
            ("H", h),
            ("L", length),
            ("S", splits),
            ("C", chunk),
        ],
        grid=(32, h, b * length * splits),
        threadgroup=(32, 1, 1),
        output_shapes=[(b, h, length, splits, d), (b, h, length, splits, 2)],
        output_dtypes=[mx.float32, mx.float32],
    )
    return _kernels()[1](
        inputs=[partial, stats],
        template=[("D", d), ("S", splits), ("T", ql.dtype)],
        grid=(d, b * h * length, 1),
        threadgroup=(min(d, 256), 1, 1),
        output_shapes=[ql.shape],
        output_dtypes=[ql.dtype],
    )[0]


def project_queries(q, weight, nope_dim):
    # Keep M=1 for each head and query; only increase the matrix batch count.
    ql = (
        q[..., :nope_dim].transpose(2, 0, 1, 3)[:, :, :, None, :] @ weight[:, :nope_dim]
    )
    return ql.squeeze(3).transpose(1, 2, 0, 3)


def project_values(out, weight, nope_dim):
    values = out.transpose(2, 0, 1, 3)[:, :, :, None, :] @ weight[
        :, nope_dim:
    ].swapaxes(-1, -2)
    return values.squeeze(3).transpose(1, 2, 0, 3)
