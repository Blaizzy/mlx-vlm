"""Fuse the incoming hyper-connection collapse and its following RMSNorm."""

from functools import lru_cache

import mlx.core as mx


@lru_cache(maxsize=1)
def _kernel():
    return mx.fast.metal_kernel(
        name="deepseek_v41_hc_pre_norm",
        input_names=["h", "mix", "weight", "eps"],
        output_names=["out"],
        ensure_row_contiguous=True,
        source=r"""
        uint row = threadgroup_position_in_grid.x;
        uint tid = thread_position_in_threadgroup.x;
        uint lane = thread_index_in_simdgroup;
        uint sg = simdgroup_index_in_threadgroup;
        threadgroup float sums[32], inv[1];
        float acc = 0.0f;
        float vals[8];
        for (int block = 0; block < 2; ++block) {
            for (int j = 0; j < 4; ++j) {
                int d = block * 4096 + int(tid) * 4 + j;
                float v = 0.0f;
                if (d < D) {
                    for (int c = 0; c < 4; ++c) {
                        // hc_pre rounds products before summing: do not use FMA.
                        volatile float product = mix[row * 4 + c] *
                            float(h[(row * 4 + c) * D + d]);
                        v = product + v;
                    }
                    // Preserve the low-precision boundary before RMSNorm.
                    v = float(T(v));
                    acc += v * v;
                }
                vals[block * 4 + j] = v;
            }
        }
        // Match RMSNorm's looped reduction (four elements per thread).
        acc = simd_sum(acc);
        if (lane == 0) sums[sg] = acc;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (sg == 0) {
            acc = simd_sum(sums[lane]);
            if (lane == 0) inv[0] = metal::precise::rsqrt(acc / float(D) + eps[0]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (int block = 0; block < 2; ++block) {
            for (int j = 0; j < 4; ++j) {
                int d = block * 4096 + int(tid) * 4 + j;
                if (d < D) out[row * D + d] = weight[d] * T(vals[block * 4 + j] * inv[0]);
            }
        }
        """,
    )


def fused_hc_pre_norm(h, mix, weight, eps):
    """Return None for shapes or dtypes requiring the ordinary array path."""
    if (
        not mx.metal.is_available()
        or mx.default_device() != mx.gpu
        or h.ndim != 4
        or h.shape[-2:] != (4, 5120)
        or not h.size
        or h.dtype not in (mx.bfloat16, mx.float16)
        or mix.shape != h.shape[:-1]
        or mix.dtype != mx.float32
        or weight.shape != (h.shape[-1],)
        or weight.dtype != h.dtype
    ):
        return None
    rows = h.shape[0] * h.shape[1]
    return _kernel()(
        inputs=[h, mix, weight, mx.array([eps], mx.float32)],
        template=[("T", h.dtype), ("D", h.shape[-1])],
        grid=(1024 * rows, 1, 1),
        threadgroup=(1024, 1, 1),
        output_shapes=[h.shape[:-2] + (h.shape[-1],)],
        output_dtypes=[h.dtype],
    )[0]
