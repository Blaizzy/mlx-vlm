"""Fuse normalization with adjacent inference operations without caching weights."""

import math
from functools import lru_cache

import mlx.core as mx
import mlx.nn as nn


@lru_cache(None)
def norm_kernel():
    return mx.fast.metal_kernel(
        name="flint_qk_norm_rope",
        input_names=["q", "k", "qw", "kw", "offset", "parameters"],
        output_names=["qn", "kn"],
        ensure_row_contiguous=False,
        source=r"""
        uint lane = thread_index_in_simdgroup;
        uint head = threadgroup_position_in_grid.y;
        uint token = threadgroup_position_in_grid.z;
        uint length = q_shape[1];
        uint batch = token / length, pos = token % length;
        bool is_query = head < QH;
        uint h = is_query ? head : head - QH;
        uint heads = is_query ? QH : KH;
        const device T* input = is_query ? q : k;
        const device T* weights = is_query ? qw : kw;
        device T* output = is_query ? qn : kn;
        size_t start = is_query
            ? size_t(batch) * q_strides[0] + size_t(pos) * q_strides[1] + h * 128 * q_strides[2]
            : size_t(batch) * k_strides[0] + size_t(pos) * k_strides[1] + h * 128 * k_strides[2];
        size_t stride = is_query ? q_strides[2] : k_strides[2];
        float values[4];
        float acc = 0.0f;
        for (int j = 0; j < 4; ++j) {
            values[j] = float(input[start + (lane * 4 + j) * stride]);
            acc += values[j] * values[j];
        }
        acc = simd_sum(acc);
        float inv = metal::precise::rsqrt(acc / 128.0f + parameters[0]);
        threadgroup T normalized[128];
        for (int j = 0; j < 4; ++j) {
            uint d = lane * 4 + j;
            // Keep the RMS rounding point before applying the learned weight.
            normalized[d] = T(weights[d * (is_query ? qw_strides[0] : kw_strides[0])]
                              * T(values[j] * inv));
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        size_t out = ((size_t(batch) * heads + h) * length + pos) * 128;
        if (ROTARY) {
            for (int j = 0; j < 2; ++j) {
                uint d = lane * 2 + j;
                float freq = metal::exp2(-(float(d) / 64.0f) * parameters[1]);
                float theta = float(int(pos) + offset[batch * offset_strides[0]]) * freq;
                float cs = metal::fast::cos(theta), sn = metal::fast::sin(theta);
                float a = float(normalized[d]), b = float(normalized[d + 64]);
                output[out + d] = T(a * cs - b * sn);
                output[out + d + 64] = T(a * sn + b * cs);
            }
        } else {
            for (int j = 0; j < 4; ++j) {
                uint d = lane * 4 + j;
                output[out + d] = normalized[d];
            }
        }
    """,
    )


@lru_cache(None)
def residual_kernel():
    return mx.fast.metal_kernel(
        name="flint_residual_rms",
        input_names=["x", "delta", "weight", "eps"],
        output_names=["residual", "normalized"],
        source=r"""
        uint row = threadgroup_position_in_grid.x;
        uint tid = thread_position_in_threadgroup.x;
        uint lane = thread_index_in_simdgroup;
        uint group = simdgroup_index_in_threadgroup;
        threadgroup float sums[32];
        threadgroup float inverse[1];
        float values[4];
        float acc = 0.0f;
        for (int j = 0; j < 4; ++j) {
            uint col = tid * 4 + j;
            size_t off = size_t(row) * D + col;
            T value = T(x[off] + delta[off]);
            residual[off] = value;
            values[j] = float(value);
            acc += values[j] * values[j];
        }
        acc = simd_sum(acc);
        if (group == 0) sums[lane] = 0;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (lane == 0) sums[group] = acc;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (group == 0) {
            acc = simd_sum(sums[lane]);
            if (lane == 0) inverse[0] = metal::precise::rsqrt(acc / D + eps[0]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (int j = 0; j < 4; ++j) {
            uint col = tid * 4 + j;
            normalized[size_t(row) * D + col] = T(T(values[j] * inverse[0]) * weight[col]);
        }
    """,
    )


@lru_cache(None)
def norm_parameters(eps, base):
    return mx.array([eps, math.log2(base)], mx.float32)


@lru_cache(None)
def epsilon(eps):
    return mx.array([eps], mx.float32)


def norm_rope(attn, q, k, offset):
    rope = attn.rope
    if (
        not mx.metal.is_available()
        or mx.default_device() != mx.gpu
        or attn.training
        or attn.head_dim != 128
        or q.dtype not in (mx.bfloat16, mx.float16)
        or k.dtype != q.dtype
        or q.size == 0
        or type(attn.q_norm) is not nn.RMSNorm
        or type(attn.k_norm) is not nn.RMSNorm
        or attn.q_norm.eps != attn.k_norm.eps
        or attn.q_norm.weight.dtype != q.dtype
        or attn.k_norm.weight.dtype != q.dtype
        or (
            rope is not None
            and (
                type(rope) is not nn.RoPE
                or rope.dims != 128
                or rope.traditional
                or rope.scale != 1.0
                or rope.base is None
            )
        )
    ):
        return None
    b, length, _ = q.shape
    if isinstance(offset, int):
        offset = mx.full((b,), offset, mx.int32)
    elif offset.shape != (b,) or offset.dtype != mx.int32:
        return None
    return norm_kernel()(
        inputs=[
            q,
            k,
            attn.q_norm.weight,
            attn.k_norm.weight,
            offset,
            norm_parameters(
                attn.q_norm.eps, rope.base if rope is not None else 10000.0
            ),
        ],
        template=[
            ("T", q.dtype),
            ("QH", attn.n_heads),
            ("KH", attn.n_kv_heads),
            ("ROTARY", rope is not None),
        ],
        grid=(32, attn.n_heads + attn.n_kv_heads, b * length),
        threadgroup=(32, 1, 1),
        output_shapes=[
            (b, attn.n_heads, length, 128),
            (b, attn.n_kv_heads, length, 128),
        ],
        output_dtypes=[q.dtype, k.dtype],
    )


def residual_norm(x, delta, norm):
    d = x.shape[-1]
    if (
        norm.training
        or type(norm) is not nn.RMSNorm
        or not mx.metal.is_available()
        or mx.default_device() != mx.gpu
        or x.dtype not in (mx.bfloat16, mx.float16)
        or delta.dtype != x.dtype
        or norm.weight.dtype != x.dtype
        or x.shape != delta.shape
        or x.size == 0
        or d % 128 != 0
        or d > 4096
    ):
        residual = x + delta
        return residual, norm(residual)
    threads = d // 4
    return residual_kernel()(
        inputs=[x, delta, norm.weight, epsilon(norm.eps)],
        template=[("T", x.dtype), ("D", d)],
        grid=(threads * (x.size // d), 1, 1),
        threadgroup=(threads, 1, 1),
        output_shapes=[x.shape, x.shape],
        output_dtypes=[x.dtype, x.dtype],
    )
