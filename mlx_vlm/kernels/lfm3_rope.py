# Copyright © 2026 Apple Inc.

"""The LFM3 rotary embedding of q or k in one kernel, read token-major and written
head-major.

It computes, bitwise, what Lfm3 RotaryEmbedding does with standard ops on x (B, L, H, D):
    angles = (offset + arange(L)) * inv_freq        # float32, inv_freq zero-padded (NoPE)
    cos, sin = cos(angles).astype(T), sin(angles).astype(T)  # both head halves
    y = x * cos + concat([-b, a]) * sin              # each op rounded to T
and writes y as (B, H, L, D). inv_freq comes from the module's own expression.
"""

import mlx.core as mx

_SOURCE = """
#pragma METAL fp contract(off)
    const uint d = thread_position_in_grid.x;
    const uint lh = thread_position_in_grid.y;
    const uint b = thread_position_in_grid.z;
    if (d >= D) return;
    const uint L = x_shape[1];
    const uint l = lh / H;
    const uint h = lh % H;
    const size_t src = ((size_t(b) * L + l) * H + h) * D;
    const uint i = d < D / 2 ? d : d - D / 2;
    const float angle = static_cast<float>(static_cast<int>(l) + offset[0]) * inv_freq[i];
    const T c = static_cast<T>(metal::precise::cos(angle));
    const T s = static_cast<T>(metal::precise::sin(angle));
    const T xd = x[src + d];
    const T o = x[src + (d < D / 2 ? d + D / 2 : d - D / 2)];
    const T r = d < D / 2 ? static_cast<T>(-static_cast<float>(o)) : o;
    const T p = static_cast<T>(static_cast<float>(xd) * static_cast<float>(c));
    const T q = static_cast<T>(static_cast<float>(r) * static_cast<float>(s));
    y[((size_t(b) * H + h) * L + l) * D + d] = static_cast<T>(static_cast<float>(p) + static_cast<float>(q));
"""

_kernel = None


def lfm3_rope(x: mx.array, inv_freq: mx.array, offset) -> mx.array:
    """x (B, L, H, D); inv_freq float32 (D/2,) zero-padded; offset an int or a 0-d/1-element
    int array. Returns the rotated x as (B, H, L, D)."""
    global _kernel
    if _kernel is None:
        _kernel = mx.fast.metal_kernel(
            name="lfm3_rope",
            input_names=["x", "inv_freq", "offset"],
            output_names=["y"],
            source=_SOURCE,
        )
    if isinstance(offset, mx.array):
        offset = offset.reshape(-1).astype(mx.int32)
    else:
        offset = mx.array([offset], dtype=mx.int32)
    b, l, h, d = x.shape
    return _kernel(
        inputs=[x, inv_freq, offset],
        template=[("T", x.dtype), ("H", h), ("D", d)],
        grid=(d, l * h, b),
        threadgroup=(d, max(1, 256 // d), 1),
        output_shapes=[(b, h, l, d)],
        output_dtypes=[x.dtype],
    )[0]
