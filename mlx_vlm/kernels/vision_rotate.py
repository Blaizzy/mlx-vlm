# Copyright © 2026 Apple Inc.

"""The LFM3-VL vision rotary embedding of q or k in one kernel, written head-major.

It computes, bitwise, what the vision attention does with standard ops:
    t = x.astype(float32); a, b = split(t, 2, axis=-1)
    y = (t * cos + concat([-b, a]) * sin).astype(x.dtype)
and writes y transposed to (heads, tokens, head_dim), the layout attention reads. Each
product is rounded to float32 on its own, as the separate multiply kernels round it.
"""

import mlx.core as mx

_SOURCE = """
#pragma METAL fp contract(off)
    const uint d = thread_position_in_grid.x;
    const uint h = thread_position_in_grid.y;
    const uint n = thread_position_in_grid.z;
    if (d >= D) return;
    const uint N = x_shape[0];
    const size_t row = (size_t(n) * H + h) * D;
    const float t = static_cast<float>(x[row + d]);
    const float o = static_cast<float>(x[row + (d < D / 2 ? d + D / 2 : d - D / 2)]);
    const float r = d < D / 2 ? -o : o;
    const float p = t * cos_[size_t(n) * D + d];
    const float q = r * sin_[size_t(n) * D + d];
    y[(size_t(h) * N + n) * D + d] = static_cast<T>(p + q);
"""

_kernel = None


def vision_rotate(x: mx.array, cos: mx.array, sin: mx.array) -> mx.array:
    """x (tokens, heads, head_dim), cos/sin float32 (tokens, head_dim). Returns the rotated
    x as (1, heads, tokens, head_dim) in x's dtype."""
    global _kernel
    if _kernel is None:
        _kernel = mx.fast.metal_kernel(
            name="lfm3_vision_rotate",
            input_names=["x", "cos_", "sin_"],
            output_names=["y"],
            source=_SOURCE,
        )
    n, h, d = x.shape
    return _kernel(
        inputs=[x, cos, sin],
        template=[("T", x.dtype), ("H", h), ("D", d)],
        grid=(d, h, n),
        threadgroup=(d, max(1, min(h, 256 // d)), 1),
        output_shapes=[(1, h, n, d)],
        output_dtypes=[x.dtype],
    )[0]
