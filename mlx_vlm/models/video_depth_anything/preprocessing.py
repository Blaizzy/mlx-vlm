"""Memory-bounded Video Depth Anything input preprocessing.

Fuses bicubic interpolation, rescaling, and normalization on Metal. Only one
frame is staged at a time, into a fresh shared MLX output. This synchronous
NumPy input boundary is not differentiable and must run outside mx.compile.
The caller must not consume the output until this function returns.
"""

from functools import lru_cache

import mlx.core as mx
import numpy as np


def _cubic_table(input_size: int, output_size: int):
    """OpenCV half-pixel coordinates: double calculation, then float32."""
    position = (
        (np.arange(output_size, dtype=np.float64) + 0.5) * (input_size / output_size)
        - 0.5
    ).astype(np.float32)
    lower = np.floor(position)
    t = position - lower
    # Match the contracted coefficient evaluation in ARM64 OpenCV builds.
    # Separate float32 operations round each product, amplifying tiny-image error.
    # Use double intermediates for the small tables, rounding each contracted
    # operation back to float32. Pixel arithmetic on Metal remains float32.

    def multiply_add(a, b, c):
        return (
            np.asarray(a, dtype=np.float64) * np.asarray(b, dtype=np.float64)
            + np.asarray(c, dtype=np.float64)
        ).astype(np.float32)

    a = np.float32(-0.75)
    one = np.float32(1)
    u = t + one
    w0 = multiply_add(multiply_add(multiply_add(a, u, -5 * a), u, 8 * a), u, -4 * a)
    w1 = multiply_add(multiply_add(a + 2, t, -(a + 3)) * t, t, one)
    r = one - t
    w2 = multiply_add(multiply_add(a + 2, r, -(a + 3)) * r, r, one)
    weights = np.stack((w0, w1, w2, one - w0 - w1 - w2))
    indices = np.clip(
        lower[None, :].astype(np.int32) + np.arange(-1, 3, dtype=np.int32)[:, None],
        0,
        input_size - 1,
    )
    return indices, weights.astype(np.float32, copy=False)


_SOURCE = r"""
    const uint index = thread_position_in_grid.x;
    const uint frames = uint(pixels_shape[0]);
    const uint input_height = uint(pixels_shape[1]);
    const uint input_width = uint(pixels_shape[2]);
    const uint output_height = uint(y_indices_shape[1]);
    const uint output_width = uint(x_indices_shape[1]);
    if (index >= frames * output_height * output_width * 3u) {
        return;
    }

    const uint channel = index % 3u;
    const uint pixel = index / 3u;
    const uint ox = pixel % output_width;
    const uint oy = (pixel / output_width) % output_height;
    const uint frame = pixel / (output_width * output_height);
    const ulong frame_offset = ulong(frame) * input_height * input_width * 3u;

    float horizontal[4];
    for (uint row = 0; row < 4u; ++row) {
        const uint iy = uint(y_indices[row * output_height + oy]);
        const ulong row_offset = frame_offset + ulong(iy) * input_width * 3u;
        const uint ix0 = uint(x_indices[ox]);
        float value = float(pixels[row_offset + ulong(ix0) * 3u + channel]) / 255.0f;
        float sum = value * x_weights[ox];
        for (uint col = 1; col < 4u; ++col) {
            const uint ix = uint(x_indices[col * output_width + ox]);
            value = float(pixels[row_offset + ulong(ix) * 3u + channel]) / 255.0f;
            sum += value * x_weights[col * output_width + ox];
        }
        horizontal[row] = sum;
    }

    float value = horizontal[0] * y_weights[oy];
    for (uint row = 1; row < 4u; ++row) {
        value += horizontal[row] * y_weights[row * output_height + oy];
    }
    normalized[index] = (value - means[channel]) / stds[channel];
"""


@lru_cache(maxsize=1)
def _kernel():
    # Construct only on first use. Importing this module launches no GPU work.
    return mx.fast.metal_kernel(
        name="video_depth_preprocess_fused",
        input_names=[
            "pixels",
            "x_indices",
            "x_weights",
            "y_indices",
            "y_weights",
            "means",
            "stds",
        ],
        output_names=["normalized"],
        source=_SOURCE,
        ensure_row_contiguous=True,
        compile_options={"math_mode": "safe"},
    )


def preprocess_frames(frames, output_size, mean, std):
    """RGB list/ndarray -> evaluated float32 (T, H, W, 3) output.

    Input values have the processor's original 0..255 scale. The only newly
    allocated window-sized tensor retained here is the returned output;
    temporary GPU storage is per frame. The MLX allocator may cache released
    buffers independently.
    """
    if len(frames) == 0:
        raise ValueError("Expected at least one RGB frame")
    if len(frames[0].shape) != 3 or frames[0].shape[-1] != 3:
        raise ValueError("Expected RGB frames with shape (H, W, 3)")
    height, width = frames[0].shape[:2]
    oh, ow = (int(value) for value in output_size)
    if min(height, width, oh, ow) <= 0:
        raise ValueError("Input and output dimensions must be positive")
    if any(frame.shape != (height, width, 3) for frame in frames):
        raise ValueError("All RGB frames must have the same shape")
    yi, yw = _cubic_table(height, oh)
    xi, xw = _cubic_table(width, ow)
    mean, std = np.asarray(mean, dtype=np.float32), np.asarray(std, dtype=np.float32)
    if mean.shape != (3,) or std.shape != (3,):
        raise ValueError("Mean and standard deviation must each have 3 values")
    mean, std = np.ascontiguousarray(mean), np.ascontiguousarray(std)
    parameters = [
        mx.array(xi),
        mx.array(xw),
        mx.array(yi),
        mx.array(yw),
        mx.array(mean),
        mx.array(std),
    ]
    kernel = _kernel()
    count = oh * ow * 3
    if count > np.iinfo(np.uint32).max:
        raise ValueError("Frame exceeds the 32-bit kernel dispatch limit")
    result = mx.zeros((len(frames), oh, ow, 3), dtype=mx.float32)
    mx.eval(result)
    # This fresh output has no consumers. CPU writes finish before it escapes;
    # each temporary is evaluated before the CPU reads its shared buffer.
    view = np.array(result, copy=False)
    for i, frame in enumerate(frames):
        raw = (
            frame if frame.dtype in (np.uint8, np.float32) else frame.astype(np.float32)
        )
        raw = np.ascontiguousarray(raw)[None]
        output = kernel(
            inputs=[mx.array(raw), *parameters],
            output_shapes=[(1, oh, ow, 3)],
            output_dtypes=[mx.float32],
            grid=(count, 1, 1),
            threadgroup=(min(256, count), 1, 1),
        )[0]
        mx.eval(output)
        view[i] = np.asarray(output)[0]
        del output
    del view
    return result
