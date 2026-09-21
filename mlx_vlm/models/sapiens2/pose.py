"""Pose top-down helpers: bbox warping and UDP heatmap decoding."""

import math
from functools import lru_cache
from typing import List, Tuple

import mlx.core as mx

from ..kernels import grid_sample
from .image import to_array


def bbox_xyxy2cs(bbox, padding: float = 1.25) -> Tuple[mx.array, mx.array]:
    """BBoxes (..., 4) ``xyxy`` -> centers (..., 2) and scales (..., 2) (w, h)."""
    x1, y1, x2, y2 = (mx.array(bbox, dtype=mx.float32)[..., i] for i in range(4))
    center = mx.stack([(x1 + x2) * 0.5, (y1 + y2) * 0.5], axis=-1)
    scale = mx.stack([(x2 - x1) * padding, (y2 - y1) * padding], axis=-1)
    return center, scale


def fix_aspect_ratio(bbox_scale: mx.array, aspect_ratio: float) -> mx.array:
    """Reshape bbox scales (..., 2) to the model input aspect ratio (w / h)."""
    w, h = bbox_scale[..., :1], bbox_scale[..., 1:]
    return mx.where(
        w > h * aspect_ratio,
        mx.concatenate([w, w / aspect_ratio], axis=-1),
        mx.concatenate([h * aspect_ratio, h], axis=-1),
    )


def warp_to_input(
    img, bbox, input_size: Tuple[int, int]
) -> Tuple[mx.array, mx.array, mx.array]:
    """Crop a person bbox to the model input via the UDP affine transform.

    ``input_size`` is (w, h). Returns the warped image (h, w, 3) float32,
    the bbox center (1, 2) and the bbox scale (1, 2) after padding and
    aspect-ratio fix.
    """
    img = to_array(img).astype(mx.float32)
    w, h = input_size
    center, scale = bbox_xyxy2cs(bbox)
    scale = fix_aspect_ratio(scale[None], aspect_ratio=w / h)[0]
    start = center - 0.5 * scale
    step = scale / mx.array([w - 1, h - 1], dtype=mx.float32)
    xs, ys = mx.meshgrid(
        mx.arange(w, dtype=mx.float32), mx.arange(h, dtype=mx.float32), indexing="xy"
    )
    src_x = start[0] + step[0] * xs
    src_y = start[1] + step[1] * ys
    # grid_sample takes align_corners-normalized coords in [-1, 1]
    in_h, in_w = img.shape[:2]
    gx = (2.0 * src_x + 1.0) / in_w - 1.0
    gy = (2.0 * src_y + 1.0) / in_h - 1.0
    grid = mx.stack([gx, gy], axis=-1)[None]

    warped = grid_sample(img[None], grid)[0]
    return warped, center[None], scale[None]


def _gaussian_kernel(kernel_size: int) -> mx.array:
    """1D Gaussian kernel with OpenCV's implicit sigma (``getGaussianKernel``)."""
    sigma = 0.3 * ((kernel_size - 1) * 0.5 - 1) + 0.8
    xs = (i - (kernel_size - 1) / 2 for i in range(kernel_size))
    k = [math.exp(-(x**2) / (2 * sigma**2)) for x in xs]
    return mx.array([v / sum(k) for v in k], dtype=mx.float32)


@lru_cache(maxsize=None)
def _blur_matrix(n: int, kernel_size: int) -> mx.array:
    """(n, n) matrix of the codec's blur along one axis, built by blurring
    the identity: zero-pad by the border, Gaussian with cv2's reflect-101
    border, crop back."""
    kernel = _gaussian_kernel(kernel_size)
    border = (kernel_size - 1) // 2
    basis = mx.pad(mx.eye(n), [(border, border), (0, 0)])
    basis = mx.pad(basis, [(border, border), (0, 0)], mode="reflect")
    rows = n + 2 * border
    blurred = sum(kernel[t] * basis[t : t + rows] for t in range(kernel_size))
    return blurred[border:-border]


def _blur(heatmaps: mx.array, kernel: int) -> mx.array:
    """Gaussian-blur channel-last (B, H, W, K) heatmaps, per keypoint."""
    B, H, W, K = heatmaps.shape
    out = _blur_matrix(H, kernel) @ heatmaps.reshape(B, H, W * K)
    out = out.reshape(B, H, W, K).transpose(0, 2, 1, 3).reshape(B, W, H * K)
    out = _blur_matrix(W, kernel) @ out
    return out.reshape(B, W, H, K).transpose(0, 2, 1, 3)


# (dx, dy) of the log-heatmap values ``_dark_udp_offset`` expects, in order.
_NEIGHBOUR_OFFSETS = ((0, 0), (1, 0), (0, 1), (1, 1), (-1, -1), (-1, 0), (0, -1))


def _dark_udp_offset(neighbours: mx.array) -> mx.array:
    """Sub-pixel offset from log-heatmap values at ``_NEIGHBOUR_OFFSETS``
    around each maximum: (7, ...) -> (..., 2)."""
    i_, ix1, iy1, ix1y1, ix1_y1_, ix1_, iy1_ = neighbours
    dx = 0.5 * (ix1 - ix1_)
    dy = 0.5 * (iy1 - iy1_)

    dxx = ix1 - 2 * i_ + ix1_
    dyy = iy1 - 2 * i_ + iy1_
    dxy = 0.5 * (ix1y1 - ix1 - iy1 + i_ + i_ - ix1_ - iy1_ + ix1_y1_)
    # Newton step, with the regularized 2x2 Hessian inverted in closed form.
    a, d = dxx + mx.finfo(mx.float32).eps, dyy + mx.finfo(mx.float32).eps
    det = a * d - dxy * dxy
    return mx.stack([(d * dx - dxy * dy) / det, (a * dy - dxy * dx) / det], axis=-1)


def udp_decode_batch(
    heatmaps: mx.array,
    input_size: Tuple[int, int],  # model input (w, h)
    blur_kernel_size: int = 11,
) -> Tuple[mx.array, mx.array]:
    """Decode channel-last (B, H, W, K) heatmaps (UDP codec, DARK refinement).

    Returns float32 keypoints (B, K, 2) in model-input pixel space and scores
    (B, K). Keypoints without a positive response are -1, as in the reference.
    """
    heatmaps = heatmaps.astype(mx.float32)
    B, H, W, K = heatmaps.shape

    flat = heatmaps.reshape(B, H * W, K)
    scores = flat.max(axis=1)
    index = flat.argmax(axis=1).astype(mx.int32)
    x, y = index % W, index // W

    blurred = _blur(heatmaps, blur_kernel_size)
    b_idx = mx.arange(B)[:, None]
    k_idx = mx.arange(K)[None, :]
    # Edge padding of the reference == clamped coordinates.
    neighbours = mx.stack(
        [
            blurred[b_idx, mx.clip(y + dy, 0, H - 1), mx.clip(x + dx, 0, W - 1), k_idx]
            for dx, dy in _NEIGHBOUR_OFFSETS
        ]
    )  # (7, B, K)
    # The codec rescales the blurred map to the original maximum, then
    # clips and takes the log; elementwise, so done on the gathered values.
    # A flat map (maximum 0) gets scale 0 and keeps the -1 sentinel.
    blur_max = blurred.max(axis=(1, 2))
    scale = mx.where(blur_max > 0, scores / blur_max, 0.0)
    neighbours = mx.log(mx.clip(neighbours * scale, 1e-3, 50.0))

    locs = mx.stack([x, y], axis=-1).astype(mx.float32)
    locs = mx.where((scores <= 0.0)[..., None], -1.0, locs)
    keypoints = locs - _dark_udp_offset(neighbours)
    # One factor per axis keeps an identity scale (and the -1 sentinel) exact.
    to_input = [input_size[0] / (W - 1), input_size[1] / (H - 1)]
    return keypoints * mx.array(to_input, dtype=mx.float32), scores


def keypoints_to_image(
    keypoints: mx.array,
    input_size: Tuple[int, int],  # model input (w, h)
    bbox_center: mx.array,
    bbox_scale: mx.array,
) -> mx.array:
    """Map decoded keypoints (N, K, 2) from the model crop back to the source
    image, given the (N, 1, 2) bbox centers and scales."""
    return (
        keypoints / mx.array(input_size, dtype=mx.float32) * bbox_scale
        + bbox_center
        - 0.5 * bbox_scale
    )


def flip_indices_from_pairs(num_keypoints: int, flip_pairs: List[List[int]]):
    """Build a channel permutation that swaps left/right keypoints."""
    indices = list(range(num_keypoints))
    for left, right in flip_pairs or []:
        indices[left], indices[right] = indices[right], indices[left]
    return mx.array(indices)
