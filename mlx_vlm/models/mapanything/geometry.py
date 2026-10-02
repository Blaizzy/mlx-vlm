"""Camera, pose and point-map geometry (ports of ``mapanything.utils.geometry``).

Quaternions are (x, y, z, w); poses are OpenCV cam2world. Everything is a lazy
MLX op, including the least-squares intrinsics fit and the edge masks.
"""

import math
from functools import reduce
from typing import Tuple

import mlx.core as mx


def _apply(matrix: mx.array, vector: mx.array) -> mx.array:
    """(..., 3, 3) @ (..., 3) with the same per-element arithmetic for every
    batch shape."""
    return (matrix * vector[..., None, :]).sum(axis=-1)


def quaternion_to_rotation_matrix(quats: mx.array) -> mx.array:
    """(..., 4) -> (..., 3, 3)."""
    q = quats / mx.linalg.norm(quats, axis=-1, keepdims=True)
    x, y, z, w = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    rows = [
        1 - 2 * (y * y + z * z),
        2 * (x * y - w * z),
        2 * (x * z + w * y),
        2 * (x * y + w * z),
        1 - 2 * (x * x + z * z),
        2 * (y * z - w * x),
        2 * (x * z - w * y),
        2 * (y * z + w * x),
        1 - 2 * (x * x + y * y),
    ]
    return mx.stack(rows, axis=-1).reshape(*quats.shape[:-1], 3, 3)


def rotation_matrix_to_quaternion(matrix: mx.array) -> mx.array:
    """(..., 3, 3) -> (..., 4) with a non-negative real part."""
    m = [matrix[..., i, j] for i in range(3) for j in range(3)]
    m00, m01, m02, m10, m11, m12, m20, m21, m22 = m
    diag = mx.stack(
        [
            1.0 + m00 + m11 + m22,
            1.0 + m00 - m11 - m22,
            1.0 - m00 + m11 - m22,
            1.0 - m00 - m11 + m22,
        ],
        axis=-1,
    )
    q_abs = mx.where(diag > 0, mx.sqrt(mx.maximum(diag, 0)), 0)
    candidates = mx.stack(
        [
            mx.stack([q_abs[..., 0] ** 2, m21 - m12, m02 - m20, m10 - m01], axis=-1),
            mx.stack([m21 - m12, q_abs[..., 1] ** 2, m10 + m01, m02 + m20], axis=-1),
            mx.stack([m02 - m20, m10 + m01, q_abs[..., 2] ** 2, m12 + m21], axis=-1),
            mx.stack([m10 - m01, m20 + m02, m21 + m12, q_abs[..., 3] ** 2], axis=-1),
        ],
        axis=-2,
    )
    candidates = candidates / (2.0 * mx.maximum(q_abs, 0.1))[..., None]
    best = mx.argmax(q_abs, axis=-1)[..., None, None]
    rijk = mx.take_along_axis(candidates, best, axis=-2)[..., 0, :]
    quats = mx.concatenate([rijk[..., 1:], rijk[..., :1]], axis=-1)
    return mx.where(quats[..., 3:] < 0, -quats, quats)


def quaternion_inverse(quats: mx.array) -> mx.array:
    conj = quats * mx.array([-1.0, -1.0, -1.0, 1.0], dtype=quats.dtype)
    return conj / (quats * quats).sum(axis=-1, keepdims=True)


def quaternion_multiply(q1: mx.array, q2: mx.array) -> mx.array:
    x1, y1, z1, w1 = (q1[..., i] for i in range(4))
    x2, y2, z2, w2 = (q2[..., i] for i in range(4))
    return mx.stack(
        [
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        ],
        axis=-1,
    )


def relative_pose(
    ref_quats: mx.array, ref_trans: mx.array, quats: mx.array, trans: mx.array
) -> Tuple[mx.array, mx.array]:
    """Express cam2world poses in the frame of the reference pose. A pose
    equal to the reference maps to a translation of exactly zero."""
    inv = quaternion_inverse(ref_quats)
    rotation = quaternion_to_rotation_matrix(inv)
    return quaternion_multiply(inv, quats), _apply(rotation, trans - ref_trans)


def pose_matrix(quats: mx.array, trans: mx.array) -> mx.array:
    """(..., 4), (..., 3) -> (..., 4, 4) cam2world."""
    top = mx.concatenate([quaternion_to_rotation_matrix(quats), trans[..., None]], -1)
    bottom = mx.broadcast_to(
        mx.array([0.0, 0.0, 0.0, 1.0], dtype=top.dtype), (*top.shape[:-2], 1, 4)
    )
    return mx.concatenate([top, bottom], axis=-2)


def pose_inverse(pose: mx.array) -> mx.array:
    rotation = pose[..., :3, :3].swapaxes(-1, -2)
    trans = -_apply(rotation, pose[..., :3, 3])
    return mx.concatenate(
        [mx.concatenate([rotation, trans[..., None]], -1), pose[..., 3:, :]], axis=-2
    )


def _pixel_grid(height: int, width: int) -> Tuple[mx.array, mx.array]:
    x = mx.arange(width, dtype=mx.float32)[None, :]
    y = mx.arange(height, dtype=mx.float32)[:, None]
    return x, y


def _focal_center(intrinsics: mx.array):
    k = intrinsics[..., None, None, :, :]
    return k[..., 0, 0], k[..., 1, 1], k[..., 0, 2], k[..., 1, 2]


def rays_from_intrinsics(intrinsics: mx.array, height: int, width: int) -> mx.array:
    """(..., 3, 3) pinhole intrinsics -> unit ray directions (..., H, W, 3)."""
    x, y = _pixel_grid(height, width)
    fx, fy, cx, cy = _focal_center(intrinsics)
    xx, yy = mx.broadcast_arrays((x - cx) / fx, (y - cy) / fy)
    rays = mx.stack([xx, yy, mx.ones_like(xx)], axis=-1)
    return rays / mx.linalg.norm(rays, axis=-1, keepdims=True)


def depth_z_to_depth_along_ray(depth_z: mx.array, rays: mx.array) -> mx.array:
    """(..., H, W) z-depth and (..., H, W, 3) rays -> (..., H, W, 1)."""
    points = depth_z[..., None] * (rays / rays[..., 2:3])
    return mx.linalg.norm(points, axis=-1, keepdims=True)


def depth_to_camera_points(depth_z: mx.array, intrinsics: mx.array) -> mx.array:
    """(..., H, W) z-depth -> camera-frame points (..., H, W, 3)."""
    x, y = _pixel_grid(*depth_z.shape[-2:])
    fx, fy, cx, cy = _focal_center(intrinsics)
    return mx.stack(
        [(x - cx) * depth_z / fx, (y - cy) * depth_z / fy, depth_z], axis=-1
    )


def intrinsics_from_rays(rays: mx.array) -> mx.array:
    """Pinhole intrinsics (..., 3, 3) fitted to ray directions (..., H, W, 3):
    a per-axis least-squares fit of ``pixel = c + f * d_xy / d_z`` on a
    ~50 x 50 subgrid, or a four-point estimate above one megapixel."""
    height, width = rays.shape[-3:-1]
    if height * width > 1_000_000:
        ch, cw = height // 2, width // 2
        qh, qw, tqh, tqw = height // 4, width // 4, 3 * height // 4, 3 * width // 4

        def plane(i, j):
            r = rays[..., i, j, :]
            return r / r[..., 2:3]

        center = plane(ch, cw)
        fx = (
            (qw - cw) / (plane(ch, qw)[..., 0] - center[..., 0])
            + (tqw - cw) / (plane(ch, tqw)[..., 0] - center[..., 0])
        ) / 2
        fy = (
            (qh - ch) / (plane(qh, cw)[..., 1] - center[..., 1])
            + (tqh - ch) / (plane(tqh, cw)[..., 1] - center[..., 1])
        ) / 2
        cx = cw - fx * center[..., 0]
        cy = ch - fy * center[..., 1]
    else:
        rows = mx.arange(0, height, max(1, height // 50))
        cols = mx.arange(0, width, max(1, width // 50))
        sampled = mx.take(mx.take(rays, rows, axis=-3), cols, axis=-2)
        ratio = sampled[..., :2] / sampled[..., 2:3]
        cx, fx = _line_fit(ratio[..., 0], cols.astype(mx.float32)[None, :])
        cy, fy = _line_fit(ratio[..., 1], rows.astype(mx.float32)[:, None])
    zeros, ones = mx.zeros_like(fx), mx.ones_like(fx)
    return mx.stack(
        [fx, zeros, cx, zeros, fy, cy, zeros, zeros, ones], axis=-1
    ).reshape(*fx.shape, 3, 3)


def _line_fit(ratio: mx.array, pixels: mx.array) -> Tuple[mx.array, mx.array]:
    """Least-squares ``pixels = c + f * ratio`` over the last two axes."""
    pixels = mx.broadcast_to(pixels, ratio.shape)
    ratio_mean = ratio.mean(axis=(-2, -1), keepdims=True)
    pixel_mean = pixels.mean(axis=(-2, -1), keepdims=True)
    dr = ratio - ratio_mean
    f = (dr * (pixels - pixel_mean)).sum(axis=(-2, -1)) / (dr * dr).sum(axis=(-2, -1))
    c = pixel_mean[..., 0, 0] - f * ratio_mean[..., 0, 0]
    return c, f


def normalize_depth(depth: mx.array) -> Tuple[mx.array, mx.array]:
    """(N, H, W, 1) depth / mean of its positive pixels, and that mean (N,)."""
    valid = depth > 0
    total = (depth * valid).sum(axis=(1, 2, 3))
    count = valid.sum(axis=(1, 2, 3))
    factor = mx.maximum(total / (count + 1e-8), 1e-8)
    return depth / factor[:, None, None, None], factor


def normalize_translations(trans: mx.array) -> Tuple[mx.array, mx.array]:
    """(B, V, 3) translations / mean norm of the non-zero ones, and that (B,)."""
    norms = mx.linalg.norm(trans, axis=-1)
    count = (norms > 0).sum(axis=1)
    factor = mx.maximum(norms.sum(axis=1) / (count + 1e-8), 1e-8)
    return trans / factor[:, None, None], factor


def log_scale(x: mx.array) -> mx.array:
    """Rescale vectors (last axis) to length ``log1p(length)``."""
    length = mx.linalg.norm(x, axis=-1, keepdims=True)
    return x / mx.maximum(length, 1e-8) * mx.log1p(length)


def _shifts3x3(x: mx.array, axes=(-2, -1)):
    """The nine (di, dj) views of a map padded by one on ``axes``."""
    ah, aw = (a % x.ndim for a in axes)
    height, width = x.shape[ah] - 2, x.shape[aw] - 2
    for di in range(3):
        for dj in range(3):
            index = [slice(None)] * x.ndim
            index[ah] = slice(di, di + height)
            index[aw] = slice(dj, dj + width)
            yield x[tuple(index)]


def _pad_hw(x: mx.array, value=0, mode="constant", axes=(-2, -1)) -> mx.array:
    widths = [(0, 0)] * x.ndim
    for a in axes:
        widths[a % x.ndim] = (1, 1)
    return mx.pad(x, widths, mode=mode, constant_values=value)


def _max_pool3x3(x: mx.array) -> mx.array:
    """3x3 stride-1 max pool of (..., H, W) that ignores the border."""
    return reduce(mx.maximum, _shifts3x3(_pad_hw(x, -mx.inf)))


def points_to_normals(points: mx.array, mask: mx.array) -> Tuple[mx.array, mx.array]:
    """Normals of (..., H, W, 3) point maps from the four neighbor pairs, and
    where at least one pair is valid."""
    p = list(_shifts3x3(_pad_hw(points, axes=(-3, -2)), axes=(-3, -2)))
    v = list(_shifts3x3(_pad_hw(mask, False)))
    up, left, right, down = (p[i] - p[4] for i in (1, 3, 5, 7))
    normals = mx.stack(
        [
            mx.linalg.cross(up, left),
            mx.linalg.cross(left, down),
            mx.linalg.cross(down, right),
            mx.linalg.cross(right, up),
        ]
    )
    normals = normals / (mx.linalg.norm(normals, axis=-1, keepdims=True) + 1e-12)
    pairs = mx.stack([v[1] & v[3], v[3] & v[7], v[7] & v[5], v[5] & v[1]]) & v[4]
    normals = (normals * pairs[..., None]).sum(axis=0)
    normals = normals / (mx.linalg.norm(normals, axis=-1, keepdims=True) + 1e-12)
    normals_mask = pairs.any(axis=0)
    return mx.where(normals_mask[..., None], normals, 0), normals_mask


def normals_edge(normals: mx.array, tol: float, mask: mx.array) -> mx.array:
    """Pixels within one pixel of a neighbor pair whose normals differ by more
    than ``tol`` degrees."""
    normals = normals / (mx.linalg.norm(normals, axis=-1, keepdims=True) + 1e-12)
    padded = _pad_hw(normals, mode="edge", axes=(-3, -2))
    padded_mask = _pad_hw(mask, mode="edge")
    angles = (
        mx.where(m, mx.arccos(mx.clip((normals * n).sum(axis=-1), -1.0, 1.0)), 0)
        for n, m in zip(_shifts3x3(padded, axes=(-3, -2)), _shifts3x3(padded_mask))
    )
    return _max_pool3x3(reduce(mx.maximum, angles)) > math.radians(tol)


def depth_edge(depth: mx.array, rtol: float, mask: mx.array) -> mx.array:
    """Pixels whose valid 3x3 neighborhood spans more than ``rtol`` of their
    depth."""
    diff = _max_pool3x3(mx.where(mask, depth, -mx.inf)) + _max_pool3x3(
        mx.where(mask, -depth, -mx.inf)
    )
    return diff / depth > rtol
