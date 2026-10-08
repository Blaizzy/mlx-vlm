"""Multi-view depth-consistency confidence (port of
``mapanything.utils.multiview_confidence``), with the view-overlap test of
``mapanything.utils.wai.intersection_check`` evaluated for all pairs on the GPU.
"""

from typing import List

import mlx.core as mx

from .geometry import _apply, depth_to_camera_points, pose_inverse

_TRIANGLES = mx.array(
    [
        [0, 1, 2],
        [0, 2, 3],
        [0, 3, 7],
        [0, 7, 4],
        [1, 2, 6],
        [1, 6, 5],
        [1, 4, 5],
        [1, 0, 4],
        [2, 6, 7],
        [2, 3, 7],
        [6, 7, 4],
        [6, 5, 4],
    ]
)
_PLANES = [[0, 1, 3], [1, 6, 2], [0, 3, 7], [2, 6, 3], [0, 5, 1], [6, 5, 4]]
# View pairs per overlap-test chunk; bounds the (pairs, 12, 12, 3) buffers.
_PAIR_CHUNK = 4096


def frustums(intrinsics: mx.array, near: mx.array, far: mx.array) -> mx.array:
    """(V, 3, 3) intrinsics and (V,) depth range -> (V, 8, 3) camera-frame
    corners: near plane first, as in ``create_frustum_from_intrinsics``."""
    half_x = intrinsics[:, 0, 2] / intrinsics[:, 0, 0]
    half_y = intrinsics[:, 1, 2] / intrinsics[:, 1, 1]
    corners = []
    for depth in (near, far):
        x, y = depth * half_x, depth * half_y
        for sx, sy in ((1, 1), (-1, 1), (-1, -1), (1, -1)):
            corners.append(mx.stack([sx * x, sy * y, depth], axis=-1))
    return mx.stack(corners, axis=1)


def _segments_hit_triangles(start, end, triangles) -> mx.array:
    """Moller-Trumbore test of segments (..., 3) against triangles (..., 3, 3)."""
    v0, v1, v2 = triangles[..., 0, :], triangles[..., 1, :], triangles[..., 2, :]
    edge1, edge2 = v1 - v0, v2 - v0
    ray = end - start
    length = mx.linalg.norm(ray, axis=-1)
    ray = ray / length[..., None]
    h = mx.linalg.cross(ray, edge2)
    a = (edge1 * h).sum(axis=-1)
    parallel = mx.abs(a) <= 1e-6
    f = mx.where(parallel, 0.0, 1.0 / mx.where(parallel, 1.0, a))
    s = start - v0
    u = f * (s * h).sum(axis=-1)
    q = mx.linalg.cross(s, edge1)
    v = f * (ray * q).sum(axis=-1)
    t = f * (edge2 * q).sum(axis=-1)
    return (u >= 0) & (u <= 1) & (v >= 0) & (u + v <= 1) & (t >= 1e-6) & (t <= length)


def _triangles_intersect(tri_a: mx.array, tri_b: mx.array) -> mx.array:
    """Whether any edge of one triangle crosses the other; (..., 3, 3) pairs."""
    hit = mx.zeros(tri_a.shape[:-2], dtype=mx.bool_)
    for i in range(3):
        j = (i + 1) % 3
        hit = hit | _segments_hit_triangles(tri_a[..., i, :], tri_a[..., j, :], tri_b)
        hit = hit | _segments_hit_triangles(tri_b[..., i, :], tri_b[..., j, :], tri_a)
    return hit


def _planes(corners: mx.array) -> mx.array:
    """(V, 8, 3) frustum corners -> (V, 6, 4) inward planes (normal, offset)."""
    planes = []
    for i, j, k in _PLANES:
        p = corners[:, i]
        normal = mx.linalg.cross(corners[:, j] - p, corners[:, k] - p)
        normal = normal / mx.linalg.norm(normal, axis=-1, keepdims=True)
        planes.append(
            mx.concatenate([normal, -(normal * p).sum(-1, keepdims=True)], -1)
        )
    return mx.stack(planes, axis=1)


def frustum_overlaps(corners: mx.array) -> mx.array:
    """(V, 8, 3) world-frame frustums -> (V, V) bool: a corner of one lies in
    the other, or their faces intersect."""
    num = corners.shape[0]
    planes = _planes(corners)
    first = corners[:, 0]
    side = (planes[:, None, :, :3] * first[None, :, None, :]).sum(-1) + planes[
        :, None, :, 3
    ]
    inside = (side >= 0).all(axis=-1)
    overlap = inside | inside.T

    triangles = mx.take(corners, _TRIANGLES.reshape(-1), axis=1).reshape(num, 12, 3, 3)
    ii, jj = mx.meshgrid(mx.arange(num), mx.arange(num), indexing="ij")
    ii, jj = ii.reshape(-1), jj.reshape(-1)
    hits = []
    for start in range(0, num * num, _PAIR_CHUNK):
        a = mx.take(triangles, ii[start : start + _PAIR_CHUNK], axis=0)
        b = mx.take(triangles, jj[start : start + _PAIR_CHUNK], axis=0)
        hit = _triangles_intersect(a[:, :, None], b[:, None, :])
        hits.append(hit.any(axis=(1, 2)))
    return overlap | mx.concatenate(hits).reshape(num, num)


def multiview_depth_confidence(
    depth_z: List[mx.array],
    intrinsics: List[mx.array],
    camera_poses: List[mx.array],
    masks: List[mx.array],
    abs_thresh: float = 0.02,
    rel_thresh: float = 0.02,
) -> List[mx.array]:
    """Per-pixel fraction of the overlapping views whose depth agrees with the
    reprojected depth of the pixel.

    Per view: ``depth_z`` (B, H, W, 1), ``intrinsics`` (B, 3, 3),
    ``camera_poses`` (B, 4, 4) cam2world, ``masks`` (B, H, W) bool. Returns
    (B, H, W) float32 confidences; views without overlap get ones.
    """
    num = len(depth_z)
    depth = mx.stack([d[..., 0] for d in depth_z]).astype(mx.float32)
    K = mx.stack(intrinsics).astype(mx.float32)
    poses = mx.stack(camera_poses).astype(mx.float32)
    height, width = depth.shape[-2:]
    if num < 2:
        return [mx.ones_like(depth[0])]

    valid_depth = depth[:, 0] > 0
    near = mx.where(valid_depth, depth[:, 0], mx.inf).min(axis=(1, 2))
    far = mx.where(valid_depth, depth[:, 0], -mx.inf).max(axis=(1, 2))
    has_depth = valid_depth.any(axis=(1, 2))
    near = mx.where(has_depth, near, 0.1)
    far = mx.where(has_depth, far, 100.0)
    corners = frustums(K[:, 0], near, far)
    rotation, trans = poses[:, 0, None, :3, :3], poses[:, 0, None, :3, 3]
    overlap = frustum_overlaps(_apply(rotation, corners) + trans)
    overlap = overlap & ~mx.eye(num, dtype=mx.bool_)

    points = depth_to_camera_points(depth, K)
    world = (
        _apply(poses[..., None, None, :3, :3], points) + poses[..., None, None, :3, 3]
    )
    world2cam = pose_inverse(poses)
    scale = mx.array([width - 1, height - 1], dtype=mx.float32)
    batch = mx.arange(depth.shape[1])[None, :, None, None]

    confidences = []
    for src in range(num):
        in_target = (
            _apply(world2cam[:, :, None, None, :3, :3], world[src][None])
            + world2cam[:, :, None, None, :3, 3]
        )
        projected = _apply(K[:, :, None, None], in_target)
        z = projected[..., 2]
        xy = projected[..., :2] / mx.maximum(z, 1e-6)[..., None]
        x, y = xy[..., 0], xy[..., 1]
        inside = (x >= 0) & (x < width) & (y >= 0) & (y < height) & (z > 0.04)
        valid = inside & (depth[src] > 0) & masks[src][None]
        grid = mx.clip(2.0 * xy / scale - 1.0, -1.0, 1.0)
        pixel = mx.round((grid + 1.0) / 2.0 * scale).astype(mx.int32)
        flat = (batch * height + pixel[..., 1]) * width + pixel[..., 0]
        sampled = mx.take_along_axis(
            depth.reshape(num, -1), flat.reshape(num, -1), axis=1
        ).reshape(z.shape)
        error = mx.abs(z - sampled)
        agree = error < abs_thresh + rel_thresh * z
        counted = valid & overlap[src][:, None, None, None]
        inliers = (agree & counted).sum(axis=0).astype(mx.float32)
        outliers = (~agree & counted).sum(axis=0).astype(mx.float32)
        confidence = inliers / (inliers + outliers + 1e-10)
        confidences.append(mx.where(overlap[src].any(), confidence, 1.0))
    return confidences
