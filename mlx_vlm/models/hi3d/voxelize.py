"""Surface voxelization of the coarse mesh (NumPy port of Hi3D's Triton kernel).

A voxel is occupied when the triangle overlaps its unit cube by the
separating-axis test of Akenine-Möller. Arithmetic is float32, as in the
kernel.
"""

import logging

import numpy as np

DEFAULT_PAD = 3
MIN_RESOLUTION = 2 * DEFAULT_PAD + 1
# Candidate voxels tested per batch of triangles.
CANDIDATE_CHUNK = 1 << 22


def normalize_vertices(vertices, resolution, pad=DEFAULT_PAD, eps=1e-6):
    """Center the mesh at ``resolution / 2`` and scale its longest side to
    ``resolution - 2 * pad - 2 * eps`` voxels, in float32."""
    v = np.asarray(vertices, dtype=np.float32)
    f32 = np.float32
    lower = v.min(axis=0) - f32(eps)
    upper = v.max(axis=0) + f32(eps)
    center = (lower + upper) / f32(2.0)
    scale = f32(resolution - pad * 2 - 2 * eps) / (upper - lower).max()
    return (v - center) * scale + f32(resolution / 2.0)


def _axis_overlaps(v0, v1, v2, axis):
    p0, p1, p2 = ((v * axis).sum(-1) for v in (v0, v1, v2))
    lower = np.minimum(np.minimum(p0, p1), p2)
    upper = np.maximum(np.maximum(p0, p1), p2)
    radius = np.float32(0.5) * np.abs(axis).sum(-1)
    return (lower <= radius) & (upper >= -radius)


def _overlaps(a, b, c, voxels):
    """Triangle ``(a, b, c)`` rows against unit voxels at integer ``voxels``."""
    center = voxels.astype(np.float32) + np.float32(0.5)
    v0, v1, v2 = a - center, b - center, c - center
    half = np.float32(0.5)
    hit = np.ones(len(voxels), dtype=bool)
    for axis in range(3):
        coords = np.stack([v0[:, axis], v1[:, axis], v2[:, axis]])
        hit &= (coords.min(0) <= half) & (coords.max(0) >= -half)
    edges = (v1 - v0, v2 - v1, v0 - v2)
    zero = np.zeros(len(voxels), np.float32)
    for e in edges:
        for axis in (
            np.stack([zero, e[:, 2], -e[:, 1]], -1),
            np.stack([-e[:, 2], zero, e[:, 0]], -1),
            np.stack([e[:, 1], -e[:, 0], zero], -1),
        ):
            hit &= _axis_overlaps(v0, v1, v2, axis)
    e0, e1 = edges[0], edges[1]
    normal = np.stack(
        [
            e0[:, 1] * e1[:, 2] - e0[:, 2] * e1[:, 1],
            e0[:, 2] * e1[:, 0] - e0[:, 0] * e1[:, 2],
            e0[:, 0] * e1[:, 1] - e0[:, 1] * e1[:, 0],
        ],
        -1,
    )
    offset = -(normal * v0).sum(-1)
    return hit & (np.abs(offset) <= half * np.abs(normal).sum(-1))


def voxelize(vertices, faces, resolution):
    """Occupied voxels ``(M, 3)`` int32 of a mesh already in voxel units.

    Rows are ordered by the kernel's packed key ``z << 22 | y << 11 | x``.
    """
    v = np.asarray(vertices, dtype=np.float32)
    tri = v[np.asarray(faces, dtype=np.int64)]
    start = np.maximum(np.floor(tri.min(axis=1)).astype(np.int64), 0)
    end = np.minimum(np.floor(tri.max(axis=1)).astype(np.int64), resolution - 1)
    extent = np.maximum(end - start + 1, 0)
    counts = extent.prod(axis=1)
    keys = []
    bounds = np.cumsum(counts)
    first = 0
    while first < len(tri):
        base = bounds[first - 1] if first else 0
        last = max(
            int(np.searchsorted(bounds, base + CANDIDATE_CHUNK, "right")), first + 1
        )
        ids = np.repeat(np.arange(first, last), counts[first:last])
        if len(ids):
            local = np.arange(len(ids)) - np.repeat(
                np.cumsum(counts[first:last]) - counts[first:last], counts[first:last]
            )
            ext = extent[ids]
            x = local % ext[:, 0]
            y = local // ext[:, 0] % ext[:, 1]
            z = local // (ext[:, 0] * ext[:, 1])
            voxels = start[ids] + np.stack([x, y, z], -1)
            hit = _overlaps(tri[ids, 0], tri[ids, 1], tri[ids, 2], voxels)
            voxels = voxels[hit]
            keys.append((voxels[:, 2] << 22) | (voxels[:, 1] << 11) | voxels[:, 0])
        first = last
    if not keys:
        return np.zeros((0, 3), np.int32)
    keys = np.unique(np.concatenate(keys))
    mask = (1 << 11) - 1
    return np.stack([keys & mask, (keys >> 11) & mask, keys >> 22], -1).astype(np.int32)


def adaptive_voxelize(vertices, faces, resolution, budget):
    """Voxelize at ``resolution``, shrinking the grid by 5% until at most
    ``budget`` voxels are occupied, as the released pipeline does."""
    scale = 1.0
    while True:
        current = int(resolution * scale)
        if current < MIN_RESOLUTION:
            raise ValueError(
                f"no grid of at least {MIN_RESOLUTION} voxels fits {budget} voxels"
            )
        coords = voxelize(normalize_vertices(vertices, current), faces, current)
        if len(coords) <= budget:
            break
        scale *= 0.95
    if current != resolution:
        logging.warning(
            "Hi3D voxelization settled at resolution %d instead of %d to fit "
            "%d voxels",
            current,
            resolution,
            budget,
        )
    return coords
