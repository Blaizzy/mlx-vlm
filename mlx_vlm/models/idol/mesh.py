"""Approximate surface extraction from Idol's anisotropic Gaussian density."""

import numpy as np


def reconstruct_mesh(gaussian, *, resolution=128, threshold=0.5):
    """March an opacity-weighted density field and transfer DC vertex colors.

    This is a lossy surface reconstruction, not a conversion of splat rendering.
    The grid is bounded by ``resolution**3``; each Gaussian touches only its
    three-sigma neighborhood. A voxel-sized filter keeps sub-voxel splats sampled.
    """
    if not isinstance(resolution, int) or not 32 <= resolution <= 256:
        raise ValueError("mesh_resolution must be an integer between 32 and 256")
    if not np.isfinite(threshold) or threshold <= 0:
        raise ValueError("mesh_threshold must be finite and positive")
    try:
        import trimesh
        from scipy.spatial import cKDTree
        from skimage.measure import marching_cubes
    except ImportError as error:
        raise ImportError(
            "GLB and OBJ export require mesh dependencies: pip install 'mlx-vlm[three_d]'"
        ) from error

    xyz, scales, rotation, opacity, features = [
        np.asarray(value, dtype=np.float32)
        for value in (
            gaussian.xyz,
            gaussian.scaling,
            gaussian.rotation,
            gaussian.opacity,
            gaussian.features_dc,
        )
    ]
    if not all(
        np.isfinite(value).all() for value in (xyz, scales, rotation, opacity, features)
    ):
        raise ValueError("Cannot reconstruct a mesh from non-finite Gaussians")
    keep = opacity.reshape(-1) > 0.005
    if not keep.any():
        raise ValueError("No opaque Gaussians available for surface reconstruction")
    xyz, scales, rotation = xyz[keep], scales[keep], rotation[keep]
    opacity = opacity.reshape(-1)[keep]
    colors = np.clip(0.5 + 0.28209479177387814 * features[keep].reshape(-1, 3), 0, 1)
    norms = np.linalg.norm(rotation, axis=-1, keepdims=True)
    if (scales <= 0).any() or (norms == 0).any():
        raise ValueError("Gaussian scales and rotation norms must be positive")
    w, x, y, z = (rotation / norms).T
    rotations = np.stack(
        [
            1 - 2 * (y * y + z * z),
            2 * (x * y - w * z),
            2 * (x * z + w * y),
            2 * (x * y + w * z),
            1 - 2 * (x * x + z * z),
            2 * (y * z - w * x),
            2 * (x * z - w * y),
            2 * (y * z + w * x),
            1 - 2 * (x * x + y * y),
        ],
        axis=-1,
    ).reshape(-1, 3, 3)
    covariance = (rotations * scales[:, None, :] ** 2) @ rotations.transpose(0, 2, 1)
    radii = 3 * np.sqrt(np.diagonal(covariance, axis1=1, axis2=2))
    lower, upper = (xyz - radii).min(axis=0), (xyz + radii).max(axis=0)
    span = float((upper - lower).max())
    spacing = span / (resolution - 5)
    origin = (lower + upper) / 2 - span / 2 - 2 * spacing
    covariance += np.eye(3, dtype=np.float32) * (spacing**2 / 12)
    inverse = np.linalg.inv(covariance)
    radii = 3 * np.sqrt(np.diagonal(covariance, axis1=1, axis2=2))
    starts = np.maximum(np.floor((xyz - radii - origin) / spacing).astype(int), 0)
    stops = np.minimum(
        np.ceil((xyz + radii - origin) / spacing).astype(int) + 1, resolution
    )
    density = np.zeros((resolution,) * 3, dtype=np.float32)
    axis = np.arange(resolution, dtype=np.float32) * spacing
    for center, alpha, precision, start, stop in zip(
        xyz, opacity, inverse, starts, stops
    ):
        a, b, c = [slice(int(lo), int(hi)) for lo, hi in zip(start, stop)]
        dx = (axis[a] + origin[0] - center[0])[:, None, None]
        dy = (axis[b] + origin[1] - center[1])[None, :, None]
        dz = (axis[c] + origin[2] - center[2])[None, None, :]
        distance = (
            precision[0, 0] * dx**2
            + precision[1, 1] * dy**2
            + precision[2, 2] * dz**2
            + 2 * precision[0, 1] * dx * dy
            + 2 * precision[0, 2] * dx * dz
            + 2 * precision[1, 2] * dy * dz
        )
        density[a, b, c] += alpha * np.exp(-0.5 * distance) * (distance <= 9)

    if not density.min() < threshold < density.max():
        raise ValueError(
            "No surface at mesh_threshold; lower it or increase mesh_resolution"
        )
    vertices, faces, _, _ = marching_cubes(
        density, level=threshold, spacing=(spacing,) * 3, allow_degenerate=False
    )
    vertices += origin
    _, nearest = cKDTree(xyz).query(vertices)
    vertex_colors = np.rint(colors[nearest] * 255).astype(np.uint8)
    # Match PLY's proper rotation from z-up to y-up (preserves face winding).
    vertices = vertices[:, [0, 2, 1]] * np.array([1, -1, 1])
    mesh = trimesh.Trimesh(
        vertices=vertices, faces=faces, vertex_colors=vertex_colors, process=False
    )
    mesh.fix_normals(multibody=True)
    return mesh
