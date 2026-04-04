"""Perspective camera projection for SAM 3D Body."""

import mlx.core as mx


def perspective_projection(
    points_3d: mx.array,
    focal_length: mx.array,
    camera_center: mx.array,
) -> mx.array:
    """Project 3D points to 2D using perspective projection.

    Args:
        points_3d: (B, N, 3) 3D points
        focal_length: (B, 2) or scalar focal lengths (fx, fy)
        camera_center: (B, 2) or scalar principal point (cx, cy)
    Returns:
        (B, N, 2) projected 2D points
    """
    projected = points_3d[..., :2] / points_3d[..., 2:3]
    projected = projected * focal_length[:, None, :] + camera_center[:, None, :]
    return projected
