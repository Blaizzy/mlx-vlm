"""Camera decoding and depth unprojection for VGGT-Omega (lazy MLX)."""

from typing import Tuple

import mlx.core as mx

from ..moge3.geometry import intrinsics_from_focal_center
from ..sam3d_body.mhr_utils import quat_to_rotmat


def encoding_to_camera(
    pose_enc: mx.array, image_size_hw: Tuple[int, int]
) -> Tuple[mx.array, mx.array]:
    """Decode (..., 9) pose encodings to OpenCV camera-from-world extrinsics
    (..., 3, 4) and pinhole intrinsics (..., 3, 3) in pixels.

    The quaternion is XYZW and need not be unit length.
    """
    H, W = image_size_hw
    translation, quat = pose_enc[..., :3], pose_enc[..., 3:7]
    fov_h, fov_w = pose_enc[..., 7], pose_enc[..., 8]

    quat = quat / mx.linalg.norm(quat, axis=-1, keepdims=True)
    extrinsics = mx.concatenate([quat_to_rotmat(quat), translation[..., None]], -1)

    fy = (H / 2.0) / mx.tan(fov_h / 2.0)
    fx = (W / 2.0) / mx.tan(fov_w / 2.0)
    zeros = mx.zeros_like(fx)
    intrinsics = intrinsics_from_focal_center(fx, fy, zeros + W / 2, zeros + H / 2)
    return extrinsics, intrinsics


def unproject_depth(
    depth: mx.array, extrinsics: mx.array, intrinsics: mx.array
) -> mx.array:
    """World points (S, H, W, 3) from depth maps (S, H, W) and the cameras
    of ``encoding_to_camera``."""
    S, H, W = depth.shape
    x = mx.arange(W, dtype=mx.float32)[None, None, :]
    y = mx.arange(H, dtype=mx.float32)[None, :, None]
    fx, fy = intrinsics[:, 0, 0, None, None], intrinsics[:, 1, 1, None, None]
    cx, cy = intrinsics[:, 0, 2, None, None], intrinsics[:, 1, 2, None, None]
    camera = mx.stack(
        [(x - cx) / fx * depth, (y - cy) / fy * depth, depth], axis=-1
    )  # (S, H, W, 3)

    rotation, translation = extrinsics[:, :, :3], extrinsics[:, :, 3]
    # R^T (p - t), as a row-vector product per frame.
    points = (camera - translation[:, None, None, :]).reshape(S, H * W, 3)
    return (points @ rotation).reshape(S, H, W, 3)
