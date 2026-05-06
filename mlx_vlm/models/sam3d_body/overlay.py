"""Single-image overlay helpers for SAM 3D Body predictions.

Two flavors of overlay on top of the original BGR frame:

    draw_skeleton_overlay  — 2D skeleton + joints, pure OpenCV (no extra deps)
    render_mesh_overlay    — photorealistic mesh via pyrender + trimesh (optional)

Both take a prediction dict from `SAM3DBodyEstimator.predict()` and the original
frame, and return an annotated frame of the same shape.

Example:
    from mlx_vlm.models.sam3d_body.estimator import SAM3DBodyEstimator
    from mlx_vlm.models.sam3d_body.overlay import render_mesh_overlay, load_faces

    estimator = SAM3DBodyEstimator("/path/to/weights/")
    result = estimator.predict(image_rgb, bbox=[...])
    faces = load_faces("/path/to/weights/")
    out_bgr = render_mesh_overlay(result, frame_bgr, faces)
"""

from __future__ import annotations

import math
from pathlib import Path

import cv2
import numpy as np

from .video import draw_bbox, draw_skeleton, project_keypoints_perspective

_LIGHT_BLUE = (0.65098039, 0.74117647, 0.85882353)


def compute_cam_t(
    camera: np.ndarray, bbox, img_w: int, img_h: int, fov_deg: float = 60.0
):
    """Convert weak-perspective (scale, tx, ty) -> 3D translation + focal length.

    Matches the PyTorch reference's conversion used for photorealistic mesh
    rendering. Sign flip on scale/ty aligns the MLX camera frame with pyrender.

    Returns:
        cam_t: (3,) float — camera translation
        focal_length: float — pixel focal length derived from image diagonal & fov
    """
    pred_cam = np.asarray(camera, dtype=np.float32).copy()
    pred_cam[[0, 2]] *= -1
    s, tx, ty = pred_cam

    bbox_size = max(bbox[2] - bbox[0], bbox[3] - bbox[1])
    focal_length = img_h / (2 * math.tan(math.radians(fov_deg / 2)))
    bbox_center = np.array([(bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2])

    bs = bbox_size * s + 1e-8
    tz = 2 * focal_length / bs
    cx = 2 * (bbox_center[0] - img_w / 2) / bs
    cy = 2 * (bbox_center[1] - img_h / 2) / bs

    return np.array([tx + cx, ty + cy, tz], dtype=np.float32), float(focal_length)


def load_faces(weights_dir: str) -> np.ndarray:
    """Load MHR face indices from the converted safetensors weights.

    Faces are stored under `head_pose.faces` in the safetensors file.
    A cached copy is kept next to the weights for fast reuse.
    """
    from safetensors import safe_open

    weights_dir = Path(weights_dir)
    cache = weights_dir / "faces.npy"
    if cache.exists():
        return np.load(cache)

    safetensors_path = weights_dir / "model.safetensors"
    if not safetensors_path.exists():
        index = weights_dir / "model.safetensors.index.json"
        if not index.exists():
            raise FileNotFoundError(f"No safetensors found in {weights_dir}")
        import json

        with open(index) as fh:
            weight_map = json.load(fh)["weight_map"]
        shard = weight_map.get("head_pose.faces")
        if shard is None:
            raise KeyError("head_pose.faces not present in safetensors index")
        safetensors_path = weights_dir / shard

    with safe_open(str(safetensors_path), framework="numpy") as f:
        faces = f.get_tensor("head_pose.faces")

    try:
        np.save(cache, faces)
    except OSError:
        pass
    return faces


def draw_skeleton_overlay(result: dict, frame_bgr: np.ndarray) -> np.ndarray:
    """Draw the projected 2D skeleton + bbox on a copy of the BGR frame.

    Args:
        result: SAM3DBodyEstimator.predict() output (needs pred_keypoints_3d,
                pred_camera, bbox).
        frame_bgr: (H, W, 3) uint8 BGR frame (OpenCV convention).

    Returns:
        (H, W, 3) uint8 BGR frame with skeleton drawn.
    """
    h, w = frame_bgr.shape[:2]
    kp2d = project_keypoints_perspective(
        result["pred_keypoints_3d"], result["pred_camera"], result["bbox"], w, h
    )
    annotated = frame_bgr.copy()
    annotated = draw_bbox(annotated, result["bbox"])
    annotated = draw_skeleton(annotated, kp2d)
    return annotated


def render_mesh_overlay(
    result: dict,
    frame_bgr: np.ndarray,
    faces: np.ndarray,
    fov_deg: float = 60.0,
    color_bgr: tuple = _LIGHT_BLUE,
) -> np.ndarray:
    """Render the predicted mesh onto the original frame via pyrender.

    Requires `pyrender` and `trimesh` (optional mlx-vlm extras).

    Args:
        result: SAM3DBodyEstimator.predict() output (needs pred_vertices,
                pred_camera, bbox).
        frame_bgr: (H, W, 3) uint8 BGR frame.
        faces: (F, 3) int32 triangle indices — load with load_faces().
        fov_deg: assumed camera field of view for back-projection.
        color_bgr: base mesh color in BGR (0-1 range).

    Returns:
        (H, W, 3) uint8 BGR frame with mesh rendered over the subject.
    """
    try:
        import pyrender
        import trimesh
    except ImportError as exc:
        raise ImportError(
            "render_mesh_overlay requires 'pyrender' and 'trimesh'. "
            "Install with `pip install pyrender trimesh`. "
            "Use draw_skeleton_overlay for a no-extra-deps alternative."
        ) from exc

    h, w = frame_bgr.shape[:2]
    image = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0

    vertices = np.asarray(result["pred_vertices"], dtype=np.float32)
    cam_t, focal_length = compute_cam_t(
        result["pred_camera"], result["bbox"], w, h, fov_deg
    )
    cam_t = cam_t.copy()
    cam_t[0] *= -1.0  # flip x for pyrender camera frame

    material = pyrender.MetallicRoughnessMaterial(
        metallicFactor=0.0,
        alphaMode="OPAQUE",
        baseColorFactor=(color_bgr[2], color_bgr[1], color_bgr[0], 1.0),
    )
    tmesh = trimesh.Trimesh(vertices.copy(), faces.copy())
    tmesh.apply_transform(
        trimesh.transformations.rotation_matrix(np.radians(180), [1, 0, 0])
    )
    rmesh = pyrender.Mesh.from_trimesh(tmesh, material=material)

    scene = pyrender.Scene(bg_color=[0, 0, 0, 0], ambient_light=(0.3, 0.3, 0.3))
    scene.add(rmesh)

    camera_pose = np.eye(4)
    camera_pose[:3, 3] = cam_t
    scene.add(
        pyrender.IntrinsicsCamera(
            fx=focal_length, fy=focal_length, cx=w / 2, cy=h / 2, zfar=1e12
        ),
        pose=camera_pose,
    )

    # Three-point directional lighting around the subject
    for theta, phi in zip([np.pi / 6] * 3, [0, 2 * np.pi / 3, 4 * np.pi / 3]):
        xp = np.sin(theta) * np.cos(phi)
        yp = np.sin(theta) * np.sin(phi)
        zp = np.cos(theta)
        z = np.array([xp, yp, zp])
        z /= np.linalg.norm(z)
        x = np.array([-z[1], z[0], 0.0])
        if np.linalg.norm(x) == 0:
            x = np.array([1.0, 0.0, 0.0])
        x /= np.linalg.norm(x)
        y = np.cross(z, x)
        mat = np.eye(4)
        mat[:3, :3] = np.c_[x, y, z]
        scene.add_node(
            pyrender.Node(
                light=pyrender.DirectionalLight(color=np.ones(3), intensity=1.0),
                matrix=mat,
            )
        )

    renderer = pyrender.OffscreenRenderer(viewport_width=w, viewport_height=h)
    try:
        color, depth = renderer.render(scene, flags=pyrender.RenderFlags.RGBA)
    finally:
        renderer.delete()

    valid = (depth > 0).astype(np.float32)[:, :, None]
    color_f = color[:, :, :3].astype(np.float32) / 255.0
    rgb_out = color_f * valid + image * (1 - valid)
    return cv2.cvtColor((rgb_out * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
