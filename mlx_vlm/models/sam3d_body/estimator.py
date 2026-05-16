"""High-level estimator: load model, preprocess image, run inference."""

import math
from pathlib import Path

import mlx.core as mx
import numpy as np

from .batch_prep import get_cliff_condition, prepare_image
from .config import SAM3DConfig
from .model import SAM3DBody


def detect_persons(image_rgb, threshold=0.5):
    """Detect persons in image using torchvision's Faster R-CNN.

    Args:
        image_rgb: (H, W, 3) RGB uint8 numpy array
        threshold: detection confidence threshold

    Returns:
        list of [x1, y1, x2, y2] bounding boxes, sorted by area descending
    """
    try:
        import torch
        import torchvision
    except ImportError:
        return []

    model = torchvision.models.detection.fasterrcnn_mobilenet_v3_large_fpn(
        weights=torchvision.models.detection.FasterRCNN_MobileNet_V3_Large_FPN_Weights.DEFAULT
    )
    model.eval()

    # Convert to tensor: (H, W, 3) uint8 -> (3, H, W) float [0, 1]
    img_tensor = torch.from_numpy(image_rgb.copy()).permute(2, 0, 1).float() / 255.0

    with torch.no_grad():
        predictions = model([img_tensor])[0]

    # Filter for person class (class 1 in COCO)
    person_mask = predictions["labels"] == 1
    scores = predictions["scores"][person_mask]
    boxes = predictions["boxes"][person_mask]

    # Threshold
    keep = scores > threshold
    boxes = boxes[keep].numpy()
    scores = scores[keep].numpy()

    if len(boxes) == 0:
        return []

    # Sort by area (largest first — main subject is usually largest)
    areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    order = np.argsort(-areas)
    boxes = boxes[order]

    return boxes.tolist()


# Cache the detection model to avoid reloading per frame
_cached_detector = None


def _get_detector():
    """Get or create a cached person detection model."""
    global _cached_detector
    if _cached_detector is not None:
        return _cached_detector

    try:
        import torchvision
    except ImportError:
        return None

    model = torchvision.models.detection.fasterrcnn_mobilenet_v3_large_fpn(
        weights=torchvision.models.detection.FasterRCNN_MobileNet_V3_Large_FPN_Weights.DEFAULT
    )
    model.eval()
    _cached_detector = model
    return model


def detect_persons_cached(image_rgb, threshold=0.5):
    """Like detect_persons but reuses a cached model instance.

    Args:
        image_rgb: (H, W, 3) RGB uint8 numpy array
        threshold: detection confidence threshold

    Returns:
        list of [x1, y1, x2, y2] bounding boxes, sorted by area descending
    """
    import torch

    model = _get_detector()
    if model is None:
        return []

    img_tensor = torch.from_numpy(image_rgb.copy()).permute(2, 0, 1).float() / 255.0

    with torch.no_grad():
        predictions = model([img_tensor])[0]

    person_mask = predictions["labels"] == 1
    scores = predictions["scores"][person_mask]
    boxes = predictions["boxes"][person_mask]

    keep = scores > threshold
    boxes = boxes[keep].numpy()

    if len(boxes) == 0:
        return []

    areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    order = np.argsort(-areas)
    boxes = boxes[order]

    return boxes.tolist()


def make_default_intrinsics(img_h, img_w):
    """Build a default camera intrinsic matrix from image dimensions.

    Uses image diagonal as focal length, matching PyTorch's default.
    Returns: (3, 3) float32 numpy array
    """
    focal = math.sqrt(img_h**2 + img_w**2)
    return np.array(
        [
            [focal, 0, img_w / 2],
            [0, focal, img_h / 2],
            [0, 0, 1],
        ],
        dtype=np.float32,
    )


class SAM3DBodyEstimator:
    """End-to-end SAM 3D Body inference.

    Usage:
        estimator = SAM3DBodyEstimator('/tmp/sam3d-mlx-weights/')
        result = estimator.predict(image_rgb, bbox=[100, 50, 400, 450])
    """

    def __init__(self, weights_dir: str, config: SAM3DConfig = None):
        self.weights_dir = Path(weights_dir)

        # Load config from weights dir if available, otherwise use defaults
        config_path = self.weights_dir / "config.json"
        if config is None and config_path.exists():
            self.config = SAM3DConfig.load(config_path)
        elif config is not None:
            self.config = config
        else:
            self.config = SAM3DConfig()

        self.model = SAM3DBody(self.config)
        self.model.load_all_weights(str(self.weights_dir))
        # Freeze all parameters for inference
        self.model.eval()

    def predict(
        self,
        image: np.ndarray,
        bbox: list | None = None,
        cam_int: np.ndarray | None = None,
        auto_detect: bool = True,
    ) -> dict:
        """Run body estimation on a single image.

        Args:
            image: (H, W, 3) RGB uint8 numpy array
            bbox: [x1, y1, x2, y2] bounding box. If None, auto-detect or use full image.
            cam_int: (3, 3) camera intrinsic matrix. If None, uses default from FOV=60.
            auto_detect: if True and bbox is None, run person detection

        Returns:
            dict with keys:
                pred_vertices: (18439, 3) float32 numpy
                pred_keypoints_3d: (70, 3) float32 numpy
                pred_joint_coords: (127, 3) float32 numpy
                pred_camera: (3,) float32 numpy [scale, tx, ty]
                bbox: [x1, y1, x2, y2] the bbox used (useful when auto-detected)
        """
        h, w = image.shape[:2]

        # Auto-detect person if no bbox given
        if bbox is None and auto_detect:
            detections = detect_persons_cached(image, threshold=0.5)
            if detections:
                bbox = detections[0]  # Use largest person

        if bbox is None:
            bbox = [0, 0, w, h]

        # Build camera intrinsics if not provided
        if cam_int is None:
            cam_int = make_default_intrinsics(h, w)

        # Preprocess
        processed = prepare_image(
            image,
            bbox,
            image_size=self.config.image_size,
            mean=self.config.image_mean,
            std=self.config.image_std,
        )
        image_mx = mx.array(processed)

        # CLIFF condition (normalize by focal length, not image dims)
        focal_length = float(cam_int[0, 0])
        cliff = get_cliff_condition(bbox, (h, w), focal_length=focal_length)
        cliff_mx = mx.array(cliff[None])  # (1, 3)

        # Camera intrinsics as MLX array
        cam_int_mx = mx.array(cam_int)

        # Forward pass with ray conditioning and intermediate predictions
        body_output, pred_cam = self.model(
            image_mx,
            cliff_condition=cliff_mx,
            bbox=bbox,
            img_size=(h, w),
            cam_int=cam_int_mx,
        )

        # Evaluate lazily computed arrays (batch for efficiency)
        mx.eval(
            body_output["pred_vertices"],
            body_output["pred_keypoints_3d"],
            body_output["pred_joint_coords"],
            body_output["pred_model_params"],
            body_output["pred_shape"],
            pred_cam,
        )

        return {
            "pred_vertices": np.array(body_output["pred_vertices"][0]),
            "pred_keypoints_3d": np.array(body_output["pred_keypoints_3d"][0]),
            "pred_joint_coords": np.array(body_output["pred_joint_coords"][0]),
            "pred_camera": np.array(pred_cam[0]),
            "pred_pose": np.array(body_output["pred_model_params"][0, :136]),
            "pred_shape": np.array(body_output["pred_shape"][0]),
            "bbox": bbox,
        }

    def predict_batch(
        self,
        images: list[np.ndarray],
        bboxes: list[list],
    ) -> list[dict]:
        """Run inference on multiple crops. Each image+bbox is processed independently."""
        results = []
        for img, bbox in zip(images, bboxes):
            results.append(self.predict(img, bbox))
        return results


def write_obj(
    vertices: np.ndarray,
    faces: np.ndarray,
    path: str,
):
    """Write a mesh to OBJ format.

    Args:
        vertices: (V, 3) float
        faces: (F, 3) int, 0-indexed
        path: output .obj file path
    """
    with open(path, "w") as f:
        for v in vertices:
            f.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")
        # OBJ faces are 1-indexed
        for face in faces:
            f.write(f"f {face[0]+1} {face[1]+1} {face[2]+1}\n")
