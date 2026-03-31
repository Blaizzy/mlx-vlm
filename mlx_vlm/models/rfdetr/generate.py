"""RF-DETR inference pipeline."""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Union

import mlx.core as mx
import numpy as np
from PIL import Image

from .processing_rfdetr import COCO_CLASSES, RFDETRProcessor


@dataclass
class DetectionResult:
    """Detection output container."""
    boxes: np.ndarray  # (N, 4) xyxy format in pixel coordinates
    scores: np.ndarray  # (N,) confidence scores
    labels: np.ndarray  # (N,) integer class indices
    class_names: List[str] = field(default_factory=list)


def box_cxcywh_to_xyxy(boxes: np.ndarray) -> np.ndarray:
    """Convert center-format boxes to corner-format.

    Args:
        boxes: (..., 4) in [cx, cy, w, h] format
    Returns:
        (..., 4) in [x1, y1, x2, y2] format
    """
    cx, cy, w, h = boxes[..., 0], boxes[..., 1], boxes[..., 2], boxes[..., 3]
    x1 = cx - w / 2
    y1 = cy - h / 2
    x2 = cx + w / 2
    y2 = cy + h / 2
    return np.stack([x1, y1, x2, y2], axis=-1)


def postprocess(
    pred_logits: np.ndarray,
    pred_boxes: np.ndarray,
    original_size: Tuple[int, int],
    score_threshold: float = 0.5,
    num_select: int = 300,
    class_names: Optional[List[str]] = None,
) -> DetectionResult:
    """Post-process model outputs to detections.

    Args:
        pred_logits: (B, Q, num_classes) raw class logits
        pred_boxes: (B, Q, 4) boxes in cxcywh normalized [0, 1]
        original_size: (H, W) original image dimensions
        score_threshold: minimum confidence threshold
        num_select: max detections to return
        class_names: optional class name list
    Returns:
        DetectionResult for first image in batch
    """
    if class_names is None:
        class_names = COCO_CLASSES

    # Sigmoid on logits
    scores = 1 / (1 + np.exp(-pred_logits[0]))  # (Q, num_classes)

    # Get max score per query and corresponding class
    max_scores = scores.max(axis=-1)  # (Q,)
    max_classes = scores.argmax(axis=-1)  # (Q,)

    # Top-K selection
    if num_select < len(max_scores):
        topk_idx = np.argpartition(-max_scores, num_select)[:num_select]
    else:
        topk_idx = np.arange(len(max_scores))

    max_scores = max_scores[topk_idx]
    max_classes = max_classes[topk_idx]
    boxes = pred_boxes[0][topk_idx]  # (K, 4)

    # Filter by threshold
    keep = max_scores > score_threshold
    max_scores = max_scores[keep]
    max_classes = max_classes[keep]
    boxes = boxes[keep]

    # Convert boxes: cxcywh -> xyxy
    boxes = box_cxcywh_to_xyxy(boxes)

    # Scale to original image size
    orig_h, orig_w = original_size
    boxes[:, [0, 2]] *= orig_w
    boxes[:, [1, 3]] *= orig_h

    # Clip to image bounds
    boxes[:, [0, 2]] = np.clip(boxes[:, [0, 2]], 0, orig_w)
    boxes[:, [1, 3]] = np.clip(boxes[:, [1, 3]], 0, orig_h)

    # Map class indices to names
    names = [class_names[c] if c < len(class_names) else f"class_{c}" for c in max_classes]

    return DetectionResult(
        boxes=boxes,
        scores=max_scores,
        labels=max_classes,
        class_names=names,
    )


class RFDETRPredictor:
    """High-level inference wrapper for RF-DETR."""

    def __init__(
        self,
        model,
        processor: RFDETRProcessor,
        score_threshold: float = 0.5,
        class_names: Optional[List[str]] = None,
    ):
        self.model = model
        self.processor = processor
        self.score_threshold = score_threshold
        self.class_names = class_names or COCO_CLASSES

    def predict(
        self,
        image: Union[Image.Image, np.ndarray, str],
        score_threshold: Optional[float] = None,
    ) -> DetectionResult:
        """Run detection on a single image.

        Args:
            image: PIL Image, numpy array, or file path
            score_threshold: override default threshold
        Returns:
            DetectionResult with boxes, scores, labels
        """
        if isinstance(image, str):
            image = Image.open(image).convert("RGB")
        elif isinstance(image, np.ndarray):
            image = Image.fromarray(image)

        threshold = score_threshold if score_threshold is not None else self.score_threshold

        # Preprocess
        inputs = self.processor.preprocess_image(image)
        pixel_values = mx.array(inputs["pixel_values"])

        # Run model
        outputs = self.model(pixel_values)
        mx.eval(outputs["pred_logits"], outputs["pred_boxes"])

        # Post-process
        pred_logits = np.array(outputs["pred_logits"])
        pred_boxes = np.array(outputs["pred_boxes"])

        return postprocess(
            pred_logits,
            pred_boxes,
            original_size=inputs["original_size"],
            score_threshold=threshold,
            num_select=self.processor.num_select,
            class_names=self.class_names,
        )
