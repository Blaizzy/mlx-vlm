"""RTMDet inference: multi-level point-based decode + class-agnostic NMS.

`RTMDetPredictor.predict(image)` returns a `DetectionResult` dataclass with
(x1, y1, x2, y2) boxes in the ORIGINAL image's pixel coords.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Union

import mlx.core as mx
import numpy as np
from PIL import Image

from .processing_rtmdet import RTMDetProcessor


@dataclass
class DetectionResult:
    boxes: np.ndarray      # (N, 4) xyxy in original image coords
    scores: np.ndarray     # (N,)
    labels: np.ndarray     # (N,) class indices
    original_size: Tuple[int, int] = (0, 0)  # (orig_h, orig_w)


def _nms_class_agnostic(
    boxes: np.ndarray, scores: np.ndarray, iou_thr: float
) -> np.ndarray:
    """Plain greedy NMS over xyxy boxes, returns kept indices."""
    if boxes.size == 0:
        return np.empty((0,), dtype=np.int64)
    x1 = boxes[:, 0]; y1 = boxes[:, 1]; x2 = boxes[:, 2]; y2 = boxes[:, 3]
    areas = (x2 - x1).clip(min=0) * (y2 - y1).clip(min=0)
    order = scores.argsort()[::-1]
    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        if order.size == 1:
            break
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        w = np.clip(xx2 - xx1, 0, None)
        h = np.clip(yy2 - yy1, 0, None)
        inter = w * h
        union = areas[i] + areas[order[1:]] - inter
        iou = inter / np.maximum(union, 1e-9)
        order = order[1:][iou <= iou_thr]
    return np.array(keep, dtype=np.int64)


def _decode_level(cls_logits: np.ndarray, reg_pred: np.ndarray,
                  stride: int, score_thr: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Decode one FPN level.

    Args:
      cls_logits: (H, W, C) raw per-pixel class logits
      reg_pred  : (H, W, 4) per-pixel distance prediction (l, t, r, b) in stride-normalized units
      stride    : pixel stride of this level (8, 16 or 32)
    Returns:
      boxes  : (N, 4) xyxy in RESIZED-image coords (before letterbox undo)
      scores : (N,)
      labels : (N,) class index
    """
    H, W, C = cls_logits.shape
    scores = 1.0 / (1.0 + np.exp(-cls_logits))  # sigmoid
    # flatten to (H*W, C)
    scores_flat = scores.reshape(-1, C)
    reg_flat = reg_pred.reshape(-1, 4)
    # Pick the max class per anchor point, then score-threshold
    best_cls = scores_flat.argmax(axis=-1)
    best_score = scores_flat[np.arange(scores_flat.shape[0]), best_cls]
    keep = best_score > score_thr
    if not keep.any():
        empty = np.zeros((0, 4), dtype=np.float32)
        return empty, np.zeros((0,), dtype=np.float32), np.zeros((0,), dtype=np.int64)

    # Anchor points at stride-grid cell centers
    ys, xs = np.meshgrid(np.arange(H), np.arange(W), indexing="ij")
    cx = (xs.reshape(-1) + 0.5) * stride
    cy = (ys.reshape(-1) + 0.5) * stride

    # Distance decoding: bbox_preds arrive from the head already in image-space
    # pixels (head applies `.exp() * stride`).  Box = (cx - l, cy - t, cx + r, cy + b).
    ltrb = reg_flat[keep]
    cx_k = cx[keep]; cy_k = cy[keep]
    x1 = cx_k - ltrb[:, 0]
    y1 = cy_k - ltrb[:, 1]
    x2 = cx_k + ltrb[:, 2]
    y2 = cy_k + ltrb[:, 3]
    boxes = np.stack([x1, y1, x2, y2], axis=-1).astype(np.float32)
    return boxes, best_score[keep].astype(np.float32), best_cls[keep].astype(np.int64)


def _stride_unused_marker():  # (no-op, kept for future debug)
    return None


class RTMDetPredictor:
    def __init__(self, model, processor: RTMDetProcessor,
                 score_threshold: float = 0.3, nms_iou_threshold: float = 0.65,
                 max_detections: int = 100):
        self.model = model
        self.processor = processor
        self.score_threshold = score_threshold
        self.nms_iou_threshold = nms_iou_threshold
        self.max_detections = max_detections
        self.strides = list(getattr(model.config, "strides", [8, 16, 32]))

    def predict(self, image: Union[Image.Image, np.ndarray]) -> DetectionResult:
        inputs = self.processor.preprocess_image(image)
        pv = mx.array(inputs["pixel_values"])
        orig_h, orig_w = inputs["original_size"]
        scale = float(inputs["scale"])

        out = self.model(pv)
        cls_outs, reg_outs = out["cls_outs"], out["reg_outs"]
        mx.eval(cls_outs, reg_outs)

        all_boxes, all_scores, all_labels = [], [], []
        for level, (c, r) in enumerate(zip(cls_outs, reg_outs)):
            boxes, scores, labels = _decode_level(
                np.array(c)[0], np.array(r)[0],
                stride=self.strides[level],
                score_thr=self.score_threshold,
            )
            all_boxes.append(boxes)
            all_scores.append(scores)
            all_labels.append(labels)

        boxes = np.concatenate(all_boxes, axis=0)
        scores = np.concatenate(all_scores, axis=0)
        labels = np.concatenate(all_labels, axis=0)
        if boxes.size == 0:
            return DetectionResult(boxes, scores, labels, original_size=(orig_h, orig_w))

        keep = _nms_class_agnostic(boxes, scores, self.nms_iou_threshold)
        if keep.size > self.max_detections:
            keep = keep[:self.max_detections]
        boxes = boxes[keep] / scale     # undo letterbox scaling
        scores = scores[keep]
        labels = labels[keep]

        # Clip to original image bounds
        boxes[:, [0, 2]] = np.clip(boxes[:, [0, 2]], 0, orig_w)
        boxes[:, [1, 3]] = np.clip(boxes[:, [1, 3]], 0, orig_h)

        return DetectionResult(boxes, scores, labels, original_size=(orig_h, orig_w))
