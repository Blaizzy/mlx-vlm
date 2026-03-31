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
    masks: Optional[np.ndarray] = None  # (N, H, W) binary masks if segmentation enabled


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
    pred_masks: Optional[np.ndarray] = None,
    nms_threshold: float = 0.5,
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

    # Top-K selection — track indices into original Q for mask lookup
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
    # Track which original query indices survived
    final_query_idx = topk_idx[keep]

    # Convert boxes: cxcywh -> xyxy
    boxes = box_cxcywh_to_xyxy(boxes)

    # Scale to original image size
    orig_h, orig_w = original_size
    boxes[:, [0, 2]] *= orig_w
    boxes[:, [1, 3]] *= orig_h

    # Clip to image bounds
    boxes[:, [0, 2]] = np.clip(boxes[:, [0, 2]], 0, orig_w)
    boxes[:, [1, 3]] = np.clip(boxes[:, [1, 3]], 0, orig_h)

    # Apply NMS per class
    if nms_threshold < 1.0 and len(boxes) > 0:
        nms_keep = _nms_per_class(boxes, max_scores, max_classes, nms_threshold)
        boxes = boxes[nms_keep]
        max_scores = max_scores[nms_keep]
        max_classes = max_classes[nms_keep]
        final_query_idx = final_query_idx[nms_keep]

    # Map class indices to names
    names = [class_names[c] if c < len(class_names) else f"class_{c}" for c in max_classes]

    # Process masks — only for surviving detections
    result_masks = None
    if pred_masks is not None and len(final_query_idx) > 0:
        masks = pred_masks[0][final_query_idx]  # (N_kept, mH, mW)
        result_masks = _resize_masks(masks, orig_h, orig_w)

    return DetectionResult(
        boxes=boxes,
        scores=max_scores,
        labels=max_classes,
        class_names=names,
        masks=result_masks,
    )


def _nms_per_class(
    boxes: np.ndarray,
    scores: np.ndarray,
    classes: np.ndarray,
    iou_threshold: float = 0.5,
) -> np.ndarray:
    """Per-class Non-Maximum Suppression.

    Args:
        boxes: (N, 4) xyxy format
        scores: (N,)
        classes: (N,) class indices
        iou_threshold: IoU threshold for suppression
    Returns:
        indices to keep
    """
    keep = []
    for cls in np.unique(classes):
        cls_mask = classes == cls
        cls_indices = np.where(cls_mask)[0]
        cls_boxes = boxes[cls_mask]
        cls_scores = scores[cls_mask]

        # Sort by score descending
        order = np.argsort(-cls_scores)
        cls_indices = cls_indices[order]
        cls_boxes = cls_boxes[order]

        while len(cls_indices) > 0:
            keep.append(cls_indices[0])
            if len(cls_indices) == 1:
                break

            # Compute IoU of first box with rest
            ious = _box_iou(cls_boxes[0:1], cls_boxes[1:])[0]
            # Keep boxes with IoU below threshold
            remaining = ious < iou_threshold
            cls_indices = cls_indices[1:][remaining]
            cls_boxes = cls_boxes[1:][remaining]

    if len(keep) == 0:
        return np.array([], dtype=np.int64)

    # Sort by score for consistent output
    keep = np.array(keep)
    keep = keep[np.argsort(-scores[keep])]
    return keep


def _box_iou(boxes1: np.ndarray, boxes2: np.ndarray) -> np.ndarray:
    """Compute IoU between two sets of xyxy boxes.

    Args:
        boxes1: (M, 4), boxes2: (N, 4)
    Returns:
        (M, N) IoU matrix
    """
    x1 = np.maximum(boxes1[:, None, 0], boxes2[None, :, 0])
    y1 = np.maximum(boxes1[:, None, 1], boxes2[None, :, 1])
    x2 = np.minimum(boxes1[:, None, 2], boxes2[None, :, 2])
    y2 = np.minimum(boxes1[:, None, 3], boxes2[None, :, 3])

    inter = np.maximum(0, x2 - x1) * np.maximum(0, y2 - y1)
    area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
    area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])
    union = area1[:, None] + area2[None, :] - inter

    return inter / (union + 1e-6)


def _resize_masks(masks: np.ndarray, target_h: int, target_w: int) -> np.ndarray:
    """Resize mask logits to target size and binarize.

    Args:
        masks: (N, mH, mW) mask logits
        target_h, target_w: output dimensions
    Returns:
        (N, target_h, target_w) binary uint8 masks
    """
    N, mH, mW = masks.shape
    # Bilinear interpolation via grid sampling
    y_coords = np.linspace(0, mH - 1, target_h)
    x_coords = np.linspace(0, mW - 1, target_w)
    yy, xx = np.meshgrid(y_coords, x_coords, indexing="ij")

    y0 = np.clip(np.floor(yy).astype(int), 0, mH - 1)
    y1 = np.clip(y0 + 1, 0, mH - 1)
    x0 = np.clip(np.floor(xx).astype(int), 0, mW - 1)
    x1 = np.clip(x0 + 1, 0, mW - 1)

    fy = yy - y0
    fx = xx - x0

    resized = np.zeros((N, target_h, target_w), dtype=np.float32)
    for i in range(N):
        m = masks[i]
        resized[i] = (
            m[y0, x0] * (1 - fy) * (1 - fx)
            + m[y0, x1] * (1 - fy) * fx
            + m[y1, x0] * fy * (1 - fx)
            + m[y1, x1] * fy * fx
        )

    # Binarize: sigmoid > 0.5 → logit > 0
    return (resized > 0).astype(np.uint8)


class RFDETRPredictor:
    """High-level inference wrapper for RF-DETR."""

    def __init__(
        self,
        model,
        processor: RFDETRProcessor,
        score_threshold: float = 0.5,
        nms_threshold: float = 0.5,
        class_names: Optional[List[str]] = None,
        exclude_classes: Optional[List[str]] = None,
    ):
        self.model = model
        self.processor = processor
        self.score_threshold = score_threshold
        self.nms_threshold = nms_threshold
        self.class_names = class_names or COCO_CLASSES
        self.exclude_classes = set(exclude_classes) if exclude_classes else set()

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
        to_eval = [outputs["pred_logits"], outputs["pred_boxes"]]
        if "pred_masks" in outputs:
            to_eval.append(outputs["pred_masks"])
        mx.eval(*to_eval)

        # Post-process
        pred_logits = np.array(outputs["pred_logits"])
        pred_boxes = np.array(outputs["pred_boxes"])
        pred_masks = np.array(outputs["pred_masks"]) if "pred_masks" in outputs else None

        result = postprocess(
            pred_logits,
            pred_boxes,
            original_size=inputs["original_size"],
            score_threshold=threshold,
            num_select=self.processor.num_select,
            class_names=self.class_names,
            pred_masks=pred_masks,
            nms_threshold=self.nms_threshold,
        )

        # Filter excluded classes
        if self.exclude_classes and len(result.scores) > 0:
            keep = np.array([n not in self.exclude_classes for n in result.class_names])
            result = DetectionResult(
                boxes=result.boxes[keep],
                scores=result.scores[keep],
                labels=result.labels[keep],
                class_names=[n for n, k in zip(result.class_names, keep) if k],
                masks=result.masks[keep] if result.masks is not None else None,
            )

        return result

    def predict_video(
        self,
        video_path: str,
        output_path: str,
        score_threshold: Optional[float] = None,
        max_frames: Optional[int] = None,
        show_fps: bool = False,
    ) -> Dict:
        """Run detection/segmentation on a video and save annotated output.

        Args:
            video_path: path to input video
            output_path: path to save annotated video
            score_threshold: override default threshold
            max_frames: limit number of frames (None = full video)
            show_fps: overlay FPS counter on video
        Returns:
            dict with stats: fps, total_frames, avg_detections
        """
        import cv2
        import time

        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if max_frames:
            total = min(total, max_frames)

        writer = cv2.VideoWriter(
            output_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h)
        )

        frame_idx = 0
        det_counts = []
        t_start = time.perf_counter()

        while frame_idx < total:
            ret, frame = cap.read()
            if not ret:
                break

            img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            result = self.predict(img, score_threshold=score_threshold)
            det_counts.append(len(result.scores))

            # Draw masks
            if result.masks is not None and len(result.scores) > 0:
                for i in range(len(result.scores)):
                    mask = result.masks[i].astype(bool)
                    color = _class_color(result.class_names[i])
                    frame[mask] = (
                        np.array(frame[mask], dtype=np.float32) * 0.5
                        + np.array(color, dtype=np.float32) * 0.5
                    ).astype(np.uint8)

            # Draw boxes + labels
            for i in range(len(result.scores)):
                box = result.boxes[i].astype(int)
                color = _class_color(result.class_names[i])
                cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), color, 2)
                label = f"{result.class_names[i]} {result.scores[i]:.2f}"
                (tw, th), _ = cv2.getTextSize(
                    label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
                )
                cv2.rectangle(
                    frame,
                    (box[0], box[1] - th - 6),
                    (box[0] + tw, box[1]),
                    color,
                    -1,
                )
                cv2.putText(
                    frame, label, (box[0], box[1] - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1,
                )

            # FPS overlay
            if show_fps and frame_idx > 0:
                elapsed = time.perf_counter() - t_start
                cur_fps = frame_idx / elapsed
                cv2.putText(
                    frame, f"{cur_fps:.1f} FPS", (10, 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2,
                )

            writer.write(frame)
            frame_idx += 1

        cap.release()
        writer.release()

        elapsed = time.perf_counter() - t_start
        avg_fps = frame_idx / elapsed
        stats = {
            "total_frames": frame_idx,
            "elapsed_seconds": elapsed,
            "fps": avg_fps,
            "avg_detections": np.mean(det_counts) if det_counts else 0,
            "output_path": output_path,
        }
        return stats


# Per-class BGR colors for visualization
_COLOR_MAP = {
    "person": (50, 50, 255), "car": (255, 200, 0), "truck": (100, 255, 0),
    "bus": (0, 100, 255), "motorcycle": (200, 0, 255), "bicycle": (0, 255, 100),
    "traffic light": (255, 255, 0), "cat": (48, 59, 255), "dog": (48, 59, 255),
    "remote": (255, 122, 0), "couch": (89, 199, 52),
}


def _class_color(name: str) -> Tuple[int, int, int]:
    """Get a BGR color for a class name."""
    if name in _COLOR_MAP:
        return _COLOR_MAP[name]
    # Deterministic hash-based color
    h = hash(name) % 360
    # HSV to BGR via simple conversion
    import colorsys
    r, g, b = colorsys.hsv_to_rgb(h / 360, 0.8, 0.9)
    return (int(b * 255), int(g * 255), int(r * 255))


def _draw_result_on_image(image: Image.Image, result: DetectionResult) -> Image.Image:
    """Draw detection boxes, labels, and masks on a PIL image."""
    import cv2

    frame = np.ascontiguousarray(np.array(image)[..., ::-1])  # RGB -> BGR

    # Draw masks
    if result.masks is not None and len(result.scores) > 0:
        for i in range(len(result.scores)):
            mask = result.masks[i].astype(bool)
            color = _class_color(result.class_names[i])
            frame[mask] = (
                np.array(frame[mask], dtype=np.float32) * 0.5
                + np.array(color, dtype=np.float32) * 0.5
            ).astype(np.uint8)

    # Draw boxes + labels
    for i in range(len(result.scores)):
        box = result.boxes[i].astype(int)
        color = _class_color(result.class_names[i])
        cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), color, 2)
        label = f"{result.class_names[i]} {result.scores[i]:.2f}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(frame, (box[0], box[1] - th - 6), (box[0] + tw, box[1]), color, -1)
        cv2.putText(frame, label, (box[0], box[1] - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    return Image.fromarray(frame[..., ::-1])  # BGR -> RGB


def main():
    """RF-DETR CLI for detection and segmentation on images and videos.

    Usage:
        # Detect on image
        python -m mlx_vlm.models.rfdetr.generate --image photo.jpg --model ./rfdetr-base-mlx

        # Segment on image
        python -m mlx_vlm.models.rfdetr.generate --image photo.jpg --model ./rfdetr-seg-small-mlx

        # Detect on video
        python -m mlx_vlm.models.rfdetr.generate --video traffic.mp4 --model ./rfdetr-base-mlx

        # Segment on video
        python -m mlx_vlm.models.rfdetr.generate --video traffic.mp4 --model ./rfdetr-seg-small-mlx

        # With options
        python -m mlx_vlm.models.rfdetr.generate --video input.mp4 --model ./rfdetr-seg-small-mlx \\
            --output out.mp4 --threshold 0.3 --nms-threshold 0.5 --exclude couch bed --show-fps
    """
    import argparse
    from pathlib import Path
    from mlx_vlm.utils import load_model

    parser = argparse.ArgumentParser(description="RF-DETR detection/segmentation")
    parser.add_argument("--image", type=str, help="Input image path")
    parser.add_argument("--video", type=str, help="Input video path")
    parser.add_argument("--model", type=str, required=True, help="Model directory")
    parser.add_argument("--output", type=str, help="Output path (default: auto-named)")
    parser.add_argument("--threshold", type=float, default=0.3, help="Score threshold")
    parser.add_argument("--nms-threshold", type=float, default=0.5, help="NMS IoU threshold")
    parser.add_argument("--exclude", nargs="+", default=[], help="Classes to exclude")
    parser.add_argument("--show-fps", action="store_true", help="Show FPS overlay on video")
    parser.add_argument("--max-frames", type=int, default=None, help="Max video frames")
    args = parser.parse_args()

    if not args.image and not args.video:
        parser.error("Provide --image or --video")

    # Load model
    model_path = Path(args.model)
    print(f"Loading model from {model_path}...")
    model = load_model(model_path)
    processor = RFDETRProcessor.from_pretrained(str(model_path))
    predictor = RFDETRPredictor(
        model, processor,
        score_threshold=args.threshold,
        nms_threshold=args.nms_threshold,
        exclude_classes=args.exclude or None,
    )

    if args.image:
        # Image inference
        image = Image.open(args.image).convert("RGB")
        result = predictor.predict(image)

        print(f"{len(result.scores)} detections:")
        for i in range(len(result.scores)):
            box = result.boxes[i]
            mask_info = f"  mask_px={result.masks[i].sum()}" if result.masks is not None else ""
            print(f"  {result.class_names[i]:20s} {result.scores[i]:.3f}  "
                  f"[{box[0]:.0f},{box[1]:.0f},{box[2]:.0f},{box[3]:.0f}]{mask_info}")

        # Save annotated image
        out_path = args.output or args.image.rsplit(".", 1)[0] + "_rfdetr.jpg"
        annotated = _draw_result_on_image(image, result)
        annotated.save(out_path, quality=95)
        print(f"Saved to {out_path}")

    elif args.video:
        # Video inference
        out_path = args.output or args.video.rsplit(".", 1)[0] + "_rfdetr.mp4"
        stats = predictor.predict_video(
            args.video, out_path,
            show_fps=args.show_fps,
            max_frames=args.max_frames,
        )
        print(f"{stats['total_frames']} frames in {stats['elapsed_seconds']:.1f}s "
              f"({stats['fps']:.1f} FPS, {stats['avg_detections']:.1f} avg dets)")
        print(f"Saved to {stats['output_path']}")


if __name__ == "__main__":
    main()
