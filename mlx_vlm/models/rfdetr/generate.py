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
    names = [
        class_names[c] if c < len(class_names) else f"class_{c}" for c in max_classes
    ]

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
    """Resize mask logits to target size and binarize with smooth edges.

    Args:
        masks: (N, mH, mW) mask logits
        target_h, target_w: output dimensions
    Returns:
        (N, target_h, target_w) binary uint8 masks
    """
    import cv2

    N = masks.shape[0]
    out = np.empty((N, target_h, target_w), dtype=np.uint8)
    for i in range(N):
        # Cubic interpolation for smoother upscale of logits
        resized = cv2.resize(
            masks[i], (target_w, target_h), interpolation=cv2.INTER_CUBIC
        )
        out[i] = (resized > 0).astype(np.uint8)
    return out


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

        threshold = (
            score_threshold if score_threshold is not None else self.score_threshold
        )

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
        pred_masks = (
            np.array(outputs["pred_masks"]) if "pred_masks" in outputs else None
        )

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

    def predict_bgr(
        self,
        bgr: np.ndarray,
        score_threshold: Optional[float] = None,
    ) -> DetectionResult:
        """Fast prediction from BGR numpy array (skips PIL, for video/camera).

        Args:
            bgr: (H, W, 3) BGR uint8 array from cv2
            score_threshold: override default threshold
        Returns:
            DetectionResult
        """
        threshold = (
            score_threshold if score_threshold is not None else self.score_threshold
        )

        inputs = self.processor.preprocess_bgr(bgr)
        pixel_values = mx.array(inputs["pixel_values"])

        outputs = self.model(pixel_values)
        to_eval = [outputs["pred_logits"], outputs["pred_boxes"]]
        if "pred_masks" in outputs:
            to_eval.append(outputs["pred_masks"])
        mx.eval(*to_eval)

        pred_logits = np.array(outputs["pred_logits"])
        pred_boxes = np.array(outputs["pred_boxes"])
        pred_masks = (
            np.array(outputs["pred_masks"]) if "pred_masks" in outputs else None
        )

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
        annotator=None,
        task: str = "auto",
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
        import time

        import cv2

        # Build default annotator if none provided
        if annotator is not None:
            ann = annotator
        else:
            ann = _get_annotator(None, task if task != "auto" else "detect")

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

        from tqdm import tqdm

        det_counts = []
        t_start = time.perf_counter()
        pbar = tqdm(total=total, desc="Processing", unit="frame")

        frame_idx = 0
        while frame_idx < total:
            ret, frame = cap.read()
            if not ret:
                break

            result = self.predict_bgr(frame, score_threshold=score_threshold)
            det_counts.append(len(result.scores))

            # Strip masks for detect-only task
            if task == "detect":
                result = DetectionResult(
                    boxes=result.boxes,
                    scores=result.scores,
                    labels=result.labels,
                    class_names=result.class_names,
                )

            # Annotate frame
            if ann is not None and len(result.scores) > 0:
                frame = ann.annotate(frame, _to_annotator_result(result))

            # FPS overlay
            if show_fps and frame_idx > 0:
                elapsed = time.perf_counter() - t_start
                cur_fps = frame_idx / elapsed
                cv2.putText(
                    frame,
                    f"{cur_fps:.1f} FPS",
                    (10, 28),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 255, 0),
                    2,
                )

            writer.write(frame)
            frame_idx += 1
            pbar.update(1)

        pbar.close()

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

    def predict_realtime(
        self,
        source: str = "0",
        score_threshold: Optional[float] = None,
        annotator=None,
        task: str = "auto",
    ):
        """Run realtime detection/segmentation from camera or video with display.

        Camera: threaded reader so cap.read() doesn't block inference.
        Video file: single-threaded with frame pacing.
        Press 'q' to quit.

        Args:
            source: camera index ("0") or video path
            score_threshold: override default threshold
            annotator: annotator chain (default: auto based on task)
            task: "detect", "segment", or "auto"
        """
        import threading
        import time

        import cv2

        if annotator is None:
            annotator = _get_annotator(None, task if task != "auto" else "detect")

        is_camera = source.isdigit()
        cap = cv2.VideoCapture(int(source) if is_camera else source)
        if not cap.isOpened():
            print(f"Error: cannot open {source}")
            return

        # Set camera to 1280x720 (16:9) for widescreen display
        if is_camera:
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        print(f"Source: {W}x{H} @ {fps:.0f}fps {'(camera)' if is_camera else ''}")
        print("Press 'q' to quit")

        threshold = score_threshold or self.score_threshold
        fps_counter = 0
        fps_t0 = time.perf_counter()
        display_fps = 0.0
        infer_fps = 0.0
        frame_interval = 1.0 / fps

        # Threaded camera reader — always has the latest frame ready
        if is_camera:
            latest_frame = [None]
            cam_lock = threading.Lock()
            running = [True]

            def _cam_reader():
                while running[0]:
                    ret, f = cap.read()
                    if ret:
                        with cam_lock:
                            latest_frame[0] = f

            reader = threading.Thread(target=_cam_reader, daemon=True)
            reader.start()

        while True:
            if is_camera:
                with cam_lock:
                    frame = latest_frame[0]
                if frame is None:
                    time.sleep(0.001)
                    continue
            else:
                ret, frame = cap.read()
                if not ret:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    continue

            # Inference
            t0 = time.perf_counter()
            result = self.predict_bgr(frame, score_threshold=threshold)
            t1 = time.perf_counter()
            infer_fps = 1.0 / max(t1 - t0, 1e-6)

            if task == "detect":
                result = DetectionResult(
                    boxes=result.boxes,
                    scores=result.scores,
                    labels=result.labels,
                    class_names=result.class_names,
                )

            # Annotate
            if len(result.scores) > 0:
                out = annotator.annotate(frame, _to_annotator_result(result))
            else:
                out = frame

            # FPS counter
            fps_counter += 1
            now = time.perf_counter()
            if now - fps_t0 >= 0.5:
                display_fps = fps_counter / (now - fps_t0)
                fps_counter = 0
                fps_t0 = now

            cv2.putText(
                out,
                f"Infer: {infer_fps:.0f} FPS | Loop: {display_fps:.0f} FPS | {len(result.scores)} obj",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2,
            )

            cv2.imshow("RF-DETR Realtime", out)

            # Frame pacing for video files
            if not is_camera:
                elapsed = time.perf_counter() - t0
                wait_ms = max(1, int((frame_interval - elapsed) * 1000))
            else:
                wait_ms = 1
            if cv2.waitKey(wait_ms) & 0xFF == ord("q"):
                break

        if is_camera:
            running[0] = False
        cap.release()
        cv2.destroyAllWindows()
        print("Done")


def _to_annotator_result(result: DetectionResult):
    """Adapt DetectionResult for SAM3 annotators (expects list labels, not numpy)."""

    class _AnnResult:
        __slots__ = ("boxes", "scores", "masks", "labels", "class_names")

    r = _AnnResult()
    r.boxes = result.boxes
    r.scores = result.scores
    r.masks = result.masks
    # SAM3 LabelAnnotator uses result.labels for display text
    r.labels = list(result.class_names)
    r.class_names = result.class_names
    return r


def _get_annotator(
    name: Optional[str],
    task: str,
    opacity: float = 0.5,
    contour_thickness: int = 1,
):
    """Build an annotator chain. Reuses SAM3's annotator system."""
    from ..sam3.generate import build_annotator

    if name:
        return build_annotator(
            name, opacity=opacity, contour_thickness=contour_thickness
        )

    from ..sam3.annotators import BoxAnnotator, LabelAnnotator, MaskAnnotator

    if task == "segment":
        return (
            MaskAnnotator(opacity=opacity, contour_thickness=contour_thickness)
            + BoxAnnotator()
            + LabelAnnotator()
        )
    else:
        return BoxAnnotator() + LabelAnnotator()


def main():
    """RF-DETR CLI for detection and segmentation on images and videos.

    Usage:
        python -m mlx_vlm.models.rfdetr.generate --task detect --image photo.jpg --model ./rfdetr-base-mlx
        python -m mlx_vlm.models.rfdetr.generate --task segment --video traffic.mp4 --model ./rfdetr-seg-small-mlx
        python -m mlx_vlm.models.rfdetr.generate --task realtime --model ./rfdetr-base-mlx
        python -m mlx_vlm.models.rfdetr.generate --task realtime --video traffic.mp4 --model ./rfdetr-seg-small-mlx
    """
    import argparse

    from mlx_vlm.generate import wired_limit
    from mlx_vlm.utils import get_model_path, load_model

    from ..sam3.generate import ANNOTATOR_PRESETS

    parser = argparse.ArgumentParser(description="RF-DETR detection/segmentation")
    parser.add_argument(
        "--task",
        default="auto",
        choices=["auto", "detect", "segment", "track", "realtime"],
        help="Task: detect (image), segment (image+masks), track (video to file), realtime (live display)",
    )
    parser.add_argument("--image", type=str, help="Input image path")
    parser.add_argument(
        "--video", type=str, help="Input video path or camera index (0)"
    )
    parser.add_argument("--model", type=str, required=True, help="Model directory")
    parser.add_argument("--output", type=str, help="Output path (default: auto-named)")
    parser.add_argument("--threshold", type=float, default=0.3, help="Score threshold")
    parser.add_argument(
        "--nms-threshold", type=float, default=0.5, help="NMS IoU threshold"
    )
    parser.add_argument("--exclude", nargs="+", default=[], help="Classes to exclude")
    parser.add_argument(
        "--show-boxes",
        action="store_true",
        default=True,
        help="Show bounding boxes and labels (default: on)",
    )
    parser.add_argument(
        "--show-fps", action="store_true", help="Show FPS overlay on video"
    )
    parser.add_argument("--max-frames", type=int, default=None, help="Max video frames")
    parser.add_argument(
        "--annotator",
        default=None,
        help="Annotation style. Presets: "
        + ", ".join(ANNOTATOR_PRESETS.keys())
        + ". Or chain: MaskAnnotator+BoxAnnotator+LabelAnnotator",
    )
    parser.add_argument("--opacity", type=float, default=0.5, help="Mask opacity")
    parser.add_argument(
        "--contour-thickness", type=int, default=1, help="Mask contour thickness"
    )
    args = parser.parse_args()

    if args.task == "track" and not args.video:
        parser.error("--task track requires --video")
    if args.task not in ("realtime", "track") and not args.image and not args.video:
        parser.error("Provide --image or --video (or use --task realtime/track)")

    # Load model
    model_path = get_model_path(args.model)
    print(f"Loading model from {model_path}...")
    model = load_model(model_path)
    processor = RFDETRProcessor.from_pretrained(str(model_path))

    # Determine task
    has_seg = model.segmentation_head is not None
    task = args.task
    if task == "auto":
        task = "segment" if has_seg else "detect"
    if task in ("segment", "realtime") and not has_seg:
        if task == "segment":
            print("Warning: model has no segmentation head, falling back to detect")
        task = "detect" if task == "segment" else task

    predictor = RFDETRPredictor(
        model,
        processor,
        score_threshold=args.threshold,
        nms_threshold=args.nms_threshold,
        exclude_classes=args.exclude or None,
    )

    # Build annotator
    effective_task = "segment" if has_seg and task != "detect" else "detect"
    annotator = _get_annotator(
        args.annotator,
        effective_task,
        opacity=args.opacity,
        contour_thickness=args.contour_thickness,
    )

    with wired_limit(model):
        if task == "realtime" or (args.task == "realtime"):
            source = args.video or "0"
            predictor.predict_realtime(
                source=source,
                annotator=annotator,
                task="segment" if has_seg else "detect",
            )

        elif args.image:
            import time

            image = Image.open(args.image).convert("RGB")

            # Warmup
            _ = predictor.predict(image)
            mx.eval(mx.zeros(1))

            mem_before = mx.get_peak_memory() / 1e6
            t0 = time.perf_counter()
            result = predictor.predict(image)
            t1 = time.perf_counter()
            peak_mem = mx.get_peak_memory() / 1e6

            if task == "detect":
                result = DetectionResult(
                    boxes=result.boxes,
                    scores=result.scores,
                    labels=result.labels,
                    class_names=result.class_names,
                )

            print(f"{len(result.scores)} detections:")
            for i in range(len(result.scores)):
                box = result.boxes[i]
                mask_info = (
                    f"  mask_px={result.masks[i].sum()}"
                    if result.masks is not None
                    else ""
                )
                print(
                    f"  {result.class_names[i]:20s} {result.scores[i]:.3f}  "
                    f"[{box[0]:.0f},{box[1]:.0f},{box[2]:.0f},{box[3]:.0f}]{mask_info}"
                )

            print(f"\nPerformance:")
            print(f"  Inference: {(t1-t0)*1000:.1f} ms ({1/(t1-t0):.1f} FPS)")
            print(f"  Peak memory: {peak_mem:.1f} MB")

            out_path = args.output or args.image.rsplit(".", 1)[0] + "_rfdetr.jpg"
            scene = np.ascontiguousarray(np.array(image)[..., ::-1])
            scene = annotator.annotate(scene, _to_annotator_result(result))
            Image.fromarray(scene[..., ::-1]).save(out_path, quality=95)
            print(f"Saved to {out_path}")

        elif args.video or task == "track":
            import time

            mx.reset_peak_memory()
            t_start = time.perf_counter()
            out_path = args.output or args.video.rsplit(".", 1)[0] + "_rfdetr.mp4"
            stats = predictor.predict_video(
                args.video,
                out_path,
                show_fps=args.show_fps,
                max_frames=args.max_frames,
                annotator=annotator,
                task=task,
            )
            peak_mem = mx.get_peak_memory() / 1e6

            print(
                f"{stats['total_frames']} frames in {stats['elapsed_seconds']:.1f}s "
                f"({stats['fps']:.1f} FPS, {stats['avg_detections']:.1f} avg dets)"
            )
            print(f"Peak memory: {peak_mem:.1f} MB")
            print(f"Saved to {stats['output_path']}")


if __name__ == "__main__":
    main()
