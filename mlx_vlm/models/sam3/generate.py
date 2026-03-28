"""SAM3 Inference Pipeline: detection, segmentation, and video tracking.

Usage:
    # Python API
    from mlx_vlm.models.sam3.generate import Sam3Predictor, Sam3VideoPredictor

    predictor = Sam3Predictor(model, processor)
    result = predictor.predict(image, text_prompt="a dog")

    # Video tracking CLI
    python -m mlx_vlm.models.sam3.generate --video input.mp4 --prompt "a car"
"""

import argparse

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import mlx.core as mx
import mlx.nn as nn
import numpy as np
from PIL import Image


@dataclass
class DetectionResult:
    """Single detection result."""

    boxes: np.ndarray  # (N, 4) xyxy format
    masks: np.ndarray  # (N, H, W) binary masks
    scores: np.ndarray  # (N,) confidence scores
    labels: Optional[List[str]] = None


@dataclass
class TrackingResult:
    """Per-frame tracking result."""

    frame_idx: int
    masks: np.ndarray  # (N_obj, H, W) binary masks
    scores: np.ndarray  # (N_obj,) confidence scores
    object_ids: List[int] = None


# ---------------------------------------------------------------------------
# Image Predictor
# ---------------------------------------------------------------------------


class Sam3Predictor:
    """Image-level segmentation predictor."""

    def __init__(self, model, processor, score_threshold: float = 0.5):
        self.model = model
        self.processor = processor
        self.score_threshold = score_threshold
        self._text_cache = {}  # prompt -> (inputs_embeds, attention_mask)

    def _get_input_embeddings(self, text_prompt: str):
        """Get text features, using cache if same prompt was seen before."""
        if text_prompt not in self._text_cache:
            text_inputs = self.processor.preprocess_text(text_prompt)
            input_ids = mx.array(text_inputs["input_ids"])
            attention_mask = mx.array(text_inputs["attention_mask"])
            inputs_embeds, attention_mask = self.model.get_input_embeddings(
                input_ids, attention_mask
            )
            self._text_cache[text_prompt] = (inputs_embeds, attention_mask)
        return self._text_cache[text_prompt]

    def predict(
        self,
        image: Union[Image.Image, np.ndarray],
        text_prompt: str,
        boxes: Optional[np.ndarray] = None,
        score_threshold: Optional[float] = None,
    ) -> DetectionResult:
        """Run detection + segmentation on a single image.

        Text encoding is cached — repeated calls with the same prompt
        skip the text encoder entirely (~10ms saved per call).

        Args:
            image: PIL Image or numpy array (H, W, 3)
            text_prompt: text description of objects to detect
            boxes: optional (N, 4) box prompts in xyxy format
            score_threshold: confidence threshold (default: self.score_threshold)
        Returns:
            DetectionResult with boxes, masks, and scores
        """
        threshold = score_threshold or self.score_threshold

        # Preprocess image
        inputs = self.processor.preprocess_image(image)
        pixel_values = mx.array(inputs["pixel_values"])

        # Get cached text features
        inputs_embeds, attention_mask = self._get_input_embeddings(text_prompt)

        box_input = None
        if boxes is not None:
            box_input = mx.array(boxes)[None]  # (1, N, 4)

        # Run detection (text encoding skipped via cache)
        outputs = self.model.detect(
            pixel_values,
            attention_mask=attention_mask,
            boxes=box_input,
            inputs_embeds=inputs_embeds,
        )
        mx.eval(outputs)

        # Post-process
        return self._postprocess(
            outputs,
            image_size=(
                image.size if isinstance(image, Image.Image) else image.shape[:2]
            ),
            threshold=threshold,
        )

    def _postprocess(
        self,
        outputs: Dict[str, mx.array],
        image_size: Tuple[int, int],
        threshold: float = 0.5,
    ) -> DetectionResult:
        """Post-process model outputs into final detections."""
        pred_logits = np.array(outputs["pred_logits"][0])  # (Q,) or (Q, 1)
        pred_boxes = np.array(outputs["pred_boxes"][0])  # (Q, 4) xyxy
        pred_masks = np.array(outputs["pred_masks"][0])  # (Q, H_mask, W_mask)

        scores = _sigmoid(pred_logits).squeeze()  # (Q,)
        if "presence_logits" in outputs and outputs["presence_logits"] is not None:
            pres = _sigmoid(np.array(outputs["presence_logits"][0]))
            scores = scores * pres  # broadcast (Q,) * (1,)

        # Filter by score
        keep = scores > threshold
        scores = scores[keep]
        boxes = pred_boxes[keep]
        masks = pred_masks[keep]

        if len(scores) == 0:
            return DetectionResult(
                boxes=np.zeros((0, 4)),
                masks=np.zeros((0, *image_size[::-1])),
                scores=np.zeros((0,)),
            )

        # Boxes already in xyxy normalized format
        if isinstance(image_size, tuple) and len(image_size) == 2:
            W, H = image_size
        else:
            H, W = image_size

        # Scale boxes to image size
        boxes[:, [0, 2]] *= W
        boxes[:, [1, 3]] *= H
        boxes = np.clip(boxes, 0, max(H, W))
        boxes_xyxy = boxes

        # Resize masks to image size
        masks_resized = _resize_masks(masks, (H, W))
        masks_binary = (masks_resized > 0).astype(np.uint8)

        return DetectionResult(
            boxes=boxes_xyxy,
            masks=masks_binary,
            scores=scores,
        )


# ---------------------------------------------------------------------------
# Video Predictor
# ---------------------------------------------------------------------------


class Sam3VideoPredictor:
    """Video-level tracking predictor using SAM3 tracker."""

    def __init__(self, model, processor, score_threshold: float = 0.5):
        self.model = model
        self.processor = processor
        self.score_threshold = score_threshold

        # State
        self._frames = []
        self._frame_features = []
        self._memory_bank = {}  # object_id -> list of memories
        self._object_prompts = {}  # object_id -> {frame_idx, type, data}
        self._next_object_id = 1

    def set_video(self, frames: List[Union[Image.Image, np.ndarray]]):
        """Set video frames."""
        self._frames = frames
        self._frame_features = [None] * len(frames)
        self._memory_bank = {}
        self._object_prompts = {}

    def add_text_prompt(
        self,
        text: str,
        frame_idx: int = 0,
        object_id: Optional[int] = None,
    ) -> int:
        """Add a text prompt to detect and track an object.

        Returns:
            object_id for the tracked object
        """
        if object_id is None:
            object_id = self._next_object_id
            self._next_object_id += 1

        self._object_prompts[object_id] = {
            "type": "text",
            "text": text,
            "frame_idx": frame_idx,
        }
        return object_id

    def add_point_prompt(
        self,
        points: np.ndarray,
        labels: np.ndarray,
        frame_idx: int,
        object_id: Optional[int] = None,
    ) -> int:
        """Add point prompts for an object."""
        if object_id is None:
            object_id = self._next_object_id
            self._next_object_id += 1

        self._object_prompts[object_id] = {
            "type": "points",
            "points": points,
            "labels": labels,
            "frame_idx": frame_idx,
        }
        return object_id

    def add_box_prompt(
        self,
        box: np.ndarray,
        frame_idx: int,
        object_id: Optional[int] = None,
    ) -> int:
        """Add a box prompt for an object."""
        if object_id is None:
            object_id = self._next_object_id
            self._next_object_id += 1

        self._object_prompts[object_id] = {
            "type": "box",
            "box": box,
            "frame_idx": frame_idx,
        }
        return object_id

    def propagate(
        self,
        start_frame: int = 0,
        end_frame: Optional[int] = None,
    ) -> List[TrackingResult]:
        """Propagate tracking through video frames.

        Returns:
            List of TrackingResult, one per frame
        """
        if end_frame is None:
            end_frame = len(self._frames)

        results = []

        for frame_idx in range(start_frame, end_frame):
            frame = self._frames[frame_idx]

            # Get backbone features
            features = self._get_frame_features(frame_idx)

            # Get active objects at this frame
            frame_masks = {}
            frame_scores = {}

            for obj_id, prompt in self._object_prompts.items():
                if prompt["frame_idx"] == frame_idx:
                    # Initialize from prompt
                    mask, score = self._init_object(obj_id, features, prompt)
                    frame_masks[obj_id] = mask
                    frame_scores[obj_id] = score

                    # Create initial memory
                    self._update_memory(obj_id, features, mask)

                elif frame_idx > prompt["frame_idx"] and obj_id in self._memory_bank:
                    # Track from memory
                    mask, score = self._track_object(obj_id, features)
                    frame_masks[obj_id] = mask
                    frame_scores[obj_id] = score

                    # Update memory
                    self._update_memory(obj_id, features, mask)

            # Build frame result
            if frame_masks:
                obj_ids = sorted(frame_masks.keys())
                all_masks = np.stack([frame_masks[i] for i in obj_ids])
                all_scores = np.array([frame_scores[i] for i in obj_ids])
            else:
                obj_ids = []
                H = (
                    self._frames[0].size[1]
                    if isinstance(self._frames[0], Image.Image)
                    else self._frames[0].shape[0]
                )
                W = (
                    self._frames[0].size[0]
                    if isinstance(self._frames[0], Image.Image)
                    else self._frames[0].shape[1]
                )
                all_masks = np.zeros((0, H, W))
                all_scores = np.zeros((0,))

            results.append(
                TrackingResult(
                    frame_idx=frame_idx,
                    masks=all_masks,
                    scores=all_scores,
                    object_ids=obj_ids,
                )
            )

        return results

    def _get_frame_features(self, frame_idx: int) -> mx.array:
        """Get or compute backbone features for a frame."""
        if self._frame_features[frame_idx] is not None:
            return self._frame_features[frame_idx]

        frame = self._frames[frame_idx]
        inputs = self.processor.preprocess_image(frame)
        pixel_values = mx.array(inputs["pixel_values"])

        features = self.model.detector_model.vision_encoder.backbone(pixel_values)
        mx.eval(features)
        self._frame_features[frame_idx] = features
        return features

    def _init_object(
        self,
        obj_id: int,
        features: mx.array,
        prompt: dict,
    ) -> Tuple[np.ndarray, float]:
        """Initialize an object from a prompt."""
        if prompt["type"] == "text":
            # Use detector for text prompts (with text caching)
            frame = self._frames[prompt["frame_idx"]]
            inputs = self.processor.preprocess_image(frame)
            pixel_values = mx.array(inputs["pixel_values"])

            # Cache text features for reuse across frames
            text = prompt["text"]
            if not hasattr(self, "_text_cache"):
                self._text_cache = {}
            if text not in self._text_cache:
                text_inputs = self.processor.preprocess_text(text)
                input_ids = mx.array(text_inputs["input_ids"])
                attention_mask = mx.array(text_inputs["attention_mask"])
                tf, am = self.model.get_input_embeddings(input_ids, attention_mask)
                self._text_cache[text] = (tf, am)
            inputs_embeds, attention_mask = self._text_cache[text]

            outputs = self.model.detect(
                pixel_values,
                attention_mask=attention_mask,
                inputs_embeds=inputs_embeds,
            )
            mx.eval(outputs)

            # Take best detection
            logits = np.array(outputs["pred_logits"][0]).squeeze()
            masks = np.array(outputs["pred_masks"][0])
            scores = _sigmoid(logits)
            if "presence_logits" in outputs and outputs["presence_logits"] is not None:
                pres = _sigmoid(np.array(outputs["presence_logits"][0]))
                scores = scores * pres
            best_idx = np.argmax(scores)
            mask = (masks[best_idx] > 0).astype(np.uint8)
            score = scores[best_idx]
            return mask, float(score)

        elif prompt["type"] == "points":
            points = mx.array(prompt["points"])[None]
            labels = mx.array(prompt["labels"])[None]

            tracker_fpn = self.model.tracker_neck(features)
            track_features = tracker_fpn[2]

            result = self.model.tracker_model.track_step(
                current_features=track_features,
                prompt_points=(points, labels),
            )
            mx.eval(result)

            mask = np.array(result["pred_masks"][0, 0])
            mask = (mask > 0).astype(np.uint8)
            score = float(np.array(result["iou_scores"][0, 0]))
            return mask, score

        elif prompt["type"] == "box":
            box = mx.array(prompt["box"])[None, None]  # (1, 1, 4)

            tracker_fpn = self.model.tracker_neck(features)
            track_features = tracker_fpn[2]

            result = self.model.tracker_model.track_step(
                current_features=track_features,
                prompt_boxes=box,
            )
            mx.eval(result)

            mask = np.array(result["pred_masks"][0, 0])
            mask = (mask > 0).astype(np.uint8)
            score = float(np.array(result["iou_scores"][0, 0]))
            return mask, score

        return np.zeros((1, 1)), 0.0

    def _track_object(
        self,
        obj_id: int,
        features: mx.array,
    ) -> Tuple[np.ndarray, float]:
        """Track an object using memory bank."""
        tracker_fpn = self.model.tracker_neck(features)
        track_features = tracker_fpn[2]
        high_res = [tracker_fpn[0], tracker_fpn[1]] if len(tracker_fpn) > 1 else None

        memory_list = self._memory_bank.get(obj_id, [])

        result = self.model.tracker_model.track_step(
            current_features=track_features,
            memory_bank=memory_list,
            multimask_output=False,
            high_res_features=high_res,
        )
        mx.eval(result)

        mask = np.array(result["pred_masks"][0, 0])
        mask = (mask > 0).astype(np.uint8)
        score = float(np.array(result["iou_scores"][0, 0]))
        return mask, score

    def _update_memory(
        self,
        obj_id: int,
        features: mx.array,
        mask: np.ndarray,
    ):
        """Update memory bank for an object."""
        if obj_id not in self._memory_bank:
            self._memory_bank[obj_id] = []

        tracker_fpn = self.model.tracker_neck(features)
        track_features = tracker_fpn[2]

        mask_mx = mx.array(mask.astype(np.float32))[None, :, :, None]  # (1, H, W, 1)

        # Resize mask to 1152x1152 so downsampler (stride=16) produces 72x72
        # matching the backbone feature spatial dimensions
        target = 1152
        if mask_mx.shape[1] != target or mask_mx.shape[2] != target:
            up = nn.Upsample(
                scale_factor=(target / mask_mx.shape[1], target / mask_mx.shape[2]),
                mode="nearest",
            )
            mask_mx = up(mask_mx)

        memory = self.model.tracker_model.memory_encoder(track_features, mask_mx)
        mx.eval(memory)
        B_m, H_m, W_m, C_m = memory.shape
        memory_flat = memory.reshape(1, H_m * W_m, C_m)

        self._memory_bank[obj_id].append(memory_flat)

        # Keep only last N memories
        max_mem = self.model.config.tracker_config.num_maskmem
        if len(self._memory_bank[obj_id]) > max_mem:
            self._memory_bank[obj_id] = self._memory_bank[obj_id][-max_mem:]


# ---------------------------------------------------------------------------
# Utility Functions
# ---------------------------------------------------------------------------


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1 / (1 + np.exp(-x))


def _cxcywh_to_xyxy(boxes: np.ndarray) -> np.ndarray:
    """Convert (cx, cy, w, h) to (x1, y1, x2, y2)."""
    cx, cy, w, h = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    x1 = cx - w / 2
    y1 = cy - h / 2
    x2 = cx + w / 2
    y2 = cy + h / 2
    return np.stack([x1, y1, x2, y2], axis=1)


def _nms(
    boxes: np.ndarray, scores: np.ndarray, iou_threshold: float = 0.5
) -> np.ndarray:
    """Non-maximum suppression."""
    if len(boxes) == 0:
        return np.array([], dtype=np.int64)

    boxes_xyxy = _cxcywh_to_xyxy(boxes)
    x1, y1, x2, y2 = (
        boxes_xyxy[:, 0],
        boxes_xyxy[:, 1],
        boxes_xyxy[:, 2],
        boxes_xyxy[:, 3],
    )
    areas = (x2 - x1) * (y2 - y1)

    order = scores.argsort()[::-1]
    keep = []

    while len(order) > 0:
        i = order[0]
        keep.append(i)

        if len(order) == 1:
            break

        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0, xx2 - xx1)
        h = np.maximum(0, yy2 - yy1)
        inter = w * h

        iou = inter / (areas[i] + areas[order[1:]] - inter)
        remaining = np.where(iou <= iou_threshold)[0]
        order = order[remaining + 1]

    return np.array(keep, dtype=np.int64)


def _box_iou(box1: np.ndarray, box2: np.ndarray) -> float:
    """IoU between two xyxy boxes."""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    a1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    a2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    return inter / max(a1 + a2 - inter, 1e-6)


def _filter_by_regions(
    result: DetectionResult, regions: np.ndarray, iou_thresh: float = 0.1
) -> DetectionResult:
    """Keep only detections that overlap with any of the input box regions.

    Args:
        result: Detection result to filter
        regions: (N, 4) xyxy boxes defining regions of interest
        iou_thresh: Minimum IoU with any region to keep a detection
    """
    if len(result.scores) == 0:
        return result
    keep = []
    for i in range(len(result.scores)):
        for region in regions:
            if _box_iou(result.boxes[i], region) > iou_thresh:
                keep.append(i)
                break
    if not keep:
        return DetectionResult(
            boxes=np.zeros((0, 4)), masks=np.zeros((0, 0, 0)), scores=np.zeros((0,))
        )
    return DetectionResult(
        boxes=result.boxes[keep],
        masks=result.masks[keep],
        scores=result.scores[keep],
    )


def _resize_masks(masks: np.ndarray, target_size: Tuple[int, int]) -> np.ndarray:
    """Resize masks to target size using bilinear interpolation."""
    from PIL import Image as PILImage

    H, W = target_size
    resized = []
    for mask in masks:
        pil_mask = PILImage.fromarray(mask.astype(np.float32))
        pil_mask = pil_mask.resize((W, H), PILImage.BILINEAR)
        resized.append(np.array(pil_mask))
    return np.stack(resized) if resized else np.zeros((0, H, W))


# ---------------------------------------------------------------------------
# NMS (public API)
# ---------------------------------------------------------------------------


def nms(result: DetectionResult, iou_thresh: float = 0.5) -> DetectionResult:
    """Remove duplicate detections via Non-Maximum Suppression."""
    if len(result.scores) == 0:
        return result
    boxes, scores, masks = result.boxes, result.scores, result.masks
    order = np.argsort(-scores)
    keep = []
    for i in order:
        discard = False
        for j in keep:
            x1 = max(boxes[i][0], boxes[j][0])
            y1 = max(boxes[i][1], boxes[j][1])
            x2 = min(boxes[i][2], boxes[j][2])
            y2 = min(boxes[i][3], boxes[j][3])
            inter = max(0, x2 - x1) * max(0, y2 - y1)
            a_i = (boxes[i][2] - boxes[i][0]) * (boxes[i][3] - boxes[i][1])
            a_j = (boxes[j][2] - boxes[j][0]) * (boxes[j][3] - boxes[j][1])
            if inter / max(a_i + a_j - inter, 1e-6) > iou_thresh:
                discard = True
                break
        if not discard:
            keep.append(i)
    return DetectionResult(
        boxes=boxes[keep], masks=masks[keep], scores=scores[keep]
    )


# ---------------------------------------------------------------------------
# Video Tracking CLI
# ---------------------------------------------------------------------------

COLORS_BGR = [
    (181, 120, 31),
    (13, 128, 255),
    (43, 161, 43),
    (41, 38, 214),
    (189, 102, 148),
    (74, 87, 140),
]


def draw_frame(frame_bgr, masks, scores, boxes, prompt, H, W, show_boxes=True):
    """Draw masks, contours, boxes, and labels on a BGR frame.

    Args:
        show_boxes: If False, draw only mask overlays + contours (no boxes/labels).
    """
    import cv2

    out = frame_bgr.copy()
    for i in range(len(scores)):
        color = COLORS_BGR[i % len(COLORS_BGR)]
        mask = masks[i]
        if mask.shape[0] != H or mask.shape[1] != W:
            mask = cv2.resize(
                mask.astype(np.float32), (W, H), interpolation=cv2.INTER_NEAREST
            )
        binary = mask > 0

        for c in range(3):
            out[:, :, c] = np.where(
                binary,
                (out[:, :, c].astype(np.float32) * 0.55 + color[c] * 0.45).astype(
                    np.uint8
                ),
                out[:, :, c],
            )

        contours, _ = cv2.findContours(
            binary.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        cv2.drawContours(out, contours, -1, color, 2)

        if show_boxes:
            x1, y1, x2, y2 = boxes[i].astype(int)
            cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)

            label = f"{prompt} {scores[i]:.2f}"
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
            cv2.rectangle(
                out,
                (x1, max(y1 - th - 10, 0)),
                (x1 + tw + 6, max(y1, th + 10)),
                color,
                -1,
            )
            cv2.putText(
                out,
                label,
                (x1 + 3, max(y1 - 4, th + 6)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2,
            )
    return out


def track_video(
    video_path: str,
    prompt: str,
    output: Optional[str] = None,
    model_path: str = "facebook/sam3",
    threshold: float = 0.15,
    nms_thresh: float = 0.5,
    every: int = 2,
    boxes: Optional[str] = None,
    show_boxes: bool = True,
):
    """Track objects in a video file.

    Args:
        video_path: Input video path
        prompt: Text prompt for detection
        output: Output video path (default: <input>_tracked.mp4)
        model_path: Model path or HF repo
        threshold: Detection score threshold
        nms_thresh: NMS IoU threshold
        every: Run detection every N frames
        boxes: Box prompts as "x1,y1,x2,y2" or "x1,y1,x2,y2;..." in pixel coords
        show_boxes: Draw bounding boxes and labels on output
    """
    import cv2

    from mlx_vlm.models.sam3.processing_sam3 import Sam3Processor
    from mlx_vlm.utils import get_model_path, load_model

    if output is None:
        p = Path(video_path)
        output = str(p.parent / f"{p.stem}_tracked{p.suffix}")

    # Parse box prompts
    box_array = None
    if boxes is not None:
        box_list = []
        for b in boxes.split(";"):
            coords = [float(x) for x in b.split(",")]
            if len(coords) == 4:
                box_list.append(coords)
        if box_list:
            box_array = np.array(box_list)
            print(f"Box prompts: {box_array.tolist()}")

    print(f"Loading model: {model_path}")
    mp = get_model_path(model_path)
    model = load_model(mp)
    processor = Sam3Processor.from_pretrained(str(mp))
    predictor = Sam3Predictor(model, processor, score_threshold=threshold)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: cannot open {video_path}")
        return

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"Video: {total_frames} frames, {fps:.1f} fps, {W}x{H}")
    print(f'Prompt: "{prompt}", detect every {every} frames, threshold {threshold}')

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output, fourcc, fps, (W, H))

    latest_masks = np.array([])
    latest_scores = np.array([])
    latest_boxes = np.array([])

    for fi in range(total_frames):
        ret, frame_bgr = cap.read()
        if not ret:
            break

        if fi % every == 0:
            frame_pil = Image.fromarray(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
            result = predictor.predict(frame_pil, text_prompt=prompt)
            if len(result.scores) > 0:
                result = nms(result, nms_thresh)
                # Filter to only objects inside the input box regions
                if box_array is not None:
                    result = _filter_by_regions(result, box_array)
                latest_masks = result.masks
                latest_scores = result.scores
                latest_boxes = result.boxes
            else:
                latest_masks = np.array([])
                latest_scores = np.array([])
                latest_boxes = np.array([])

            if fi % 40 == 0:
                print(f"  Frame {fi}/{total_frames}: {len(latest_scores)} detections")

        out = draw_frame(
            frame_bgr, latest_masks, latest_scores, latest_boxes, prompt, H, W,
            show_boxes=show_boxes,
        )
        writer.write(out)

    writer.release()
    cap.release()
    print(f"\nSaved: {output}")


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def _load_predictor(model_path, threshold):
    """Shared model loading for CLI tasks."""
    from mlx_vlm.models.sam3.processing_sam3 import Sam3Processor
    from mlx_vlm.utils import get_model_path, load_model

    print(f"Loading model: {model_path}")
    mp = get_model_path(model_path)
    model = load_model(mp)
    processor = Sam3Processor.from_pretrained(str(mp))
    return Sam3Predictor(model, processor, score_threshold=threshold)


def _draw_boxes_only(frame_bgr, scores, boxes, prompt, H, W):
    """Draw boxes and labels only (no masks)."""
    import cv2

    out = frame_bgr.copy()
    for i in range(len(scores)):
        color = COLORS_BGR[i % len(COLORS_BGR)]
        x1, y1, x2, y2 = boxes[i].astype(int)
        cv2.rectangle(out, (x1, y1), (x2, y2), color, 3)
        label = f"{prompt} {scores[i]:.2f}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
        cv2.rectangle(
            out,
            (x1, max(y1 - th - 10, 0)),
            (x1 + tw + 6, max(y1, th + 10)),
            color,
            -1,
        )
        cv2.putText(
            out, label, (x1 + 3, max(y1 - 4, th + 6)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2,
        )
    return out


def run_image(
    image_path: str,
    prompt: str,
    task: str = "segment",
    output: Optional[str] = None,
    model_path: str = "facebook/sam3",
    threshold: float = 0.3,
    nms_thresh: float = 0.5,
    boxes: Optional[str] = None,
    show_boxes: bool = True,
):
    """Run detection or segmentation on an image.

    Args:
        image_path: Input image path
        prompt: Text prompt
        task: "detect" (boxes only) or "segment" (masks + boxes)
        output: Output image path
        model_path: Model path or HF repo
        threshold: Score threshold
        nms_thresh: NMS IoU threshold
        boxes: Comma-separated box coords "x1,y1,x2,y2" or "x1,y1,x2,y2;x1,y1,x2,y2"
        show_boxes: Draw bounding boxes and labels on output (default: True)
    """
    import cv2

    suffix = "_detected" if task == "detect" else "_segmented"
    if output is None:
        p = Path(image_path)
        output = str(p.parent / f"{p.stem}{suffix}{p.suffix}")

    predictor = _load_predictor(model_path, threshold)

    image = Image.open(image_path).convert("RGB")
    W, H = image.size
    print(f"Image: {W}x{H}")
    print(f'Task: {task}, prompt: "{prompt}", threshold {threshold}')

    # Parse box prompts
    box_array = None
    if boxes is not None:
        box_list = []
        for b in boxes.split(";"):
            coords = [float(x) for x in b.split(",")]
            if len(coords) == 4:
                box_list.append(coords)
        if box_list:
            box_array = np.array(box_list)
            print(f"Box prompts: {box_array.tolist()}")

    result = predictor.predict(image, text_prompt=prompt)
    if len(result.scores) > 0:
        result = nms(result, nms_thresh)
        # Filter to only objects inside the input box regions
        if box_array is not None:
            result = _filter_by_regions(result, box_array)

    print(f"Detections: {len(result.scores)}")
    for i in range(len(result.scores)):
        x1, y1, x2, y2 = result.boxes[i]
        line = f"  [{result.scores[i]:.2f}] box=({x1:.0f}, {y1:.0f}, {x2:.0f}, {y2:.0f})"
        if task == "segment":
            line += f"  mask={int(result.masks[i].sum())}px"
        print(line)

    frame_bgr = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    if task == "detect":
        out = _draw_boxes_only(frame_bgr, result.scores, result.boxes, prompt, H, W)
    else:
        out = draw_frame(
            frame_bgr, result.masks, result.scores, result.boxes, prompt, H, W,
            show_boxes=show_boxes,
        )
    cv2.imwrite(output, out)
    print(f"Saved: {output}")


def main():
    parser = argparse.ArgumentParser(
        description="SAM3: detection, segmentation, and video tracking",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""examples:
  # Object detection (boxes only)
  python -m mlx_vlm.models.sam3.generate --task detect --image photo.jpg --prompt "a cat"

  # Instance segmentation (masks only)
  python -m mlx_vlm.models.sam3.generate --task segment --image photo.jpg --prompt "a cat"

  # Instance segmentation with boxes overlaid
  python -m mlx_vlm.models.sam3.generate --task segment --image photo.jpg --prompt "a cat" --show-boxes

  # Box-guided segmentation
  python -m mlx_vlm.models.sam3.generate --task segment --image photo.jpg --prompt "a cat" --boxes "10,50,300,400"

  # Video tracking
  python -m mlx_vlm.models.sam3.generate --task track --video input.mp4 --prompt "a car"
""",
    )
    parser.add_argument(
        "--task",
        choices=["detect", "segment", "track"],
        default="segment",
        help="Task: detect (boxes), segment (masks), track (video) (default: segment)",
    )
    parser.add_argument("--image", default=None, help="Input image path (detect/segment)")
    parser.add_argument("--video", default=None, help="Input video path (track)")
    parser.add_argument("--prompt", required=True, help="Text prompt (e.g. 'a car')")
    parser.add_argument(
        "--boxes", default=None,
        help="Box prompts as 'x1,y1,x2,y2' or 'x1,y1,x2,y2;x1,y1,x2,y2' in pixel coords",
    )
    parser.add_argument(
        "--show-boxes", action="store_true",
        help="Draw bounding boxes and labels on segment/track output",
    )
    parser.add_argument("--output", default=None, help="Output path (default: auto-named)")
    parser.add_argument("--model", default="facebook/sam3", help="Model path or HF repo")
    parser.add_argument(
        "--threshold", type=float, default=None,
        help="Score threshold (default: 0.3 image, 0.15 video)",
    )
    parser.add_argument("--nms-thresh", type=float, default=0.5, help="NMS IoU threshold")
    parser.add_argument("--every", type=int, default=2, help="Detect every N frames (track only)")
    args = parser.parse_args()

    if args.task in ("detect", "segment"):
        if args.image is None:
            parser.error("--image is required for --task detect/segment")
        run_image(
            image_path=args.image,
            prompt=args.prompt,
            task=args.task,
            output=args.output,
            model_path=args.model,
            threshold=args.threshold if args.threshold is not None else 0.3,
            nms_thresh=args.nms_thresh,
            boxes=args.boxes,
            show_boxes=args.show_boxes if args.task == "segment" else True,
        )
    elif args.task == "track":
        if args.video is None:
            parser.error("--video is required for --task track")
        track_video(
            video_path=args.video,
            prompt=args.prompt,
            output=args.output,
            model_path=args.model,
            threshold=args.threshold if args.threshold is not None else 0.15,
            nms_thresh=args.nms_thresh,
            every=args.every,
            boxes=args.boxes,
            show_boxes=args.show_boxes,
        )


if __name__ == "__main__":
    main()
