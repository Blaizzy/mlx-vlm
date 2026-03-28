"""SAM3 Inference Pipeline for image and video segmentation.

Usage:
    from mlx_vlm.models.sam3.generate import Sam3Predictor, Sam3VideoPredictor

    # Image segmentation
    predictor = Sam3Predictor(model, processor)
    results = predictor.predict(image, text_prompt="a dog")

    # Video tracking
    video_predictor = Sam3VideoPredictor(model, processor)
    video_predictor.set_video(frames)
    video_predictor.add_text_prompt("a person", frame_idx=0)
    results = video_predictor.propagate()
"""

import math
from dataclasses import dataclass
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

    def predict(
        self,
        image: Union[Image.Image, np.ndarray],
        text_prompt: str,
        boxes: Optional[np.ndarray] = None,
        score_threshold: Optional[float] = None,
    ) -> DetectionResult:
        """Run detection + segmentation on a single image.

        Args:
            image: PIL Image or numpy array (H, W, 3)
            text_prompt: text description of objects to detect
            boxes: optional (N, 4) box prompts in xyxy format
            score_threshold: confidence threshold (default: self.score_threshold)
        Returns:
            DetectionResult with boxes, masks, and scores
        """
        threshold = score_threshold or self.score_threshold

        # Preprocess
        inputs = self.processor.preprocess_image(image)
        text_inputs = self.processor.preprocess_text(text_prompt)

        pixel_values = mx.array(inputs["pixel_values"])
        input_ids = mx.array(text_inputs["input_ids"])
        attention_mask = mx.array(text_inputs["attention_mask"])

        box_input = None
        if boxes is not None:
            box_input = mx.array(boxes)[None]  # (1, N, 4)

        # Run detection
        outputs = self.model.detect(
            pixel_values,
            input_ids,
            attention_mask,
            boxes=box_input,
        )
        mx.eval(outputs)

        # Post-process
        return self._postprocess(
            outputs,
            image_size=image.size if isinstance(image, Image.Image) else image.shape[:2],
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
                H = self._frames[0].size[1] if isinstance(self._frames[0], Image.Image) else self._frames[0].shape[0]
                W = self._frames[0].size[0] if isinstance(self._frames[0], Image.Image) else self._frames[0].shape[1]
                all_masks = np.zeros((0, H, W))
                all_scores = np.zeros((0,))

            results.append(TrackingResult(
                frame_idx=frame_idx,
                masks=all_masks,
                scores=all_scores,
                object_ids=obj_ids,
            ))

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
            # Use detector for text prompts
            frame = self._frames[prompt["frame_idx"]]
            inputs = self.processor.preprocess_image(frame)
            text_inputs = self.processor.preprocess_text(prompt["text"])

            pixel_values = mx.array(inputs["pixel_values"])
            input_ids = mx.array(text_inputs["input_ids"])
            attention_mask = mx.array(text_inputs["attention_mask"])

            outputs = self.model.detect(pixel_values, input_ids, attention_mask)
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


def _nms(boxes: np.ndarray, scores: np.ndarray, iou_threshold: float = 0.5) -> np.ndarray:
    """Non-maximum suppression."""
    if len(boxes) == 0:
        return np.array([], dtype=np.int64)

    boxes_xyxy = _cxcywh_to_xyxy(boxes)
    x1, y1, x2, y2 = boxes_xyxy[:, 0], boxes_xyxy[:, 1], boxes_xyxy[:, 2], boxes_xyxy[:, 3]
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
