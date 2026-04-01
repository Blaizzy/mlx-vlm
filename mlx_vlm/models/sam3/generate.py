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
    track_ids: Optional[np.ndarray] = None  # (N,) stable IDs for color consistency


@dataclass
class TrackingResult:
    """Per-frame tracking result."""

    frame_idx: int
    masks: np.ndarray  # (N_obj, H, W) binary masks
    scores: np.ndarray  # (N_obj,) confidence scores
    object_ids: List[int] = None


class SimpleTracker:
    """Assigns stable IDs to detections across frames using IoU matching."""

    def __init__(self, iou_threshold: float = 0.3, max_lost: int = 10):
        self.iou_threshold = iou_threshold
        self.max_lost = max_lost
        self._next_id = 0
        self._tracks = {}  # id -> {"box": ndarray, "lost": int}

    def update(self, result: DetectionResult) -> DetectionResult:
        if len(result.scores) == 0:
            for tid in list(self._tracks):
                self._tracks[tid]["lost"] += 1
                if self._tracks[tid]["lost"] > self.max_lost:
                    del self._tracks[tid]
            return result

        new_boxes = result.boxes
        track_ids = list(self._tracks.keys())
        assigned = np.full(len(new_boxes), -1, dtype=int)

        if track_ids:
            old_boxes = np.stack([self._tracks[t]["box"] for t in track_ids])
            ious = self._box_iou(new_boxes, old_boxes)

            for _ in range(min(len(new_boxes), len(track_ids))):
                i, j = np.unravel_index(np.argmax(ious), ious.shape)
                if ious[i, j] < self.iou_threshold:
                    break
                assigned[i] = track_ids[j]
                ious[i, :] = -1
                ious[:, j] = -1

        matched_track_ids = set(assigned[assigned >= 0])
        for tid in track_ids:
            if tid in matched_track_ids:
                self._tracks[tid]["lost"] = 0
            else:
                self._tracks[tid]["lost"] += 1
                if self._tracks[tid]["lost"] > self.max_lost:
                    del self._tracks[tid]

        ids = []
        for i in range(len(new_boxes)):
            if assigned[i] >= 0:
                tid = int(assigned[i])
            else:
                tid = self._next_id
                self._next_id += 1
            self._tracks[tid] = {"box": new_boxes[i], "lost": 0}
            ids.append(tid)

        result.track_ids = np.array(ids)
        return result

    @staticmethod
    def _box_iou(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        x1 = np.maximum(a[:, None, 0], b[None, :, 0])
        y1 = np.maximum(a[:, None, 1], b[None, :, 1])
        x2 = np.minimum(a[:, None, 2], b[None, :, 2])
        y2 = np.minimum(a[:, None, 3], b[None, :, 3])
        inter = np.maximum(0, x2 - x1) * np.maximum(0, y2 - y1)
        a_area = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
        b_area = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
        union = a_area[:, None] + b_area[None, :] - inter
        return inter / (union + 1e-6)


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


def predict_multi(
    predictor: Sam3Predictor,
    image,
    prompts: List[str],
    boxes: Optional[np.ndarray] = None,
    score_threshold: Optional[float] = None,
) -> DetectionResult:
    """Run vision backbone ONCE, then text+DETR per prompt. Merge with labels.

    For N prompts, cost = 1x ViT + Nx (text + DETR) instead of Nx full pipeline.
    """
    if len(prompts) == 1:
        result = predictor.predict(
            image,
            text_prompt=prompts[0],
            boxes=boxes,
            score_threshold=score_threshold,
        )
        if len(result.scores) > 0:
            result = nms(result)
            result.labels = [prompts[0]] * len(result.scores)
        else:
            result.labels = []
        return result

    # Run vision backbone once
    inputs = predictor.processor.preprocess_image(image)
    pixel_values = mx.array(inputs["pixel_values"])

    det = predictor.model.detector_model
    fpn_features = det.vision_encoder(pixel_values)
    fpn_pos = [det._pos_enc(feat) for feat in fpn_features]
    fpn_trimmed = fpn_features[:-1]
    fpn_pos_trimmed = fpn_pos[:-1]

    encoder_feat = fpn_trimmed[-1]
    B, H_f, W_f, D = encoder_feat.shape
    src = encoder_feat.reshape(B, H_f * W_f, D)
    pos_flat = fpn_pos_trimmed[-1].reshape(B, H_f * W_f, D)
    mx.eval(src, pos_flat)

    threshold = score_threshold or predictor.score_threshold
    image_size = image.size if hasattr(image, "size") else image.shape[:2]

    all_boxes, all_masks, all_scores, all_labels = [], [], [], []

    # Run text + DETR per prompt (cheap: ~15ms each vs 80ms ViT)
    for prompt in prompts:
        inputs_embeds, attention_mask = predictor._get_input_embeddings(prompt)

        encoded = det.detr_encoder(src, pos_flat, inputs_embeds, attention_mask)
        mx.eval(encoded)

        hs, ref_boxes, presence_logits = det.detr_decoder(
            vision_features=encoded,
            inputs_embeds=inputs_embeds,
            vision_pos_encoding=pos_flat,
            text_mask=attention_mask,
            spatial_shape=(H_f, W_f),
        )

        pred_boxes_cxcywh = ref_boxes[-1]
        cx = pred_boxes_cxcywh[..., 0]
        cy = pred_boxes_cxcywh[..., 1]
        w = pred_boxes_cxcywh[..., 2]
        h = pred_boxes_cxcywh[..., 3]
        pred_boxes_xyxy = mx.stack(
            [cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], axis=-1
        )

        all_logits = det.dot_product_scoring(hs, inputs_embeds, attention_mask)
        pred_logits = all_logits[-1].squeeze(-1)
        presence = presence_logits[-1]

        last_hs = hs[-1]
        seg_out = det.mask_decoder(
            last_hs,
            list(fpn_trimmed),
            encoder_hidden_states=encoded,
            prompt_features=inputs_embeds,
            prompt_mask=attention_mask,
        )
        mx.eval(pred_logits, pred_boxes_xyxy, seg_out, presence)

        # Postprocess — ensure batch dim is present
        outputs = {
            "pred_logits": pred_logits if pred_logits.ndim == 2 else pred_logits[None],
            "pred_boxes": (
                pred_boxes_xyxy if pred_boxes_xyxy.ndim == 3 else pred_boxes_xyxy[None]
            ),
            "pred_masks": seg_out["pred_masks"],
            "presence_logits": presence if presence.ndim == 2 else presence[None],
        }
        result = predictor._postprocess(outputs, image_size, threshold)
        if len(result.scores) > 0:
            result = nms(result)
            all_boxes.append(result.boxes)
            all_masks.append(result.masks)
            all_scores.append(result.scores)
            all_labels.extend([prompt] * len(result.scores))

    if not all_scores:
        return DetectionResult(
            boxes=np.zeros((0, 4)),
            masks=np.zeros((0, 1, 1), dtype=np.uint8),
            scores=np.zeros((0,)),
            labels=[],
        )

    return DetectionResult(
        boxes=np.concatenate(all_boxes),
        masks=np.concatenate(all_masks),
        scores=np.concatenate(all_scores),
        labels=all_labels,
    )


def _get_backbone_features(model, pixel_values: mx.array) -> mx.array:
    """Run ViT backbone only (no FPN neck)."""
    features = model.detector_model.vision_encoder.backbone(pixel_values)
    mx.eval(features)
    return features


def _get_det_features(model, backbone_features: mx.array):
    """Run FPN neck + flatten for DETR. Returns (src, pos_flat, fpn_trimmed, spatial)."""
    det = model.detector_model
    fpn_features = det.vision_encoder.neck(backbone_features)
    fpn_pos = [det._pos_enc(feat) for feat in fpn_features]
    fpn_trimmed = fpn_features[:-1]

    encoder_feat = fpn_trimmed[-1]
    B, H_f, W_f, D = encoder_feat.shape
    src = encoder_feat.reshape(B, H_f * W_f, D)
    pos_flat = fpn_pos[:-1][-1].reshape(B, H_f * W_f, D)
    mx.eval(src, pos_flat)
    return src, pos_flat, fpn_trimmed, (H_f, W_f)


def _detect_with_backbone(
    predictor: Sam3Predictor,
    backbone_features: mx.array,
    prompts: List[str],
    image_size,
    threshold: float,
    encoder_cache: Optional[Dict] = None,
) -> DetectionResult:
    """Run detection on pre-computed backbone features.

    With encoder_cache: skips DETR encoder when backbone + text are unchanged.
    All scoring stays in MLX until final conversion.
    """
    det = predictor.model.detector_model
    src, pos_flat, fpn_trimmed, spatial = _get_det_features(
        predictor.model, backbone_features
    )
    H_f, W_f = spatial

    W, H = (
        image_size if isinstance(image_size, tuple) else (image_size[1], image_size[0])
    )
    all_boxes, all_masks, all_scores, all_labels = [], [], [], []
    for prompt in prompts:
        inputs_embeds, attention_mask = predictor._get_input_embeddings(prompt)

        # DETR encoder — use cache if available (caller controls invalidation)
        cached = encoder_cache.get(prompt) if encoder_cache is not None else None
        if cached is not None:
            encoded = cached["encoded"]
        else:
            encoded = det.detr_encoder(src, pos_flat, inputs_embeds, attention_mask)
            mx.eval(encoded)
            if encoder_cache is not None:
                encoder_cache[prompt] = {"encoded": encoded}

        hs, ref_boxes, presence_logits = det.detr_decoder(
            vision_features=encoded,
            inputs_embeds=inputs_embeds,
            vision_pos_encoding=pos_flat,
            text_mask=attention_mask,
            spatial_shape=(H_f, W_f),
        )

        # Box conversion + scoring in MLX
        pred_boxes_cxcywh = ref_boxes[-1]
        cx, cy, w, h = (
            pred_boxes_cxcywh[..., 0],
            pred_boxes_cxcywh[..., 1],
            pred_boxes_cxcywh[..., 2],
            pred_boxes_cxcywh[..., 3],
        )
        pred_boxes_xyxy = mx.stack(
            [cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], axis=-1
        )

        all_logits = det.dot_product_scoring(hs, inputs_embeds, attention_mask)
        pred_logits = all_logits[-1].squeeze(-1)
        presence = presence_logits[-1]

        last_hs = hs[-1]
        seg_out = det.mask_decoder(
            last_hs,
            list(fpn_trimmed),
            encoder_hidden_states=encoded,
            prompt_features=inputs_embeds,
            prompt_mask=attention_mask,
        )

        # Single eval, scoring in MLX
        scores = mx.sigmoid(pred_logits[0].squeeze())
        if presence is not None:
            scores = scores * mx.sigmoid(presence[0])
        boxes = pred_boxes_xyxy[0] * mx.array([W, H, W, H], dtype=pred_boxes_xyxy.dtype)
        boxes = mx.clip(boxes, 0, max(H, W))

        mx.eval(scores, boxes, seg_out)

        # Single numpy conversion
        scores_np = np.array(scores)
        keep = scores_np > threshold
        if not keep.any():
            continue

        boxes_np = np.array(boxes)[keep]
        masks_np = np.array(seg_out["pred_masks"][0])[keep]
        masks_resized = _resize_masks(masks_np, (H, W))
        masks_binary = (masks_resized > 0).astype(np.uint8)

        result = DetectionResult(
            boxes=boxes_np, masks=masks_binary, scores=scores_np[keep]
        )
        if len(result.scores) > 0:
            result = nms(result)
            all_boxes.append(result.boxes)
            all_masks.append(result.masks)
            all_scores.append(result.scores)
            all_labels.extend([prompt] * len(result.scores))

    if not all_scores:
        return DetectionResult(
            boxes=np.zeros((0, 4)),
            masks=np.zeros((0, H, W), dtype=np.uint8),
            scores=np.zeros((0,)),
            labels=[],
        )

    return DetectionResult(
        boxes=np.concatenate(all_boxes),
        masks=np.concatenate(all_masks),
        scores=np.concatenate(all_scores),
        labels=all_labels,
    )


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
            boxes=np.zeros((0, 4)),
            masks=np.zeros((0, 0, 0)),
            scores=np.zeros((0,)),
            labels=[],
        )
    labels = [result.labels[i] for i in keep] if result.labels else None
    return DetectionResult(
        boxes=result.boxes[keep],
        masks=result.masks[keep],
        scores=result.scores[keep],
        labels=labels,
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
    labels = [result.labels[i] for i in keep] if result.labels else None
    return DetectionResult(
        boxes=boxes[keep],
        masks=masks[keep],
        scores=scores[keep],
        labels=labels,
    )


COLORS_BGR = [
    (181, 120, 31),
    (13, 128, 255),
    (43, 161, 43),
    (41, 38, 214),
    (189, 102, 148),
    (74, 87, 140),
]


def draw_frame(
    frame_bgr, masks, scores, boxes, prompt, H, W, show_boxes=True, labels=None
):
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
                mask.astype(np.float32), (W, H), interpolation=cv2.INTER_LINEAR
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

        # Label text
        lbl = labels[i] if labels and i < len(labels) else prompt
        label = f"{lbl} {scores[i]:.2f}"

        if show_boxes:
            x1, y1, x2, y2 = boxes[i].astype(int)
            cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
            lx, ly = x1, max(y1 - 8, 12)
        else:
            # Place label at top of mask bounding box
            ys, xs = np.where(binary)
            if len(ys) > 0:
                lx, ly = int(xs.min()), max(int(ys.min()) - 8, 12)
            else:
                lx, ly = 10, 30

        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.rectangle(
            out,
            (lx, max(ly - th - 6, 0)),
            (lx + tw + 6, ly + 4),
            color,
            -1,
        )
        cv2.putText(
            out,
            label,
            (lx + 3, ly),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2,
        )
    return out


def track_video(
    video_path: str,
    prompts: List[str],
    output: Optional[str] = None,
    model_path: str = "facebook/sam3",
    threshold: float = 0.15,
    nms_thresh: float = 0.5,
    every: int = 2,
    boxes: Optional[str] = None,
    show_boxes: bool = True,
    resolution: int = 1008,
    annotator_name: Optional[str] = None,
    backbone_every: int = 1,
    opacity: float = 0.6,
    contour_thickness: int = 2,
):
    """Track objects in a video file."""
    import cv2

    from mlx_vlm.generate import wired_limit
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
    if resolution != 1008:
        processor.image_size = resolution
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
    prompt_str = " + ".join(prompts)
    print(
        f"Prompts: {prompts}, detect every {every} frames, threshold {threshold}, resolution {resolution}"
    )

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output, fourcc, fps, (W, H))

    latest_result = DetectionResult(
        boxes=np.array([]),
        masks=np.array([]),
        scores=np.array([]),
        labels=[],
    )

    id_tracker = SimpleTracker()
    backbone_cache = None
    encoder_cache = {}
    detect_count = 0
    import time as _time

    t_start = _time.perf_counter()

    with wired_limit(model):
        for fi in range(total_frames):
            ret, frame_bgr = cap.read()
            if not ret:
                break

            if fi % every == 0:
                frame_pil = Image.fromarray(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
                inputs = predictor.processor.preprocess_image(frame_pil)
                pixel_values = mx.array(inputs["pixel_values"])

                if detect_count % backbone_every == 0 or backbone_cache is None:
                    backbone_cache = _get_backbone_features(model, pixel_values)
                    encoder_cache.clear()

                result = _detect_with_backbone(
                    predictor,
                    backbone_cache,
                    prompts,
                    frame_pil.size,
                    threshold,
                    encoder_cache=encoder_cache,
                )
                if box_array is not None and len(result.scores) > 0:
                    result = _filter_by_regions(result, box_array)
                latest_result = id_tracker.update(result)
                detect_count += 1

                if fi % 40 == 0:
                    elapsed = _time.perf_counter() - t_start
                    fps_actual = (fi + 1) / elapsed if elapsed > 0 else 0
                    print(
                        f"  Frame {fi}/{total_frames}: {len(latest_result.scores)} det, "
                        f"{fps_actual:.1f} fps"
                    )

            if annotator_name and len(latest_result.scores) > 0:
                ann = build_annotator(
                    annotator_name, opacity=opacity, contour_thickness=contour_thickness
                )
                out = ann.annotate(frame_bgr, latest_result)
            else:
                out = draw_frame(
                    frame_bgr,
                    latest_result.masks,
                    latest_result.scores,
                    latest_result.boxes,
                    prompt_str,
                    H,
                    W,
                    show_boxes=show_boxes,
                    labels=latest_result.labels,
                )
            writer.write(out)

    writer.release()
    cap.release()
    print(f"\nSaved: {output}")


def track_video_realtime(
    video_path: str,
    prompts: List[str],
    model_path: str = "facebook/sam3",
    threshold: float = 0.15,
    nms_thresh: float = 0.5,
    boxes: Optional[str] = None,
    show_boxes: bool = True,
    resolution: int = 1008,
    bg_image: Optional[str] = None,
    recompute_backbone_every: int = 5,
    annotator_name: Optional[str] = None,
    opacity: float = 0.6,
    contour_thickness: int = 2,
):
    """Track objects in a video with a real-time preview window.

    Optimized with backbone caching, DETR encoder caching, and fast overlay.
    Press 'q' to quit.
    """
    import threading
    import time

    import cv2

    from mlx_vlm.generate import wired_limit
    from mlx_vlm.models.sam3.processing_sam3 import Sam3Processor
    from mlx_vlm.utils import get_model_path, load_model

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

    print(f"Loading model: {model_path}")
    mp = get_model_path(model_path)
    model = load_model(mp)
    processor = Sam3Processor.from_pretrained(str(mp))
    if resolution != 1008:
        processor.image_size = resolution
    predictor = Sam3Predictor(model, processor, score_threshold=threshold)

    source = int(video_path) if video_path.isdigit() else video_path
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print(f"Error: cannot open {video_path}")
        return

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"Video: {fps:.1f} fps, {W}x{H}")
    prompt_str = " + ".join(prompts)
    print(
        f"Prompts: {prompts}, threshold {threshold}, resolution {resolution}x{resolution}"
    )

    # Load background image for bg swap mode
    bg_frame = None
    if bg_image is not None:
        bg_pil = Image.open(bg_image).convert("RGB").resize((W, H))
        bg_frame = cv2.cvtColor(np.array(bg_pil), cv2.COLOR_RGB2BGR)
        print(f"Background swap: {bg_image} ({W}x{H})")

    print("Press 'q' to quit")

    import queue

    frame_buffer = queue.Queue(maxsize=10)

    # Shared state
    lock = threading.Lock()
    empty_result = DetectionResult(
        boxes=np.zeros((0, 4)),
        masks=np.zeros((0, H, W), dtype=np.uint8),
        scores=np.zeros((0,)),
        labels=[],
    )
    latest = {
        "result": empty_result,
        "n_obj": 0,
        "fps": 0.0,
    }
    latest_frame = {"bgr": None}
    running = {"active": True}

    # --- Thread 1: Read frames into ring buffer ---
    is_camera = str(video_path).isdigit()
    frame_interval = 1.0 / fps if not is_camera else 0

    def reader_loop():
        next_frame_time = time.perf_counter()
        while running["active"]:
            # Pace to native FPS for video files, read ASAP for camera
            if not is_camera:
                now = time.perf_counter()
                if now < next_frame_time:
                    time.sleep(max(0, next_frame_time - now - 0.001))
                    continue

            ret, frame = cap.read()
            if not ret:
                if is_camera:
                    continue  # camera drop, retry
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                next_frame_time = time.perf_counter()
                continue

            with lock:
                latest_frame["bgr"] = frame

            if frame_buffer.full():
                try:
                    frame_buffer.get_nowait()
                except queue.Empty:
                    pass
            frame_buffer.put(frame)

            if not is_camera:
                next_frame_time += frame_interval

    # --- Thread 2: Optimized inference (backbone + encoder caching, fast overlay) ---
    def inference_loop():
        backbone_cache = {"features": None, "frame_idx": -1}
        encoder_cache = {}
        inference_count = 0
        id_tracker = SimpleTracker()

        while running["active"]:
            with lock:
                latest_bgr = latest_frame["bgr"]
                latest_frame["bgr"] = None

            if latest_bgr is None:
                time.sleep(0.005)
                continue

            t0 = time.perf_counter()
            frame_pil = Image.fromarray(cv2.cvtColor(latest_bgr, cv2.COLOR_BGR2RGB))
            image_size = frame_pil.size

            inputs = predictor.processor.preprocess_image(frame_pil)
            pixel_values = mx.array(inputs["pixel_values"])

            # Backbone (cached or fresh)
            need_backbone = (
                inference_count % recompute_backbone_every == 0
                or backbone_cache["features"] is None
            )
            if need_backbone:
                backbone_features = _get_backbone_features(model, pixel_values)
                backbone_cache["features"] = backbone_features
                encoder_cache.clear()
            else:
                backbone_features = backbone_cache["features"]

            # Detection with cached backbone + encoder
            result = _detect_with_backbone(
                predictor,
                backbone_features,
                prompts,
                image_size,
                threshold,
                encoder_cache=encoder_cache,
            )
            if len(result.scores) > 0:
                result = nms(result, nms_thresh)
                if box_array is not None:
                    result = _filter_by_regions(result, box_array)

            dt = time.perf_counter() - t0

            result = id_tracker.update(result)
            with lock:
                latest["result"] = result
                latest["n_obj"] = len(result.scores)
                latest["fps"] = 1.0 / max(dt, 1e-6)

            inference_count += 1

    # Start threads with wired limit for GPU memory management
    with wired_limit(model):
        reader_thread = threading.Thread(target=reader_loop, daemon=True)
        inference_thread = threading.Thread(target=inference_loop, daemon=True)
        reader_thread.start()
        inference_thread.start()

        display_fps_counter = 0
        display_fps_t0 = time.perf_counter()
        display_fps_val = 0.0

        if annotator_name:
            display_ann = build_annotator(
                annotator_name, opacity=opacity, contour_thickness=contour_thickness
            )
        else:
            from mlx_vlm.models.sam3.annotators import (
                BoxAnnotator,
                LabelAnnotator,
                MaskAnnotator,
            )

            display_ann = MaskAnnotator() + BoxAnnotator() + LabelAnnotator()

        while True:
            try:
                frame_bgr = frame_buffer.get(timeout=0.05)
            except queue.Empty:
                continue

            with lock:
                result = latest["result"]
                det_fps = latest["fps"]
                n_obj = latest["n_obj"]

            if len(result.scores) > 0:
                out = display_ann.annotate(frame_bgr.copy(), result)
            else:
                out = frame_bgr

            # Measure display FPS
            display_fps_counter += 1
            now = time.perf_counter()
            if now - display_fps_t0 >= 0.5:
                display_fps_val = display_fps_counter / (now - display_fps_t0)
                display_fps_counter = 0
                display_fps_t0 = now

            cv2.putText(
                out,
                f"Detect: {det_fps:.1f} FPS | Display: {display_fps_val:.0f} FPS | {n_obj} obj",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0),
                2,
            )

            cv2.imshow(f"SAM3 Tracking - {prompt_str}", out)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break

        running["active"] = False
        inference_thread.join(timeout=2)

    cap.release()
    cv2.destroyAllWindows()
    print("Done")


def _load_predictor(model_path, threshold, resolution=1008):
    """Shared model loading for CLI tasks."""
    from mlx_vlm.models.sam3.processing_sam3 import Sam3Processor
    from mlx_vlm.utils import get_model_path, load_model

    print(f"Loading model: {model_path}")
    mp = get_model_path(model_path)
    model = load_model(mp)
    processor = Sam3Processor.from_pretrained(str(mp))
    if resolution != 1008:
        processor.image_size = resolution
        print(f"Resolution: {resolution}x{resolution}")
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
            out,
            label,
            (x1 + 3, max(y1 - 4, th + 6)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
        )
    return out


ANNOTATOR_PRESETS = {
    "box": "BoxAnnotator+LabelAnnotator",
    "corner": "BoxCornerAnnotator+LabelAnnotator",
    "round": "RoundBoxAnnotator+LabelAnnotator",
    "mask": "MaskAnnotator+LabelAnnotator",
    "mask+box": "MaskAnnotator+BoxAnnotator+LabelAnnotator",
    "halo": "HaloAnnotator+LabelAnnotator",
    "halo+box": "HaloAnnotator+BoxAnnotator+LabelAnnotator",
    "color": "ColorAnnotator+LabelAnnotator",
    "ellipse": "EllipseAnnotator+LabelAnnotator",
    "triangle": "TriangleAnnotator+LabelAnnotator",
    "dot": "DotAnnotator+LabelAnnotator",
    "circle": "CircleAnnotator+LabelAnnotator",
    "bar": "PercentageBarAnnotator+BoxAnnotator",
    "blur": "BlurAnnotator",
    "pixelate": "PixelateAnnotator",
    "background": "BackgroundOverlayAnnotator+LabelAnnotator",
}


def build_annotator(name: str, opacity: float = 0.6, contour_thickness: int = 2):
    """Build an annotator from a preset name or explicit chain.

    Append ``+bg`` to blur/pixelate to target the background instead of objects.

    Examples:
        build_annotator("mask+box")
        build_annotator("blur+bg")          # blur background
        build_annotator("pixelate+bg")      # pixelate background
        build_annotator("halo", opacity=0.8, contour_thickness=3)
    """
    from mlx_vlm.models.sam3 import annotators as ann_module

    # Handle +bg modifier: strip it, resolve preset, re-append
    bg_modifier = False
    base_name = name
    if "+bg" in name:
        base_name = name.replace("+bg", "").strip("+")
        bg_modifier = True

    spec = ANNOTATOR_PRESETS.get(base_name, base_name)
    if bg_modifier:
        spec += "+bg"

    parts = [p.strip() for p in spec.split("+")]
    chain = None
    last = None
    for part in parts:
        if part == "bg":
            # Apply background=True to the previous annotator
            if last is not None and hasattr(last, "background"):
                last.background = True
            continue
        cls = getattr(ann_module, part, None)
        if cls is None:
            raise ValueError(
                f"Unknown annotator: {part}. "
                f"Presets: {list(ANNOTATOR_PRESETS.keys())}"
            )
        kwargs = {}
        if hasattr(cls, "opacity"):
            kwargs["opacity"] = opacity
        if hasattr(cls, "contour_thickness"):
            kwargs["contour_thickness"] = contour_thickness
        a = cls(**kwargs)
        last = a
        chain = a if chain is None else chain + a
    return chain


def _parse_boxes(boxes_str: Optional[str]) -> Optional[np.ndarray]:
    """Parse box string 'x1,y1,x2,y2;...' into numpy array."""
    if boxes_str is None:
        return None
    box_list = []
    for b in boxes_str.split(";"):
        coords = [float(x) for x in b.split(",")]
        if len(coords) == 4:
            box_list.append(coords)
    return np.array(box_list) if box_list else None


def run_image(
    image_path: str,
    prompts: List[str],
    task: str = "segment",
    output: Optional[str] = None,
    model_path: str = "facebook/sam3",
    threshold: float = 0.3,
    nms_thresh: float = 0.5,
    boxes: Optional[str] = None,
    show_boxes: bool = True,
    resolution: int = 1008,
    annotator_name: Optional[str] = None,
    opacity: float = 0.6,
    contour_thickness: int = 2,
):
    """Run detection or segmentation on an image."""
    import cv2

    from mlx_vlm.generate import wired_limit

    suffix = "_detected" if task == "detect" else "_segmented"
    if output is None:
        p = Path(image_path)
        output = str(p.parent / f"{p.stem}{suffix}{p.suffix}")

    predictor = _load_predictor(model_path, threshold, resolution)
    box_array = _parse_boxes(boxes)

    image = Image.open(image_path).convert("RGB")
    W, H = image.size
    print(f"Image: {W}x{H}")
    print(f"Task: {task}, prompts: {prompts}, threshold {threshold}")

    with wired_limit(predictor.model):
        result = predict_multi(predictor, image, prompts, boxes=box_array)
    if box_array is not None and len(result.scores) > 0:
        result = _filter_by_regions(result, box_array)

    print(f"Detections: {len(result.scores)}")
    for i in range(len(result.scores)):
        x1, y1, x2, y2 = result.boxes[i]
        lbl = result.labels[i] if result.labels else prompts[0]
        line = f"  [{result.scores[i]:.2f}] {lbl} box=({x1:.0f}, {y1:.0f}, {x2:.0f}, {y2:.0f})"
        if task == "segment":
            line += f"  mask={int(result.masks[i].sum())}px"
        print(line)

    frame_bgr = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    prompt_str = " + ".join(prompts)
    if annotator_name:
        ann = build_annotator(
            annotator_name, opacity=opacity, contour_thickness=contour_thickness
        )
        out = ann.annotate(frame_bgr, result)
    elif task == "detect":
        out = _draw_boxes_only(frame_bgr, result.scores, result.boxes, prompt_str, H, W)
    else:
        out = draw_frame(
            frame_bgr,
            result.masks,
            result.scores,
            result.boxes,
            prompt_str,
            H,
            W,
            show_boxes=show_boxes,
            labels=result.labels,
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

  # Real-time webcam (live preview window, press 'q' to quit)
  python -m mlx_vlm.models.sam3.generate --task realtime --prompt "a person"

  # Real-time webcam with background swap
  python -m mlx_vlm.models.sam3.generate --task realtime --prompt "a person" --bg-image beach.jpg
""",
    )
    parser.add_argument(
        "--task",
        choices=["detect", "segment", "track", "realtime"],
        default="segment",
        help="Task: detect, segment, track, realtime (default: segment)",
    )
    parser.add_argument(
        "--image", default=None, help="Input image path (detect/segment)"
    )
    parser.add_argument(
        "--video",
        default=None,
        help="Input video path (track only)",
    )
    parser.add_argument(
        "--prompt",
        required=True,
        nargs="+",
        help="Text prompt(s). Pass multiple to track different objects: --prompt 'a car' 'a person'",
    )
    parser.add_argument(
        "--boxes",
        default=None,
        help="Box prompts as 'x1,y1,x2,y2' or 'x1,y1,x2,y2;x1,y1,x2,y2' in pixel coords",
    )
    parser.add_argument(
        "--show-boxes",
        action="store_true",
        help="Draw bounding boxes and labels on segment/track output",
    )
    parser.add_argument(
        "--output", default=None, help="Output path (default: auto-named)"
    )
    parser.add_argument(
        "--model", default="facebook/sam3", help="Model path or HF repo"
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Score threshold (default: 0.3 image, 0.15 video)",
    )
    parser.add_argument(
        "--nms-thresh", type=float, default=0.5, help="NMS IoU threshold"
    )
    parser.add_argument(
        "--every", type=int, default=1, help="Detect every N frames (track only)"
    )
    parser.add_argument(
        "--resolution",
        type=int,
        default=1008,
        help="Input resolution (default: 1008). Lower = faster: 336 (~8 FPS), 504 (~3 FPS)",
    )
    parser.add_argument(
        "--bg-image",
        default=None,
        help="Background image for realtime bg swap (replaces area outside detected objects)",
    )
    parser.add_argument(
        "--annotator",
        default=None,
        help=(
            "Annotation style. Presets: "
            + ", ".join(ANNOTATOR_PRESETS.keys())
            + ". Or chain: BoxAnnotator+LabelAnnotator"
        ),
    )
    parser.add_argument(
        "--backbone-every",
        type=int,
        default=1,
        help="Re-run ViT backbone every N detection frames (default: 1, increase for speed)",
    )
    parser.add_argument(
        "--opacity",
        type=float,
        default=0.5,
        help="Mask/overlay opacity (default: 0.5)",
    )
    parser.add_argument(
        "--contour-thickness",
        type=int,
        default=1,
        help="Mask contour thickness, 0 to disable (default: 1)",
    )
    args = parser.parse_args()

    prompts = args.prompt  # list of 1+ prompts

    if args.task in ("detect", "segment"):
        if args.image is None:
            parser.error("--image is required for --task detect/segment")
        run_image(
            image_path=args.image,
            prompts=prompts,
            task=args.task,
            output=args.output,
            model_path=args.model,
            threshold=args.threshold if args.threshold is not None else 0.3,
            nms_thresh=args.nms_thresh,
            boxes=args.boxes,
            show_boxes=args.show_boxes if args.task == "segment" else True,
            resolution=args.resolution,
            annotator_name=args.annotator,
            opacity=args.opacity,
            contour_thickness=args.contour_thickness,
        )
    elif args.task == "track":
        if args.video is None:
            parser.error("--video is required for --task track")
        track_video(
            video_path=args.video,
            prompts=prompts,
            output=args.output,
            model_path=args.model,
            threshold=args.threshold if args.threshold is not None else 0.15,
            nms_thresh=args.nms_thresh,
            every=args.every,
            boxes=args.boxes,
            show_boxes=args.show_boxes,
            resolution=args.resolution,
            annotator_name=args.annotator,
            backbone_every=args.backbone_every,
            opacity=args.opacity,
            contour_thickness=args.contour_thickness,
        )
    elif args.task == "realtime":
        track_video_realtime(
            video_path="0",
            prompts=prompts,
            model_path=args.model,
            threshold=args.threshold if args.threshold is not None else 0.5,
            nms_thresh=args.nms_thresh if args.nms_thresh != 0.5 else 0.3,
            boxes=args.boxes,
            show_boxes=args.show_boxes,
            resolution=args.resolution,
            bg_image=args.bg_image,
            annotator_name=args.annotator,
            opacity=args.opacity,
            contour_thickness=args.contour_thickness,
        )


if __name__ == "__main__":
    main()
