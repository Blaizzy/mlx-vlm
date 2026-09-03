"""Sapiens2 inference helper.

Unlike generative VLMs, Sapiens2 produces dense per-pixel or per-keypoint
predictions.  We expose `Sapiens2Predictor`, which supports three call modes:

1. **Whole-image** — the default.  The input image is resized directly to
   1024 × 768 and fed to the model.  Correct for single-person images where
   the subject fills the frame.

2. **Top-down with a detector** — pass any `detector` callable on construction.
   The detector must be a thing that, given a PIL image, returns either a list
   of (x1, y1, x2, y2) boxes, a list of (x1, y1, x2, y2, score) tuples, or an
   object with a `.boxes` numpy array (so `RTMDetPredictor` /
   `RFDETRPredictor.predict()` results work directly).  Each detected person
   is cropped with 4:3 aspect-ratio padding, run through pose, and the
   keypoints are stitched back to full-image coordinates.

3. **Top-down with explicit boxes** — pass `person_boxes=[...]` to `predict`
   directly, bypassing the detector.
"""

from dataclasses import dataclass, field
from typing import Any, Callable, List, Optional, Tuple, Union

import mlx.core as mx
import numpy as np
from PIL import Image

from .processing_sapiens2 import Sapiens2Processor


Box = Tuple[float, float, float, float]  # x1, y1, x2, y2


@dataclass
class Sapiens2Result:
    task: str  # "pose" | "seg" | "normal" | "pointmap"
    original_size: Tuple[int, int]  # (orig_h, orig_w)

    # pose
    keypoints: Optional[np.ndarray] = None      # (N_kpt, 2) xy in original coords
    keypoint_scores: Optional[np.ndarray] = None  # (N_kpt,)

    # seg
    mask: Optional[np.ndarray] = None
    seg_logits: Optional[np.ndarray] = None

    # normal
    normal: Optional[np.ndarray] = None

    # pointmap
    pointmap: Optional[np.ndarray] = None
    scale: Optional[float] = None


@dataclass
class PersonPose:
    """Per-person pose prediction from a top-down pipeline."""
    box: Box                              # detected / supplied person bbox in original coords
    keypoints: np.ndarray                 # (N_kpt, 2) xy in ORIGINAL image coords
    keypoint_scores: np.ndarray           # (N_kpt,)
    detector_score: Optional[float] = None


@dataclass
class Sapiens2PoseMultiResult:
    """Wraps a list of per-person pose results for multi-person images."""
    task: str = "pose"
    original_size: Tuple[int, int] = (0, 0)
    persons: List[PersonPose] = field(default_factory=list)


def _bilinear_resize_np(x: np.ndarray, size: Tuple[int, int]) -> np.ndarray:
    """(C, H, W) → (C, out_h, out_w) via PIL bilinear."""
    C, _, _ = x.shape
    out_h, out_w = size
    out = np.empty((C, out_h, out_w), dtype=x.dtype)
    for c in range(C):
        im = Image.fromarray(x[c])
        im = im.resize((out_w, out_h), Image.BILINEAR)
        out[c] = np.asarray(im)
    return out


def _normalize_detector_output(
    detector_output: Any, filter_label: Optional[int] = None
) -> List[Tuple[Box, Optional[float]]]:
    """Coerce any reasonable detector return type into a list of (box, score).

    Accepted forms:
      - list/tuple of (x1, y1, x2, y2) or (x1, y1, x2, y2, score)
      - numpy array of shape (N, 4) or (N, 5)
      - object with `.boxes` ndarray (N, 4) and optional `.scores` / `.labels`
        (covers RTMDet's `DetectionResult` and RFDETR's `DetectionResult`).
    """
    # Object with .boxes attribute
    if hasattr(detector_output, "boxes"):
        boxes = np.asarray(detector_output.boxes)
        scores = getattr(detector_output, "scores", None)
        labels = getattr(detector_output, "labels", None)
        out = []
        for i in range(boxes.shape[0]):
            if filter_label is not None and labels is not None:
                if int(labels[i]) != filter_label:
                    continue
            s = float(scores[i]) if scores is not None else None
            out.append((tuple(boxes[i, :4].tolist()), s))
        return out
    # Raw iterable
    results: List[Tuple[Box, Optional[float]]] = []
    for row in detector_output:
        row = np.asarray(row, dtype=np.float32)
        if row.size == 4:
            results.append((tuple(row.tolist()), None))
        elif row.size >= 5:
            results.append((tuple(row[:4].tolist()), float(row[4])))
    return results


def _expand_box_to_aspect(
    box: Box, aspect: float, padding: float, img_size: Tuple[int, int]
) -> Box:
    """Expand `box` (x1, y1, x2, y2) to target `aspect` (W/H) and inflate by
    `padding` (e.g. 1.25 for a 25% margin around the person).  Clips to image
    bounds AFTER expansion — callers should letterbox-pad missing sides.
    """
    x1, y1, x2, y2 = box
    cx = (x1 + x2) / 2.0
    cy = (y1 + y2) / 2.0
    w = max(x2 - x1, 1.0)
    h = max(y2 - y1, 1.0)
    if w / h > aspect:
        h = w / aspect
    else:
        w = h * aspect
    w *= padding
    h *= padding
    return cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2


def _crop_with_letterbox(
    image: Image.Image, box: Box, target_size: Tuple[int, int],
    pad_value: int = 0,
) -> Tuple[Image.Image, float, Tuple[int, int]]:
    """Crop the image to `box` (which may extend outside the image), letterbox
    into a canvas of exactly `target_size = (H, W)`, then return:
      - cropped_resized: PIL image of size (W, H)
      - scale          : resize factor applied after crop
      - crop_origin    : (x_min, y_min) in the ORIGINAL image where the crop began

    The crop region is the requested `box`; out-of-image pixels are filled with
    `pad_value`.  Keypoints predicted in the crop's local coordinates map back
    to the original image via:   orig_xy = crop_origin + crop_xy / scale
    """
    x1, y1, x2, y2 = box
    w = int(round(x2 - x1))
    h = int(round(y2 - y1))
    x1_i, y1_i = int(round(x1)), int(round(y1))

    # Paste the cropped region onto a blank canvas of size (w, h) so that
    # boxes partially outside the image are handled correctly.
    canvas = Image.new("RGB", (w, h), (pad_value, pad_value, pad_value))
    # Region in the original image
    ox1 = max(0, x1_i); oy1 = max(0, y1_i)
    ox2 = min(image.width, x1_i + w); oy2 = min(image.height, y1_i + h)
    if ox2 > ox1 and oy2 > oy1:
        region = image.crop((ox1, oy1, ox2, oy2))
        canvas.paste(region, (ox1 - x1_i, oy1 - y1_i))

    H, W = target_size
    # Final resize to the pose model's expected input (non-aspect-preserving
    # since the box is already at target aspect ratio).
    out = canvas.resize((W, H), Image.BILINEAR)
    scale = W / w  # same as H / h since aspect matches
    return out, scale, (x1_i, y1_i)


class Sapiens2Predictor:
    # Sapiens2 expects 1024 × 768 (H × W) — aspect 3/4 (W/H).
    POSE_TARGET_HW: Tuple[int, int] = (1024, 768)
    POSE_ASPECT: float = 768.0 / 1024.0  # 0.75 (W/H)
    POSE_PAD_FACTOR: float = 1.25  # expand bbox by 25% for margin

    def __init__(
        self,
        model,
        processor: Sapiens2Processor,
        detector: Optional[Callable[[Image.Image], Any]] = None,
        detector_class_filter: Optional[int] = None,
    ):
        """
        Args:
            model: loaded Sapiens2 Model (one of pose/seg/normal/pointmap).
            processor: the image preprocessor.
            detector: optional callable with `.predict(image)` returning boxes.
                Only consulted when task == "pose".
            detector_class_filter: if the detector returns class labels, keep
                only boxes whose label equals this value.  Useful for general
                detectors like RF-DETR (COCO person class index = 1).
        """
        self.model = model
        self.processor = processor
        self.task = getattr(model.config, "task", "seg")
        self.detector = detector
        self.detector_class_filter = detector_class_filter

    # ────────────────────────── Public API ──────────────────────────────────

    def predict(
        self,
        image: Union[Image.Image, np.ndarray],
        person_boxes: Optional[List[Box]] = None,
    ) -> Union[Sapiens2Result, Sapiens2PoseMultiResult]:
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)
        image = image.convert("RGB")

        # Top-down pose path (only when task == pose AND we have boxes / a detector)
        if self.task == "pose":
            boxes = self._resolve_boxes(image, person_boxes)
            if boxes is not None:
                return self._predict_pose_top_down(image, boxes)

        # Otherwise: whole-image path
        return self._predict_whole_image(image)

    # ───────────────────────── Whole-image path ─────────────────────────────

    def _predict_whole_image(self, image: Image.Image) -> Sapiens2Result:
        inputs = self.processor.preprocess_image(image)
        pixel_values = mx.array(inputs["pixel_values"])
        orig_h, orig_w = inputs["original_size"]

        outputs = self.model(pixel_values)
        mx.eval(outputs)

        if self.task == "pose":
            return self._postprocess_pose(outputs, (orig_h, orig_w))
        if self.task == "seg":
            return self._postprocess_seg(outputs, (orig_h, orig_w))
        if self.task == "normal":
            return self._postprocess_normal(outputs, (orig_h, orig_w))
        if self.task == "pointmap":
            return self._postprocess_pointmap(outputs, (orig_h, orig_w))
        raise ValueError(f"Unknown task: {self.task}")

    # ───────────────────────── Top-down pose path ───────────────────────────

    def _resolve_boxes(
        self, image: Image.Image, person_boxes: Optional[List[Box]]
    ) -> Optional[List[Tuple[Box, Optional[float]]]]:
        if person_boxes is not None:
            return [(tuple(b), None) for b in person_boxes]
        if self.detector is not None:
            det_out = self.detector.predict(image)
            return _normalize_detector_output(
                det_out, filter_label=self.detector_class_filter
            )
        return None

    def _predict_pose_top_down(
        self, image: Image.Image, boxes_and_scores: List[Tuple[Box, Optional[float]]]
    ) -> Sapiens2PoseMultiResult:
        persons: List[PersonPose] = []
        H_t, W_t = self.POSE_TARGET_HW
        for (box, det_score) in boxes_and_scores:
            expanded = _expand_box_to_aspect(
                box, aspect=self.POSE_ASPECT, padding=self.POSE_PAD_FACTOR,
                img_size=image.size,
            )
            crop, scale, (ox, oy) = _crop_with_letterbox(
                image, expanded, target_size=(H_t, W_t),
            )

            # Preprocess the crop (same ImageNet-on-[0,255] norm as whole-image)
            pixel_values = self.processor.preprocess_image(crop)["pixel_values"]
            out = self.model(mx.array(pixel_values))
            mx.eval(out)

            hm = np.array(out)[0]  # (H_out, W_out, K)
            H_out, W_out, K = hm.shape
            flat = hm.reshape(-1, K)
            idx = flat.argmax(axis=0)
            scores = flat[idx, np.arange(K)]
            ys = (idx // W_out).astype(np.float32)
            xs = (idx %  W_out).astype(np.float32)

            # heatmap → crop pixel coords → original image pixel coords.
            # The crop was resized to (W_t, H_t), and the crop spans `expanded`
            # in original-image space.
            crop_w = expanded[2] - expanded[0]
            crop_h = expanded[3] - expanded[1]
            xs_crop = xs * (W_t / W_out)
            ys_crop = ys * (H_t / H_out)
            xs_orig = expanded[0] + xs_crop * (crop_w / W_t)
            ys_orig = expanded[1] + ys_crop * (crop_h / H_t)
            kp = np.stack([xs_orig, ys_orig], axis=-1).astype(np.float32)

            persons.append(PersonPose(
                box=tuple(box), keypoints=kp,
                keypoint_scores=scores, detector_score=det_score,
            ))

        return Sapiens2PoseMultiResult(
            task="pose", original_size=(image.height, image.width), persons=persons,
        )

    # ─────────────────────── Whole-image postprocessors ─────────────────────

    def _postprocess_pose(self, heatmaps: mx.array, orig) -> Sapiens2Result:
        hm = np.array(heatmaps)[0]  # (H, W, K)
        H, W, K = hm.shape
        flat = hm.reshape(-1, K)
        idx = np.argmax(flat, axis=0)
        ys = idx // W
        xs = idx % W
        scores = flat[idx, np.arange(K)]
        orig_h, orig_w = orig
        xs = xs.astype(np.float32) * (orig_w / W)
        ys = ys.astype(np.float32) * (orig_h / H)
        return Sapiens2Result(
            task="pose", original_size=orig,
            keypoints=np.stack([xs, ys], axis=-1), keypoint_scores=scores,
        )

    def _postprocess_seg(self, logits: mx.array, orig) -> Sapiens2Result:
        arr = np.array(logits)[0].transpose(2, 0, 1)  # (C, H, W)
        orig_h, orig_w = orig
        if arr.shape[1:] != orig:
            arr = _bilinear_resize_np(arr, (orig_h, orig_w))
        mask = np.argmax(arr, axis=0).astype(np.int32)
        return Sapiens2Result(task="seg", original_size=orig, mask=mask, seg_logits=arr)

    def _postprocess_normal(self, normal: mx.array, orig) -> Sapiens2Result:
        arr = np.array(normal)[0].transpose(2, 0, 1)  # (3, H, W)
        orig_h, orig_w = orig
        if arr.shape[1:] != orig:
            arr = _bilinear_resize_np(arr, (orig_h, orig_w))
        norm = np.linalg.norm(arr, axis=0, keepdims=True)
        arr = arr / np.where(norm < 1e-8, 1.0, norm)
        return Sapiens2Result(task="normal", original_size=orig, normal=arr)

    def _postprocess_pointmap(self, outputs, orig) -> Sapiens2Result:
        pointmap, scale = outputs
        pm = np.array(pointmap)[0].transpose(2, 0, 1)
        orig_h, orig_w = orig
        if pm.shape[1:] != orig:
            pm = _bilinear_resize_np(pm, (orig_h, orig_w))
        sc = float(np.array(scale).reshape(-1)[0])
        return Sapiens2Result(
            task="pointmap", original_size=orig, pointmap=pm, scale=sc,
        )
