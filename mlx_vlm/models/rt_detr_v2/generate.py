"""RT-DETRv2 postprocessing and predictor.

`RTDetrV2Predictor.predict_batch` returns a `DetectionResult` per image
with vectorized fields: `boxes` (xyxy pixel coords), `scores`, `labels`
(integer class ids), and `class_names`. Output schema matches the
existing `mlx_vlm/models/rfdetr/generate.py:DetectionResult` so detection
models in this repo share a result type.

Label handling: the predictor reads `model.config.id2label` to map class
ids to human-readable strings. Override via the `labels` constructor arg
to use a different vocabulary. If neither is set, `class_names` holds
stringified integer ids.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Sequence, Union

import mlx.core as mx
import numpy as np

from .processing_rt_detr_v2 import ImageInput, RTDetrV2Processor


@dataclass
class DetectionResult:
    """Per-image detection output.

    Mirrors `mlx_vlm.models.rfdetr.generate.DetectionResult`.
    """

    boxes: np.ndarray  # (N, 4) xyxy pixel coordinates in the original image
    scores: np.ndarray  # (N,) confidence scores in [0, 1]
    labels: np.ndarray  # (N,) integer class ids
    class_names: List[str] = field(default_factory=list)


LabelMap = Union[Sequence[str], Dict[int, str], Dict[str, str]]


class RTDetrV2Predictor:
    """Inference wrapper over `Model` + `RTDetrV2Processor`."""

    DEFAULT_THRESHOLD: float = 0.3

    def __init__(
        self,
        model,
        processor: Optional[RTDetrV2Processor] = None,
        threshold: float = DEFAULT_THRESHOLD,
        labels: Optional[LabelMap] = None,
    ) -> None:
        self.model = model
        self.processor = processor or RTDetrV2Processor()
        self.threshold = threshold
        self.labels = _resolve_labels(labels, getattr(model, "config", None))

    def predict(self, image: ImageInput) -> DetectionResult:
        return self.predict_batch([image])[0]

    def predict_batch(self, images: Iterable[ImageInput]) -> List[DetectionResult]:
        out = self.processor(images)
        result = self.model(out.pixel_values)
        mx.eval(result["pred_logits"], result["pred_boxes"])
        logits = np.asarray(result["pred_logits"])
        boxes = np.asarray(result["pred_boxes"])
        return [
            self._decode_one(logits[i], boxes[i], w, h)
            for i, (w, h) in enumerate(out.original_sizes)
        ]

    def _decode_one(
        self,
        logits: np.ndarray,
        boxes: np.ndarray,
        img_w: int,
        img_h: int,
    ) -> DetectionResult:
        """Top-K extraction across the flat (queries x labels) score space.

        Matches `RTDetrImageProcessor.post_process_object_detection` for
        `use_focal_loss=True` models: a single query can yield multiple
        detections if it scores above threshold on multiple labels.
        """
        Q, num_labels = logits.shape
        scores = 1.0 / (1.0 + np.exp(-logits))
        flat = scores.reshape(-1)

        k = min(Q, flat.size)
        top_idx = np.argpartition(-flat, k - 1)[:k]
        top_scores = flat[top_idx]
        order = np.argsort(-top_scores)
        top_idx, top_scores = top_idx[order], top_scores[order]

        top_query = top_idx // num_labels
        top_label = top_idx % num_labels

        keep = top_scores >= self.threshold
        if not keep.any():
            empty = np.zeros((0, 4), dtype=np.float32)
            return DetectionResult(
                boxes=empty,
                scores=np.zeros((0,), dtype=np.float32),
                labels=np.zeros((0,), dtype=np.int64),
                class_names=[],
            )
        top_query = top_query[keep]
        top_label = top_label[keep].astype(np.int64)
        top_scores = top_scores[keep].astype(np.float32)

        sel = boxes[top_query]
        cx, cy = sel[:, 0] * img_w, sel[:, 1] * img_h
        bw, bh = sel[:, 2] * img_w, sel[:, 3] * img_h
        x1 = np.clip(cx - bw / 2.0, 0.0, img_w)
        y1 = np.clip(cy - bh / 2.0, 0.0, img_h)
        x2 = np.clip(cx + bw / 2.0, 0.0, img_w)
        y2 = np.clip(cy + bh / 2.0, 0.0, img_h)
        xyxy = np.stack([x1, y1, x2, y2], axis=-1).astype(np.float32)

        if self.labels is not None:
            class_names = [self.labels[int(lid)] for lid in top_label]
        else:
            class_names = [str(int(lid)) for lid in top_label]

        return DetectionResult(
            boxes=xyxy,
            scores=top_scores,
            labels=top_label,
            class_names=class_names,
        )


def _resolve_labels(labels: Optional[LabelMap], config) -> Optional[List[str]]:
    """Build an ordered list of label strings.

    Priority: user-supplied `labels` arg > `config.id2label` > None (numeric
    fallback at decode time).
    """
    if labels is not None:
        if isinstance(labels, dict):
            return _sort_id_dict(labels)
        return list(labels)
    if config is not None and getattr(config, "id2label", None):
        return _sort_id_dict(config.id2label)
    return None


def _sort_id_dict(d: Dict) -> List[str]:
    """Order an id->label dict by integer-cast key. HF stores keys as
    strings, but user-supplied dicts may use ints; handle both."""
    return [d[k] for k in sorted(d, key=lambda x: int(x))]
