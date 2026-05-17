"""RT-DETRv2 postprocessing and predictor.

`RTDetrV2Predictor.predict_batch` returns a list of detections per image.
Each detection is a dict with keys `l, t, r, b, label, confidence`, where
the box is in original image pixel coordinates.

Label handling: the predictor reads `model.config.id2label` to map model
class ids to human-readable strings. Override via the `labels` constructor
arg to use a different vocabulary. Numeric labels are used as a fallback
if `id2label` is missing.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Set, Union

import numpy as np

from .processing_rt_detr_v2 import ImageInput, RTDetrV2Processor


@dataclass
class Detection:
    l: float
    t: float
    r: float
    b: float
    label: str
    confidence: float


LabelMap = Union[Sequence[str], Dict[int, str], Dict[str, str]]


class RTDetrV2Predictor:
    """Inference wrapper over `Model` + `RTDetrV2Processor`."""

    DEFAULT_THRESHOLD: float = 0.3

    def __init__(
        self,
        model,
        processor: Optional[RTDetrV2Processor] = None,
        threshold: float = DEFAULT_THRESHOLD,
        blacklist: Optional[Set[str]] = None,
        labels: Optional[LabelMap] = None,
    ) -> None:
        self.model = model
        self.processor = processor or RTDetrV2Processor()
        self.threshold = threshold
        self.blacklist = blacklist or frozenset()
        self.labels = _resolve_labels(labels, getattr(model, "config", None))

    def predict(self, image: ImageInput) -> List[dict]:
        return self.predict_batch([image])[0]

    def predict_batch(self, images: Iterable[ImageInput]) -> List[List[dict]]:
        out = self.processor(images)
        result = self.model(out.pixel_values)
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
    ) -> List[dict]:
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
            return []
        top_query = top_query[keep]
        top_label = top_label[keep]
        top_scores = top_scores[keep]

        sel = boxes[top_query]
        cx, cy = sel[:, 0] * img_w, sel[:, 1] * img_h
        bw, bh = sel[:, 2] * img_w, sel[:, 3] * img_h
        l = np.clip(cx - bw / 2.0, 0.0, img_w)
        t = np.clip(cy - bh / 2.0, 0.0, img_h)
        r = np.clip(cx + bw / 2.0, 0.0, img_w)
        b = np.clip(cy + bh / 2.0, 0.0, img_h)

        results: List[dict] = []
        for i, lid in enumerate(top_label):
            label = self.labels[int(lid)] if self.labels else str(int(lid))
            if label in self.blacklist:
                continue
            results.append(
                {
                    "l": float(l[i]),
                    "t": float(t[i]),
                    "r": float(r[i]),
                    "b": float(b[i]),
                    "label": label,
                    "confidence": float(top_scores[i]),
                }
            )
        return results


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
