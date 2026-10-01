"""Shared typed API and result containers for extraction models."""

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, List, Optional, Union

import mlx.core as mx
import numpy as np

Array = Union[mx.array, np.ndarray]


@dataclass
class DetectionResult:
    """Boxes, scores and optional masks predicted for one image or frame."""

    boxes: Array  # (N, 4) xyxy in pixel coordinates
    scores: Array  # (N,)
    labels: Optional[Array] = None  # (N,) integer class ids
    class_names: List[str] = field(default_factory=list)  # class id -> name
    label_names: Optional[List[str]] = None  # (N,) per-detection text
    masks: Optional[Array] = None  # (N, H, W) binary
    track_ids: Optional[Array] = None  # (N,) stable ids across frames
    image: Optional[Any] = None

    def __len__(self) -> int:
        return 0 if self.scores is None else len(self.scores)


@dataclass
class TrackingResult:
    """Masks and scores for the objects tracked in one frame."""

    frame_idx: int
    masks: Array  # (N, H, W) binary
    scores: Array  # (N,)
    object_ids: Optional[List[int]] = None


def cxcywh_to_xyxy(boxes: Array) -> Array:
    """Convert center-format boxes to corner format along the last axis."""
    cx, cy, w, h = boxes[..., 0], boxes[..., 1], boxes[..., 2], boxes[..., 3]
    stack = mx.stack if isinstance(boxes, mx.array) else np.stack
    return stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], axis=-1)


def detection_outputs(result: "DetectionResult") -> dict:
    """Name the populated fields of a detection result for `extract`."""
    named = {"boxes": result.boxes, "scores": result.scores}
    for field_name in ("labels", "masks", "track_ids"):
        value = getattr(result, field_name)
        if value is not None:
            named[field_name] = value
    if result.class_names:
        named["class_names"] = result.class_names
    if result.label_names is not None:
        named["label_names"] = result.label_names
    return named


def extract(model, processor, inputs, task=None, **kwargs):
    """Predict named structured outputs using a model's native extraction tasks."""
    supported = getattr(model, "extraction_types", ())
    if isinstance(supported, str):
        raise ValueError("extraction_types must be a sequence of task names")
    if not supported:
        raise ValueError("This model does not support extraction prediction")
    if not callable(getattr(model, "extract_task", None)):
        raise ValueError(
            "This model declares extraction_types but implements no extract_task"
        )
    if inputs is None:
        raise ValueError("Extraction requires inputs")
    if task is None:
        if len(supported) != 1:
            raise ValueError(
                f"This model serves {tuple(supported)}; pass task= to select one"
            )
        task = supported[0]
    if task not in supported:
        raise ValueError(f"This model does not support {task!r} extraction")
    outputs = model.extract_task(processor, inputs, task=task, **kwargs)
    if not isinstance(outputs, Mapping):
        raise ValueError(f"{task!r} extraction must return a mapping of named outputs")
    return dict(outputs)
