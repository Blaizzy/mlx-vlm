"""Shared typed API and result containers for extraction models."""

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, List, Optional, Union

import mlx.core as mx
import numpy as np

Array = Union[mx.array, np.ndarray]

#: Reserved output name for anything an extraction result carries that is not
#: an array: class names, a mesh, a pose, a flag.
METADATA = "metadata"


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


def detection_outputs(result) -> dict:
    """Name the populated fields of a detection result for `extract`.

    sam3 still carries its own DetectionResult, so fields are read
    defensively until it moves onto the shared one.
    """
    named = {"boxes": result.boxes, "scores": result.scores}
    for field_name in ("labels", "masks", "track_ids"):
        value = getattr(result, field_name, None)
        if value is not None and not isinstance(value, list):
            named[field_name] = value
    extra = {}
    if getattr(result, "class_names", None):
        extra["class_names"] = result.class_names
    labels = getattr(result, "labels", None)
    if isinstance(labels, list):
        extra["label_names"] = labels
    if getattr(result, "label_names", None) is not None:
        extra["label_names"] = result.label_names
    if extra:
        named[METADATA] = extra
    return named


def split_outputs(result: Mapping) -> dict:
    """Name the array values and gather everything else under `METADATA`."""
    arrays, extra = {}, {}
    for name, value in result.items():
        if name == METADATA and isinstance(value, Mapping):
            extra.update(value)
        elif isinstance(value, (mx.array, np.ndarray)):
            arrays[name] = value
        else:
            extra[name] = value
    if extra:
        arrays[METADATA] = extra
    return arrays


def describe_outputs(outputs):
    """Summarise named arrays as shapes and dtypes, listing metadata keys."""
    described = {
        name: {
            "shape": list(np.asarray(value).shape),
            "dtype": str(np.asarray(value).dtype),
        }
        for name, value in outputs.items()
        if name != METADATA
    }
    summary = {"outputs": described}
    if METADATA in outputs:
        summary[METADATA] = sorted(outputs[METADATA])
    return summary


def extract(model, processor, inputs, task=None, **kwargs):
    """Predict named structured outputs using a model's native extraction tasks."""
    supported = getattr(model, "extraction_types", ())
    if isinstance(supported, str):
        raise ValueError("extraction_types must be a sequence of task names")
    if not supported:
        raise ValueError("This model does not support extraction prediction")
    if not callable(getattr(model, "extract", None)):
        raise ValueError(
            "This model declares extraction_types but implements no extract"
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
    outputs = model.extract(processor, inputs, task=task, **kwargs)
    if not isinstance(outputs, Mapping):
        raise ValueError(f"{task!r} extraction must return a mapping of named outputs")
    for name, value in outputs.items():
        if name == METADATA:
            if not isinstance(value, Mapping):
                raise ValueError(
                    f"{METADATA!r} must be a mapping, got {type(value).__name__}"
                )
            continue
        if not isinstance(value, (mx.array, np.ndarray)):
            raise ValueError(
                f"{task!r} extraction returned {name!r} as {type(value).__name__}; "
                f"named outputs are arrays, so anything else belongs in {METADATA!r}"
            )
    return dict(outputs)
