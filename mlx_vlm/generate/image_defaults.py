"""Model-owned image sampling defaults, available without loading weights."""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal


@dataclass(frozen=True, slots=True)
class ImageSamplingDefaults:
    steps: int
    guidance: float

    def __post_init__(self) -> None:
        if (
            isinstance(self.steps, bool)
            or not isinstance(self.steps, int)
            or self.steps < 1
        ):
            raise ValueError("Default image steps must be a positive integer")
        if not math.isfinite(self.guidance) or self.guidance < 0:
            raise ValueError("Default image guidance must be finite and non-negative")

    @classmethod
    def from_config(cls, config: Any) -> ImageSamplingDefaults:
        return cls(config.default_steps, config.default_guidance)


def image_metadata_path(model: str, model_path: Path | None = None) -> Path:
    """Resolve configuration files only; never fetch weights or executable code."""
    if model_path is not None:
        return Path(model_path).expanduser()
    from ..utils import get_model_path

    return get_model_path(
        str(Path(model).expanduser()) if Path(model).expanduser().exists() else model,
        allow_patterns=["*.json"],
    )


def resolve_image_defaults(
    model: str, task: Literal["generate", "edit"] = "generate"
) -> ImageSamplingDefaults:
    """Return effective sampling defaults for a local checkpoint, Hub ID or alias.

    Only JSON metadata may be downloaded. Model providers implement
    ``resolve_defaults`` and expose ``default_sampling`` on loaded models using
    the same config/variant values. No pipeline or model weights are loaded.
    """
    from .edit_image import _image_edit_model_class_for_type
    from .image import (
        _image_model_class_for_type,
        _local_image_model_types,
        _model_type_from_id,
        _normalize_image_task,
    )

    task = _normalize_image_task(task)
    if not isinstance(model, str) or not model.strip():
        raise ValueError("Image model must be specified")
    model = model.strip()
    lookup = (
        _image_model_class_for_type
        if task == "generate"
        else _image_edit_model_class_for_type
    )
    root = Path(model).expanduser()
    model_path = root if root.is_dir() else None
    # Recognized aliases can resolve defaults directly from their variant table.
    if model_path is None:
        cls = lookup(_model_type_from_id(model))
        if cls is not None and cls.supports_model(model):
            return cls.resolve_defaults(model)
        other_lookup = (
            _image_edit_model_class_for_type
            if task == "generate"
            else _image_model_class_for_type
        )
        other_cls = other_lookup(_model_type_from_id(model))
        if other_cls is not None and other_cls.supports_model(model):
            raise ValueError(f"Image model {model} does not support task {task!r}")
        model_path = image_metadata_path(model)
    # Checkpoint metadata takes precedence over a directory/repository name.
    candidates = _local_image_model_types(str(model_path))
    if not candidates:
        raise ValueError(f"Cannot identify image model metadata: {model}")
    for model_type in candidates:
        cls = lookup(model_type)
        if cls is not None:
            return cls.resolve_defaults(model, model_path=model_path)
    raise ValueError(f"Image model {model} does not support task {task!r}")
