from __future__ import annotations

import importlib
import random
from collections.abc import Sequence
from dataclasses import dataclass, field, replace
from functools import lru_cache
from pathlib import Path
from typing import Any, ClassVar, Literal, Protocol

from .image import (
    DEFAULT_IMAGE_FORMAT,
    DEFAULT_IMAGE_GUIDANCE,
    DEFAULT_IMAGE_STEPS,
    ImageGenerationResult,
    _image_model_class_for_type,
    _image_model_type_from_manifest,
    _load_json_file,
    _local_image_model_types,
    _model_type_from_id,
    _normalize_model_type,
    _resolve_image_model_path,
)


@dataclass(slots=True)
class ImageEditRequest:
    prompt: str
    image_paths: tuple[str | Path, ...]
    seed: int | None = None
    steps: int = DEFAULT_IMAGE_STEPS
    width: int | None = None
    height: int | None = None
    guidance: float = DEFAULT_IMAGE_GUIDANCE
    output_format: Literal["png"] = DEFAULT_IMAGE_FORMAT
    extra: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if isinstance(self.image_paths, (str, Path)):
            self.image_paths = (self.image_paths,)
        else:
            self.image_paths = tuple(self.image_paths)


class ImageEditModel(Protocol):
    is_image_edit_model: ClassVar[bool]
    model_type: ClassVar[str]
    model_id: str
    family: str

    @classmethod
    def supports_model(cls, model: str) -> bool: ...

    @classmethod
    def from_model_id(cls, model: str, **kwargs: Any) -> "ImageEditModel": ...

    def edit(self, request: ImageEditRequest) -> ImageGenerationResult: ...


@lru_cache(maxsize=128)
def _image_edit_model_class_for_type(model_type: str | None) -> type[Any] | None:
    normalized = _normalize_model_type(model_type)
    if normalized is None:
        return None
    package_name = f"mlx_vlm.models.{normalized}"
    module_name = f"{package_name}.model"
    try:
        module = importlib.import_module(module_name)
    except ModuleNotFoundError as exc:
        if exc.name in {package_name, module_name}:
            return None
        raise
    for value in vars(module).values():
        if (
            isinstance(value, type)
            and getattr(value, "is_image_edit_model", False)
            and getattr(value, "model_type", None) == normalized
        ):
            return value
    return None


def _supported_image_edit_model_class(
    model_type: str | None, model: str
) -> type[Any] | None:
    model_class = _image_edit_model_class_for_type(model_type)
    if (
        model_class is not None
        and getattr(model_class, "is_image_edit_model", False)
        and model_class.supports_model(model)
    ):
        return model_class
    return None


def image_edit_model_class(model: str | None) -> type[Any] | None:
    if model is None:
        return None

    model_path = Path(model).expanduser()
    local_model_types = (
        _local_image_model_types(str(model_path)) if model_path.exists() else ()
    )
    for model_type in (_model_type_from_id(model), *local_model_types):
        model_class = _image_edit_model_class_for_type(model_type)
        if model_class is not None and getattr(
            model_class, "is_image_edit_model", False
        ):
            return model_class

    return None


def is_image_edit_model(model: str | None) -> bool:
    if model is None:
        return False

    for model_type in (*_local_image_model_types(model), _model_type_from_id(model)):
        if _supported_image_edit_model_class(model_type, model) is not None:
            return True
    return False


def load_image_edit_model(
    model: str | None,
    **kwargs: Any,
) -> ImageEditModel:
    if model is None:
        raise ValueError("Image edit model must be specified")
    force_download = kwargs.pop("force_download", False)
    revision = kwargs.pop("revision", None)
    alias_model_class = _image_edit_model_class_for_type(_model_type_from_id(model))
    model_path = Path(model).expanduser()
    if (
        not model_path.exists()
        and alias_model_class is not None
        and getattr(alias_model_class, "is_image_edit_model", False)
        and alias_model_class.supports_model(model)
    ):
        return alias_model_class.from_model_id(
            model,
            force_download=force_download,
            revision=revision,
            **kwargs,
        )
    try:
        resolved_path = _resolve_image_model_path(
            model,
            revision=revision,
            force_download=force_download,
        )
    except Exception:
        if alias_model_class is not None and getattr(
            alias_model_class, "is_image_edit_model", False
        ):
            return alias_model_class.from_model_id(model, **kwargs)
        raise
    local_model_types = (
        _local_image_model_types(str(resolved_path))
        if resolved_path is not None
        else ()
    )
    # If the repo's authoritative type (manifest first, then model id) maps
    # to a known image model that has no edit-capable class, stop here:
    # Bonsai repos carry FLUX.2-named components, so the candidate scan
    # below would otherwise select the flux2 edit class and fail variant
    # inference on Bonsai weights. Raise the clear unsupported-edit message
    # instead of the confusing flux2 variant error. (A type with an edit
    # class, e.g. flux2 or mage_flow, skips the guard and keeps today's
    # behavior.)
    authoritative_type = _model_type_from_id(model)
    if resolved_path is not None:
        manifest = _load_json_file(Path(resolved_path) / "manifest.json")
        if manifest is not None:
            authoritative_type = (
                _image_model_type_from_manifest(manifest) or authoritative_type
            )
    if _image_edit_model_class_for_type(authoritative_type) is None and (
        _image_model_class_for_type(authoritative_type) is not None
    ):
        raise ValueError(
            "Model "
            + model
            + " is a text-to-image generation model and does not support "
            "image editing with reference images. Image editing requires a "
            "model that implements edit support (FLUX.2 or Mage-Flow-Edit). "
            "Send the request to /v1/images/generations without a reference "
            "image, or use an edit-capable model."
        )
    model_class = None
    for model_type in (*local_model_types, _model_type_from_id(model)):
        model_class = _image_edit_model_class_for_type(model_type)
        if model_class is not None and getattr(
            model_class, "is_image_edit_model", False
        ):
            break
    if model_class is None:
        model_class = alias_model_class
    if model_class is not None:
        if resolved_path is not None:
            kwargs.setdefault("model_path", resolved_path)
        return model_class.from_model_id(model, **kwargs)
    raise ValueError(
        "Model "
        + model
        + " is a text-to-image generation model and does not support "
        "image editing with reference images. Image editing requires a "
        "model that implements edit support (FLUX.2 or Mage-Flow-Edit). "
        "Send the request to /v1/images/generations without a reference "
        "image, or use an edit-capable model."
    )


def _request_from_prompt(
    prompt: str,
    *,
    image_paths: Sequence[str | Path],
    **kwargs: Any,
) -> ImageEditRequest:
    supported = {
        "seed",
        "steps",
        "width",
        "height",
        "guidance",
        "output_format",
        "extra",
    }
    request_kwargs = {key: value for key, value in kwargs.items() if key in supported}
    extra = dict(request_kwargs.pop("extra", {}) or {})
    for key, value in kwargs.items():
        if key not in supported and value is not None:
            extra[key] = value
    request_kwargs["extra"] = extra
    return ImageEditRequest(
        prompt=prompt, image_paths=tuple(image_paths), **request_kwargs
    )


def edit_image(
    model: ImageEditModel,
    request: ImageEditRequest | str,
    *,
    image_paths: Sequence[str | Path] | None = None,
    output_path: str | Path | None = None,
    **kwargs: Any,
) -> ImageGenerationResult:
    if isinstance(request, str):
        if image_paths is None:
            raise ValueError("image_paths are required when request is a prompt string")
        request = _request_from_prompt(request, image_paths=image_paths, **kwargs)
    elif kwargs:
        extra = dict(request.extra)
        extra.update({key: value for key, value in kwargs.items() if value is not None})
        request = replace(request, extra=extra)

    if not request.image_paths:
        raise ValueError("At least one reference image path is required")
    if request.seed is None:
        request = replace(request, seed=random.randrange(2**32))

    data = model.edit(request)
    if output_path is not None:
        data.save(output_path)
    return data


__all__ = [
    "ImageEditModel",
    "ImageEditRequest",
    "edit_image",
    "image_edit_model_class",
    "is_image_edit_model",
    "load_image_edit_model",
]
