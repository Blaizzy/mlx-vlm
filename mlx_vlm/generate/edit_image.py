from __future__ import annotations

import argparse
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
    _local_image_model_types,
    _model_type_from_id,
    _normalize_model_type,
    _prompt_to_image_text,
    _resolve_image_model_path,
    parse_size,
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
    raise ValueError(f"Image edit model {model} is not supported")


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


def parse_image_edit_arguments(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Edit an image with a supported image-editing model."
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="The path to the local model directory or Hugging Face repo.",
    )
    parser.add_argument(
        "--image",
        type=str,
        nargs="+",
        required=True,
        help="Path of one or more reference images.",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        nargs="+",
        required=True,
        help="Edit prompt.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output path for the edited image.",
    )
    parser.add_argument(
        "--size",
        type=str,
        default=None,
        help="Edited image size as WIDTHxHEIGHT. Defaults to the first reference image size.",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=DEFAULT_IMAGE_STEPS,
        help="Number of image edit inference steps.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Seed for image editing. Defaults to a random 32-bit seed.",
    )
    parser.add_argument(
        "--guidance",
        type=float,
        default=DEFAULT_IMAGE_GUIDANCE,
        help="Classifier-free guidance for image editing.",
    )
    return parser.parse_args(argv)


def run_image_edit_cli(args: Any | None = None) -> None:
    if args is None:
        args = parse_image_edit_arguments()
    width = height = None
    if args.size is not None:
        width, height = parse_size(args.size)
    seed = args.seed if args.seed is not None else random.randrange(2**32)
    prompt = _prompt_to_image_text(args.prompt)
    if not prompt:
        raise ValueError("--prompt must not be empty for image editing")

    output_path = (
        Path(args.output).expanduser()
        if args.output is not None
        else Path("outputs") / f"edit-{seed}.png"
    )
    model = load_image_edit_model(args.model)
    request = ImageEditRequest(
        prompt=prompt,
        image_paths=tuple(args.image),
        seed=seed,
        steps=args.steps,
        width=width,
        height=height,
        guidance=args.guidance,
    )
    result = edit_image(
        model,
        request,
        output_path=output_path,
    )
    print(
        f"Saved {result.path} seed={result.seed} "
        f"size={result.width}x{result.height} steps={result.steps} "
        f"variant={result.variant}"
    )


__all__ = [
    "ImageEditModel",
    "ImageEditRequest",
    "edit_image",
    "image_edit_model_class",
    "is_image_edit_model",
    "load_image_edit_model",
    "parse_image_edit_arguments",
    "run_image_edit_cli",
]
