from __future__ import annotations

import base64
import importlib
import json
import random
import re
from collections.abc import Sequence
from dataclasses import dataclass, field, replace
from functools import lru_cache
from io import BytesIO
from pathlib import Path
from typing import Any, ClassVar, Literal, Protocol

import mlx.core as mx
import numpy as np
from PIL import Image

from ..utils import get_model_path

DEFAULT_IMAGE_SIZE = "512x512"
DEFAULT_IMAGE_STEPS = 4
DEFAULT_IMAGE_GUIDANCE = 1.0
DEFAULT_IMAGE_FORMAT = "png"
DEFAULT_IMAGE_TASK = "generate"
IMAGE_DOWNLOAD_PATTERNS = (
    "*.json",
    "*.safetensors",
    "*.txt",
    "*.jinja",
    "*.model",
    "*.tiktoken",
    "**/*.json",
    "**/*.safetensors",
    "**/*.txt",
    "**/*.jinja",
    "**/*.model",
    "**/*.tiktoken",
)
IMAGE_METADATA_DOWNLOAD_PATTERNS = (
    "model_index.json",
    "config.json",
    "manifest.json",
    "**/config.json",
)

ImageOutputFormat = Literal["b64_json", "path"]
ImageTask = Literal["generate", "edit"]
ImageArrayLayout = Literal["HWC"]
ImageArrayRange = Literal["uint8_0_255"]
ImageColorSpace = Literal["RGB"]


@dataclass(slots=True)
class ImageGenerationRequest:
    prompt: str
    seed: int | None = None
    steps: int = DEFAULT_IMAGE_STEPS
    width: int = 512
    height: int = 512
    guidance: float = DEFAULT_IMAGE_GUIDANCE
    output_format: Literal["png"] = DEFAULT_IMAGE_FORMAT
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class ImageGenerationResult:
    array: mx.array
    seed: int
    width: int
    height: int
    steps: int
    model: str
    family: str
    guidance: float
    variant: str | None = None
    prompt_tokens: int | None = None
    peak_memory: float = 0.0
    path: Path | None = None
    output_format: Literal["png"] = DEFAULT_IMAGE_FORMAT
    layout: ImageArrayLayout = "HWC"
    value_range: ImageArrayRange = "uint8_0_255"
    color_space: ImageColorSpace = "RGB"
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        mx.eval(self.array)

    @property
    def image(self) -> Image.Image:
        return self.to_pil()

    def to_pil(self) -> Image.Image:
        if self.layout != "HWC" or self.color_space != "RGB":
            raise ValueError(
                f"Cannot convert image layout={self.layout!r} "
                f"color_space={self.color_space!r} to PIL"
            )
        return Image.fromarray(np.array(self.array))

    def to_png_bytes(self) -> bytes:
        buffer = BytesIO()
        self.to_pil().save(buffer, format="PNG")
        return buffer.getvalue()

    def to_b64_json(self) -> str:
        return base64.b64encode(self.to_png_bytes()).decode("ascii")

    def save(self, path: str | Path) -> Path:
        output_path = Path(path).expanduser()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        self.to_pil().save(output_path)
        self.path = output_path
        return output_path


class ImageGenerationModel(Protocol):
    is_image_generation_model: ClassVar[bool]
    model_type: ClassVar[str]
    model_id: str
    family: str

    @classmethod
    def supports_model(cls, model: str) -> bool: ...

    @classmethod
    def from_model_id(cls, model: str, **kwargs: Any) -> "ImageGenerationModel": ...

    def generate(self, request: ImageGenerationRequest) -> ImageGenerationResult: ...


def parse_size(value: str) -> tuple[int, int]:
    normalized = value.lower().replace("×", "x")
    try:
        width_s, height_s = normalized.split("x", 1)
        width = int(width_s)
        height = int(height_s)
    except ValueError as exc:
        raise ValueError(f"Size must be WIDTHxHEIGHT, got {value!r}") from exc
    return width, height


def _model_type_from_id(model: str) -> str:
    model_id = model.strip().lower().rstrip("/")
    if not model_id:
        return ""
    name = model_id.rsplit("/", 1)[-1]
    model_type = name.split("-", 1)[0]
    return {
        "ternary": "bonsai",
        "2bit": "bonsai",
        "flux.2": "flux2",
        "flux2": "flux2",
        "klein": "flux2",
    }.get(model_type, model_type)


def _normalize_model_type(model_type: Any | None) -> str | None:
    if model_type is None:
        return None
    normalized = str(model_type).strip().lower().replace("-", "_")
    return normalized if normalized and normalized.isidentifier() else None


def _add_model_type(candidates: list[str], model_type: Any | None) -> None:
    normalized = _normalize_model_type(model_type)
    if normalized is not None and normalized not in candidates:
        candidates.append(normalized)


def _model_types_from_class_name(class_name: str) -> tuple[str, ...]:
    tokens = re.findall(r"[A-Z][a-z0-9]*|[A-Z]+(?=[A-Z]|$)", class_name)
    candidates: list[str] = []
    for end in range(1, len(tokens) + 1):
        _add_model_type(candidates, "_".join(tokens[:end]))
    for token in tokens:
        _add_model_type(candidates, token)
    return tuple(candidates)


def _image_model_types_from_metadata(metadata: dict[str, Any]) -> tuple[str, ...]:
    candidates: list[str] = []
    _add_model_type(candidates, metadata.get("model_type"))
    candidates.extend(
        candidate
        for candidate in _model_types_from_class_name(
            str(metadata.get("_class_name") or "")
        )
        if candidate not in candidates
    )
    for component in ("transformer", "vae", "scheduler", "text_encoder"):
        value = metadata.get(component)
        component_class = ""
        if isinstance(value, Sequence) and not isinstance(value, str) and value:
            component_class = str(value[-1])
        elif isinstance(value, dict):
            component_class = str(value.get("_class_name") or value.get("class") or "")
        candidates.extend(
            candidate
            for candidate in _model_types_from_class_name(component_class)
            if candidate not in candidates
        )
    return tuple(candidates)


def _manifest_paths(metadata: dict[str, Any]) -> set[str]:
    paths = set()
    for entry in metadata.get("files", ()):
        if isinstance(entry, str):
            paths.add(entry)
        elif isinstance(entry, dict):
            path = (
                entry.get("remote_path")
                or entry.get("path")
                or entry.get("filename")
                or entry.get("name")
            )
            if path is not None:
                paths.add(str(path))
    return paths


def _image_model_type_from_manifest(metadata: dict[str, Any]) -> str | None:
    paths = _manifest_paths(metadata)
    if (
        "transformer-packed-mflux/diffusion_pytorch_model.safetensors" in paths
        and "text_encoder-mlx-4bit/model.safetensors" in paths
        and "tokenizer/tokenizer.json" in paths
    ):
        return "bonsai"
    return None


def _load_json_file(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        value = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return None
    return value if isinstance(value, dict) else None


def _local_image_model_types(model: str) -> tuple[str, ...]:
    root = Path(model).expanduser()
    if not root.exists():
        return ()

    candidates: list[str] = []
    for filename in ("model_index.json", "config.json"):
        metadata = _load_json_file(root / filename)
        if metadata is not None:
            for model_type in _image_model_types_from_metadata(metadata):
                _add_model_type(candidates, model_type)

    manifest = _load_json_file(root / "manifest.json")
    if manifest is not None:
        _add_model_type(candidates, _image_model_type_from_manifest(manifest))

    for config_path in sorted(root.glob("*/config.json")):
        metadata = _load_json_file(config_path)
        if metadata is not None:
            for model_type in _image_model_types_from_metadata(metadata):
                _add_model_type(candidates, model_type)
    return tuple(candidates)


@lru_cache(maxsize=128)
def _image_model_class_for_type(model_type: str | None) -> type[Any] | None:
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
            and getattr(value, "is_image_generation_model", False)
            and getattr(value, "model_type", None) == normalized
        ):
            return value
    return None


def _supported_image_model_class(
    model_type: str | None, model: str
) -> type[Any] | None:
    model_class = _image_model_class_for_type(model_type)
    if (
        model_class is not None
        and model_class.is_image_generation_model
        and model_class.supports_model(model)
    ):
        return model_class
    return None


def _resolve_image_model_path(
    model: str,
    *,
    revision: str | None = None,
    force_download: bool = False,
) -> Path | None:
    model_path = Path(model).expanduser()
    if model_path.exists():
        return model_path
    return get_model_path(
        model,
        revision=revision,
        force_download=force_download,
        allow_patterns=list(IMAGE_DOWNLOAD_PATTERNS),
    )


def _resolve_image_model_metadata_path(model: str) -> Path | None:
    model_path = Path(model).expanduser()
    if model_path.exists():
        return model_path
    return get_model_path(
        model,
        allow_patterns=list(IMAGE_METADATA_DOWNLOAD_PATTERNS),
    )


def image_generation_model_class(model: str | None) -> type[Any] | None:
    if model is None:
        return None

    alias_model_class = _supported_image_model_class(_model_type_from_id(model), model)
    if alias_model_class is not None:
        return alias_model_class

    try:
        resolved_path = _resolve_image_model_metadata_path(model)
    except Exception:
        resolved_path = None
    local_model_types = (
        _local_image_model_types(str(resolved_path))
        if resolved_path is not None
        else ()
    )
    for model_type in (
        *local_model_types,
        _model_type_from_id(model),
    ):
        model_class = _image_model_class_for_type(model_type)
        if model_class is not None and model_class.is_image_generation_model:
            return model_class

    return None


def is_image_generation_model(model: str | None) -> bool:
    if model is None:
        return False

    for model_type in (*_local_image_model_types(model), _model_type_from_id(model)):
        if _supported_image_model_class(model_type, model) is not None:
            return True
    return False


def load_image_generation_model(
    model: str | None,
    **kwargs: Any,
) -> ImageGenerationModel:
    if model is None:
        raise ValueError("Image generation model must be specified")
    force_download = kwargs.pop("force_download", False)
    revision = kwargs.pop("revision", None)
    alias_model_class = _image_model_class_for_type(_model_type_from_id(model))
    try:
        resolved_path = _resolve_image_model_path(
            model,
            revision=revision,
            force_download=force_download,
        )
    except Exception:
        if (
            alias_model_class is not None
            and alias_model_class.is_image_generation_model
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
        model_class = _image_model_class_for_type(model_type)
        if model_class is not None and model_class.is_image_generation_model:
            break
    if model_class is None:
        model_class = alias_model_class
    if model_class is not None:
        if resolved_path is not None:
            kwargs.setdefault("model_path", resolved_path)
        return model_class.from_model_id(model, **kwargs)
    raise ValueError(f"Image generation model {model} is not supported")


def _normalize_image_task(task: str | None) -> ImageTask:
    normalized = (task or DEFAULT_IMAGE_TASK).strip().lower()
    if normalized not in {"generate", "edit"}:
        raise ValueError(f"Image task must be 'generate' or 'edit', got {task!r}")
    return normalized  # type: ignore[return-value]


def load_image_model(
    model: str | None,
    *,
    task: ImageTask = DEFAULT_IMAGE_TASK,
    **kwargs: Any,
) -> ImageGenerationModel | Any:
    task = _normalize_image_task(task)
    if task == "edit":
        from .edit_image import load_image_edit_model

        return load_image_edit_model(model, **kwargs)
    return load_image_generation_model(model, **kwargs)


def _request_from_prompt(prompt: str, **kwargs: Any) -> ImageGenerationRequest:
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
    return ImageGenerationRequest(prompt=prompt, **request_kwargs)


def generate_image(
    model: ImageGenerationModel | Any,
    request: ImageGenerationRequest | str | Any,
    *,
    task: ImageTask = DEFAULT_IMAGE_TASK,
    image_paths: Sequence[str | Path] | None = None,
    output_path: str | Path | None = None,
    **kwargs: Any,
) -> ImageGenerationResult:
    task = _normalize_image_task(task)
    if task == "edit":
        from .edit_image import ImageEditRequest, edit_image

        if isinstance(request, ImageGenerationRequest):
            if image_paths is None:
                raise ValueError(
                    "image_paths are required when editing from "
                    "ImageGenerationRequest"
                )
            request = ImageEditRequest(
                prompt=request.prompt,
                image_paths=tuple(image_paths),
                seed=request.seed,
                steps=request.steps,
                width=request.width,
                height=request.height,
                guidance=request.guidance,
                output_format=request.output_format,
                extra=dict(request.extra),
            )
        return edit_image(
            model,
            request,
            image_paths=image_paths,
            output_path=output_path,
            **kwargs,
        )

    if isinstance(request, str):
        request = _request_from_prompt(request, **kwargs)
    elif kwargs:
        extra = dict(request.extra)
        extra.update({key: value for key, value in kwargs.items() if value is not None})
        request = replace(request, extra=extra)

    if request.seed is None:
        request = replace(request, seed=random.randrange(2**32))

    data = model.generate(request)
    if output_path is not None:
        data.save(output_path)
    return data


def image_to_png_bytes(image: Image.Image | mx.array | ImageGenerationResult) -> bytes:
    if isinstance(image, ImageGenerationResult):
        return image.to_png_bytes()
    if isinstance(image, mx.array):
        image = Image.fromarray(np.array(image))
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def image_to_b64_json(image: Image.Image | mx.array | ImageGenerationResult) -> str:
    return base64.b64encode(image_to_png_bytes(image)).decode("ascii")


def _prompt_to_image_text(prompt: str | Sequence[str]) -> str:
    if isinstance(prompt, str):
        return prompt
    return " ".join(str(part) for part in prompt)


def _validate_image_generation_args(args: Any, *, task: ImageTask) -> None:
    incompatible = []
    if task == "generate" and getattr(args, "image", None) is not None:
        incompatible.append("--image")
    if task == "edit" and not getattr(args, "image", None):
        raise ValueError("Image editing requires at least one --image path")
    for name in ("audio", "video", "adapter_path", "draft_model"):
        if getattr(args, name, None) is not None:
            incompatible.append(f"--{name.replace('_', '-')}")
    if getattr(args, "chat", False):
        incompatible.append("--chat")
    if getattr(args, "resize_shape", None) is not None:
        incompatible.append("--resize-shape")
    if getattr(args, "eos_tokens", None) is not None:
        incompatible.append("--eos-tokens")
    if getattr(args, "kv_bits", None) is not None:
        incompatible.append("--kv-bits")
    if getattr(args, "max_kv_size", None) is not None:
        incompatible.append("--max-kv-size")
    if getattr(args, "processor_kwargs", None):
        incompatible.append("--processor-kwargs")
    if getattr(args, "quantize_activations", False):
        incompatible.append("--quantize-activations")
    if getattr(args, "skip_special_tokens", False):
        incompatible.append("--skip-special-tokens")
    if getattr(args, "thinking_budget", None) is not None:
        incompatible.append("--thinking-budget")
    if getattr(args, "max_denoising_steps", None) is not None:
        incompatible.append("--max-denoising-steps")
    if getattr(args, "diffusion_full_canvas", False):
        incompatible.append("--diffusion-full-canvas")
    if incompatible:
        joined = ", ".join(incompatible)
        raise ValueError(f"Image {task} does not support: {joined}")


def run_image_generation_cli(args: Any) -> None:
    task = _normalize_image_task(getattr(args, "task", DEFAULT_IMAGE_TASK))
    _validate_image_generation_args(args, task=task)
    seed = args.seed if args.seed is not None else random.randrange(2**32)
    prompt = _prompt_to_image_text(args.prompt)
    if not prompt:
        raise ValueError(f"--prompt must not be empty for image {task}")

    load_kwargs = {
        "force_download": getattr(args, "force_download", False),
        "revision": getattr(args, "revision", None),
    }
    if task == "edit":
        from .edit_image import ImageEditRequest

        width = height = None
        size = getattr(args, "size", None)
        if size is not None:
            width, height = parse_size(size)
        output_path = (
            Path(args.output).expanduser()
            if args.output is not None
            else Path("outputs") / f"edit-{seed}.png"
        )
        model = load_image_model(args.model, task="edit", **load_kwargs)
        request = ImageEditRequest(
            prompt=prompt,
            image_paths=tuple(args.image),
            seed=seed,
            steps=args.steps,
            width=width,
            height=height,
            guidance=args.guidance,
            extra=dict(getattr(args, "gen_kwargs", {}) or {}),
        )
        result = generate_image(
            model,
            request,
            task="edit",
            output_path=output_path,
        )
    else:
        width, height = parse_size(getattr(args, "size", None) or DEFAULT_IMAGE_SIZE)
        output_path = (
            Path(args.output).expanduser()
            if args.output is not None
            else Path("outputs") / f"image-{seed}.png"
        )
        model = load_image_model(args.model, task="generate", **load_kwargs)
        extra = dict(getattr(args, "gen_kwargs", {}) or {})
        prompt_expansion_model = getattr(args, "prompt_expansion_model", None)
        if prompt_expansion_model is not None:
            extra["prompt_expansion_model"] = prompt_expansion_model
        request = ImageGenerationRequest(
            prompt=prompt,
            seed=seed,
            steps=args.steps,
            width=width,
            height=height,
            guidance=args.guidance,
            extra=extra,
        )
        result = generate_image(
            model,
            request,
            task="generate",
            output_path=output_path,
        )
    print(
        f"Saved {result.path} seed={result.seed} "
        f"size={result.width}x{result.height} steps={result.steps} "
        f"variant={result.variant}"
    )


__all__ = [
    "DEFAULT_IMAGE_FORMAT",
    "DEFAULT_IMAGE_GUIDANCE",
    "DEFAULT_IMAGE_SIZE",
    "DEFAULT_IMAGE_STEPS",
    "DEFAULT_IMAGE_TASK",
    "ImageArrayLayout",
    "ImageArrayRange",
    "ImageColorSpace",
    "ImageGenerationModel",
    "ImageGenerationRequest",
    "ImageGenerationResult",
    "ImageOutputFormat",
    "ImageTask",
    "generate_image",
    "image_generation_model_class",
    "image_to_b64_json",
    "image_to_png_bytes",
    "is_image_generation_model",
    "load_image_generation_model",
    "load_image_model",
    "parse_size",
    "run_image_generation_cli",
]
