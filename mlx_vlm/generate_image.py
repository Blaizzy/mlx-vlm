from __future__ import annotations

import base64
import random
from dataclasses import dataclass, field, replace
from io import BytesIO
from pathlib import Path
from typing import Any, ClassVar, Literal, Protocol

import mlx.core as mx
import numpy as np
from PIL import Image

DEFAULT_IMAGE_SIZE = "512x512"
DEFAULT_IMAGE_STEPS = 4
DEFAULT_IMAGE_GUIDANCE = 1.0
DEFAULT_IMAGE_FORMAT = "png"

ImageOutputFormat = Literal["b64_json", "path"]
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
    return {"ternary": "bonsai", "2bit": "bonsai"}.get(model_type, model_type)


def image_generation_model_class(model: str | None) -> type[Any] | None:
    if model is None:
        return None

    model_type = _model_type_from_id(model)
    if model_type == "bonsai":
        from .models.bonsai.model import BonsaiImageGenerationModel

        if (
            BonsaiImageGenerationModel.is_image_generation_model
            and BonsaiImageGenerationModel.supports_model(model)
        ):
            return BonsaiImageGenerationModel
        return None

    model_path = Path(model).expanduser()
    if model_path.exists():
        from .models.bonsai.model import BonsaiImageGenerationModel

        if (
            BonsaiImageGenerationModel.is_image_generation_model
            and BonsaiImageGenerationModel.supports_model(model)
        ):
            return BonsaiImageGenerationModel

    return None


def is_image_generation_model(model: str | None) -> bool:
    if model is None:
        return False
    return image_generation_model_class(model) is not None


def load_image_generation_model(
    model: str | None,
    **kwargs: Any,
) -> ImageGenerationModel:
    if model is None:
        raise ValueError("Image generation model must be specified")
    model_class = image_generation_model_class(model)
    if model_class is not None:
        return model_class.from_model_id(model, **kwargs)
    raise ValueError(f"Image generation model {model} is not supported")


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
    model: ImageGenerationModel,
    request: ImageGenerationRequest | str,
    *,
    output_path: str | Path | None = None,
    **kwargs: Any,
) -> ImageGenerationResult:
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


__all__ = [
    "DEFAULT_IMAGE_FORMAT",
    "DEFAULT_IMAGE_GUIDANCE",
    "DEFAULT_IMAGE_SIZE",
    "DEFAULT_IMAGE_STEPS",
    "ImageArrayLayout",
    "ImageArrayRange",
    "ImageColorSpace",
    "ImageGenerationModel",
    "ImageGenerationRequest",
    "ImageGenerationResult",
    "ImageOutputFormat",
    "generate_image",
    "image_generation_model_class",
    "image_to_b64_json",
    "image_to_png_bytes",
    "is_image_generation_model",
    "load_image_generation_model",
    "parse_size",
]
