from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, ClassVar

import mlx.core as mx

from mlx_vlm.generate.image import (
    ImageGenerationModel,
    ImageGenerationRequest,
    ImageGenerationResult,
)

from .config import (
    IDEOGRAM_4_FP8_REPO_ID,
    Ideogram4Variant,
    get_variant,
    variant_from_local_path,
)
from .download import validate_model_layout
from .pipeline import Ideogram4ImagePipeline


def resolve_variant(model: str | Ideogram4Variant | None) -> Ideogram4Variant:
    if isinstance(model, Ideogram4Variant):
        return model
    if model is None:
        return get_variant(IDEOGRAM_4_FP8_REPO_ID)
    model_path = Path(str(model)).expanduser()
    if model_path.exists():
        return variant_from_local_path(model_path)
    return get_variant(str(model))


def can_load(model: str) -> bool:
    model_path = Path(model).expanduser()
    if model_path.exists():
        try:
            validate_model_layout(model_path)
            return True
        except (FileNotFoundError, ValueError):
            return False
    try:
        resolve_variant(model)
        return True
    except ValueError:
        return False


@dataclass(slots=True)
class Ideogram4ImageGenerationModel(ImageGenerationModel):
    is_image_generation_model: ClassVar[bool] = True
    model_type: ClassVar[str] = "ideogram4"
    pipeline: Ideogram4ImagePipeline
    model_id: str
    family: str = "ideogram4"

    @property
    def variant(self) -> str:
        return self.pipeline.variant.name

    def generate(self, request: ImageGenerationRequest) -> ImageGenerationResult:
        seed = 0 if request.seed is None else request.seed
        array, metadata = self.pipeline.generate_array(
            request.prompt,
            seed=seed,
            steps=request.steps,
            width=request.width,
            height=request.height,
            guidance=request.guidance,
            **request.extra,
        )
        return ImageGenerationResult(
            array=array,
            seed=seed,
            width=request.width,
            height=request.height,
            steps=int(metadata.get("steps", request.steps)),
            model=self.model_id,
            family=self.family,
            variant=self.variant,
            guidance=float(metadata.get("guidance", request.guidance)),
            prompt_tokens=metadata.get("prompt_tokens"),
            peak_memory=mx.get_peak_memory() / 1e9,
            metadata=metadata,
        )

    @classmethod
    def supports_model(cls, model: str) -> bool:
        return can_load(model)

    @classmethod
    def from_model_id(
        cls,
        model: str = "ideogram-ai/ideogram-4-fp8",
        **kwargs: Any,
    ) -> "Ideogram4ImageGenerationModel":
        model_path_arg = kwargs.pop("model_path", None)
        model_path = (
            Path(model).expanduser()
            if (
                model_path_arg is None
                and isinstance(model, str)
                and Path(model).expanduser().exists()
            )
            else model_path_arg
        )
        pipeline = Ideogram4ImagePipeline.from_pretrained(
            resolve_variant(model),
            model_path=model_path,
            download=kwargs.pop("download", True),
            token=kwargs.pop("token", None),
            revision=kwargs.pop("revision", None),
            force_download=kwargs.pop("force_download", False),
        )
        return cls(pipeline=pipeline, model_id=str(model))


def load(
    model: str = "ideogram-ai/ideogram-4-fp8", **kwargs: Any
) -> Ideogram4ImageGenerationModel:
    return Ideogram4ImageGenerationModel.from_model_id(model, **kwargs)


__all__ = [
    "Ideogram4ImageGenerationModel",
    "can_load",
    "load",
    "resolve_variant",
]
