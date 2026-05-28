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

from .config import BonsaiVariant, get_variant
from .download import validate_model_layout
from .pipeline import BonsaiImage


def resolve_variant(model: str | BonsaiVariant | None) -> BonsaiVariant:
    if isinstance(model, BonsaiVariant):
        return model
    if model is None:
        return get_variant("ternary")

    model_s = str(model)
    model_path = Path(model_s).expanduser()
    if model_path.exists():
        return get_variant("ternary")

    return get_variant(model_s)


def can_load(model: str) -> bool:
    model_path = Path(model).expanduser()
    if model_path.exists():
        try:
            validate_model_layout(model_path)
            return True
        except FileNotFoundError:
            return False
    try:
        resolve_variant(model)
        return True
    except ValueError:
        return False


@dataclass(slots=True)
class BonsaiImageGenerationModel(ImageGenerationModel):
    is_image_generation_model: ClassVar[bool] = True
    model_type: ClassVar[str] = "bonsai"
    pipeline: BonsaiImage
    model_id: str
    family: str = "bonsai"

    @property
    def variant(self) -> str:
        return self.pipeline.variant.name

    def count_prompt_tokens(self, prompt: str) -> int | None:
        try:
            return self.pipeline.tokenizer.count_tokens(prompt)
        except Exception:
            return None

    def generate(self, request: ImageGenerationRequest) -> ImageGenerationResult:
        seed = 0 if request.seed is None else request.seed
        max_sequence_length = request.extra.get("max_sequence_length", None)
        tiled_vae = request.extra.get("tiled_vae", None)
        array = self.pipeline.generate_array(
            request.prompt,
            seed=seed,
            steps=request.steps,
            width=request.width,
            height=request.height,
            guidance=request.guidance,
            max_sequence_length=max_sequence_length,
            tiled_vae=tiled_vae,
        )
        return ImageGenerationResult(
            array=array,
            seed=seed,
            width=request.width,
            height=request.height,
            steps=request.steps,
            model=self.model_id,
            family=self.family,
            variant=self.variant,
            guidance=request.guidance,
            prompt_tokens=self.count_prompt_tokens(request.prompt),
            peak_memory=mx.get_peak_memory() / 1e9,
            metadata={"model_path": str(self.pipeline.model_path)},
        )

    @classmethod
    def supports_model(cls, model: str) -> bool:
        return can_load(model)

    @classmethod
    def from_model_id(
        cls,
        model: str = "bonsai-ternary",
        **kwargs: Any,
    ) -> "BonsaiImageGenerationModel":
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
        pipeline = BonsaiImage.from_pretrained(
            resolve_variant(model),
            model_path=model_path,
            download=kwargs.pop("download", True),
            token=kwargs.pop("token", None),
            evict_text_encoder=kwargs.pop("evict_text_encoder", True),
            evict_transformer=kwargs.pop("evict_transformer", False),
            bucketed_seq_len=kwargs.pop("bucketed_seq_len", True),
            tiled_vae=kwargs.pop("tiled_vae", "auto"),
            max_sequence_length=kwargs.pop("max_sequence_length", 512),
        )
        return cls(pipeline=pipeline, model_id=str(model))


def load(model: str = "bonsai-ternary", **kwargs: Any) -> BonsaiImageGenerationModel:
    return BonsaiImageGenerationModel.from_model_id(model, **kwargs)


__all__ = [
    "BonsaiImageGenerationModel",
    "can_load",
    "load",
    "resolve_variant",
]
