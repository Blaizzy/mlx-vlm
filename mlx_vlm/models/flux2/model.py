from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, ClassVar

import mlx.core as mx

from mlx_vlm.generate.edit_image import ImageEditModel, ImageEditRequest
from mlx_vlm.generate.image import (
    ImageGenerationModel,
    ImageGenerationRequest,
    ImageGenerationResult,
)

from .config import Flux2Variant, get_variant, variant_from_local_path
from .download import validate_model_layout
from .pipeline import Flux2Image, Flux2ImageEdit


def resolve_variant(model: str | Flux2Variant | None) -> Flux2Variant:
    if isinstance(model, Flux2Variant):
        return model
    if model is None:
        return get_variant("flux2-klein-4b")

    model_s = str(model)
    model_path = Path(model_s).expanduser()
    if model_path.exists():
        return variant_from_local_path(model_path)

    return get_variant(model_s)


def _can_load(model: str, *, require_edit: bool = False) -> bool:
    model_path = Path(model).expanduser()
    if model_path.exists():
        try:
            validate_model_layout(model_path)
            variant = variant_from_local_path(model_path)
            if require_edit and not variant.supports_edit:
                return False
            return True
        except (FileNotFoundError, ValueError):
            return False
    try:
        variant = resolve_variant(model)
        if require_edit and not variant.supports_edit:
            return False
        return True
    except ValueError:
        return False


def can_load(model: str) -> bool:
    return _can_load(model)


def can_edit(model: str) -> bool:
    return _can_load(model, require_edit=True)


@dataclass(slots=True)
class Flux2ImageGenerationModel(ImageGenerationModel):
    is_image_generation_model: ClassVar[bool] = True
    model_type: ClassVar[str] = "flux2"
    pipeline: Flux2Image
    model_id: str
    family: str = "flux2"

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
            metadata={
                "model_path": str(self.pipeline.model_path),
                "architecture": "dense",
                "vae_variant": "full",
            },
        )

    @classmethod
    def supports_model(cls, model: str) -> bool:
        return can_load(model)

    @classmethod
    def from_model_id(
        cls,
        model: str = "flux2-klein-4b",
        **kwargs: Any,
    ) -> "Flux2ImageGenerationModel":
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
        pipeline = Flux2Image.from_pretrained(
            resolve_variant(model),
            model_path=model_path,
            download=kwargs.pop("download", True),
            token=kwargs.pop("token", None),
            revision=kwargs.pop("revision", None),
            force_download=kwargs.pop("force_download", False),
            evict_text_encoder=kwargs.pop("evict_text_encoder", True),
            evict_transformer=kwargs.pop("evict_transformer", False),
            bucketed_seq_len=kwargs.pop("bucketed_seq_len", True),
            tiled_vae=kwargs.pop("tiled_vae", "auto"),
            max_sequence_length=kwargs.pop("max_sequence_length", 512),
        )
        return cls(pipeline=pipeline, model_id=str(model))


@dataclass(slots=True)
class Flux2ImageEditModel(ImageEditModel):
    is_image_edit_model: ClassVar[bool] = True
    model_type: ClassVar[str] = "flux2"
    pipeline: Flux2ImageEdit
    model_id: str
    family: str = "flux2"

    @property
    def variant(self) -> str:
        return self.pipeline.variant.name

    def count_prompt_tokens(self, prompt: str) -> int | None:
        try:
            return self.pipeline.tokenizer.count_tokens(prompt)
        except Exception:
            return None

    def edit(self, request: ImageEditRequest) -> ImageGenerationResult:
        seed = 0 if request.seed is None else request.seed
        max_sequence_length = request.extra.get("max_sequence_length", None)
        tiled_vae = request.extra.get("tiled_vae", None)
        array = self.pipeline.edit_array(
            request.prompt,
            request.image_paths,
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
            width=array.shape[1],
            height=array.shape[0],
            steps=request.steps,
            model=self.model_id,
            family=self.family,
            variant=self.variant,
            guidance=request.guidance,
            prompt_tokens=self.count_prompt_tokens(request.prompt),
            peak_memory=mx.get_peak_memory() / 1e9,
            metadata={
                "model_path": str(self.pipeline.model_path),
                "architecture": "dense",
                "vae_variant": "full",
                "reference_count": len(request.image_paths),
                "uses_reference_kv_cache": self.pipeline.variant.uses_reference_kv_cache,
            },
        )

    @classmethod
    def supports_model(cls, model: str) -> bool:
        return can_edit(model)

    @classmethod
    def from_model_id(
        cls,
        model: str = "flux2-klein-9b-kv",
        **kwargs: Any,
    ) -> "Flux2ImageEditModel":
        variant = resolve_variant(model)
        if not variant.supports_edit:
            raise ValueError(f"{variant.repo_id} does not support image editing")
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
        pipeline = Flux2ImageEdit.from_pretrained(
            variant,
            model_path=model_path,
            download=kwargs.pop("download", True),
            token=kwargs.pop("token", None),
            revision=kwargs.pop("revision", None),
            force_download=kwargs.pop("force_download", False),
            evict_text_encoder=kwargs.pop("evict_text_encoder", True),
            evict_transformer=kwargs.pop("evict_transformer", False),
            bucketed_seq_len=kwargs.pop("bucketed_seq_len", True),
            tiled_vae=kwargs.pop("tiled_vae", "auto"),
            max_sequence_length=kwargs.pop("max_sequence_length", 512),
        )
        return cls(pipeline=pipeline, model_id=str(model))


def load(model: str = "flux2-klein-4b", **kwargs: Any) -> Flux2ImageGenerationModel:
    return Flux2ImageGenerationModel.from_model_id(model, **kwargs)


def load_edit(model: str = "flux2-klein-9b-kv", **kwargs: Any) -> Flux2ImageEditModel:
    return Flux2ImageEditModel.from_model_id(model, **kwargs)


__all__ = [
    "Flux2ImageEditModel",
    "Flux2ImageGenerationModel",
    "can_edit",
    "can_load",
    "load",
    "load_edit",
    "resolve_variant",
]
