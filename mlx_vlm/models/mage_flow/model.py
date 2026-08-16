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

from .config import MageFlowVariant, get_variant, variant_from_local_path
from .download import validate_model_layout
from .pipeline import MageFlowPipeline


def resolve_variant(model: str | MageFlowVariant | None) -> MageFlowVariant:
    if isinstance(model, MageFlowVariant):
        return model
    if model is None:
        return get_variant()
    path = Path(model).expanduser()
    if path.exists():
        return variant_from_local_path(path)
    return get_variant(model)


def _can_load(model: str, *, task: str) -> bool:
    path = Path(model).expanduser()
    try:
        if path.exists():
            validate_model_layout(path)
        variant = resolve_variant(model)
        return variant.task == task
    except (FileNotFoundError, ValueError):
        return False


@dataclass(slots=True)
class MageFlowImageGenerationModel(ImageGenerationModel):
    is_image_generation_model: ClassVar[bool] = True
    model_type: ClassVar[str] = "mage_flow"
    pipeline: MageFlowPipeline
    model_id: str
    family: str = "mage_flow"

    @property
    def variant(self) -> str:
        return self.pipeline.variant.name

    def generate(self, request: ImageGenerationRequest) -> ImageGenerationResult:
        seed = 0 if request.seed is None else request.seed
        array = self.pipeline.generate_array(
            request.prompt,
            seed=seed,
            steps=request.steps,
            width=request.width,
            height=request.height,
            guidance=request.guidance,
            negative_prompt=request.extra.get("negative_prompt", " "),
            static_shift=float(request.extra.get("static_shift", 6.0)),
            renormalization=bool(request.extra.get("renormalization", False)),
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
            prompt_tokens=self.pipeline.count_prompt_tokens(request.prompt),
            peak_memory=mx.get_peak_memory() / 1e9,
            metadata={
                "model_path": str(self.pipeline.model_path),
                "architecture": "native-resolution-mmdit",
                "default_steps": self.pipeline.variant.default_steps,
                "default_guidance": self.pipeline.variant.default_guidance,
            },
        )

    @classmethod
    def supports_model(cls, model: str) -> bool:
        return _can_load(model, task="generate")

    @classmethod
    def from_model_id(
        cls, model: str = "mage-flow", **kwargs: Any
    ) -> "MageFlowImageGenerationModel":
        variant = resolve_variant(model)
        if not variant.supports_generation:
            raise ValueError(f"{variant.repo_id} is an image-edit checkpoint")
        model_path_arg = kwargs.pop("model_path", None)
        model_path = (
            Path(model).expanduser()
            if model_path_arg is None and Path(model).expanduser().exists()
            else model_path_arg
        )
        pipeline = MageFlowPipeline.from_pretrained(
            variant,
            model_path=model_path,
            download=kwargs.pop("download", True),
            token=kwargs.pop("token", None),
            revision=kwargs.pop("revision", None),
            force_download=kwargs.pop("force_download", False),
            evict_text_encoder=kwargs.pop("evict_text_encoder", True),
            evict_transformer=kwargs.pop("evict_transformer", False),
            max_sequence_length=kwargs.pop("max_sequence_length", 2048),
            sample_posterior=kwargs.pop("sample_posterior", True),
        )
        return cls(pipeline=pipeline, model_id=str(model))


@dataclass(slots=True)
class MageFlowImageEditModel(ImageEditModel):
    is_image_edit_model: ClassVar[bool] = True
    model_type: ClassVar[str] = "mage_flow"
    pipeline: MageFlowPipeline
    model_id: str
    family: str = "mage_flow"

    @property
    def variant(self) -> str:
        return self.pipeline.variant.name

    def edit(self, request: ImageEditRequest) -> ImageGenerationResult:
        seed = 0 if request.seed is None else request.seed
        array = self.pipeline.edit_array(
            request.prompt,
            request.image_paths,
            seed=seed,
            steps=request.steps,
            width=request.width,
            height=request.height,
            guidance=request.guidance,
            negative_prompt=request.extra.get("negative_prompt", " "),
            max_size=request.extra.get("max_size"),
            static_shift=float(request.extra.get("static_shift", 6.0)),
            vl_cond_long_edge=request.extra.get("vl_cond_long_edge", 384),
            renormalization=bool(request.extra.get("renormalization", False)),
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
            prompt_tokens=self.pipeline.count_prompt_tokens(request.prompt, edit=True),
            peak_memory=mx.get_peak_memory() / 1e9,
            metadata={
                "model_path": str(self.pipeline.model_path),
                "architecture": "native-resolution-mmdit",
                "reference_count": len(request.image_paths),
                "default_steps": self.pipeline.variant.default_steps,
                "default_guidance": self.pipeline.variant.default_guidance,
            },
        )

    @classmethod
    def supports_model(cls, model: str) -> bool:
        return _can_load(model, task="edit")

    @classmethod
    def from_model_id(
        cls, model: str = "mage-flow-edit", **kwargs: Any
    ) -> "MageFlowImageEditModel":
        variant = resolve_variant(model)
        if not variant.supports_edit:
            raise ValueError(f"{variant.repo_id} is a text-to-image checkpoint")
        model_path_arg = kwargs.pop("model_path", None)
        model_path = (
            Path(model).expanduser()
            if model_path_arg is None and Path(model).expanduser().exists()
            else model_path_arg
        )
        pipeline = MageFlowPipeline.from_pretrained(
            variant,
            model_path=model_path,
            download=kwargs.pop("download", True),
            token=kwargs.pop("token", None),
            revision=kwargs.pop("revision", None),
            force_download=kwargs.pop("force_download", False),
            evict_text_encoder=kwargs.pop("evict_text_encoder", True),
            evict_transformer=kwargs.pop("evict_transformer", False),
            max_sequence_length=kwargs.pop("max_sequence_length", 2048),
            sample_posterior=kwargs.pop("sample_posterior", True),
        )
        return cls(pipeline=pipeline, model_id=str(model))


def load(model: str = "mage-flow", **kwargs: Any) -> MageFlowImageGenerationModel:
    return MageFlowImageGenerationModel.from_model_id(model, **kwargs)


def load_edit(model: str = "mage-flow-edit", **kwargs: Any) -> MageFlowImageEditModel:
    return MageFlowImageEditModel.from_model_id(model, **kwargs)


__all__ = [
    "MageFlowImageEditModel",
    "MageFlowImageGenerationModel",
    "load",
    "load_edit",
    "resolve_variant",
]
