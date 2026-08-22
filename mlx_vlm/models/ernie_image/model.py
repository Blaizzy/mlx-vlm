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

from .config import ErnieImageVariant, get_variant, variant_from_local_path
from .download import validate_model_layout
from .pipeline import ErnieImagePipeline


def resolve_variant(
    model: str | ErnieImageVariant | None,
) -> ErnieImageVariant:
    if isinstance(model, ErnieImageVariant):
        return model
    if model is None:
        return get_variant()
    path = Path(model).expanduser()
    if path.exists():
        return variant_from_local_path(path)
    return get_variant(model)


@dataclass(slots=True)
class ErnieImageGenerationModel(ImageGenerationModel):
    is_image_generation_model: ClassVar[bool] = True
    model_type: ClassVar[str] = "ernie_image"
    pipeline: ErnieImagePipeline
    model_id: str
    family: str = "ernie_image"

    @property
    def variant(self) -> str:
        return self.pipeline.variant.name

    @property
    def default_steps(self) -> int:
        return self.pipeline.variant.default_steps

    @property
    def default_guidance(self) -> float:
        return self.pipeline.variant.default_guidance

    def generate(self, request: ImageGenerationRequest) -> ImageGenerationResult:
        seed = 0 if request.seed is None else request.seed
        steps = self.default_steps if request.steps is None else request.steps
        guidance = (
            self.default_guidance if request.guidance is None else request.guidance
        )
        array = self.pipeline.generate_array(
            request.prompt,
            seed=seed,
            steps=steps,
            width=request.width,
            height=request.height,
            guidance=guidance,
            negative_prompt=str(request.extra.get("negative_prompt", "")),
        )
        metadata = {
            "model_path": str(self.pipeline.model_path),
            "architecture": "single-stream-dit",
            "default_steps": self.default_steps,
            "default_guidance": self.default_guidance,
            "classifier_free_guidance": guidance > 1.0,
            "prompt_enhancement": self.pipeline._should_enhance_prompt(),
        }
        if quantization := self.pipeline.quantization_config:
            metadata["quantization"] = quantization
        if self.pipeline.last_revised_prompt is not None:
            metadata["revised_prompt"] = self.pipeline.last_revised_prompt
        return ImageGenerationResult(
            array=array,
            seed=seed,
            width=request.width,
            height=request.height,
            steps=steps,
            model=self.model_id,
            family=self.family,
            variant=self.variant,
            guidance=guidance,
            prompt_tokens=self.pipeline.count_prompt_tokens(
                self.pipeline.last_revised_prompt or request.prompt
            ),
            peak_memory=mx.get_peak_memory() / 1e9,
            metadata=metadata,
        )

    @classmethod
    def supports_model(cls, model: str) -> bool:
        path = Path(model).expanduser()
        try:
            if path.exists():
                validate_model_layout(path)
                variant_from_local_path(path)
            else:
                get_variant(model)
            return True
        except (FileNotFoundError, ValueError):
            return False

    @classmethod
    def from_model_id(
        cls,
        model: str = "ernie-image-turbo",
        **kwargs: Any,
    ) -> "ErnieImageGenerationModel":
        model_path_arg = kwargs.pop("model_path", None)
        model_path = (
            Path(model).expanduser()
            if model_path_arg is None and Path(model).expanduser().exists()
            else model_path_arg
        )
        variant = resolve_variant(model_path if model_path is not None else model)
        pipeline = ErnieImagePipeline.from_pretrained(
            variant,
            model_path=model_path,
            download=kwargs.pop("download", True),
            token=kwargs.pop("token", None),
            revision=kwargs.pop("revision", None),
            force_download=kwargs.pop("force_download", False),
            evict_text_encoder=kwargs.pop("evict_text_encoder", True),
            evict_transformer=kwargs.pop("evict_transformer", False),
            max_sequence_length=kwargs.pop("max_sequence_length", 2048),
            prompt_cache_size=kwargs.pop("prompt_cache_size", 2),
            use_prompt_enhancer=kwargs.pop("use_prompt_enhancer", None),
            prompt_enhancer_max_tokens=kwargs.pop("prompt_enhancer_max_tokens", None),
        )
        return cls(pipeline=pipeline, model_id=str(model))


@dataclass(slots=True)
class ErnieImageEditModel(ImageEditModel):
    is_image_edit_model: ClassVar[bool] = True
    model_type: ClassVar[str] = "ernie_image"
    pipeline: ErnieImagePipeline
    model_id: str
    family: str = "ernie_image"

    @property
    def variant(self) -> str:
        return self.pipeline.variant.name

    @property
    def default_steps(self) -> int:
        return self.pipeline.variant.default_steps

    @property
    def default_guidance(self) -> float:
        # Editing is generic SDEdit rather than native instruction conditioning,
        # so it needs classifier-free guidance to actually apply the requested
        # change. Turbo ships at guidance 1.0 for generation (no CFG), which
        # barely edits; the generation recommendation is left untouched and only
        # this edit default is raised.
        return self.pipeline.variant.edit_default_guidance

    def edit(self, request: ImageEditRequest) -> ImageGenerationResult:
        if len(request.image_paths) != 1:
            raise ValueError("ERNIE-Image img2img accepts exactly one source image")
        seed = 0 if request.seed is None else request.seed
        steps = request.steps or self.default_steps
        guidance = (
            self.default_guidance if request.guidance is None else request.guidance
        )
        # ``image_strength`` is the canonical key; ``strength`` is accepted as
        # an alias so the edit command lines up with the z_image model and
        # matches the naming used in the documentation.
        raw_strength = request.extra.get(
            "image_strength", request.extra.get("strength", 0.6)
        )
        image_strength = float(raw_strength)
        array = self.pipeline.edit_array(
            request.prompt,
            request.image_paths[0],
            seed=seed,
            steps=steps,
            width=request.width,
            height=request.height,
            guidance=guidance,
            negative_prompt=str(request.extra.get("negative_prompt", "")),
            image_strength=image_strength,
        )
        metadata = {
            "model_path": str(self.pipeline.model_path),
            "architecture": "single-stream-dit-img2img",
            "classifier_free_guidance": guidance > 1.0,
            "image_strength": image_strength,
            "prompt_enhancement": self.pipeline._should_enhance_prompt(for_edit=True),
            "native_instruction_edit": False,
            "reference_count": 1,
        }
        if quantization := self.pipeline.quantization_config:
            metadata["quantization"] = quantization
        if self.pipeline.last_revised_prompt is not None:
            metadata["revised_prompt"] = self.pipeline.last_revised_prompt
        return ImageGenerationResult(
            array=array,
            seed=seed,
            width=array.shape[1],
            height=array.shape[0],
            steps=steps,
            model=self.model_id,
            family=self.family,
            variant=self.variant,
            guidance=guidance,
            prompt_tokens=self.pipeline.count_prompt_tokens(
                self.pipeline.last_revised_prompt or request.prompt
            ),
            peak_memory=mx.get_peak_memory() / 1e9,
            metadata=metadata,
        )

    @classmethod
    def supports_model(cls, model: str) -> bool:
        return ErnieImageGenerationModel.supports_model(model)

    @classmethod
    def from_model_id(
        cls,
        model: str = "ernie-image-turbo",
        **kwargs: Any,
    ) -> "ErnieImageEditModel":
        # The prompt enhancer is a text-to-image feature: it rewrites a short
        # prompt into a standalone scene caption and never sees the source
        # image, so it dilutes edit intent. Default it off for the edit task;
        # callers can still opt back in with ``use_prompt_enhancer=True``.
        kwargs.setdefault("use_prompt_enhancer", False)
        generation = ErnieImageGenerationModel.from_model_id(model, **kwargs)
        return cls(pipeline=generation.pipeline, model_id=generation.model_id)


def load(model: str = "ernie-image-turbo", **kwargs: Any) -> ErnieImageGenerationModel:
    return ErnieImageGenerationModel.from_model_id(model, **kwargs)


def load_edit(model: str = "ernie-image-turbo", **kwargs: Any) -> ErnieImageEditModel:
    return ErnieImageEditModel.from_model_id(model, **kwargs)


__all__ = [
    "ErnieImageEditModel",
    "ErnieImageGenerationModel",
    "load",
    "load_edit",
    "resolve_variant",
]
