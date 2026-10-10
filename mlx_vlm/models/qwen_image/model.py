from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, ClassVar

import mlx.core as mx

from mlx_vlm.generate.edit_image import ImageEditRequest
from mlx_vlm.generate.image import (
    ImageGenerationModel,
    ImageGenerationRequest,
    ImageGenerationResult,
)
from mlx_vlm.generate.image_defaults import ImageSamplingDefaults

from .config import QwenImageVariant, get_variant, variant_from_local_path
from .download import validate_model_layout
from .pipeline import QwenImagePipeline


def resolve_variant(model: str | QwenImageVariant | None) -> QwenImageVariant:
    if isinstance(model, QwenImageVariant):
        return model
    if model is None:
        return get_variant()
    path = Path(model).expanduser()
    if path.exists():
        return variant_from_local_path(path)
    try:
        return get_variant(model)
    except ValueError:
        name = model.rstrip("/").rsplit("/", 1)[-1]
        if name != model:
            return get_variant(name)
        raise


def _can_load(model: str) -> bool:
    path = Path(model).expanduser()
    try:
        if path.exists():
            validate_model_layout(path)
        resolve_variant(model)
        return True
    except (FileNotFoundError, ValueError):
        return False


@dataclass(slots=True)
class QwenImageGenerationModel(ImageGenerationModel):
    is_image_generation_model: ClassVar[bool] = True
    model_type: ClassVar[str] = "qwen_image"
    pipeline: QwenImagePipeline
    model_id: str
    family: str = "qwen_image"

    default_sampling: ClassVar[ImageSamplingDefaults] = ImageSamplingDefaults(30, 1.0)

    @classmethod
    def resolve_defaults(
        cls, model: str, *, model_path: Path | None = None
    ) -> ImageSamplingDefaults:
        return cls.default_sampling

    @property
    def variant(self) -> str:
        return self.pipeline.variant.name

    def generate(self, request: ImageGenerationRequest) -> ImageGenerationResult:
        seed = 0 if request.seed is None else request.seed
        defaults = self.default_sampling
        steps = request.resolve_steps(defaults.steps)
        guidance = request.resolve_guidance(defaults.guidance)
        array = self.pipeline.generate_array(
            request.prompt,
            seed=seed,
            steps=steps,
            width=request.width,
            height=request.height,
            guidance=guidance,
            negative_prompt=request.extra.get("negative_prompt", " "),
            num_images=int(request.extra.get("num_images", 1)),
        )
        metadata = {"model_path": str(self.pipeline.model_path)}
        if self.pipeline.quantization_config:
            metadata["quantization"] = self.pipeline.quantization_config
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
            prompt_tokens=self.pipeline.count_prompt_tokens(request.prompt),
            peak_memory=mx.get_peak_memory() / 1e9,
            metadata=metadata,
        )

    @classmethod
    def supports_model(cls, model: str) -> bool:
        return _can_load(model)

    @classmethod
    def from_model_id(
        cls, model: str = "qwen-image-2.1", **kwargs: Any
    ) -> "QwenImageGenerationModel":
        model_path_arg = kwargs.pop("model_path", None)
        local = Path(model).expanduser()
        model_path = (
            local if model_path_arg is None and local.exists() else model_path_arg
        )
        try:
            variant = resolve_variant(model)
        except ValueError:
            if model_path is None:
                raise
            variant = variant_from_local_path(model_path)
        pipeline = QwenImagePipeline.from_pretrained(
            variant,
            model_path=model_path,
            download=kwargs.pop("download", True),
            token=kwargs.pop("token", None),
            revision=kwargs.pop("revision", None),
            force_download=kwargs.pop("force_download", False),
        )
        return cls(pipeline=pipeline, model_id=str(model))


@dataclass(slots=True)
class QwenImageEditModel(QwenImageGenerationModel):
    is_image_generation_model: ClassVar[bool] = False
    is_image_edit_model: ClassVar[bool] = True

    default_sampling: ClassVar[ImageSamplingDefaults] = ImageSamplingDefaults(40, 1.0)

    @classmethod
    def resolve_defaults(
        cls, model: str, *, model_path: Path | None = None
    ) -> ImageSamplingDefaults:
        return cls.default_sampling

    def edit(self, request: ImageEditRequest) -> ImageGenerationResult:
        seed = 0 if request.seed is None else request.seed
        defaults = self.default_sampling
        steps = request.resolve_steps(defaults.steps)
        guidance = request.resolve_guidance(defaults.guidance)
        array = self.pipeline.edit_array(
            request.prompt,
            request.image_paths,
            seed=seed,
            steps=steps,
            width=request.width,
            height=request.height,
            guidance=guidance,
            negative_prompt=request.extra.get("negative_prompt", " "),
            output_resolution=request.extra.get("output_resolution", 1024),
            use_kv_cache=request.extra.get("use_kv_cache", True),
        )
        metadata = {
            "model_path": str(self.pipeline.model_path),
            "reference_count": len(request.image_paths),
        }
        if self.pipeline.quantization_config:
            metadata["quantization"] = self.pipeline.quantization_config
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
            color_space="RGBA",
            prompt_tokens=self.pipeline.count_prompt_tokens(request.prompt),
            peak_memory=mx.get_peak_memory() / 1e9,
            metadata=metadata,
        )


__all__ = ["QwenImageEditModel", "QwenImageGenerationModel"]
