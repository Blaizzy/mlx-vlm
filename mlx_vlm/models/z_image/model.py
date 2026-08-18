"""Z-Image model implementing the ImageGenerationModel protocol."""

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

from .config import detect_z_image_layout
from .pipeline import ZImagePipeline

# Known model IDs / aliases
_KNOWN_IDS = {
    "tongyi-mai/z-image",
    "tongyi-mai/z-image-turbo",
    "z-image",
    "z-image-turbo",
    "z_image",
}


@dataclass(slots=True)
class ZImageGenerationModel(ImageGenerationModel):
    is_image_generation_model: ClassVar[bool] = True
    model_type: ClassVar[str] = "z_image"
    pipeline: ZImagePipeline
    model_id: str
    family: str = "z_image"

    @property
    def variant(self) -> str | None:
        return self.pipeline.config.variant

    def generate(self, request: ImageGenerationRequest) -> ImageGenerationResult:
        seed = 0 if request.seed is None else request.seed
        steps = request.resolve_steps(self.pipeline.config.default_steps)
        guidance = request.resolve_guidance(self.pipeline.config.default_guidance)
        if self.variant == "turbo" and not 0.0 <= guidance <= 1.0:
            raise ValueError(
                "Z-Image Turbo does not support classifier-free guidance; "
                "use a guidance value between 0 and 1 to keep it disabled"
            )
        array = self.pipeline.generate_array(
            request.prompt,
            seed=seed,
            steps=steps,
            width=request.width,
            height=request.height,
            guidance=guidance,
            negative_prompt=request.extra.get("negative_prompt"),
            cfg_truncation=float(request.extra.get("cfg_truncation", 1.0)),
        )
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
            metadata={
                "model_path": str(self.pipeline.model_path),
                "architecture": "z-image-dit",
                "guidance_mode": ("classifier-free" if guidance > 1.0 else "disabled"),
            },
        )

    @classmethod
    def supports_model(cls, model: str) -> bool:
        # Check if local path with Z-Image layout
        path = Path(model).expanduser()
        if path.exists():
            return detect_z_image_layout(path)
        # Check known aliases
        return model.strip().lower().rstrip("/") in _KNOWN_IDS

    @classmethod
    def from_model_id(cls, model: str, **kwargs: Any) -> ZImageGenerationModel:
        model_path = kwargs.pop("model_path", None)
        if model_path is None:
            path = Path(model).expanduser()
            if path.exists():
                model_path = path
            else:
                raise FileNotFoundError(
                    f"Z-Image requires a local model path. Got: {model}"
                )
        pipeline = ZImagePipeline(
            model_path,
            evict_text_encoder=kwargs.pop("evict_text_encoder", True),
        )
        return cls(pipeline=pipeline, model_id=str(model))


@dataclass(slots=True)
class ZImageEditModel(ImageEditModel):
    is_image_edit_model: ClassVar[bool] = True
    model_type: ClassVar[str] = "z_image"
    pipeline: ZImagePipeline
    model_id: str
    family: str = "z_image"

    @property
    def variant(self) -> str | None:
        return self.pipeline.config.variant

    def edit(self, request: ImageEditRequest) -> ImageGenerationResult:
        seed = 0 if request.seed is None else request.seed
        steps = request.resolve_steps(8 if self.variant == "turbo" else 50)
        guidance = request.resolve_guidance(self.pipeline.config.default_guidance)
        if self.variant == "turbo" and not 0.0 <= guidance <= 1.0:
            raise ValueError(
                "Z-Image Turbo does not support classifier-free guidance; "
                "use a guidance value between 0 and 1 to keep it disabled"
            )
        array = self.pipeline.edit_array(
            request.prompt,
            request.image_paths,
            seed=seed,
            steps=steps,
            width=request.width,
            height=request.height,
            guidance=guidance,
            negative_prompt=request.extra.get("negative_prompt"),
            cfg_truncation=float(request.extra.get("cfg_truncation", 1.0)),
            strength=float(request.extra.get("strength", 0.6)),
        )
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
            prompt_tokens=self.pipeline.count_prompt_tokens(request.prompt),
            peak_memory=mx.get_peak_memory() / 1e9,
            metadata={
                "model_path": str(self.pipeline.model_path),
                "architecture": "z-image-img2img",
                "reference_count": 1,
                "strength": float(request.extra.get("strength", 0.6)),
            },
        )

    @classmethod
    def supports_model(cls, model: str) -> bool:
        return ZImageGenerationModel.supports_model(model)

    @classmethod
    def from_model_id(cls, model: str, **kwargs: Any) -> ZImageEditModel:
        generation_model = ZImageGenerationModel.from_model_id(model, **kwargs)
        return cls(
            pipeline=generation_model.pipeline, model_id=generation_model.model_id
        )


def load(model: str, **kwargs: Any) -> ZImageGenerationModel:
    return ZImageGenerationModel.from_model_id(model, **kwargs)


def load_edit(model: str, **kwargs: Any) -> ZImageEditModel:
    return ZImageEditModel.from_model_id(model, **kwargs)


__all__ = ["ZImageEditModel", "ZImageGenerationModel", "load", "load_edit"]
