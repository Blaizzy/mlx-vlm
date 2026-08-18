"""Z-Image model implementing the ImageGenerationModel protocol."""

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

from .config import ZImageConfig, detect_z_image_layout
from .pipeline import ZImagePipeline

# Known model IDs / aliases
_KNOWN_IDS = {
    "tongyi-mai/z-image-turbo",
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
        return "turbo"

    def generate(self, request: ImageGenerationRequest) -> ImageGenerationResult:
        seed = 0 if request.seed is None else request.seed
        steps = request.steps or 9
        array = self.pipeline.generate_array(
            request.prompt,
            seed=seed,
            steps=steps,
            width=request.width,
            height=request.height,
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
            guidance=request.guidance,
            prompt_tokens=self.pipeline.count_prompt_tokens(request.prompt),
            peak_memory=mx.get_peak_memory() / 1e9,
            metadata={
                "model_path": str(self.pipeline.model_path),
                "architecture": "z-image-dit",
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
    def from_model_id(cls, model: str, **kwargs: Any) -> "ZImageGenerationModel":
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


def load(model: str, **kwargs: Any) -> ZImageGenerationModel:
    return ZImageGenerationModel.from_model_id(model, **kwargs)


__all__ = ["ZImageGenerationModel", "load"]
