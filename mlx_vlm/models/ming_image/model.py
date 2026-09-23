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

from .config import detect_ming_image_layout
from .pipeline import MingImagePipeline

_KNOWN_IDS = {
    "inclusionai/ming-image-0.1-design",
    "ming-image-0.1-design",
    "ming-image",
    "ming_image",
}


@dataclass(slots=True)
class MingImageGenerationModel(ImageGenerationModel):
    is_image_generation_model: ClassVar[bool] = True
    model_type: ClassVar[str] = "ming_image"
    pipeline: MingImagePipeline
    model_id: str
    family: str = "ming_image"

    def generate(self, request: ImageGenerationRequest) -> ImageGenerationResult:
        seed = 0 if request.seed is None else request.seed
        steps = request.resolve_steps(self.pipeline.config.default_steps)
        guidance = request.resolve_guidance(self.pipeline.config.default_guidance)
        array = self.pipeline.generate_array(
            request.prompt,
            seed=seed,
            steps=steps,
            width=request.width,
            height=request.height,
            guidance=guidance,
        )
        return ImageGenerationResult(
            array=array,
            seed=seed,
            width=request.width,
            height=request.height,
            steps=steps,
            model=self.model_id,
            family=self.family,
            guidance=guidance,
            color_space="RGBA",
            prompt_tokens=self.pipeline.count_prompt_tokens(request.prompt),
            peak_memory=mx.get_peak_memory() / 1e9,
            metadata={
                "model_path": str(self.pipeline.model_path),
                "architecture": "ming-image-nextdit",
            },
        )

    @classmethod
    def supports_model(cls, model: str) -> bool:
        path = Path(model).expanduser()
        if path.exists():
            return detect_ming_image_layout(path)
        return model.strip().lower().rstrip("/") in _KNOWN_IDS

    @classmethod
    def from_model_id(cls, model: str, **kwargs: Any) -> "MingImageGenerationModel":
        model_path = kwargs.pop("model_path", None)
        local = Path(model).expanduser()
        if model_path is None and local.exists():
            model_path = local
        pipeline = MingImagePipeline.from_pretrained(
            model_path=model_path,
            repo_id=model if "/" in str(model) else None,
            download=kwargs.pop("download", True),
            token=kwargs.pop("token", None),
            revision=kwargs.pop("revision", None),
            force_download=kwargs.pop("force_download", False),
            evict_text_encoder=kwargs.pop("evict_text_encoder", True),
        )
        return cls(pipeline=pipeline, model_id=str(model))


def load(model: str, **kwargs: Any) -> MingImageGenerationModel:
    return MingImageGenerationModel.from_model_id(model, **kwargs)


__all__ = ["MingImageGenerationModel", "load"]
