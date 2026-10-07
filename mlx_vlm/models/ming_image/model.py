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
from mlx_vlm.generate.image_defaults import ImageSamplingDefaults, image_metadata_path

from .config import MingImageConfig, detect_ming_image_layout
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

    @property
    def default_sampling(self) -> ImageSamplingDefaults:
        return ImageSamplingDefaults.from_config(self.pipeline.config)

    @classmethod
    def resolve_defaults(
        cls, model: str, *, model_path: Path | None = None
    ) -> ImageSamplingDefaults:
        config = MingImageConfig.from_model_path(image_metadata_path(model, model_path))
        return ImageSamplingDefaults.from_config(config)

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
            num_images=int(request.extra.get("num_images", 1)),
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
        if model_path is None:
            path = Path(model).expanduser()
            if not path.exists():
                raise FileNotFoundError(
                    f"Ming-Image requires a local model path. Got: {model}"
                )
            model_path = path
        pipeline = MingImagePipeline(
            model_path, evict_text_encoder=kwargs.pop("evict_text_encoder", True)
        )
        return cls(pipeline=pipeline, model_id=str(model))


def load(model: str, **kwargs: Any) -> MingImageGenerationModel:
    return MingImageGenerationModel.from_model_id(model, **kwargs)


__all__ = ["MingImageGenerationModel", "load"]
