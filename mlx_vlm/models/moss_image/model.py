from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import ClassVar

import mlx.core as mx

from ...generate.image import (
    ImageGenerationModel,
    ImageGenerationRequest,
    ImageGenerationResult,
)
from ...generate.image_defaults import ImageSamplingDefaults
from .config import detect_moss_image_layout
from .pipeline import MOSSImagePipeline

_KNOWN_IDS = {
    "openmoss-team/moss-image-sft",
    "openmoss-team/moss-image-base",
    "moss-image",
    "moss_image",
}


@dataclass(slots=True)
class MOSSImageGenerationModel(ImageGenerationModel):
    is_image_generation_model: ClassVar[bool] = True
    model_type: ClassVar[str] = "moss_image"
    pipeline: MOSSImagePipeline
    model_id: str
    family: str = "moss_image"

    @property
    def default_sampling(self):
        return ImageSamplingDefaults(50, 3.0)

    @classmethod
    def resolve_defaults(cls, model, *, model_path=None):
        return ImageSamplingDefaults(50, 3.0)

    @classmethod
    def supports_model(cls, model):
        if Path(model).expanduser().exists():
            return detect_moss_image_layout(model)
        return model.strip().lower().rstrip("/") in _KNOWN_IDS

    @classmethod
    def from_model_id(cls, model, **kwargs):
        model_path = kwargs.pop("model_path", None)
        if model_path is None:
            model_path = Path(model).expanduser()
            if not model_path.is_dir():
                raise FileNotFoundError(
                    f"MOSS-Image requires a local model path: {model}"
                )
        pipeline = MOSSImagePipeline(
            model_path, evict_text_encoder=kwargs.pop("evict_text_encoder", True)
        )
        return cls(pipeline, str(model))

    def generate(self, request: ImageGenerationRequest) -> ImageGenerationResult:
        seed = 0 if request.seed is None else request.seed
        steps = request.resolve_steps(self.default_sampling.steps)
        guidance = request.resolve_guidance(self.default_sampling.guidance)
        options = {
            key: request.extra[key]
            for key in (
                "negative_prompt",
                "num_images",
                "max_sequence_length",
                "uncond_skip_layers",
                "slg_layers",
                "slg_scale",
                "slg_start",
                "slg_stop",
            )
            if key in request.extra
        }
        array = self.pipeline.generate_array(
            request.prompt,
            seed=seed,
            steps=steps,
            width=request.width,
            height=request.height,
            guidance=guidance,
            **options,
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
            prompt_tokens=self.pipeline.count_prompt_tokens(request.prompt),
            peak_memory=mx.get_peak_memory() / 1e9,
            metadata={
                "model_path": str(self.pipeline.model_path),
                "architecture": "moss-image-dit",
            },
        )


def load(model, **kwargs):
    return MOSSImageGenerationModel.from_model_id(model, **kwargs)
