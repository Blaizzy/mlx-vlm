from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import ClassVar

import mlx.core as mx

from mlx_vlm.generate.edit_image import ImageEditRequest
from mlx_vlm.generate.image import (
    ImageGenerationModel,
    ImageGenerationRequest,
    ImageGenerationResult,
)
from mlx_vlm.generate.image_defaults import ImageSamplingDefaults, image_metadata_path

from .config import read_config
from .pipeline import LLaDAImagePipeline
from .scheduler import LLaDAImageScheduler


@dataclass
class LLaDAImageGenerationModel(ImageGenerationModel):
    is_image_generation_model: ClassVar[bool] = True
    is_image_edit_model: ClassVar[bool] = True
    model_type: ClassVar[str] = "llada_image"
    family: ClassVar[str] = "llada_image"
    pipeline: LLaDAImagePipeline
    model_id: str

    @property
    def default_sampling(self) -> ImageSamplingDefaults:
        return ImageSamplingDefaults.from_config(self.pipeline.scheduler)

    @classmethod
    def resolve_defaults(
        cls, model: str, *, model_path: Path | None = None
    ) -> ImageSamplingDefaults:
        path = image_metadata_path(model, model_path)
        scheduler = LLaDAImageScheduler(
            read_config(path / "scheduler/scheduler_config.json")
        )
        return ImageSamplingDefaults.from_config(scheduler)

    @property
    def variant(self) -> str:
        return (
            "llada-image-turbo"
            if self.pipeline.scheduler.use_uniform_sigmas
            else "llada-image"
        )

    @property
    def default_steps(self) -> int:
        return self.pipeline.scheduler.default_steps

    @property
    def default_guidance(self) -> float:
        return self.pipeline.scheduler.default_guidance

    def generate(self, request: ImageGenerationRequest) -> ImageGenerationResult:
        mode = request.extra.get("generation_mode", "text")
        if mode not in {"text", "vq"}:
            raise ValueError(
                "Generation supports text or vq mode; use edit_image for editing"
            )
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
            negative_prompt=request.extra.get("negative_prompt"),
            stochastic_sampling=request.extra.get("stochastic_sampling"),
            generation_mode=mode,
        )
        return self._result(request, array, seed, steps, guidance, mode)

    def edit(self, request: ImageEditRequest) -> ImageGenerationResult:
        if len(request.image_paths) != 1:
            raise ValueError("LLaDA-Image editing requires exactly one source image")
        if request.extra.get("generation_mode", "editing") != "editing":
            raise ValueError("Image editing requires generation_mode='editing'")
        seed = 0 if request.seed is None else request.seed
        defaults = self.default_sampling
        steps = request.resolve_steps(defaults.steps)
        guidance = request.resolve_guidance(defaults.guidance)
        array = self.pipeline.edit_array(
            request.prompt,
            request.image_paths[0],
            seed=seed,
            steps=steps,
            width=request.width,
            height=request.height,
            guidance=guidance,
            negative_prompt=request.extra.get("negative_prompt"),
            stochastic_sampling=request.extra.get("stochastic_sampling"),
        )
        return self._result(request, array, seed, steps, guidance, "editing")

    def _result(self, request, array, seed, steps, guidance, mode):
        return ImageGenerationResult(
            array=array,
            seed=seed,
            steps=steps,
            width=array.shape[1],
            height=array.shape[0],
            model=self.model_id,
            family=self.family,
            variant=self.variant,
            guidance=guidance,
            prompt_tokens=len(self.pipeline.tokenize(request.prompt)),
            peak_memory=mx.get_peak_memory() / 1e9,
            metadata={
                "model_path": str(self.pipeline.model_path),
                "generation_mode": mode,
            },
        )

    @classmethod
    def supports_model(cls, model: str) -> bool:
        try:
            return (
                read_config(Path(model).expanduser() / "model_index.json").get(
                    "_class_name"
                )
                == "LLaDAImagePipeline"
            )
        except (OSError, ValueError):
            return False

    @classmethod
    def from_model_id(cls, model: str = "inclusionAI/LLaDA-Image-Turbo", **kwargs):
        path = kwargs.pop("model_path", None)
        if path is None:
            path = Path(model).expanduser()
            if not path.exists():
                from huggingface_hub import snapshot_download

                path = snapshot_download(
                    model,
                    token=kwargs.pop("token", None),
                    revision=kwargs.pop("revision", None),
                    force_download=kwargs.pop("force_download", False),
                    local_files_only=not kwargs.pop("download", True),
                    allow_patterns=[
                        "model_index.json",
                        "scheduler/*.json",
                        "tokenizer/*",
                        "text_encoder/*.json",
                        "text_encoder/*.safetensors",
                        "queryformer/*",
                        "text_projection/*",
                        "sigvq/*",
                        "transformer/*",
                        "vae/*",
                    ],
                )
        pipeline = LLaDAImagePipeline(
            path,
            evict_text_encoder=kwargs.pop("evict_text_encoder", True),
            evict_transformer=kwargs.pop("evict_transformer", True),
            max_sequence_length=kwargs.pop("max_sequence_length", 2048),
            prompt_cache_size=kwargs.pop("prompt_cache_size", 2),
        )
        return cls(pipeline=pipeline, model_id=str(model))


def load(model: str = "inclusionAI/LLaDA-Image-Turbo", **kwargs):
    return LLaDAImageGenerationModel.from_model_id(model, **kwargs)


def load_edit(model: str = "inclusionAI/LLaDA-Image-Turbo", **kwargs):
    return LLaDAImageGenerationModel.from_model_id(model, **kwargs)
