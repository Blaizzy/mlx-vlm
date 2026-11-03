from dataclasses import dataclass
from pathlib import Path
from typing import ClassVar

import mlx.core as mx

from mlx_vlm.generate.edit_image import ImageEditModel, ImageEditRequest
from mlx_vlm.generate.image import (
    ImageGenerationModel,
    ImageGenerationRequest,
    ImageGenerationResult,
)
from mlx_vlm.generate.image_defaults import ImageSamplingDefaults, image_metadata_path

from .config import Flux3Config, repo_id
from .pipeline import Flux3Image


def can_load(model):
    root = Path(model).expanduser()
    if root.is_dir():
        try:
            Flux3Config.from_path(root)
            return True
        except (OSError, ValueError):
            return False
    return repo_id(model) in {
        "black-forest-labs/flux-3-image-klein",
        "black-forest-labs/flux-3-image-klein-base",
    }


class Flux3ModelMixin:
    model_type: ClassVar[str] = "flux3"

    @property
    def default_sampling(self):
        return ImageSamplingDefaults.from_config(self.pipeline.config)

    @classmethod
    def resolve_defaults(cls, model, *, model_path=None):
        root = Path(model).expanduser()
        if model_path is not None or root.is_dir():
            config = Flux3Config.from_path(model_path or root)
        elif can_load(model):
            base = repo_id(model).endswith("-base")
            config = Flux3Config(
                default_steps=50 if base else 16,
                default_guidance=3.5 if base else 1.0,
                flow_map=not base,
            )
        else:
            config = Flux3Config.from_path(image_metadata_path(model))
        return ImageSamplingDefaults.from_config(config)

    @property
    def variant(self):
        return "flux3-klein" if self.pipeline.config.flow_map else "flux3-klein-base"

    def count_prompt_tokens(self, prompt):
        return self.pipeline.tokenizer.count_tokens(prompt)

    @classmethod
    def supports_model(cls, model):
        return can_load(model)

    @classmethod
    def from_model_id(cls, model="black-forest-labs/flux-3-image-klein", **kwargs):
        return cls(
            pipeline=Flux3Image.from_pretrained(repo_id(model), **kwargs),
            model_id=str(model),
        )

    def _result(self, request, array, seed, steps, guidance):
        return ImageGenerationResult(
            array=array,
            seed=seed,
            width=array.shape[-2],
            height=array.shape[-3],
            steps=steps,
            guidance=guidance,
            model=self.model_id,
            family=self.family,
            variant=self.variant,
            prompt_tokens=self.count_prompt_tokens(
                getattr(self.pipeline, "last_prompt_metadata", {}).get(
                    "revised_prompt", request.prompt
                )
            ),
            peak_memory=mx.get_peak_memory() / 1e9,
            metadata={
                "model_path": str(self.pipeline.model_path),
                "architecture": "dense",
                "flow_map": self.pipeline.config.flow_map,
                "reference_count": len(getattr(request, "image_paths", ())),
                **getattr(self.pipeline, "last_prompt_metadata", {}),
            },
        )


@dataclass
class Flux3ImageGenerationModel(Flux3ModelMixin, ImageGenerationModel):
    is_image_generation_model: ClassVar[bool] = True
    pipeline: Flux3Image
    model_id: str
    family: str = "flux3"

    def generate(self, request: ImageGenerationRequest):
        seed = 0 if request.seed is None else request.seed
        steps = request.resolve_steps(self.default_sampling.steps)
        guidance = request.resolve_guidance(self.default_sampling.guidance)
        array = self.pipeline.generate_array(
            request.prompt,
            seed=seed,
            steps=steps,
            width=request.width,
            height=request.height,
            guidance=guidance,
            max_sequence_length=request.extra.get("max_sequence_length"),
            num_images=int(request.extra.get("num_images", 1)),
            upsample_prompt=request.extra.get("upsample_prompt"),
            prompt_expansion_model=request.extra.get("prompt_expansion_model"),
            boxes=request.extra.get("boxes"),
        )
        return self._result(request, array, seed, steps, guidance)


@dataclass
class Flux3ImageEditModel(Flux3ModelMixin, ImageEditModel):
    is_image_edit_model: ClassVar[bool] = True
    pipeline: Flux3Image
    model_id: str
    family: str = "flux3"

    def edit(self, request: ImageEditRequest):
        seed = 0 if request.seed is None else request.seed
        steps = request.resolve_steps(self.default_sampling.steps)
        guidance = request.resolve_guidance(self.default_sampling.guidance)
        array = self.pipeline.edit_array(
            request.prompt,
            request.image_paths,
            seed=seed,
            steps=steps,
            width=request.width,
            height=request.height,
            guidance=guidance,
            max_sequence_length=request.extra.get("max_sequence_length"),
            upsample_prompt=request.extra.get("upsample_prompt"),
            prompt_expansion_model=request.extra.get("prompt_expansion_model"),
            boxes=request.extra.get("boxes"),
        )
        return self._result(request, array, seed, steps, guidance)


def load(model="black-forest-labs/flux-3-image-klein", **kwargs):
    return Flux3ImageGenerationModel.from_model_id(model, **kwargs)


def load_edit(model="black-forest-labs/flux-3-image-klein", **kwargs):
    return Flux3ImageEditModel.from_model_id(model, **kwargs)
