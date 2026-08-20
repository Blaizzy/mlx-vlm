from __future__ import annotations

import gc
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import mlx.core as mx
import numpy as np
from PIL import Image

from .config import MageFlowVariant, get_variant, validate_dimensions
from .download import download_model, validate_model_layout
from .scheduler import FlowMatchEulerDiscreteScheduler
from .text_encoder import EDIT_TEMPLATE, GENERATION_TEMPLATE
from .weights import load_text_encoder, load_transformer, load_vae


@dataclass(frozen=True, slots=True)
class MageFlowRuntimeConfig:
    evict_text_encoder: bool = True
    evict_transformer: bool = False
    max_sequence_length: int = 2048
    sample_posterior: bool = True


def _image_array(decoded: mx.array) -> mx.array:
    decoded = mx.clip(decoded[0], -1.0, 1.0)
    return mx.round(127.5 * (decoded + 1.0)).astype(mx.uint8)


def _load_pil(image: str | Path | Image.Image) -> Image.Image:
    if isinstance(image, Image.Image):
        return image.convert("RGB")
    with Image.open(Path(image).expanduser()) as opened:
        return opened.convert("RGB")


def _resize_reference(image: Image.Image, height: int, width: int) -> mx.array:
    resized = image.convert("RGB").resize((width, height), Image.Resampling.BICUBIC)
    array = np.asarray(resized, dtype=np.float32) / 127.5 - 1.0
    return mx.array(array, dtype=mx.bfloat16)


def _make_divisible_by_16(value: int) -> int:
    return max(16, 16 * (value // 16))


def _edit_dimensions(
    image: Image.Image,
    *,
    width: int | None,
    height: int | None,
    max_size: int | None,
) -> tuple[int, int]:
    if width is not None or height is not None:
        if width is None or height is None:
            raise ValueError("width and height must be supplied together for editing")
        return width, height
    target = max_size or max(image.size)
    if image.height >= image.width:
        height = target
        width = round(image.width * target / image.height)
    else:
        width = target
        height = round(image.height * target / image.width)
    return _make_divisible_by_16(width), _make_divisible_by_16(height)


class MageFlowPipeline:
    def __init__(
        self,
        *,
        variant: str | MageFlowVariant,
        model_path: str | Path,
        runtime_config: MageFlowRuntimeConfig | None = None,
    ) -> None:
        self.variant = get_variant(variant)
        self.model_path = validate_model_layout(model_path)
        self.runtime_config = runtime_config or MageFlowRuntimeConfig()
        self.text_encoder = load_text_encoder(
            self.model_path,
            max_length=self.runtime_config.max_sequence_length,
        )
        self.tokenizer = self.text_encoder.tokenizer
        self.transformer = None
        self.vae = None
        self.prompt_cache: dict[str, mx.array] = {}

    @classmethod
    def from_pretrained(
        cls,
        variant: str | MageFlowVariant = "mage-flow",
        *,
        model_path: str | Path | None = None,
        download: bool = True,
        token: str | None = None,
        revision: str | None = None,
        force_download: bool = False,
        evict_text_encoder: bool = True,
        evict_transformer: bool = False,
        max_sequence_length: int = 2048,
        sample_posterior: bool = True,
    ) -> "MageFlowPipeline":
        spec = get_variant(variant)
        if model_path is None:
            if not download:
                raise FileNotFoundError(
                    f"No local model_path was provided for {spec.repo_id}"
                )
            model_path = download_model(
                spec,
                token=token,
                revision=revision,
                force_download=force_download,
            )
        return cls(
            variant=spec,
            model_path=model_path,
            runtime_config=MageFlowRuntimeConfig(
                evict_text_encoder=evict_text_encoder,
                evict_transformer=evict_transformer,
                max_sequence_length=max_sequence_length,
                sample_posterior=sample_posterior,
            ),
        )

    def count_prompt_tokens(self, prompt: str, *, edit: bool = False) -> int:
        formatted = (EDIT_TEMPLATE if edit else GENERATION_TEMPLATE).format(prompt)
        return len(self.tokenizer(formatted, truncation=False)["input_ids"])

    def _ensure_text_encoder(self):
        if self.text_encoder is None:
            self.text_encoder = load_text_encoder(
                self.model_path,
                max_length=self.runtime_config.max_sequence_length,
            )
        return self.text_encoder

    def _evict_text_encoder(self) -> None:
        if self.runtime_config.evict_text_encoder:
            self.text_encoder = None
            gc.collect()
            mx.clear_cache()

    def _encode_generation_pair(
        self, prompt: str, negative_prompt: str | None
    ) -> tuple[mx.array, mx.array | None]:
        encoder = self._ensure_text_encoder()
        if prompt in self.prompt_cache:
            positive = self.prompt_cache[prompt]
        else:
            positive = encoder.encode(prompt)
            mx.eval(positive)
            self.prompt_cache[prompt] = positive
        negative = None
        if negative_prompt is not None:
            cache_key = f"\0negative\0{negative_prompt}"
            if cache_key in self.prompt_cache:
                negative = self.prompt_cache[cache_key]
            else:
                negative = encoder.encode(negative_prompt)
                mx.eval(negative)
                self.prompt_cache[cache_key] = negative
        del encoder
        self._evict_text_encoder()
        return positive, negative

    def _encode_edit_pair(
        self,
        prompt: str,
        negative_prompt: str | None,
        images: Sequence[Image.Image],
        *,
        vl_cond_long_edge: int | None,
    ) -> tuple[mx.array, mx.array | None]:
        encoder = self._ensure_text_encoder()
        positive = encoder.encode_edit(
            prompt, images, vl_cond_long_edge=vl_cond_long_edge
        )
        negative = (
            encoder.encode_edit(
                negative_prompt,
                images,
                vl_cond_long_edge=vl_cond_long_edge,
            )
            if negative_prompt is not None
            else None
        )
        mx.eval(positive, negative)
        del encoder
        self._evict_text_encoder()
        return positive, negative

    def _ensure_components(self, *, require_encoder: bool) -> None:
        if self.transformer is None:
            self.transformer = load_transformer(self.model_path)
        if self.vae is None or (
            require_encoder and getattr(self.vae, "dconv_encoder", None) is None
        ):
            self.vae = load_vae(self.model_path, include_encoder=require_encoder)

    def _predict(
        self,
        *,
        latents: mx.array,
        text: mx.array,
        sigma: mx.array,
        image_shapes: Sequence[tuple[int, int, int]],
    ) -> mx.array:
        return self.transformer(
            img=latents,
            txt=text,
            timesteps=mx.full((latents.shape[0],), sigma, dtype=latents.dtype),
            img_shapes=image_shapes,
        )

    def _guided_velocity(
        self,
        *,
        latents: mx.array,
        positive: mx.array,
        negative: mx.array | None,
        sigma: mx.array,
        guidance: float,
        image_shapes: Sequence[tuple[int, int, int]],
        renormalization: bool,
    ) -> mx.array:
        conditional = self._predict(
            latents=latents,
            text=positive,
            sigma=sigma,
            image_shapes=image_shapes,
        )
        if negative is None:
            return conditional
        unconditional = self._predict(
            latents=latents,
            text=negative,
            sigma=sigma,
            image_shapes=image_shapes,
        )
        combined = unconditional + guidance * (conditional - unconditional)
        if renormalization:
            cond_norm = mx.linalg.norm(conditional, axis=-1, keepdims=True)
            combined_norm = mx.linalg.norm(combined, axis=-1, keepdims=True)
            combined = combined * cond_norm / (combined_norm + 1e-6)
        return combined

    def generate_array(
        self,
        prompt: str,
        *,
        seed: int = 42,
        steps: int = 20,
        width: int = 1024,
        height: int = 1024,
        guidance: float = 5.0,
        negative_prompt: str = " ",
        static_shift: float = 6.0,
        renormalization: bool = False,
    ) -> mx.array:
        if not self.variant.supports_generation:
            raise ValueError(f"{self.variant.repo_id} is an image-edit checkpoint")
        validate_dimensions(width=width, height=height)
        if not prompt:
            raise ValueError("prompt must not be empty")
        if steps < 1:
            raise ValueError(f"steps must be >= 1, got {steps}")
        positive, negative = self._encode_generation_pair(
            prompt,
            negative_prompt if guidance > 1.0 and negative_prompt else None,
        )
        self._ensure_components(require_encoder=False)
        grid_h, grid_w = height // 16, width // 16
        latents = mx.random.normal(
            (1, grid_h * grid_w, 128),
            key=mx.random.key(seed),
            dtype=mx.bfloat16,
        )
        shapes = [(1, grid_h, grid_w)]
        scheduler = FlowMatchEulerDiscreteScheduler(
            num_inference_steps=steps, shift=static_shift
        )
        for index in range(steps):
            velocity = self._guided_velocity(
                latents=latents,
                positive=positive,
                negative=negative,
                sigma=scheduler.sigmas[index],
                guidance=guidance,
                image_shapes=shapes,
                renormalization=renormalization,
            )
            latents = scheduler.step(
                velocity=velocity, step_index=index, latents=latents
            )
            mx.eval(latents)
        decoded = self.vae.decode(latents.reshape(1, grid_h, grid_w, 128))
        mx.eval(decoded)
        self._evict_after_generation()
        return _image_array(decoded)

    def edit_array(
        self,
        prompt: str,
        image_paths: Sequence[str | Path | Image.Image],
        *,
        seed: int = 42,
        steps: int = 30,
        width: int | None = None,
        height: int | None = None,
        guidance: float = 5.0,
        negative_prompt: str = " ",
        max_size: int | None = None,
        static_shift: float = 6.0,
        vl_cond_long_edge: int | None = 384,
        renormalization: bool = False,
    ) -> mx.array:
        if not self.variant.supports_edit:
            raise ValueError(f"{self.variant.repo_id} is a text-to-image checkpoint")
        if not prompt:
            raise ValueError("prompt must not be empty")
        if steps < 1:
            raise ValueError(f"steps must be >= 1, got {steps}")
        if not image_paths:
            raise ValueError("At least one reference image is required")
        images = [_load_pil(image) for image in image_paths]
        width, height = _edit_dimensions(
            images[0],
            width=width,
            height=height,
            max_size=max_size,
        )
        validate_dimensions(width=width, height=height)
        positive, negative = self._encode_edit_pair(
            prompt,
            (negative_prompt or " ") if guidance > 1.0 else None,
            images,
            vl_cond_long_edge=vl_cond_long_edge,
        )
        self._ensure_components(require_encoder=True)
        references = mx.stack(
            [_resize_reference(image, height, width) for image in images], axis=0
        )
        reference_latents = self.vae.encode(
            references,
            sample_posterior=self.runtime_config.sample_posterior,
            key=mx.random.key(seed),
        )
        mx.eval(reference_latents)
        grid_h, grid_w = height // 16, width // 16
        target = mx.random.normal(
            (1, grid_h * grid_w, 128),
            key=mx.random.key(seed),
            dtype=mx.bfloat16,
        )
        reference_tokens = reference_latents.reshape(
            len(images), grid_h * grid_w, 128
        ).reshape(1, len(images) * grid_h * grid_w, 128)
        target_length = target.shape[1]
        shapes = [(1, grid_h, grid_w)] * (1 + len(images))
        scheduler = FlowMatchEulerDiscreteScheduler(
            num_inference_steps=steps, shift=static_shift
        )
        for index in range(steps):
            combined = mx.concatenate([target, reference_tokens], axis=1)
            velocity = self._guided_velocity(
                latents=combined,
                positive=positive,
                negative=negative,
                sigma=scheduler.sigmas[index],
                guidance=guidance,
                image_shapes=shapes,
                renormalization=renormalization,
            )
            target = scheduler.step(
                velocity=velocity[:, :target_length],
                step_index=index,
                latents=target,
            )
            mx.eval(target)
        decoded = self.vae.decode(target.reshape(1, grid_h, grid_w, 128))
        mx.eval(decoded)
        self._evict_after_generation()
        return _image_array(decoded)

    def _evict_after_generation(self) -> None:
        if self.runtime_config.evict_transformer:
            self.transformer = None
            self.vae = None
            gc.collect()
            mx.clear_cache()


__all__ = ["MageFlowPipeline", "MageFlowRuntimeConfig"]
