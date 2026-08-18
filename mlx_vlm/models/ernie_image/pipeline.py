from __future__ import annotations

import gc
import math
from dataclasses import dataclass
from pathlib import Path

import mlx.core as mx
import numpy as np
from PIL import Image

from mlx_vlm.models.flux2.latent import patchify_latents

from .config import ErnieImageVariant, get_variant, validate_dimensions
from .download import download_model, validate_model_layout
from .scheduler import ErnieImageFlowMatchScheduler
from .text_encoder import ErnieImageTokenizer
from .weights import (
    load_prompt_enhancer,
    load_text_encoder,
    load_transformer,
    load_vae,
)


@dataclass(frozen=True, slots=True)
class ErnieImageRuntimeConfig:
    evict_text_encoder: bool = True
    evict_transformer: bool = False
    max_sequence_length: int = 2048
    use_prompt_enhancer: bool | None = None
    prompt_enhancer_max_tokens: int | None = None


def _pad_text(hidden_states: list[mx.array]) -> tuple[mx.array, mx.array]:
    if not hidden_states:
        raise ValueError("At least one text embedding is required")
    lengths = mx.array([item.shape[1] for item in hidden_states], dtype=mx.int32)
    max_length = max(int(item.shape[1]) for item in hidden_states)
    padded = [
        mx.pad(
            item,
            [(0, 0), (0, max_length - item.shape[1]), (0, 0)],
        )
        for item in hidden_states
    ]
    return mx.concatenate(padded, axis=0), lengths


def _to_image_array(decoded: mx.array) -> mx.array:
    images = mx.clip(decoded / 2 + 0.5, 0, 1)
    images = images.transpose(0, 2, 3, 1).astype(mx.float32)
    image = mx.round(images * 255).astype(mx.uint8)[0]
    mx.eval(image)
    return image


def _load_edit_image(
    image: str | Path | Image.Image,
    *,
    width: int | None,
    height: int | None,
) -> tuple[mx.array, int, int]:
    if isinstance(image, Image.Image):
        source = image.convert("RGB")
    else:
        with Image.open(Path(image).expanduser()) as opened:
            source = opened.convert("RGB")
    if width is None and height is None:
        max_scale = min(2048 / source.width, 2048 / source.height)
        scale = min(
            1.0,
            math.sqrt((1024 * 1024) / (source.width * source.height)),
            max_scale,
        )
        min_scale = 256 / min(source.width, source.height)
        if min_scale <= max_scale:
            scale = max(scale, min_scale)
        width = _round_edit_dimension(source.width * scale)
        height = _round_edit_dimension(source.height * scale)
    elif width is None or height is None:
        raise ValueError("width and height must be supplied together for editing")
    validate_dimensions(width=width, height=height)
    source = source.resize((width, height), Image.Resampling.LANCZOS)
    pixels = np.asarray(source, dtype=np.float32) / 127.5 - 1.0
    pixels = np.transpose(pixels, (2, 0, 1))[None, ...]
    return mx.array(pixels), width, height


def _round_edit_dimension(value: float) -> int:
    return max(16, min(2048, int(value) // 16 * 16))


class ErnieImagePipeline:
    def __init__(
        self,
        *,
        variant: str | ErnieImageVariant,
        model_path: str | Path,
        runtime_config: ErnieImageRuntimeConfig | None = None,
    ) -> None:
        self.variant = get_variant(variant)
        self.model_path = validate_model_layout(model_path)
        self.runtime_config = runtime_config or ErnieImageRuntimeConfig()
        self.tokenizer = ErnieImageTokenizer(
            self.model_path, max_length=self.runtime_config.max_sequence_length
        )
        self.text_encoder = load_text_encoder(self.model_path)
        self.component_quantization: dict[str, dict] = {}
        if config := getattr(self.text_encoder, "quantization_config", None):
            self.component_quantization["text_encoder"] = dict(config)
        self.prompt_enhancer = None
        self.transformer = None
        self.vae = None
        self.prompt_cache: dict[str, mx.array] = {}
        self.last_revised_prompt: str | None = None

    @classmethod
    def from_pretrained(
        cls,
        variant: str | ErnieImageVariant = "ernie-image-turbo",
        *,
        model_path: str | Path | None = None,
        download: bool = True,
        token: str | None = None,
        revision: str | None = None,
        force_download: bool = False,
        evict_text_encoder: bool = True,
        evict_transformer: bool = False,
        max_sequence_length: int = 2048,
        use_prompt_enhancer: bool | None = None,
        prompt_enhancer_max_tokens: int | None = None,
    ) -> "ErnieImagePipeline":
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
            runtime_config=ErnieImageRuntimeConfig(
                evict_text_encoder=evict_text_encoder,
                evict_transformer=evict_transformer,
                max_sequence_length=max_sequence_length,
                use_prompt_enhancer=use_prompt_enhancer,
                prompt_enhancer_max_tokens=prompt_enhancer_max_tokens,
            ),
        )

    @property
    def quantization_config(self) -> dict[str, dict] | None:
        components = dict(self.component_quantization)
        for name in ("text_encoder", "transformer", "vae"):
            component = getattr(self, name, None)
            config = getattr(component, "quantization_config", None)
            if config:
                components[name] = dict(config)
        return components or None

    def count_prompt_tokens(self, prompt: str) -> int:
        return self.tokenizer.count_tokens(prompt)

    def _ensure_text_encoder(self):
        if self.text_encoder is None:
            self.text_encoder = load_text_encoder(self.model_path)
        return self.text_encoder

    def _encode_prompt(self, prompt: str) -> mx.array:
        cached = self.prompt_cache.get(prompt)
        if cached is not None:
            return cached
        input_ids = self.tokenizer.encode(prompt)
        hidden_states = self._ensure_text_encoder()(input_ids)
        mx.eval(hidden_states)
        self.prompt_cache[prompt] = hidden_states
        return hidden_states

    def _encode_prompts(self, prompts: list[str]) -> tuple[mx.array, mx.array]:
        hidden_states = [self._encode_prompt(prompt) for prompt in prompts]
        text, lengths = _pad_text(hidden_states)
        mx.eval(text, lengths)
        if self.runtime_config.evict_text_encoder:
            self.text_encoder = None
            gc.collect()
            mx.clear_cache()
        return text, lengths

    def _ensure_components(self, *, require_encoder: bool = False) -> None:
        if self.transformer is None:
            self.transformer = load_transformer(self.model_path)
            if config := getattr(self.transformer, "quantization_config", None):
                self.component_quantization["transformer"] = dict(config)
        if self.vae is None or (
            require_encoder and getattr(self.vae, "encoder", None) is None
        ):
            self.vae = load_vae(
                self.model_path, include_encoder=require_encoder
            )
            if config := getattr(self.vae, "quantization_config", None):
                self.component_quantization["vae"] = dict(config)

    def _should_enhance_prompt(self) -> bool:
        available = (
            bool(list((self.model_path / "pe").glob("*.safetensors")))
            and (self.model_path / "pe_tokenizer" / "tokenizer.json").exists()
        )
        requested = self.runtime_config.use_prompt_enhancer
        if requested is True and not available:
            raise FileNotFoundError(
                "Prompt enhancement was requested but pe/ and pe_tokenizer/ "
                "are not present in the model snapshot"
            )
        return available if requested is None else requested

    def _enhance_prompt(
        self, prompt: str, *, width: int, height: int, seed: int
    ) -> str:
        if self.prompt_enhancer is None:
            self.prompt_enhancer = load_prompt_enhancer(
                self.model_path,
                max_new_tokens=self.runtime_config.prompt_enhancer_max_tokens,
            )
        revised = self.prompt_enhancer.enhance(
            prompt, width=width, height=height, seed=seed
        )
        if self.runtime_config.evict_text_encoder:
            self.prompt_enhancer = None
            gc.collect()
            mx.clear_cache()
        return revised

    def generate(
        self,
        prompt: str,
        *,
        seed: int = 42,
        steps: int | None = None,
        width: int = 1024,
        height: int = 1024,
        guidance: float | None = None,
        negative_prompt: str = "",
    ) -> Image.Image:
        return Image.fromarray(
            np.array(
                self.generate_array(
                    prompt,
                    seed=seed,
                    steps=steps,
                    width=width,
                    height=height,
                    guidance=guidance,
                    negative_prompt=negative_prompt,
                )
            )
        )

    def generate_array(
        self,
        prompt: str,
        *,
        seed: int = 42,
        steps: int | None = None,
        width: int = 1024,
        height: int = 1024,
        guidance: float | None = None,
        negative_prompt: str = "",
    ) -> mx.array:
        if not prompt:
            raise ValueError("prompt must not be empty")
        validate_dimensions(width=width, height=height)
        steps = self.variant.default_steps if steps is None else steps
        guidance = self.variant.default_guidance if guidance is None else guidance
        if steps < 1:
            raise ValueError(f"steps must be >= 1, got {steps}")
        if guidance < 0:
            raise ValueError(f"guidance must be >= 0, got {guidance}")
        if self._should_enhance_prompt():
            prompt = self._enhance_prompt(
                prompt, width=width, height=height, seed=seed
            )
            self.last_revised_prompt = prompt
        else:
            self.last_revised_prompt = None

        do_cfg = guidance > 1.0
        prompts = [negative_prompt, prompt] if do_cfg else [prompt]
        text_hidden_states, text_lengths = self._encode_prompts(prompts)
        self._ensure_components()

        latent_height, latent_width = height // 16, width // 16
        latents = mx.random.normal(
            (1, 128, latent_height, latent_width),
            key=mx.random.key(seed),
            dtype=mx.bfloat16,
        )
        scheduler = ErnieImageFlowMatchScheduler(num_inference_steps=steps)
        latents = self._denoise(
            latents=latents,
            scheduler=scheduler,
            start_index=0,
            text_hidden_states=text_hidden_states,
            text_lengths=text_lengths,
            guidance=guidance,
        )
        return self._decode(latents)

    def edit_array(
        self,
        prompt: str,
        image: str | Path | Image.Image,
        *,
        seed: int = 42,
        steps: int | None = None,
        width: int | None = None,
        height: int | None = None,
        guidance: float | None = None,
        negative_prompt: str = "",
        image_strength: float = 0.6,
    ) -> mx.array:
        if not prompt:
            raise ValueError("prompt must not be empty")
        if not 0.0 < image_strength <= 1.0:
            raise ValueError(
                f"image_strength must be in (0, 1], got {image_strength}"
            )
        pixels, width, height = _load_edit_image(
            image, width=width, height=height
        )
        steps = self.variant.default_steps if steps is None else steps
        guidance = self.variant.default_guidance if guidance is None else guidance
        if steps < 1:
            raise ValueError(f"steps must be >= 1, got {steps}")
        if guidance < 0:
            raise ValueError(f"guidance must be >= 0, got {guidance}")
        if self._should_enhance_prompt():
            prompt = self._enhance_prompt(
                prompt, width=width, height=height, seed=seed
            )
            self.last_revised_prompt = prompt
        else:
            self.last_revised_prompt = None

        do_cfg = guidance > 1.0
        prompts = [negative_prompt, prompt] if do_cfg else [prompt]
        text_hidden_states, text_lengths = self._encode_prompts(prompts)
        self._ensure_components(require_encoder=True)

        source_latents = self.vae.encode(pixels)
        source_latents = patchify_latents(source_latents)
        mean = self.vae.bn.running_mean.reshape(1, -1, 1, 1).astype(
            source_latents.dtype
        )
        std = mx.sqrt(
            self.vae.bn.running_var.reshape(1, -1, 1, 1) + 1e-5
        ).astype(source_latents.dtype)
        source_latents = (source_latents - mean) / std
        noise = mx.random.normal(
            source_latents.shape,
            key=mx.random.key(seed),
            dtype=source_latents.dtype,
        )
        scheduler = ErnieImageFlowMatchScheduler(num_inference_steps=steps)
        denoise_steps = max(1, min(steps, round(steps * image_strength)))
        start_index = steps - denoise_steps
        sigma = scheduler.sigmas[start_index].astype(source_latents.dtype)
        latents = (1.0 - sigma) * source_latents + sigma * noise
        mx.eval(latents)
        latents = self._denoise(
            latents=latents,
            scheduler=scheduler,
            start_index=start_index,
            text_hidden_states=text_hidden_states,
            text_lengths=text_lengths,
            guidance=guidance,
        )
        return self._decode(latents)

    def _denoise(
        self,
        *,
        latents: mx.array,
        scheduler: ErnieImageFlowMatchScheduler,
        start_index: int,
        text_hidden_states: mx.array,
        text_lengths: mx.array,
        guidance: float,
    ) -> mx.array:
        do_cfg = guidance > 1.0
        for index in range(start_index, len(scheduler.timesteps)):
            latent_input = (
                mx.concatenate([latents, latents], axis=0) if do_cfg else latents
            )
            prediction = self.transformer(
                latent_input,
                timestep=mx.full(
                    (latent_input.shape[0],),
                    float(scheduler.timesteps[index].item()),
                    dtype=latents.dtype,
                ),
                text_hidden_states=text_hidden_states,
                text_lengths=text_lengths,
            )
            if do_cfg:
                unconditional, conditional = mx.split(prediction, 2, axis=0)
                prediction = unconditional + guidance * (
                    conditional - unconditional
                )
            latents = scheduler.step(
                model_output=prediction, step_index=index, sample=latents
            )
            mx.eval(latents)
        return latents

    def _decode(self, latents: mx.array) -> mx.array:
        decoded = self.vae.decode_packed_latents(latents)
        mx.eval(decoded)
        if self.runtime_config.evict_transformer:
            self.transformer = None
            self.vae = None
            gc.collect()
            mx.clear_cache()
        return _to_image_array(decoded)


__all__ = [
    "ErnieImagePipeline",
    "ErnieImageRuntimeConfig",
    "_load_edit_image",
    "_pad_text",
]
