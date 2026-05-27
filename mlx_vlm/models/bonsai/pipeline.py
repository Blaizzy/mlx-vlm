from __future__ import annotations

import gc
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import mlx.core as mx
import numpy as np
from PIL import Image

from mlx_vlm.models.bonsai.config import (
    BonsaiVariant,
    default_model_path,
    get_variant,
    validate_dimensions,
)
from mlx_vlm.models.bonsai.download import download_model, validate_model_layout
from mlx_vlm.models.bonsai.klein_fast.blocks import (
    DEFAULT_QUANT_GROUP_SIZE,
    _require_native_quantized_matmul,
)
from mlx_vlm.models.bonsai.latent import prepare_packed_latents
from mlx_vlm.models.bonsai.prompt import encode_prompt
from mlx_vlm.models.bonsai.scheduler import FlowMatchEulerDiscreteScheduler
from mlx_vlm.models.bonsai.tiling import TilingConfig
from mlx_vlm.models.bonsai.tokenizer import BonsaiTokenizer
from mlx_vlm.models.bonsai.weights import (
    load_text_encoder_4bit,
    load_transformer,
    load_vae,
)

TiledVAE = Literal["auto", "on", "off"]


@dataclass(frozen=True, slots=True)
class BonsaiRuntimeConfig:
    evict_text_encoder: bool = True
    evict_transformer: bool = False
    bucketed_seq_len: bool = True
    tiled_vae: TiledVAE = "auto"
    max_sequence_length: int = 512


class BonsaiImage:
    def __init__(
        self,
        *,
        variant: str | BonsaiVariant = "ternary",
        model_path: str | Path,
        runtime_config: BonsaiRuntimeConfig | None = None,
    ) -> None:
        self.variant = get_variant(variant)
        self.model_path = validate_model_layout(model_path)
        self.runtime_config = runtime_config or BonsaiRuntimeConfig()
        self.tokenizer = BonsaiTokenizer(self.model_path)
        self.text_encoder = load_text_encoder_4bit(self.model_path)
        self.transformer = None
        self.vae = None
        self.prompt_cache: dict[tuple[str, int, bool], tuple[mx.array, mx.array]] = {}
        _check_quantized_matmul(self.variant.precision)

    @classmethod
    def from_pretrained(
        cls,
        variant: str | BonsaiVariant = "ternary",
        *,
        model_path: str | Path | None = None,
        models_dir: str | Path | None = None,
        download: bool = True,
        token: str | None = None,
        evict_text_encoder: bool = True,
        evict_transformer: bool = False,
        bucketed_seq_len: bool = True,
        tiled_vae: TiledVAE = "auto",
        max_sequence_length: int = 512,
    ) -> "BonsaiImage":
        spec = get_variant(variant)
        if model_path is None:
            if download:
                model_path = download_model(spec, models_dir=models_dir, token=token)
            else:
                model_path = default_model_path(spec, models_dir)
        config = BonsaiRuntimeConfig(
            evict_text_encoder=evict_text_encoder,
            evict_transformer=evict_transformer,
            bucketed_seq_len=bucketed_seq_len,
            tiled_vae=tiled_vae,
            max_sequence_length=max_sequence_length,
        )
        return cls(variant=spec, model_path=model_path, runtime_config=config)

    def generate(
        self,
        prompt: str,
        *,
        seed: int = 42,
        steps: int = 4,
        width: int = 512,
        height: int = 512,
        guidance: float = 1.0,
        max_sequence_length: int | None = None,
        tiled_vae: bool | None = None,
    ) -> Image.Image:
        return _to_pil(
            self.generate_array(
                prompt,
                seed=seed,
                steps=steps,
                width=width,
                height=height,
                guidance=guidance,
                max_sequence_length=max_sequence_length,
                tiled_vae=tiled_vae,
            )
        )

    def generate_array(
        self,
        prompt: str,
        *,
        seed: int = 42,
        steps: int = 4,
        width: int = 512,
        height: int = 512,
        guidance: float = 1.0,
        max_sequence_length: int | None = None,
        tiled_vae: bool | None = None,
    ) -> mx.array:
        validate_dimensions(width=width, height=height)
        if steps < 1:
            raise ValueError(f"steps must be >= 1, got {steps}")
        if not prompt:
            raise ValueError("prompt must not be empty")

        max_seq = max_sequence_length or self.runtime_config.max_sequence_length
        prompt_embeds, text_ids = self._encode_prompt(
            prompt, max_sequence_length=max_seq
        )
        negative_prompt_embeds = None
        negative_text_ids = None
        if guidance is not None and guidance > 1.0:
            negative_prompt_embeds, negative_text_ids = self._encode_prompt(
                " ", max_sequence_length=max_seq
            )
        self._ensure_transformer_and_vae()

        latents, latent_ids, latent_height, latent_width = prepare_packed_latents(
            seed=seed,
            height=height,
            width=width,
        )
        scheduler = FlowMatchEulerDiscreteScheduler(
            image_seq_len=(height // 16) * (width // 16),
            num_inference_steps=steps,
        )
        predict = self._predict
        for i in range(steps):
            noise = predict(
                latents=latents,
                latent_ids=latent_ids,
                prompt_embeds=prompt_embeds,
                text_ids=text_ids,
                guidance=guidance,
                timestep=scheduler.timesteps[i],
            )
            if negative_prompt_embeds is not None and negative_text_ids is not None:
                negative_noise = predict(
                    latents=latents,
                    latent_ids=latent_ids,
                    prompt_embeds=negative_prompt_embeds,
                    text_ids=negative_text_ids,
                    guidance=guidance,
                    timestep=scheduler.timesteps[i],
                )
                noise = negative_noise + guidance * (noise - negative_noise)
            latents = scheduler.step(noise=noise, step_index=i, latents=latents)
            mx.eval(latents)

        packed = latents.reshape(
            latents.shape[0], latent_height, latent_width, latents.shape[-1]
        )
        packed = packed.transpose(0, 3, 1, 2)
        decoded = self.vae.decode_packed_latents(  # type: ignore[union-attr]
            packed,
            tiling_config=self._resolve_tiling(
                height=height, width=width, override=tiled_vae
            ),
        )
        if self.runtime_config.evict_transformer:
            self.transformer = None
            self.vae = None
            gc.collect()
            mx.clear_cache()
        return _to_image_array(decoded)

    def _encode_prompt(
        self, prompt: str, *, max_sequence_length: int
    ) -> tuple[mx.array, mx.array]:
        key = (prompt, max_sequence_length, self.runtime_config.bucketed_seq_len)
        cached = self.prompt_cache.get(key)
        if cached is not None:
            return cached
        if self.text_encoder is None:
            self.text_encoder = load_text_encoder_4bit(self.model_path)
        embeds, text_ids = encode_prompt(
            prompt=prompt,
            tokenizer=self.tokenizer,
            text_encoder=self.text_encoder,
            max_sequence_length=max_sequence_length,
            bucketed=self.runtime_config.bucketed_seq_len,
        )
        mx.eval(embeds, text_ids)
        self.prompt_cache[key] = (embeds, text_ids)
        if self.runtime_config.evict_text_encoder:
            self.text_encoder = None
            gc.collect()
            mx.clear_cache()
        return embeds, text_ids

    def _ensure_transformer_and_vae(self) -> None:
        if self.transformer is None:
            self.transformer = load_transformer(self.model_path, self.variant.precision)
        if self.vae is None:
            self.vae = load_vae()

    def _predict(
        self,
        *,
        latents: mx.array,
        latent_ids: mx.array,
        prompt_embeds: mx.array,
        text_ids: mx.array,
        guidance: float,
        timestep: mx.array,
    ) -> mx.array:
        return self.transformer(  # type: ignore[operator]
            hidden_states=latents,
            encoder_hidden_states=prompt_embeds,
            timestep=timestep,
            img_ids=latent_ids,
            txt_ids=text_ids,
            guidance=None,
        )

    def _resolve_tiling(
        self, *, height: int, width: int, override: bool | None
    ) -> TilingConfig | None:
        if override is True:
            return TilingConfig()
        if override is False:
            return None
        mode = self.runtime_config.tiled_vae
        if mode == "on":
            return TilingConfig()
        if mode == "off":
            return None
        return (
            TilingConfig()
            if max(height, width) >= 2 * TilingConfig().vae_decode_tile_size
            else None
        )


def _to_image_array(decoded_latents: mx.array) -> mx.array:
    images = mx.clip(decoded_latents / 2 + 0.5, 0, 1)
    images = mx.transpose(images, (0, 2, 3, 1)).astype(mx.float32)
    images = (images * 255).round().astype(mx.uint8)
    image = images[0]
    mx.eval(image)
    return image


def _to_pil(image_array: mx.array) -> Image.Image:
    array = np.array(image_array)
    return Image.fromarray(array)


def _check_quantized_matmul(precision: str) -> None:
    bits = 1 if precision == "1bit" else 2 if precision == "2bit" else None
    if bits is None:
        return
    try:
        _require_native_quantized_matmul(bits, DEFAULT_QUANT_GROUP_SIZE)
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(
            "The installed MLX runtime does not support the Bonsai "
            f"{precision} quantized matmul path. Use the PrismML MLX revision "
            "pinned by the original demo, or another MLX build with this kernel."
        ) from exc
