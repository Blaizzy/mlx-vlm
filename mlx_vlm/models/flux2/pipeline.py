from __future__ import annotations

import gc
import math
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import mlx.core as mx
import numpy as np
from PIL import Image

from mlx_vlm.models.flux2.config import Flux2Variant, get_variant, validate_dimensions
from mlx_vlm.models.flux2.constants import ModelConfig
from mlx_vlm.models.flux2.download import download_model, validate_model_layout
from mlx_vlm.models.flux2.latent import (
    pack_latents,
    patchify_latents,
    prepare_grid_ids,
    prepare_packed_latents,
    unpack_latents,
)
from mlx_vlm.models.flux2.prompt import encode_prompt
from mlx_vlm.models.flux2.scheduler import FlowMatchEulerDiscreteScheduler
from mlx_vlm.models.flux2.tiling import TilingConfig
from mlx_vlm.models.flux2.tokenizer import Flux2Tokenizer
from mlx_vlm.models.flux2.transformer.kv_cache import Flux2KVCache
from mlx_vlm.models.flux2.weights import load_text_encoder, load_transformer, load_vae

TiledVAE = Literal["auto", "on", "off"]


@dataclass(frozen=True, slots=True)
class Flux2RuntimeConfig:
    evict_text_encoder: bool = True
    evict_transformer: bool = False
    bucketed_seq_len: bool = True
    tiled_vae: TiledVAE = "auto"
    max_sequence_length: int = 512


class Flux2Image:
    def __init__(
        self,
        *,
        variant: str | Flux2Variant = "flux2-klein-4b",
        model_path: str | Path,
        runtime_config: Flux2RuntimeConfig | None = None,
    ) -> None:
        self.variant = get_variant(variant)
        self.model_path = validate_model_layout(model_path)
        self.runtime_config = runtime_config or Flux2RuntimeConfig()
        self.tokenizer = Flux2Tokenizer(self.model_path)
        self.text_encoder = load_text_encoder(self.model_path, self.variant)
        self.transformer = None
        self.vae = None
        self.prompt_cache: dict[tuple[str, int, bool], tuple[mx.array, mx.array]] = {}

    @classmethod
    def from_pretrained(
        cls,
        variant: str | Flux2Variant = "flux2-klein-4b",
        *,
        model_path: str | Path | None = None,
        download: bool = True,
        token: str | None = None,
        revision: str | None = None,
        force_download: bool = False,
        evict_text_encoder: bool = True,
        evict_transformer: bool = False,
        bucketed_seq_len: bool = True,
        tiled_vae: TiledVAE = "auto",
        max_sequence_length: int = 512,
    ) -> "Flux2Image":
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
        config = Flux2RuntimeConfig(
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
        width: int = 1024,
        height: int = 1024,
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
        width: int = 1024,
        height: int = 1024,
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
        for i in range(steps):
            noise = self._predict(
                latents=latents,
                latent_ids=latent_ids,
                prompt_embeds=prompt_embeds,
                text_ids=text_ids,
                timestep=scheduler.timesteps[i],
            )
            if negative_prompt_embeds is not None and negative_text_ids is not None:
                negative_noise = self._predict(
                    latents=latents,
                    latent_ids=latent_ids,
                    prompt_embeds=negative_prompt_embeds,
                    text_ids=negative_text_ids,
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
            self.text_encoder = load_text_encoder(self.model_path, self.variant)
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
            self.transformer = load_transformer(self.model_path, self.variant)
        if self.vae is None:
            self.vae = load_vae(self.model_path)

    def _predict(
        self,
        *,
        latents: mx.array,
        latent_ids: mx.array,
        prompt_embeds: mx.array,
        text_ids: mx.array,
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


class Flux2ImageEdit(Flux2Image):
    def edit(
        self,
        prompt: str,
        image_paths: Sequence[str | Path],
        *,
        seed: int = 42,
        steps: int = 4,
        width: int | None = None,
        height: int | None = None,
        guidance: float = 1.0,
        max_sequence_length: int | None = None,
        tiled_vae: bool | None = None,
    ) -> Image.Image:
        return _to_pil(
            self.edit_array(
                prompt,
                image_paths,
                seed=seed,
                steps=steps,
                width=width,
                height=height,
                guidance=guidance,
                max_sequence_length=max_sequence_length,
                tiled_vae=tiled_vae,
            )
        )

    def edit_array(
        self,
        prompt: str,
        image_paths: Sequence[str | Path],
        *,
        seed: int = 42,
        steps: int = 4,
        width: int | None = None,
        height: int | None = None,
        guidance: float = 1.0,
        max_sequence_length: int | None = None,
        tiled_vae: bool | None = None,
    ) -> mx.array:
        if not self.variant.supports_edit:
            raise ValueError(f"{self.variant.repo_id} does not support image editing")
        if steps < 1:
            raise ValueError(f"steps must be >= 1, got {steps}")
        if not prompt:
            raise ValueError("prompt must not be empty")
        if not image_paths:
            raise ValueError("At least one reference image path is required")

        width, height = _resolve_edit_dimensions(
            image_paths=image_paths,
            width=width,
            height=height,
        )
        validate_dimensions(width=width, height=height)

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
        reference_latents, reference_ids = self._prepare_reference_latents(
            image_paths=image_paths,
            width=width,
            height=height,
        )
        latents, latent_ids, latent_height, latent_width = prepare_packed_latents(
            seed=seed,
            height=height,
            width=width,
        )
        scheduler = FlowMatchEulerDiscreteScheduler(
            image_seq_len=(height // 16) * (width // 16),
            num_inference_steps=steps,
        )
        cache = None
        negative_cache = None
        for i in range(steps):
            if self.variant.uses_reference_kv_cache:
                noise, cache = self._predict_edit_with_kv_cache(
                    latents=latents,
                    latent_ids=latent_ids,
                    reference_latents=reference_latents,
                    reference_ids=reference_ids,
                    prompt_embeds=prompt_embeds,
                    text_ids=text_ids,
                    timestep=scheduler.timesteps[i],
                    kv_cache=cache,
                )
            else:
                noise = self._predict_edit(
                    latents=latents,
                    latent_ids=latent_ids,
                    reference_latents=reference_latents,
                    reference_ids=reference_ids,
                    prompt_embeds=prompt_embeds,
                    text_ids=text_ids,
                    timestep=scheduler.timesteps[i],
                )
            if negative_prompt_embeds is not None and negative_text_ids is not None:
                if self.variant.uses_reference_kv_cache:
                    negative_noise, negative_cache = self._predict_edit_with_kv_cache(
                        latents=latents,
                        latent_ids=latent_ids,
                        reference_latents=reference_latents,
                        reference_ids=reference_ids,
                        prompt_embeds=negative_prompt_embeds,
                        text_ids=negative_text_ids,
                        timestep=scheduler.timesteps[i],
                        kv_cache=negative_cache,
                    )
                else:
                    negative_noise = self._predict_edit(
                        latents=latents,
                        latent_ids=latent_ids,
                        reference_latents=reference_latents,
                        reference_ids=reference_ids,
                        prompt_embeds=negative_prompt_embeds,
                        text_ids=negative_text_ids,
                        timestep=scheduler.timesteps[i],
                    )
                noise = negative_noise + guidance * (noise - negative_noise)
            if i == 0:
                arrays = []
                if cache is not None:
                    arrays.extend(cache.arrays())
                if negative_cache is not None:
                    arrays.extend(negative_cache.arrays())
                if arrays:
                    mx.eval(*arrays)
            latents = scheduler.step(noise=noise, step_index=i, latents=latents)
            mx.eval(latents)

        packed = unpack_latents(
            latents, latent_height=latent_height, latent_width=latent_width
        )
        decoded = self.vae.decode_packed_latents(  # type: ignore[union-attr]
            packed,
            tiling_config=self._resolve_tiling(
                height=height, width=width, override=tiled_vae
            ),
        )
        if cache is not None:
            cache.clear()
        if negative_cache is not None:
            negative_cache.clear()
        if self.runtime_config.evict_transformer:
            self.transformer = None
            self.vae = None
            gc.collect()
            mx.clear_cache()
        return _to_image_array(decoded)

    def _ensure_transformer_and_vae(self) -> None:
        if self.transformer is None:
            self.transformer = load_transformer(self.model_path, self.variant)
        if self.vae is None:
            self.vae = load_vae(self.model_path, include_encoder=True)

    def _prepare_reference_latents(
        self,
        *,
        image_paths: Sequence[str | Path],
        width: int,
        height: int,
    ) -> tuple[mx.array, mx.array]:
        packed_refs = []
        ref_ids = []
        for index, image_path in enumerate(image_paths):
            image = _load_reference_image(image_path, width=width, height=height)
            image_array = _reference_image_array(image)
            latents = self.vae.encode(image_array)  # type: ignore[union-attr]
            latents = _crop_to_even_spatial(latents)
            latents = patchify_latents(latents)
            latents = _bn_normalize_vae_encoded_latents(
                latents, vae=self.vae  # type: ignore[arg-type]
            )
            packed_refs.append(pack_latents(latents))
            ref_ids.append(prepare_grid_ids(latents, t_coord=10 + 10 * index))

        reference_latents = mx.concatenate(packed_refs, axis=1)
        reference_ids = mx.concatenate(ref_ids, axis=1)
        mx.eval(reference_latents, reference_ids)
        return reference_latents, reference_ids

    def _predict_edit(
        self,
        *,
        latents: mx.array,
        latent_ids: mx.array,
        reference_latents: mx.array,
        reference_ids: mx.array,
        prompt_embeds: mx.array,
        text_ids: mx.array,
        timestep: mx.array,
    ) -> mx.array:
        hidden_states = mx.concatenate([latents, reference_latents], axis=1)
        img_ids = mx.concatenate([latent_ids, reference_ids], axis=1)
        noise = self.transformer(  # type: ignore[operator]
            hidden_states=hidden_states,
            encoder_hidden_states=prompt_embeds,
            timestep=timestep,
            img_ids=img_ids,
            txt_ids=text_ids,
            guidance=None,
        )
        if isinstance(noise, tuple):
            noise = noise[0]
        return noise[:, : latents.shape[1]]

    def _predict_edit_with_kv_cache(
        self,
        *,
        latents: mx.array,
        latent_ids: mx.array,
        reference_latents: mx.array,
        reference_ids: mx.array,
        prompt_embeds: mx.array,
        text_ids: mx.array,
        timestep: mx.array,
        kv_cache: Flux2KVCache | None,
    ) -> tuple[mx.array, Flux2KVCache | None]:
        if kv_cache is None:
            hidden_states = mx.concatenate([reference_latents, latents], axis=1)
            img_ids = mx.concatenate([reference_ids, latent_ids], axis=1)
            result = self.transformer(  # type: ignore[operator]
                hidden_states=hidden_states,
                encoder_hidden_states=prompt_embeds,
                timestep=timestep,
                img_ids=img_ids,
                txt_ids=text_ids,
                guidance=None,
                kv_cache_mode="extract",
                num_ref_tokens=reference_latents.shape[1],
            )
            if not isinstance(result, tuple):
                raise RuntimeError("Flux2 KV extraction did not return a cache")
            return result

        result = self.transformer(  # type: ignore[operator]
            hidden_states=latents,
            encoder_hidden_states=prompt_embeds,
            timestep=timestep,
            img_ids=latent_ids,
            txt_ids=text_ids,
            guidance=None,
            kv_cache=kv_cache,
            kv_cache_mode="cached",
        )
        if isinstance(result, tuple):
            return result
        return result, kv_cache


def _resolve_edit_dimensions(
    *,
    image_paths: Sequence[str | Path],
    width: int | None,
    height: int | None,
) -> tuple[int, int]:
    if width is not None or height is not None:
        if width is None or height is None:
            raise ValueError("Both width and height are required when either is set")
        return width, height
    with Image.open(Path(image_paths[0]).expanduser()) as image:
        image_width, image_height = image.size
    scale = min(1.0, math.sqrt((1024 * 1024) / (image_width * image_height)))
    width = _round_edit_dimension(image_width * scale)
    height = _round_edit_dimension(image_height * scale)
    return width, height


def _round_edit_dimension(value: float) -> int:
    return max(256, min(2048, int(value) // 16 * 16))


def _load_reference_image(
    image_path: str | Path,
    *,
    width: int,
    height: int,
) -> Image.Image:
    path = Path(image_path).expanduser()
    if not path.exists():
        raise FileNotFoundError(f"Reference image does not exist: {path}")
    with Image.open(path) as image:
        image = image.convert("RGB")
        if image.size != (width, height):
            image = image.resize((width, height), Image.Resampling.LANCZOS)
        return image


def _reference_image_array(image: Image.Image) -> mx.array:
    array = np.asarray(image).astype(np.float32) / 127.5 - 1.0
    array = np.transpose(array, (2, 0, 1))[None, ...]
    return mx.array(array).astype(ModelConfig.precision)


def _crop_to_even_spatial(latents: mx.array) -> mx.array:
    if latents.shape[2] % 2 != 0:
        latents = latents[:, :, :-1, :]
    if latents.shape[3] % 2 != 0:
        latents = latents[:, :, :, :-1]
    return latents


def _bn_normalize_vae_encoded_latents(latents: mx.array, *, vae) -> mx.array:
    bn_mean = vae.bn.running_mean.reshape(1, -1, 1, 1).astype(latents.dtype)
    bn_std = mx.sqrt(vae.bn.running_var.reshape(1, -1, 1, 1) + vae.bn.eps).astype(
        latents.dtype
    )
    return (latents - bn_mean) / bn_std


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
