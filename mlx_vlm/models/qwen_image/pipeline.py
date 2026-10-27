"""Generation and reference-image editing for Qwen-Image-2.1."""

from __future__ import annotations

import json
import math
from collections.abc import Sequence
from pathlib import Path

import mlx.core as mx
import numpy as np
from PIL import Image

from .config import QwenImageVariant, get_variant
from .download import download_model, validate_model_layout
from .kv_cache import QwenImageKVCache
from .scheduler import FlowMatchEulerDiscreteScheduler
from .text_encoder import QwenImageTextEncoder
from .weights import _read_quant, load_text_encoder, load_transformer, load_vae


def _image_dimensions(resolution: int, ratio: float) -> tuple[int, int]:
    width = math.sqrt(resolution * resolution * ratio)
    return max(32, round(width / 32) * 32), max(32, round(width / ratio / 32) * 32)


class QwenImagePipeline:
    def __init__(
        self,
        *,
        variant: QwenImageVariant,
        model_path: str | Path,
        text_encoder,
        transformer,
        vae,
    ) -> None:
        self.variant = variant
        self.model_path = Path(model_path)
        self.text_encoder = QwenImageTextEncoder(
            model=text_encoder, model_path=self.model_path
        )
        self.transformer = transformer
        self.vae = vae
        cfg = json.loads((self.model_path / "vae" / "config.json").read_text())
        self.z_dim = cfg["z_dim"]
        self.latents_mean = mx.array(cfg["latents_mean"]).reshape(
            1, self.z_dim, 1, 1, 1
        )
        self.latents_std = mx.array(cfg["latents_std"]).reshape(1, self.z_dim, 1, 1, 1)
        self.quantization_config = _read_quant(self.model_path / "transformer")
        scheduler_path = self.model_path / "scheduler" / "scheduler_config.json"
        scheduler_config = (
            json.loads(scheduler_path.read_text()) if scheduler_path.exists() else {}
        )
        self.scheduler_config = {
            key: scheduler_config[key]
            for key in (
                "base_shift",
                "max_shift",
                "base_image_seq_len",
                "max_image_seq_len",
                "num_train_timesteps",
                "shift_terminal",
            )
            if key in scheduler_config
        }

    @classmethod
    def from_pretrained(
        cls,
        variant: str | QwenImageVariant = "qwen-image-2.1",
        *,
        model_path: str | Path | None = None,
        download: bool = True,
        token: str | None = None,
        revision: str | None = None,
        force_download: bool = False,
    ) -> "QwenImagePipeline":
        variant = get_variant(variant)
        if model_path is None:
            if not download:
                raise ValueError("model_path is required when download=False")
            model_path = download_model(
                variant, token=token, revision=revision, force_download=force_download
            )
        model_path = validate_model_layout(model_path)
        return cls(
            variant=variant,
            model_path=model_path,
            text_encoder=load_text_encoder(model_path),
            transformer=load_transformer(model_path, variant),
            vae=load_vae(model_path),
        )

    def count_prompt_tokens(self, prompt: str) -> int:
        return len(self.text_encoder.tokenizer(prompt)["input_ids"])

    def generate_array(
        self,
        prompt: str,
        *,
        seed: int = 0,
        steps: int = 30,
        width: int = 512,
        height: int = 512,
        guidance: float = 1.0,
        negative_prompt: str = " ",
        num_images: int = 1,
    ) -> mx.array:
        """Generate one ``[H, W, 3]`` image, or ``[N, H, W, 3]`` when num_images > 1."""
        emb = self.text_encoder.encode(prompt).astype(mx.bfloat16)
        do_cfg = guidance is not None and guidance > 1.0
        neg = (
            self.text_encoder.encode(negative_prompt).astype(mx.bfloat16)
            if do_cfg
            else None
        )

        return self._sample(
            emb,
            neg,
            seed=seed,
            steps=steps,
            width=width,
            height=height,
            guidance=guidance,
            num_images=num_images,
        )[..., :3]

    def edit_array(
        self,
        prompt: str,
        image_paths: Sequence[str | Path],
        *,
        seed: int = 0,
        steps: int = 40,
        width: int | None = None,
        height: int | None = None,
        guidance: float = 1.0,
        negative_prompt: str = " ",
        output_resolution: int = 1024,
        use_kv_cache: bool = True,
    ) -> mx.array:
        """Edit one or more references; return an [H, W, 4] uint8 RGBA image."""
        if not image_paths:
            raise ValueError("At least one reference image is required")
        if output_resolution < 256:
            raise ValueError("output_resolution must be at least 256")
        references, reference_latents, reference_shapes = [], [], []
        for path in image_paths:
            with Image.open(Path(path).expanduser()) as source:
                image = source.convert("RGBA")
            size = _image_dimensions(output_resolution, image.width / image.height)
            image = image.resize(size, Image.Resampling.LANCZOS)
            references.append(image)
            pixels = mx.array(np.asarray(image).astype(np.float32) / 127.5 - 1.0)
            pixels = pixels.transpose(2, 0, 1)[None, :, None].astype(mx.bfloat16)
            mean, _ = self.vae.encode(pixels)
            normalized = (
                mean - self.latents_mean.astype(mean.dtype)
            ) / self.latents_std.astype(mean.dtype)
            h, w = normalized.shape[-2:]
            reference_shapes.append((1, h, w))
            reference_latents.append(
                normalized.reshape(1, self.z_dim, h * w).transpose(0, 2, 1)
            )
        default_width, default_height = references[-1].size
        width = (default_width if width is None else width) // 32 * 32
        height = (default_height if height is None else height) // 32 * 32
        if width < 32 or height < 32:
            raise ValueError("Output width and height must be at least 32")
        emb, image_pad_mask = self.text_encoder.encode_edit(prompt, references)
        neg = negative_image_pad_mask = None
        if guidance is not None and guidance > 1.0:
            neg, negative_image_pad_mask = self.text_encoder.encode_edit(
                negative_prompt, references
            )
            neg = neg.astype(mx.bfloat16)
        return self._sample(
            emb.astype(mx.bfloat16),
            neg,
            seed=seed,
            steps=steps,
            width=width,
            height=height,
            guidance=guidance,
            reference_latents=mx.concatenate(reference_latents, axis=1).astype(
                mx.bfloat16
            ),
            reference_image_shapes=reference_shapes,
            image_pad_mask=image_pad_mask,
            negative_image_pad_mask=negative_image_pad_mask,
            use_kv_cache=use_kv_cache,
        )

    def _sample(
        self,
        emb,
        neg,
        *,
        seed,
        steps,
        width,
        height,
        guidance,
        num_images=1,
        reference_latents=None,
        reference_image_shapes=None,
        image_pad_mask=None,
        negative_image_pad_mask=None,
        use_kv_cache=False,
    ) -> mx.array:
        z = self.z_dim
        h_lat, w_lat = height // 16, width // 16
        tokens = h_lat * w_lat
        mx.random.seed(seed)
        latents = mx.random.normal((num_images, 1, z, h_lat, w_lat)).astype(mx.bfloat16)
        latents = latents.reshape(num_images, z, tokens).transpose(0, 2, 1)
        if num_images > 1:
            emb = mx.broadcast_to(emb, (num_images, *emb.shape[1:]))
            if neg is not None:
                neg = mx.broadcast_to(neg, (num_images, *neg.shape[1:]))
        scheduler = FlowMatchEulerDiscreteScheduler(
            image_seq_len=tokens, num_inference_steps=steps, **self.scheduler_config
        )
        cache = negative_cache = None
        if (
            use_kv_cache
            and reference_latents is not None
            and steps > 1
            and self.transformer.causal_condition
        ):
            num_layers = len(self.transformer.transformer_blocks)
            cache = QwenImageKVCache(num_layers)
            if neg is not None:
                negative_cache = QwenImageKVCache(num_layers)
        for i in range(steps):
            t = scheduler.timesteps[i : i + 1].astype(latents.dtype) / 1000
            model_input = latents
            edit_kwargs = {}
            if reference_latents is not None:
                if cache is None or i == 0:
                    model_input = mx.concatenate([reference_latents, latents], axis=1)
                edit_kwargs = dict(
                    reference_image_shapes=reference_image_shapes,
                    image_pad_mask=image_pad_mask,
                )
            if cache is not None:
                edit_kwargs.update(
                    kv_cache=cache, kv_cache_mode="extract" if i == 0 else "cached"
                )
            pred = self.transformer(
                hidden_states=model_input,
                encoder_hidden_states=emb,
                timestep=t,
                img_shape=(1, h_lat, w_lat),
                **edit_kwargs,
            )
            if neg is not None:
                if reference_latents is not None:
                    edit_kwargs["image_pad_mask"] = negative_image_pad_mask
                if negative_cache is not None:
                    edit_kwargs["kv_cache"] = negative_cache
                neg_pred = self.transformer(
                    hidden_states=model_input,
                    encoder_hidden_states=neg,
                    timestep=t,
                    img_shape=(1, h_lat, w_lat),
                    **edit_kwargs,
                )
                pred = neg_pred + guidance * (pred - neg_pred)
            latents = scheduler.step(noise=pred, step_index=i, latents=latents)
            if cache is not None and i == 0:
                # Materialize compact prefix storage at the existing step boundary,
                # releasing the extraction graph before the next denoising step.
                mx.eval(
                    latents,
                    cache.arrays(),
                    [] if negative_cache is None else negative_cache.arrays(),
                )
            else:
                mx.eval(latents)

        # The VAE decoder does not need the per-layer prefix buffers.
        if cache is not None:
            cache.clear()
        if negative_cache is not None:
            negative_cache.clear()

        z_lat = (
            latents.transpose(0, 2, 1)
            .reshape(num_images, z, 1, h_lat, w_lat)
            .astype(mx.float32)
        )
        z_lat = z_lat * self.latents_std + self.latents_mean
        image = self.vae.decode(z_lat)[:, :, 0]
        image = ((mx.clip(image, -1.0, 1.0) + 1.0) / 2.0 * 255).astype(mx.uint8)
        images = image.transpose(0, 2, 3, 1)
        return images[0] if num_images == 1 else images


__all__ = ["QwenImagePipeline"]
