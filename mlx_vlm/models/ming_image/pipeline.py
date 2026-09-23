"""Text-to-image generation pipeline for Ming-Image-0.1-Design."""

from __future__ import annotations

import gc
from pathlib import Path

import mlx.core as mx
from transformers import AutoTokenizer

from mlx_vlm.models.qwen_image.scheduler import FlowMatchEulerDiscreteScheduler

from .config import MingImageConfig
from .weights import load_text_encoder, load_transformer, load_vae


def scheduler_shift(image_seq_len: int) -> tuple[int, float]:
    """Ming's dynamic-shift rule as (max_image_seq_len, max_shift) for the scheduler."""
    return (image_seq_len, 1.35) if image_seq_len >= 4096 else (4096, 1.15)


class MingImagePipeline:
    def __init__(
        self,
        model_path: str | Path,
        config: MingImageConfig | None = None,
        *,
        evict_text_encoder: bool = True,
    ) -> None:
        self.model_path = Path(model_path).expanduser()
        self.config = config or MingImageConfig.from_model_path(self.model_path)
        self.evict_text_encoder = evict_text_encoder
        self.tokenizer = AutoTokenizer.from_pretrained(
            str(self.model_path / "mllm"), local_files_only=True, use_fast=True
        )
        self.text_encoder = load_text_encoder(self.model_path, self.config)
        self.transformer = None
        self.vae = None

    def _prompt_ids(self, prompt: str) -> list[int]:
        text = self.tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,
        )
        return self.tokenizer(text, add_special_tokens=False)["input_ids"]

    def _tokenize(self, prompt: str) -> mx.array:
        cfg = self.config
        block = (
            [cfg.image_start_token]
            + [cfg.image_patch_token] * cfg.bridge.query_token_count
            + [cfg.image_end_token]
        )
        return mx.array([self._prompt_ids(prompt) + block], dtype=mx.int32)

    def count_prompt_tokens(self, prompt: str) -> int:
        return len(self._prompt_ids(prompt))

    def _ensure_components(self) -> None:
        if self.transformer is None:
            self.transformer = load_transformer(self.model_path, self.config)
        if self.vae is None:
            self.vae = load_vae(self.model_path, self.config)

    def _encode(self, prompt: str) -> tuple[mx.array, mx.array]:
        cap_feats, cap_feats_2 = self.text_encoder.encode(self._tokenize(prompt))
        cap_feats = cap_feats.astype(mx.bfloat16)
        cap_feats_2 = cap_feats_2.astype(mx.bfloat16)
        mx.eval(cap_feats, cap_feats_2)
        if self.evict_text_encoder:
            self.text_encoder = None
            gc.collect()
            mx.clear_cache()
        return cap_feats, cap_feats_2

    def generate_array(
        self,
        prompt: str,
        *,
        seed: int = 0,
        steps: int = 12,
        width: int = 1024,
        height: int = 1024,
        guidance: float = 1.0,
        num_images: int = 1,
    ) -> mx.array:
        """Generate one image ([H, W, 4]) or a batch ([N, H, W, 4]) for one prompt."""
        for name, value in (("width", width), ("height", height)):
            if value < 16 or value % 16:
                raise ValueError(
                    f"{name} must be a positive multiple of 16, got {value}"
                )
        if steps < 1:
            raise ValueError(f"steps must be at least 1, got {steps}")
        if num_images < 1:
            raise ValueError(f"num_images must be at least 1, got {num_images}")
        if guidance != 1.0:
            raise ValueError(
                "Ming-Image-0.1-Design is trained for guidance 1.0 (CFG off)"
            )

        cap_feats, cap_feats_2 = self._encode(prompt)
        self._ensure_components()

        z = self.config.vae.z_dim
        latent_h, latent_w = height // 8, width // 8
        image_seq_len = (latent_h // self.config.dit.patch_size) * (
            latent_w // self.config.dit.patch_size
        )
        latents = mx.random.normal(
            (num_images, z, 1, latent_h, latent_w),
            key=mx.random.key(seed),
            dtype=mx.float32,
        )
        max_image_seq_len, max_shift = scheduler_shift(image_seq_len)
        scheduler = FlowMatchEulerDiscreteScheduler(
            image_seq_len=image_seq_len,
            num_inference_steps=steps,
            base_shift=0.5,
            max_shift=max_shift,
            base_image_seq_len=256,
            max_image_seq_len=max_image_seq_len,
            num_train_timesteps=self.config.num_train_timesteps,
            shift_terminal=None,
        )
        patch = self.config.dit.patch_size
        conditioning = self.transformer.prepare_conditioning(
            cap_feats, cap_feats_2, (1, latent_h // patch, latent_w // patch)
        )
        mx.eval(conditioning.arrays())
        for i in range(steps):
            t = (1.0 - scheduler.sigmas[i]).reshape(1).astype(mx.bfloat16)
            pred = self.transformer.denoise(
                latents.astype(mx.bfloat16), t, conditioning
            ).astype(mx.float32)
            latents = scheduler.step(noise=-pred, step_index=i, latents=latents)
            mx.eval(latents)

        images = self._decode(latents)
        return images[0] if num_images == 1 else images

    def _decode(self, latents: mx.array) -> mx.array:
        """Decode ``[N, z, 1, h, w]`` latents to an ``[N, H, W, 4]`` uint8 array."""
        z = latents / self.config.vae.scaling_factor + self.config.vae.shift_factor
        decoded = self.vae.decode(z.astype(mx.float32))[:, :, 0]
        images = (mx.clip(decoded, -1.0, 1.0) + 1.0) / 2.0 * 255.0
        mx.eval(images)
        return images.astype(mx.uint8).transpose(0, 2, 3, 1)


__all__ = ["MingImagePipeline"]
