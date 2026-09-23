"""Text-to-image generation pipeline for Ming-Image-0.1-Design."""

from __future__ import annotations

import gc
from pathlib import Path

import mlx.core as mx
from transformers import AutoTokenizer

from .config import MingImageConfig
from .download import download_model, validate_model_layout
from .scheduler import FlowMatchEulerDiscreteScheduler
from .weights import load_text_encoder, load_transformer, load_vae


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

    @classmethod
    def from_pretrained(
        cls,
        model_path: str | Path | None = None,
        *,
        repo_id: str | None = None,
        download: bool = True,
        token: str | None = None,
        revision: str | None = None,
        force_download: bool = False,
        evict_text_encoder: bool = True,
    ) -> "MingImagePipeline":
        if model_path is None:
            if not download:
                raise ValueError("model_path is required when download=False")
            model_path = download_model(
                repo_id or "inclusionAI/Ming-Image-0.1-Design",
                token=token,
                revision=revision,
                force_download=force_download,
            )
        model_path = validate_model_layout(model_path)
        return cls(model_path, evict_text_encoder=evict_text_encoder)

    def _tokenize(self, prompt: str) -> mx.array:
        text = self.tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,
        )
        ids = self.tokenizer(text, add_special_tokens=False)["input_ids"]
        mllm = self.config.mllm
        block = (
            [mllm.image_start_token]
            + [mllm.image_patch_token] * self.config.bridge.query_token_count
            + [mllm.image_end_token]
        )
        return mx.array([ids + block], dtype=mx.int32)

    def count_prompt_tokens(self, prompt: str) -> int:
        text = self.tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,
        )
        return len(self.tokenizer(text, add_special_tokens=False)["input_ids"])

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
    ) -> mx.array:
        """Generate one image, returned as an ``[H, W, 4]`` uint8 RGBA array."""
        for name, value in (("width", width), ("height", height)):
            if value < 16 or value % 16:
                raise ValueError(
                    f"{name} must be a positive multiple of 16, got {value}"
                )
        if steps < 1:
            raise ValueError(f"steps must be at least 1, got {steps}")
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
            (1, z, 1, latent_h, latent_w), key=mx.random.key(seed), dtype=mx.float32
        )
        scheduler = FlowMatchEulerDiscreteScheduler(
            image_seq_len=image_seq_len,
            num_inference_steps=steps,
            num_train_timesteps=self.config.num_train_timesteps,
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

        return self._decode(latents)

    def _decode(self, latents: mx.array) -> mx.array:
        z = latents / self.config.vae.scaling_factor + self.config.vae.shift_factor
        decoded = self.vae.decode(z.astype(mx.float32))[:, :, 0]
        image = (mx.clip(decoded[0], -1.0, 1.0) + 1.0) / 2.0 * 255.0
        mx.eval(image)
        return image.astype(mx.uint8).transpose(1, 2, 0)


__all__ = ["MingImagePipeline"]
