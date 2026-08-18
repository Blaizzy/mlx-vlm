"""Z-Image generation pipeline."""

from __future__ import annotations

import gc
from pathlib import Path

import mlx.core as mx
import numpy as np
from transformers import AutoTokenizer

from .config import ZImageConfig
from .scheduler import FlowMatchEulerScheduler
from .weights import load_text_encoder, load_transformer, load_vae


class ZImagePipeline:
    """End-to-end text-to-image pipeline for Z-Image."""

    def __init__(
        self,
        model_path: str | Path,
        config: ZImageConfig | None = None,
        *,
        evict_text_encoder: bool = True,
    ) -> None:
        self.model_path = Path(model_path).expanduser()
        self.config = config or ZImageConfig()
        self.evict_text_encoder = evict_text_encoder
        self.text_encoder = load_text_encoder(self.model_path, self.config)
        self.tokenizer = AutoTokenizer.from_pretrained(
            str(self.model_path / "tokenizer"),
            local_files_only=True,
            trust_remote_code=True,
        )
        self.transformer = None
        self.vae = None

    def _ensure_components(self) -> None:
        if self.transformer is None:
            self.transformer = load_transformer(self.model_path, self.config)
        if self.vae is None:
            self.vae = load_vae(self.model_path, self.config)

    def _encode_prompt(self, prompt: str, max_length: int = 512) -> mx.array:
        """Tokenize and encode prompt through text encoder."""
        # Apply chat template
        messages = [{"role": "user", "content": prompt}]
        try:
            text = self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        except Exception:
            text = prompt

        tokens = self.tokenizer(
            text,
            padding="max_length",
            max_length=max_length,
            truncation=True,
            return_tensors="np",
        )
        input_ids = mx.array(tokens["input_ids"])
        attention_mask = mx.array(tokens["attention_mask"])
        hidden = self.text_encoder(input_ids, attention_mask)
        mx.eval(hidden)
        return hidden

    def _evict_encoder(self) -> None:
        if self.evict_text_encoder:
            self.text_encoder = None
            gc.collect()
            mx.clear_cache()

    def _reload_encoder(self) -> None:
        if self.text_encoder is None:
            self.text_encoder = load_text_encoder(self.model_path, self.config)

    def count_prompt_tokens(self, prompt: str) -> int:
        messages = [{"role": "user", "content": prompt}]
        try:
            text = self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        except Exception:
            text = prompt
        return len(self.tokenizer(text, truncation=False)["input_ids"])

    def generate_array(
        self,
        prompt: str,
        *,
        seed: int = 42,
        steps: int = 9,
        width: int = 1024,
        height: int = 1024,
    ) -> mx.array:
        """Generate image as uint8 [H, W, 3] array."""
        # Encode prompt
        self._reload_encoder()
        cap_feats = self._encode_prompt(prompt)
        self._evict_encoder()

        # Load components
        self._ensure_components()

        # Prepare latents [B, C, F, H, W]
        vae_scale = 8
        latent_h = 2 * (height // (vae_scale * 2))
        latent_w = 2 * (width // (vae_scale * 2))
        latents = mx.random.normal(
            (1, 16, 1, latent_h, latent_w),
            key=mx.random.key(seed),
            dtype=mx.float32,
        )

        # Denoise
        scheduler = FlowMatchEulerScheduler(steps)
        for i in range(steps):
            t = scheduler.get_model_input_timestep(i)
            noise_pred = self.transformer(latents, t, cap_feats)
            # Negate prediction per reference pipeline
            noise_pred = -noise_pred
            latents = scheduler.step(noise_pred, i, latents)
            mx.eval(latents)

        # Decode
        # Squeeze F dim and transpose to [B, H, W, C]
        z = latents.squeeze(2).transpose(0, 2, 3, 1)
        # Apply scaling
        z = (z / self.config.vae.scaling_factor) + self.config.vae.shift_factor
        decoded = self.vae.decode(z)
        mx.eval(decoded)

        # Post-process to uint8
        image = (decoded[0] / 2.0 + 0.5)
        image = mx.clip(image, 0.0, 1.0)
        return mx.round(image * 255.0).astype(mx.uint8)


__all__ = ["ZImagePipeline"]
