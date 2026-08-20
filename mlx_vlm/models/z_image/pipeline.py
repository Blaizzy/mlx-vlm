from __future__ import annotations

import gc
from collections.abc import Sequence
from pathlib import Path

import mlx.core as mx
import numpy as np
from PIL import Image
from transformers import AutoTokenizer

from mlx_vlm.models.mage_flow.scheduler import FlowMatchEulerDiscreteScheduler

from .config import ZImageConfig
from .weights import load_text_encoder, load_transformer, load_vae


def _img2img_start_index(steps: int, strength: float) -> int:
    return int(max(steps - min(steps * strength, steps), 0))


class ZImagePipeline:
    def __init__(
        self,
        model_path: str | Path,
        config: ZImageConfig | None = None,
        *,
        evict_text_encoder: bool = True,
    ) -> None:
        self.model_path = Path(model_path).expanduser()
        self.config = config or ZImageConfig.from_model_path(self.model_path)
        self.evict_text_encoder = evict_text_encoder
        self.text_encoder = load_text_encoder(self.model_path, self.config)
        self.tokenizer = AutoTokenizer.from_pretrained(
            str(self.model_path / "tokenizer"),
            local_files_only=True,
            trust_remote_code=True,
        )
        self.transformer = None
        self.vae = None

    def _format_prompt(self, prompt: str) -> str:
        return self.tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=True,
        )

    def _ensure_components(self) -> None:
        if self.transformer is None:
            self.transformer = load_transformer(self.model_path, self.config)
        if self.vae is None:
            self.vae = load_vae(self.model_path, self.config)

    def _evict_components(self) -> None:
        if self.transformer is None and self.vae is None:
            return
        self.transformer = None
        self.vae = None
        gc.collect()
        mx.clear_cache()

    def _encode_prompt(self, prompt: str, max_length: int = 512) -> mx.array:
        tokens = self.tokenizer(
            self._format_prompt(prompt),
            padding="max_length",
            max_length=max_length,
            truncation=True,
            return_tensors="np",
        )
        input_ids = mx.array(tokens["input_ids"])
        attention_mask = mx.array(tokens["attention_mask"])
        hidden = self.text_encoder(input_ids, attention_mask, output_penultimate=True)
        valid_length = int(mx.sum(attention_mask[0]).item())
        hidden = hidden[:, :valid_length]
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
        text = self._format_prompt(prompt)
        return len(self.tokenizer(text, truncation=False)["input_ids"])

    def _encode_conditioning(
        self,
        prompt: str,
        *,
        guidance: float,
        negative_prompt: str | None,
    ) -> tuple[mx.array, mx.array | None]:
        if self.evict_text_encoder:
            self._evict_components()
        self._reload_encoder()
        cap_feats = self._encode_prompt(prompt)
        negative_cap_feats = (
            self._encode_prompt(negative_prompt or "") if guidance > 1.0 else None
        )
        self._evict_encoder()
        self._ensure_components()
        return cap_feats, negative_cap_feats

    def _denoise(
        self,
        latents: mx.array,
        *,
        scheduler: FlowMatchEulerDiscreteScheduler,
        start_index: int,
        cap_feats: mx.array,
        negative_cap_feats: mx.array | None,
        guidance: float,
        cfg_truncation: float,
    ) -> mx.array:
        for i in range(start_index, len(scheduler.sigmas) - 1):
            t = (1.0 - scheduler.sigmas[i]).reshape(1)
            prediction = self.transformer(latents, t, cap_feats)
            if negative_cap_feats is not None and float(t.item()) <= cfg_truncation:
                negative_prediction = self.transformer(
                    latents,
                    t,
                    negative_cap_feats,
                )
                # Z-Image uses the conditional prediction as the guidance base.
                prediction = prediction + guidance * (prediction - negative_prediction)
            latents = scheduler.step(
                velocity=-prediction,
                step_index=i,
                latents=latents,
            )
            mx.eval(latents)
        return latents

    def _decode_latents(self, latents: mx.array) -> mx.array:
        z = latents.squeeze(2).transpose(0, 2, 3, 1)
        z = (z / self.config.vae.scaling_factor) + self.config.vae.shift_factor
        decoded = self.vae.decode(z)
        mx.eval(decoded)
        image = mx.clip(decoded[0] / 2.0 + 0.5, 0.0, 1.0)
        return mx.round(image * 255.0).astype(mx.uint8)

    def generate_array(
        self,
        prompt: str,
        *,
        seed: int = 42,
        steps: int = 9,
        width: int = 1024,
        height: int = 1024,
        guidance: float = 0.0,
        negative_prompt: str | None = None,
        cfg_truncation: float = 1.0,
    ) -> mx.array:
        for name, value in (("width", width), ("height", height)):
            if value < 16 or value % 16:
                raise ValueError(
                    f"Z-Image {name} must be a positive multiple of 16, got {value}"
                )
        if steps < 2:
            raise ValueError(f"Z-Image steps must be at least 2, got {steps}")
        if cfg_truncation < 0.0 or cfg_truncation > 1.0:
            raise ValueError(
                f"Z-Image cfg_truncation must be in [0, 1], got {cfg_truncation}"
            )

        cap_feats, negative_cap_feats = self._encode_conditioning(
            prompt,
            guidance=guidance,
            negative_prompt=negative_prompt,
        )

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
        # Upstream skips the terminal zero timestep, so N requested steps run N - 1
        # nonzero transformer forwards.
        scheduler = FlowMatchEulerDiscreteScheduler(
            num_inference_steps=steps - 1,
            shift=self.config.scheduler_shift,
        )
        latents = self._denoise(
            latents,
            scheduler=scheduler,
            start_index=0,
            cap_feats=cap_feats,
            negative_cap_feats=negative_cap_feats,
            guidance=guidance,
            cfg_truncation=cfg_truncation,
        )
        return self._decode_latents(latents)

    def edit_array(
        self,
        prompt: str,
        image_paths: Sequence[str | Path],
        *,
        seed: int = 42,
        steps: int = 8,
        width: int | None = None,
        height: int | None = None,
        guidance: float = 0.0,
        negative_prompt: str | None = None,
        cfg_truncation: float = 1.0,
        strength: float = 0.6,
    ) -> mx.array:
        if len(image_paths) != 1:
            raise ValueError("Z-Image img2img requires exactly one source image")
        if not 0.0 < strength <= 1.0:
            raise ValueError(f"Z-Image strength must be in (0, 1], got {strength}")
        if steps < 1:
            raise ValueError(f"Z-Image steps must be at least 1, got {steps}")
        if not 0.0 <= cfg_truncation <= 1.0:
            raise ValueError(
                f"Z-Image cfg_truncation must be in [0, 1], got {cfg_truncation}"
            )

        with Image.open(Path(image_paths[0]).expanduser()) as source:
            source = source.convert("RGB")
            width = width or source.width - source.width % 16
            height = height or source.height - source.height % 16
            for name, value in (("width", width), ("height", height)):
                if value < 16 or value % 16:
                    raise ValueError(
                        f"Z-Image {name} must be a positive multiple of 16, got {value}"
                    )
            source = source.resize((width, height), Image.Resampling.LANCZOS)
            pixels = mx.array(np.asarray(source), dtype=mx.float32)[None]
        pixels = pixels / 127.5 - 1.0

        cap_feats, negative_cap_feats = self._encode_conditioning(
            prompt,
            guidance=guidance,
            negative_prompt=negative_prompt,
        )
        image_latents = self.vae.encode(pixels)
        image_latents = (
            image_latents - self.config.vae.shift_factor
        ) * self.config.vae.scaling_factor
        image_latents = image_latents.transpose(0, 3, 1, 2)[:, :, None]
        mx.eval(image_latents)

        scheduler = FlowMatchEulerDiscreteScheduler(
            num_inference_steps=steps,
            shift=self.config.scheduler_shift,
        )
        # Match Diffusers' SD3-derived img2img timestep selection. Keeping the
        # product fractional before truncation starts at floor(N * (1-strength)).
        start_index = _img2img_start_index(steps, strength)
        if steps - start_index < 1:
            raise ValueError(
                f"Z-Image strength {strength} produces no denoising steps for {steps} steps"
            )
        noise = mx.random.normal(
            image_latents.shape,
            key=mx.random.key(seed),
            dtype=mx.float32,
        )
        sigma = scheduler.sigmas[start_index].astype(image_latents.dtype)
        latents = (1.0 - sigma) * image_latents + sigma * noise
        latents = self._denoise(
            latents,
            scheduler=scheduler,
            start_index=start_index,
            cap_feats=cap_feats,
            negative_cap_feats=negative_cap_feats,
            guidance=guidance,
            cfg_truncation=cfg_truncation,
        )
        return self._decode_latents(latents)


__all__ = ["ZImagePipeline"]
