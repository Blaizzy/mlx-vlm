"""Text-to-image generation pipeline for Qwen-Image-2.1."""

from __future__ import annotations

import json
from pathlib import Path

import mlx.core as mx

from .config import QwenImageVariant, get_variant
from .download import download_model, validate_model_layout
from .scheduler import FlowMatchEulerDiscreteScheduler
from .text_encoder import QwenImageTextEncoder
from .weights import _read_quant, load_text_encoder, load_transformer, load_vae


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
    ) -> mx.array:
        """Generate one image, returned as an ``[H, W, 3]`` uint8 RGB array."""
        z = self.z_dim
        h_lat, w_lat = height // 16, width // 16
        tokens = h_lat * w_lat
        emb = self.text_encoder.encode(prompt).astype(mx.bfloat16)
        do_cfg = guidance is not None and guidance > 1.0
        neg = (
            self.text_encoder.encode(negative_prompt).astype(mx.bfloat16)
            if do_cfg
            else None
        )

        mx.random.seed(seed)
        latents = mx.random.normal((1, 1, z, h_lat, w_lat)).astype(mx.bfloat16)
        latents = latents.reshape(1, z, tokens).transpose(0, 2, 1)
        scheduler = FlowMatchEulerDiscreteScheduler(
            image_seq_len=tokens, num_inference_steps=steps
        )
        for i in range(steps):
            t = mx.array([float(scheduler.sigmas[i])], dtype=mx.bfloat16)
            pred = self.transformer(
                hidden_states=latents,
                encoder_hidden_states=emb,
                timestep=t,
                img_shape=(1, h_lat, w_lat),
            )
            if do_cfg:
                neg_pred = self.transformer(
                    hidden_states=latents,
                    encoder_hidden_states=neg,
                    timestep=t,
                    img_shape=(1, h_lat, w_lat),
                )
                pred = neg_pred + guidance * (pred - neg_pred)
            latents = scheduler.step(noise=pred, step_index=i, latents=latents)
            mx.eval(latents)

        z_lat = (
            latents.transpose(0, 2, 1).reshape(1, z, 1, h_lat, w_lat).astype(mx.float32)
        )
        z_lat = z_lat * self.latents_std + self.latents_mean
        image = self.vae.decode(z_lat)[:, :, 0]
        image = ((mx.clip(image, -1.0, 1.0) + 1.0) / 2.0 * 255).astype(mx.uint8)
        return image[0].transpose(1, 2, 0)[:, :, :3]


__all__ = ["QwenImagePipeline"]
