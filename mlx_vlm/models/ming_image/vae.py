"""Ming-Image VAE: the 4-channel RGBA single-frame ``AutoencoderKLQwenImage``."""

from __future__ import annotations

import mlx.core as mx

from mlx_vlm.models.qwen_image.vae import QwenImageVAE

from .config import MingImageVAEConfig


def build_vae(config: MingImageVAEConfig) -> QwenImageVAE:
    return QwenImageVAE(
        base_dim=config.base_dim,
        decoder_base_dim=config.base_dim,
        z_dim=config.z_dim,
        dim_mult=tuple(config.dim_mult),
        num_res_blocks=config.num_res_blocks,
        temperal_downsample=tuple(config.temperal_downsample),
        in_channels=config.in_channels,
        out_channels=config.out_channels,
        is_residual=config.is_residual,
    )


def sanitize_vae_weights(weights: dict[str, mx.array]) -> dict[str, mx.array]:
    out: dict[str, mx.array] = {}
    for key, value in weights.items():
        if key.endswith(".gamma"):
            value = value.reshape(-1)
        elif value.ndim == 5:
            value = value[:, :, -1].transpose(0, 2, 3, 1)
        elif value.ndim == 4:
            value = value.transpose(0, 2, 3, 1)
        out[key] = value
    return out


__all__ = ["build_vae", "sanitize_vae_weights"]
