"""Diffusers AutoencoderKL adapter for Z-Image."""

from __future__ import annotations

import mlx.core as mx
from mlx import nn

from mlx_vlm.models.flux2.vae.decoder.decoder import Flux2Decoder
from mlx_vlm.models.flux2.vae.encoder.encoder import Flux2Encoder

from .config import ZImageVAEConfig


class ZImageVAE(nn.Module):
    """AutoencoderKL without the optional quant/post-quant convolutions."""

    def __init__(self, config: ZImageVAEConfig | None = None) -> None:
        super().__init__()
        config = config or ZImageVAEConfig()
        self.config = config
        self.encoder = Flux2Encoder(
            in_channels=config.in_channels,
            out_channels=config.latent_channels,
            block_out_channels=config.block_out_channels,
            layers_per_block=config.layers_per_block,
        )
        self.decoder = Flux2Decoder(
            in_channels=config.latent_channels,
            out_channels=config.out_channels,
            block_out_channels=config.block_out_channels,
            layers_per_block=config.layers_per_block,
        )

    def encode(self, x: mx.array) -> mx.array:
        """Encode an NHWC image and return the mean latent."""
        moments = self.encoder(x.transpose(0, 3, 1, 2))
        mean, _ = mx.split(moments, 2, axis=1)
        return mean.transpose(0, 2, 3, 1)

    def decode(self, z: mx.array) -> mx.array:
        """Decode an NHWC latent into an NHWC image."""
        return self.decoder(z.transpose(0, 3, 1, 2)).transpose(0, 2, 3, 1)


def sanitize_vae_weights(
    weights: dict[str, mx.array],
    *,
    source_layout: bool | None = None,
) -> dict[str, mx.array]:
    """Map Diffusers and earlier MLX-VLM VAE layouts to shared VAE blocks."""
    if source_layout is None:
        source_layout = "encoder.conv_in.weight" in weights
    sanitized = {}
    for key, value in weights.items():
        key = key.replace(".to_out.0.", ".to_out.")
        key = key.replace("encoder.conv_in.conv2d.", "encoder.conv_in.")
        key = key.replace("encoder.conv_out.conv2d.", "encoder.conv_out.")
        key = key.replace("encoder.conv_norm_out.norm.", "encoder.conv_norm_out.")
        key = key.replace("decoder.conv_in.conv.", "decoder.conv_in.")
        key = key.replace("decoder.conv_out.conv.", "decoder.conv_out.")
        key = key.replace("decoder.conv_norm_out.norm.", "decoder.conv_norm_out.")
        if source_layout and value.ndim == 4:
            value = value.transpose(0, 2, 3, 1)
        sanitized[key] = value
    return sanitized


__all__ = ["ZImageVAE", "sanitize_vae_weights"]
