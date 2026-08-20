from __future__ import annotations

import mlx.core as mx
from mlx import nn

from mlx_vlm.models.flux2.tiling import TilingConfig, decode_image_tiled
from mlx_vlm.models.flux2.vae.common.batch_norm_stats import Flux2BatchNormStats
from mlx_vlm.models.flux2.vae.decoder.decoder import Flux2Decoder
from mlx_vlm.models.flux2.vae.encoder.encoder import Flux2Encoder


class Flux2VAE(nn.Module):
    scaling_factor: float = 1.0
    shift_factor: float = 0.0
    latent_channels: int = 32
    spatial_scale: int = 8

    def __init__(
        self,
        decoder_block_out_channels: tuple[int, ...] = (96, 192, 384, 384),
        *,
        include_encoder: bool = False,
        encoder_block_out_channels: tuple[int, ...] = (128, 256, 512, 512),
    ):
        super().__init__()
        self.encoder = (
            Flux2Encoder(block_out_channels=encoder_block_out_channels)
            if include_encoder
            else None
        )
        self.decoder = Flux2Decoder(block_out_channels=decoder_block_out_channels)
        self.quant_conv = (
            nn.Conv2d(
                2 * self.latent_channels,
                2 * self.latent_channels,
                kernel_size=1,
                padding=0,
            )
            if include_encoder
            else None
        )
        self.post_quant_conv = nn.Conv2d(
            self.latent_channels, self.latent_channels, kernel_size=1, padding=0
        )
        self.bn = Flux2BatchNormStats(
            num_features=4 * self.latent_channels, eps=1e-4, momentum=0.1
        )

    def encode(self, image: mx.array) -> mx.array:
        if self.encoder is None or self.quant_conv is None:
            raise RuntimeError("Flux2VAE was loaded without encoder weights")
        if image.ndim == 5:
            image = image[:, :, 0, :, :]
        enc = self.encoder(image)
        enc = mx.transpose(enc, (0, 2, 3, 1))
        enc = self.quant_conv(enc)
        enc = mx.transpose(enc, (0, 3, 1, 2))
        mean, _ = mx.split(enc, 2, axis=1)
        return (mean - self.shift_factor) * self.scaling_factor

    def decode(self, latents: mx.array) -> mx.array:
        if latents.ndim == 5:
            latents = latents[:, :, 0, :, :]
        latents = (latents / self.scaling_factor) + self.shift_factor
        latents = mx.transpose(latents, (0, 2, 3, 1))
        latents = self.post_quant_conv(latents)
        latents = mx.transpose(latents, (0, 3, 1, 2))
        return self.decoder(latents)

    def decode_packed_latents(
        self,
        packed_latents: mx.array,
        tiling_config: TilingConfig | None = None,
    ) -> mx.array:
        if packed_latents.ndim == 5:
            packed_latents = packed_latents[:, :, 0, :, :]
        bn_mean = self.bn.running_mean.reshape(1, -1, 1, 1)
        bn_std = mx.sqrt(self.bn.running_var.reshape(1, -1, 1, 1) + self.bn.eps)
        latents = packed_latents * bn_std + bn_mean
        latents = self._unpatchify_latents(latents)
        if (
            tiling_config is not None
            and tiling_config.vae_decode_tiles_per_dim
            and tiling_config.vae_decode_tiles_per_dim > 1
        ):
            latent_5d = latents[:, :, None, :, :]
            overlap_px = int(tiling_config.vae_decode_overlap) * self.spatial_scale
            return decode_image_tiled(
                latent=latent_5d,
                decode_fn=self.decode,
                tile_size=(
                    tiling_config.vae_decode_tile_size,
                    tiling_config.vae_decode_tile_size,
                ),
                tile_overlap=(overlap_px, overlap_px),
                spatial_scale=self.spatial_scale,
            )
        return self.decode(latents)

    @staticmethod
    def _unpatchify_latents(latents: mx.array) -> mx.array:
        batch_size, num_channels, height, width = latents.shape
        latents = mx.reshape(
            latents, (batch_size, num_channels // 4, 2, 2, height, width)
        )
        latents = mx.transpose(latents, (0, 1, 4, 2, 5, 3))
        return mx.reshape(
            latents, (batch_size, num_channels // 4, height * 2, width * 2)
        )
