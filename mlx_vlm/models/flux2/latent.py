from __future__ import annotations

import mlx.core as mx

from mlx_vlm.models.flux2.constants import ModelConfig


def patchify_latents(latents: mx.array) -> mx.array:
    if latents.ndim == 5 and latents.shape[2] == 1:
        latents = latents[:, :, 0, :, :]
    if latents.ndim != 4:
        raise ValueError(f"Expected latents with ndim=4, got shape={latents.shape}")
    batch_size, num_channels, height, width = latents.shape
    latents = latents.reshape(batch_size, num_channels, height // 2, 2, width // 2, 2)
    latents = latents.transpose(0, 1, 3, 5, 2, 4)
    return latents.reshape(batch_size, num_channels * 4, height // 2, width // 2)


def pack_latents(latents: mx.array) -> mx.array:
    batch_size, num_channels, height, width = latents.shape
    return latents.reshape(batch_size, num_channels, height * width).transpose(0, 2, 1)


def unpack_latents(
    latents: mx.array, *, latent_height: int, latent_width: int
) -> mx.array:
    if latents.ndim == 4:
        return latents
    if latents.ndim != 3:
        raise ValueError(
            f"Expected packed latents with ndim=3, got shape={latents.shape}"
        )
    return latents.reshape(
        latents.shape[0], latent_height, latent_width, latents.shape[-1]
    ).transpose(0, 3, 1, 2)


def prepare_packed_latents(
    *,
    seed: int,
    height: int,
    width: int,
    batch_size: int = 1,
    num_latent_channels: int = 32,
    vae_scale_factor: int = 8,
) -> tuple[mx.array, mx.array, int, int]:
    height = 2 * (height // (vae_scale_factor * 2))
    width = 2 * (width // (vae_scale_factor * 2))
    latent_height = height // 2
    latent_width = width // 2
    latents = mx.random.normal(
        shape=(batch_size, num_latent_channels * 4, latent_height, latent_width),
        key=mx.random.key(seed),
    ).astype(ModelConfig.precision)
    latent_ids = prepare_grid_ids(latents, t_coord=0)
    return pack_latents(latents), latent_ids, latent_height, latent_width


def prepare_grid_ids(latents: mx.array, *, t_coord: int) -> mx.array:
    batch_size, _, height, width = latents.shape
    h_ids = mx.arange(height, dtype=mx.int32)
    w_ids = mx.arange(width, dtype=mx.int32)
    h_grid = mx.broadcast_to(mx.expand_dims(h_ids, axis=1), (height, width))
    w_grid = mx.broadcast_to(mx.expand_dims(w_ids, axis=0), (height, width))
    flat_h = h_grid.reshape(-1)
    flat_w = w_grid.reshape(-1)
    t = mx.full(flat_h.shape, t_coord, dtype=mx.int32)
    layer_ids = mx.zeros_like(flat_h)
    coords = mx.stack([t, flat_h, flat_w, layer_ids], axis=1)
    coords = mx.expand_dims(coords, axis=0)
    return mx.broadcast_to(coords, (batch_size, coords.shape[1], coords.shape[2]))
