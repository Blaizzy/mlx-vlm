from __future__ import annotations

import mlx.core as mx

from mlx_vlm.models.bonsai.constants import ModelConfig


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
    packed = latents.reshape(
        batch_size, num_latent_channels * 4, latent_height * latent_width
    )
    return packed.transpose(0, 2, 1), latent_ids, latent_height, latent_width


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
