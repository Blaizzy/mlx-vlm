from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import mlx.core as mx
import numpy as np


@dataclass(frozen=True, slots=True)
class TilingConfig:
    vae_decode_tiles_per_dim: int | None = 8
    vae_decode_tile_size: int = 128
    vae_decode_overlap: int = 8


def decode_image_tiled(
    *,
    latent: mx.array,
    decode_fn: Callable[[mx.array], mx.array],
    tile_size: tuple[int, int] = (512, 512),
    tile_overlap: tuple[int, int] = (64, 64),
    spatial_scale: int = 8,
) -> mx.array:
    batch, _, frames, height_lat, width_lat = latent.shape
    if batch != 1 or frames != 1:
        decoded = decode_fn(latent)
        return decoded[:, :, 0, :, :] if decoded.shape[2] == 1 else decoded

    scale = int(spatial_scale)
    height_out = height_lat * scale
    width_out = width_lat * scale
    tile_h, tile_w = tile_size
    overlap_h, overlap_w = tile_overlap
    latent_tile_h = max(1, tile_h // scale)
    latent_tile_w = max(1, tile_w // scale)

    if height_lat <= latent_tile_h and width_lat <= latent_tile_w:
        decoded = decode_fn(latent)
        return decoded[:, :, 0, :, :] if decoded.shape[2] == 1 else decoded

    latent_overlap_h = max(0, min(overlap_h // scale, latent_tile_h - 1))
    latent_overlap_w = max(0, min(overlap_w // scale, latent_tile_w - 1))
    stride_h = max(1, latent_tile_h - latent_overlap_h)
    stride_w = max(1, latent_tile_w - latent_overlap_w)
    ramp_h = _cos_ramp(overlap_h)
    ramp_w = _cos_ramp(overlap_w)
    out_np: np.ndarray | None = None
    count_np: np.ndarray | None = None

    for y_lat in range(0, height_lat, stride_h):
        y_lat_end = min(y_lat + latent_tile_h, height_lat)
        for x_lat in range(0, width_lat, stride_w):
            x_lat_end = min(x_lat + latent_tile_w, width_lat)
            if (y_lat > 0 and (y_lat_end - y_lat) <= latent_overlap_h) or (
                x_lat > 0 and (x_lat_end - x_lat) <= latent_overlap_w
            ):
                continue

            tile_latent = latent[:, :, :, y_lat:y_lat_end, x_lat:x_lat_end]
            decoded_tile = decode_fn(tile_latent)
            if decoded_tile.shape[2] == 1:
                decoded_tile = decoded_tile[:, :, 0, :, :]
            tile_np = np.array(decoded_tile.astype(mx.float32))[0].transpose(1, 2, 0)

            y_out = y_lat * scale
            x_out = x_lat * scale
            h_out = (y_lat_end - y_lat) * scale
            w_out = (x_lat_end - x_lat) * scale
            eff_h = min(h_out, tile_np.shape[0], height_out - y_out)
            eff_w = min(w_out, tile_np.shape[1], width_out - x_out)
            tile_np = tile_np[:eff_h, :eff_w, :]

            if out_np is None:
                out_np = np.zeros(
                    (height_out, width_out, tile_np.shape[2]), dtype=np.float32
                )
                count_np = np.zeros((height_out, width_out, 1), dtype=np.float32)

            ov_h_out = max(0, min(overlap_h, eff_h - 1))
            ov_w_out = max(0, min(overlap_w, eff_w - 1))
            wh = np.ones((eff_h,), dtype=np.float32)
            ww = np.ones((eff_w,), dtype=np.float32)
            if ov_h_out > 0:
                if y_lat > 0:
                    wh[:ov_h_out] = ramp_h[:ov_h_out]
                if y_lat_end < height_lat:
                    wh[-ov_h_out:] = 1.0 - ramp_h[:ov_h_out]
            if ov_w_out > 0:
                if x_lat > 0:
                    ww[:ov_w_out] = ramp_w[:ov_w_out]
                if x_lat_end < width_lat:
                    ww[-ov_w_out:] = 1.0 - ramp_w[:ov_w_out]

            weights = wh[:, None] * ww[None, :]
            out_np[y_out : y_out + eff_h, x_out : x_out + eff_w, :] += (
                tile_np * weights[:, :, None]
            )
            count_np[y_out : y_out + eff_h, x_out : x_out + eff_w, :] += weights[
                :, :, None
            ]

    assert out_np is not None and count_np is not None
    out_np = out_np / np.clip(count_np, 1e-6, None)
    return mx.array(out_np.transpose(2, 0, 1)[None, ...])


def _cos_ramp(n: int) -> np.ndarray:
    if n <= 0:
        return np.zeros((0,), dtype=np.float32)
    t = np.linspace(0.0, 1.0, num=n, dtype=np.float32)
    return 0.5 - 0.5 * np.cos(t * np.pi)
