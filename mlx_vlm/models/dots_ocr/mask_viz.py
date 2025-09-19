from __future__ import annotations

import os
from typing import Sequence

import mlx.core as mx
import numpy as np
from PIL import Image, ImageDraw

from .processor import build_cu_seqlens


def block_mask_from_grid(grid_thw: Sequence[Sequence[int]]) -> mx.array:
    """Return dense block mask [S,S] where blocks correspond to images."""
    totals = [int(H * W) for _, H, W in grid_thw]
    total = int(sum(totals))
    mask = mx.zeros((total, total), dtype=mx.bool_)
    offset = 0
    for _, H, W in grid_thw:
        span = H * W
        start = offset
        stop = offset + span
        mask[start:stop, start:stop] = True
        offset = stop
    return mask


def overlay_patch_grid(
    image: Image.Image,
    patch: int = 14,
    color: tuple[int, int, int] = (255, 0, 0),
    step: int = 5,
) -> Image.Image:
    """Draw every Nth patch grid line to reduce clutter."""
    width, height = image.size
    out = image.copy()
    draw = ImageDraw.Draw(out)

    step_px = patch * step
    for x in range(0, width + 1, step_px):
        draw.line([(x, 0), (x, height)], fill=color, width=1)
    for y in range(0, height + 1, step_px):
        draw.line([(0, y), (width, y)], fill=color, width=1)

    return out


def save_mask_preview(
    pixels_1chw: mx.array, grid_thw: Sequence[Sequence[int]], out_path: str
) -> str:
    """Save a side-by-side preview showing the normalized image and mask heatmap."""
    x = np.array(pixels_1chw)
    if x.ndim != 4 or x.shape[0] != 1:
        raise ValueError("pixels_1chw must have shape [1, C, H, W]")
    _, channels, height, width = x.shape
    if channels != 3:
        raise ValueError("Expected 3 channels (RGB)")

    restored = x[0].transpose(1, 2, 0)
    restored = restored.clip(-3, 3)
    min_val = restored.min()
    max_val = restored.max()
    restored = (restored - min_val) / (max_val - min_val + 1e-6)
    left = Image.fromarray((restored * 255).astype(np.uint8))

    mask = np.array(block_mask_from_grid(grid_thw))
    first_tokens = grid_thw[0][1] * grid_thw[0][2]
    region = mask[:first_tokens, :first_tokens]
    row_mean = region.mean(axis=1).reshape(grid_thw[0][1], grid_thw[0][2])
    row_img = (row_mean / (row_mean.max() + 1e-6) * 255).astype(np.uint8)
    right = Image.fromarray(row_img).resize((width, height), Image.NEAREST).convert("L")
    right = right.convert("RGB")

    canvas = Image.new("RGB", (width * 2, height))
    canvas.paste(left, (0, 0))
    canvas.paste(right, (width, 0))

    out_dir = os.path.dirname(out_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    canvas.save(out_path)
    return out_path
