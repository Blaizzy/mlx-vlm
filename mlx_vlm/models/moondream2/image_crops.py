"""Overlapping image crops for the Moondream2 vision encoder."""

import math

import mlx.core as mx
import numpy as np
from PIL import Image


def select_crop_grid(image_width, image_height, crop_size, max_crops, overlap_margin):
    margin = 2 * overlap_margin * 14
    height, width = image_height - margin, image_width - margin
    window = crop_size - margin
    if window <= 0 or max_crops < 1:
        raise ValueError("Crop size and crop budget must allow a positive crop window")
    if height <= window or width <= window:
        return (1, 1)
    min_h, min_w = math.ceil(height / window), math.ceil(width / window)
    if min_h * min_w > max_crops:
        ratio = math.sqrt(max_crops / (min_h * min_w))
        return max(1, math.floor(min_h * ratio)), max(1, math.floor(min_w * ratio))
    rows = max(math.floor(math.sqrt(max_crops * height / width)), min_h)
    columns = max(math.floor(math.sqrt(max_crops * width / height)), min_w)
    if rows * columns > max_crops:
        if columns > rows:
            columns = math.floor(max_crops / rows)
        else:
            rows = math.floor(max_crops / columns)
    return max(1, rows), max(1, columns)


def create_crops(image, crop_size, max_crops, overlap_margin):
    """Return normalized global and local crops; max_crops counts local crops."""
    image = image.convert("RGB")
    layout = select_crop_grid(*image.size, crop_size, max_crops, overlap_margin)
    rows, columns = layout
    margin = 2 * overlap_margin * 14
    window = crop_size - margin
    resized = np.asarray(
        image.resize(
            (columns * window + margin, rows * window + margin),
            Image.Resampling.LANCZOS,
        )
    )
    crops = [np.asarray(image.resize((crop_size, crop_size), Image.Resampling.LANCZOS))]
    crops.extend(
        resized[
            row * window : row * window + crop_size,
            column * window : column * window + crop_size,
        ]
        for row in range(rows)
        for column in range(columns)
    )
    pixels = mx.array(np.stack(crops)).astype(mx.bfloat16)
    # Match the reference's bfloat16 rounding at every normalization step.
    pixels = (pixels / 255.0).astype(mx.bfloat16)
    pixels = (pixels - 0.5).astype(mx.bfloat16)
    pixels = (pixels / 0.5).astype(mx.bfloat16)
    normalized = np.array(pixels.astype(mx.float32))
    return list(normalized), layout
