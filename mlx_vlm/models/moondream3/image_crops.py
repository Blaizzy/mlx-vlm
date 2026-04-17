"""Multi-crop image processing for Moondream3.

Splits images into overlapping crops for high-resolution encoding.
"""

import math

import numpy as np
from PIL import Image


def select_crop_grid(image_width, image_height, crop_size, max_crops, overlap_margin):
    """Select the optimal crop grid layout for an image.

    Returns:
        (rows, cols) grid layout, or None if single global crop suffices.
    """
    patch_size = 14
    effective_crop = crop_size - 2 * overlap_margin * patch_size

    if effective_crop <= 0:
        return None

    best_layout = None
    best_score = float("inf")

    for rows in range(1, max_crops + 1):
        for cols in range(1, max_crops + 1):
            total_crops = rows * cols + 1  # +1 for global
            if total_crops > max_crops:
                continue

            # Effective resolution covered
            eff_h = rows * effective_crop + 2 * overlap_margin * patch_size
            eff_w = cols * effective_crop + 2 * overlap_margin * patch_size

            # Scale factor to cover the image
            scale_h = eff_h / image_height
            scale_w = eff_w / image_width

            # We want the scale to be close to 1 (covering image well)
            # while minimizing number of crops
            aspect_ratio = image_width / image_height
            grid_ratio = cols / rows

            # Penalize aspect ratio mismatch and over/under-coverage
            ratio_diff = abs(math.log(aspect_ratio / grid_ratio))
            scale_diff = abs(math.log(min(scale_h, scale_w)))
            score = ratio_diff + 0.5 * scale_diff + 0.1 * total_crops

            if score < best_score:
                best_score = score
                best_layout = (rows, cols)

    return best_layout


def create_crops(image, crop_size, max_crops, overlap_margin):
    """Create multi-resolution crops from an image.

    Args:
        image: PIL Image
        crop_size: size of each crop (378)
        max_crops: maximum number of crops including global
        overlap_margin: overlap in patches between adjacent crops

    Returns:
        crops: list of numpy arrays (H, W, C) normalized to [-1, 1]
        layout: (rows, cols) or None for single crop
    """
    width, height = image.size
    patch_size = 14

    # Always create global crop
    global_img = image.resize((crop_size, crop_size), Image.BICUBIC)
    global_crop = np.array(global_img).astype(np.float32)
    global_crop = (global_crop / 255.0 - 0.5) / 0.5  # normalize to [-1, 1]

    crops = [global_crop]

    # Check if we need local crops
    if max_crops <= 1:
        return crops, None

    layout = select_crop_grid(width, height, crop_size, max_crops, overlap_margin)
    if layout is None:
        return crops, None

    rows, cols = layout
    if rows * cols + 1 > max_crops:
        return crops, None

    # Calculate overlap in pixels
    overlap_pixels = overlap_margin * patch_size

    # Calculate the region each crop covers (with overlap)
    effective_size = crop_size - 2 * overlap_pixels

    for r in range(rows):
        for c in range(cols):
            # Source region in original image coordinates
            src_y = (
                r
                * (
                    height
                    - crop_size * height / (rows * effective_size + 2 * overlap_pixels)
                )
                if rows > 1
                else 0
            )
            src_x = (
                c
                * (
                    width
                    - crop_size * width / (cols * effective_size + 2 * overlap_pixels)
                )
                if cols > 1
                else 0
            )

            # Simpler approach: evenly distribute crops
            if rows > 1:
                src_y = (
                    r
                    * (height - crop_size * height / (rows * crop_size / crop_size))
                    / max(rows - 1, 1)
                )
            else:
                src_y = 0
            if cols > 1:
                src_x = (
                    c
                    * (width - crop_size * width / (cols * crop_size / crop_size))
                    / max(cols - 1, 1)
                )
            else:
                src_x = 0

            # Scale: map crop_size pixels to image region
            scale_x = (
                width / (cols * (crop_size - 2 * overlap_pixels) + 2 * overlap_pixels)
                if cols > 0
                else 1
            )
            scale_y = (
                height / (rows * (crop_size - 2 * overlap_pixels) + 2 * overlap_pixels)
                if rows > 0
                else 1
            )
            scale = max(scale_x, scale_y)

            crop_w = int(crop_size * scale)
            crop_h = int(crop_size * scale)

            # Center the crop grid
            total_w = cols * (crop_size - 2 * overlap_pixels) + 2 * overlap_pixels
            total_h = rows * (crop_size - 2 * overlap_pixels) + 2 * overlap_pixels

            x0 = int(c * (crop_size - 2 * overlap_pixels) * scale)
            y0 = int(r * (crop_size - 2 * overlap_pixels) * scale)

            x0 = min(x0, max(width - crop_w, 0))
            y0 = min(y0, max(height - crop_h, 0))
            x1 = min(x0 + crop_w, width)
            y1 = min(y0 + crop_h, height)

            crop_img = image.crop((x0, y0, x1, y1))
            crop_img = crop_img.resize((crop_size, crop_size), Image.BICUBIC)

            crop_arr = np.array(crop_img).astype(np.float32)
            crop_arr = (crop_arr / 255.0 - 0.5) / 0.5
            crops.append(crop_arr)

    return crops, layout
