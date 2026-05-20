"""Set-of-Marks visualisation helpers for the grounded reasoning agent."""

import numpy as np
from PIL import Image, ImageDraw, ImageFont

# Colour palette — each mask gets a distinct colour (cycles after 10)
_PALETTE = [
    (255, 80, 80),  # red
    (80, 200, 80),  # green
    (80, 120, 255),  # blue
    (255, 220, 50),  # yellow
    (220, 80, 220),  # magenta
    (50, 210, 210),  # cyan
    (255, 150, 40),  # orange
    (160, 80, 255),  # purple
    (50, 210, 140),  # spring-green
    (255, 80, 160),  # deep-pink
]


def _load_font(size=14):
    try:
        return ImageFont.load_default(size=size)
    except TypeError:
        return ImageFont.load_default()


def render_som(image, masks, interior_opacity=0.40, label_radius=13):
    """Render a Set-of-Marks overlay on image.

    Each mask is drawn as a semi-transparent coloured fill with a numbered
    white-circle label at its centroid.  Returns a new PIL RGB image.
    """
    img_rgb = image.convert("RGB")
    W, H = img_rgb.size
    base_np = np.array(img_rgb, dtype=np.uint8)

    if not masks:
        return img_rgb.copy()

    sorted_ids = sorted(masks.keys())

    # Build binary mask list and per-pixel index map
    idx_map = np.full((H, W), -1, dtype=np.int32)
    binary_masks = []
    for rank, mask_id in enumerate(sorted_ids):
        m = masks[mask_id].get("mask_np")
        if m is None:
            binary_masks.append(np.zeros((H, W), dtype=np.uint8))
            continue
        if m.shape != (H, W):
            m = np.array(
                Image.fromarray(m.astype(np.uint8)).resize((W, H), Image.NEAREST)
            ).astype(np.uint8)
        binary_masks.append(m)

    # Smallest masks render on top of larger ones
    areas = [m.sum() for m in binary_masks]
    draw_order = np.argsort(areas)[::-1]

    for rank_in_order in draw_order:
        m = binary_masks[rank_in_order]
        idx_map[m > 0] = rank_in_order

    has_mask = idx_map >= 0
    if has_mask.any():
        palette_np = np.array(_PALETTE, dtype=np.uint8)
        P = len(palette_np)
        ordered_colors = palette_np[
            np.array(
                [int(draw_order[i]) % P for i in range(len(sorted_ids))], dtype=np.intp
            )
        ]

        clamped = np.where(has_mask, idx_map, 0)
        fill_rgb = ordered_colors[clamped]

        composite = base_np.copy().astype(np.float32)
        mask_3d = has_mask[:, :, np.newaxis]
        composite = np.where(
            mask_3d,
            interior_opacity * fill_rgb.astype(np.float32)
            + (1.0 - interior_opacity) * composite,
            composite,
        )

        # Single-pixel contour borders
        border = np.zeros((H, W), dtype=np.bool_)
        border[:, 1:] |= idx_map[:, 1:] != idx_map[:, :-1]
        border[:, :-1] |= idx_map[:, 1:] != idx_map[:, :-1]
        border[1:, :] |= idx_map[1:, :] != idx_map[:-1, :]
        border[:-1, :] |= idx_map[1:, :] != idx_map[:-1, :]
        border &= has_mask
        if border.any():
            bright = np.clip(
                0.65 * ordered_colors.astype(np.float32) + 89.25, 0, 255
            ).astype(np.uint8)
            composite[border] = bright[np.where(border, idx_map, 0)][border]

        result_np = np.clip(composite, 0, 255).astype(np.uint8)
    else:
        result_np = base_np.copy()

    # Draw numbered circle labels at each mask centroid
    result_pil = Image.fromarray(result_np)
    draw = ImageDraw.Draw(result_pil)
    font = _load_font(size=max(12, label_radius))

    for mask_id in sorted_ids:
        meta = masks[mask_id]
        cx_norm = meta.get("centroid_norm", {}).get("x", 0.5)
        cy_norm = meta.get("centroid_norm", {}).get("y", 0.5)
        cx_px = int(cx_norm * W)
        cy_px = int(cy_norm * H)
        r = label_radius

        draw.ellipse(
            [cx_px - r, cy_px - r, cx_px + r, cy_px + r],
            fill="white",
            outline="black",
            width=2,
        )
        draw.text((cx_px, cy_px), str(mask_id), fill="black", font=font, anchor="mm")

    return result_pil


def render_final(image, masks, selected_ids):
    """Render only selected_ids masks on image for the final answer display."""
    selected = {k: v for k, v in masks.items() if k in selected_ids}
    return render_som(image, selected)


def get_crop(image, mask_info, padding_frac=0.15):
    """Return a padded bounding-box crop of image for the given mask."""
    W, H = image.size
    bbox = mask_info.get("bbox_norm", {"x1": 0, "y1": 0, "x2": 1, "y2": 1})
    x1 = bbox["x1"] * W
    y1 = bbox["y1"] * H
    x2 = bbox["x2"] * W
    y2 = bbox["y2"] * H

    pad_x = (x2 - x1) * padding_frac
    pad_y = (y2 - y1) * padding_frac
    x1 = max(0.0, x1 - pad_x)
    y1 = max(0.0, y1 - pad_y)
    x2 = min(float(W), x2 + pad_x)
    y2 = min(float(H), y2 + pad_y)

    return image.convert("RGB").crop((int(x1), int(y1), int(x2), int(y2)))
