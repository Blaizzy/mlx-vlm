"""Utilities for extracting and visualizing MolmoPoint predictions."""

import re
from typing import List, Optional, Tuple

import numpy as np
from PIL import Image, ImageDraw

EXTRACT_POINT_TRIPLE = re.compile(
    r"<POINT_(\d+)> ?<POINT_(\d+)> ?<POINT_(\d+)> ?([0-9]+)"
)


def extract_points_from_text(
    output_text: str,
    pointing_metadata: dict,
    no_more_points_class: bool = True,
    patch_location: str = "3x3",
) -> List[Tuple[int, int, float, float]]:
    """Extract point coordinates from MolmoPoint output text.

    Args:
        output_text: Generated text containing <POINT_X> tokens
        pointing_metadata: Dict with 'token_pooling', 'subpatch_mapping', 'image_sizes'
        no_more_points_class: Whether the model has a no-more-points class
        patch_location: Location grid type ("3x3" or None)

    Returns:
        List of (object_id, image_num, x, y) tuples
    """
    pooling = pointing_metadata["token_pooling"]
    mappings = pointing_metadata["subpatch_mapping"]
    image_sizes = pointing_metadata["image_sizes"]

    n_patches, n_subpatches = pooling.shape[-2:]
    if no_more_points_class:
        n_patches += 1

    extracted_points = []
    for match in EXTRACT_POINT_TRIPLE.finditer(output_text):
        patch_id = int(match.group(1))
        subpatch_num = int(match.group(2))
        location_num = int(match.group(3))
        example_id = int(match.group(4))

        subpatch_id = subpatch_num - n_patches
        location_id = (
            location_num - n_patches - n_subpatches if patch_location else None
        )
        vit_patch_id = pooling[patch_id, subpatch_id]

        for image_ix, (mapping, (w, h)) in enumerate(zip(mappings, image_sizes)):
            patch_coords = np.argwhere(mapping == int(vit_patch_id))
            if len(patch_coords) == 1:
                p_y, p_x = patch_coords[0]
                if location_id is not None:
                    loc_x = location_id // 3
                    loc_y = location_id % 3
                    p_x += (loc_x + 0.5) * 0.33
                    p_y += (loc_y + 0.5) * 0.33
                else:
                    p_x += 0.5
                    p_y += 0.5
                extracted_points.append(
                    (
                        example_id,
                        image_ix,
                        (p_x / mapping.shape[1]) * w,
                        (p_y / mapping.shape[0]) * h,
                    )
                )
                break

    return extracted_points


def draw_points_on_image(
    image_path: str,
    points: List[Tuple[int, int, float, float]],
    output_path: Optional[str] = None,
    point_radius: int = 8,
    point_color: str = "red",
    label: bool = True,
) -> Image.Image:
    """Draw point predictions on an image.

    Args:
        image_path: Path to the original image
        points: List of (object_id, image_num, x, y) tuples
        output_path: Optional path to save the annotated image
        point_radius: Radius of the point marker
        point_color: Color of the point marker
        label: Whether to draw object ID labels

    Returns:
        Annotated PIL Image
    """
    img = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(img)

    colors = ["red", "blue", "green", "yellow", "purple", "orange", "cyan", "magenta"]

    for obj_id, img_num, x, y in points:
        color = colors[obj_id % len(colors)]
        r = point_radius

        # Draw filled circle
        draw.ellipse(
            [x - r, y - r, x + r, y + r],
            fill=color,
            outline="white",
            width=2,
        )

        # Draw crosshair
        draw.line([(x - r * 1.5, y), (x + r * 1.5, y)], fill="white", width=2)
        draw.line([(x, y - r * 1.5), (x, y + r * 1.5)], fill="white", width=2)

        if label:
            draw.text(
                (x + r + 3, y - r),
                str(obj_id),
                fill="white",
            )

    if output_path:
        img.save(output_path)
        print(f"Saved annotated image to {output_path}")

    return img
