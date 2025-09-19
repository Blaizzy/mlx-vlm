from __future__ import annotations

import argparse
from pathlib import Path

from PIL import Image

from mlx_vlm.models.dots_ocr.dots_ocr import DotsOCRConfig
from mlx_vlm.models.dots_ocr.mask_viz import overlay_patch_grid, save_mask_preview
from mlx_vlm.models.dots_ocr.processor import DotsOCRProcessor


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate mask visualization for a document page."
    )
    parser.add_argument("image", help="Path to the input image (document page)")
    parser.add_argument(
        "--out",
        default="mask_preview.png",
        help="Destination PNG for the visualization",
    )
    parser.add_argument(
        "--step",
        type=int,
        default=8,
        help="Stride of grid lines when drawing overlays",
    )
    args = parser.parse_args()

    cfg = DotsOCRConfig({})
    processor = DotsOCRProcessor(cfg)

    image_path = Path(args.image)
    image = Image.open(image_path).convert("RGB")

    overlay = overlay_patch_grid(
        image, patch=cfg.vision.patch_size, step=args.step
    )
    overlay_path = image_path.with_name("grid_overlay.png")
    overlay.save(overlay_path)

    pixels, grid = processor.process_one(image)
    preview_path = Path(args.out)
    save_mask_preview(pixels, grid, str(preview_path))

    print(f"Saved: {preview_path} and {overlay_path}")


if __name__ == "__main__":
    main()
