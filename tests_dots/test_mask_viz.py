import os
import tempfile

import numpy as np
from PIL import Image

from mlx_vlm.models.dots_ocr.dots_ocr import DotsOCRConfig
from mlx_vlm.models.dots_ocr.mask_viz import overlay_patch_grid, save_mask_preview
from mlx_vlm.models.dots_ocr.processor import DotsOCRProcessor


def test_mask_preview_saves_png(tmp_path):
    cfg = DotsOCRConfig({})
    processor = DotsOCRProcessor(cfg)
    image = Image.fromarray((np.random.rand(233, 351, 3) * 255).astype("uint8"))

    pixels, grid = processor.process_one(image)
    out_path = tmp_path / "mask.png"
    result = save_mask_preview(pixels, grid, str(out_path))

    assert result == str(out_path)
    assert out_path.exists()
    assert out_path.suffix == ".png"

    overlay = overlay_patch_grid(image, patch=14, step=8)
    assert overlay.size == image.size
