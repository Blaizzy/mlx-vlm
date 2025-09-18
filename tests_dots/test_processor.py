import numpy as np
from PIL import Image

from mlx_vlm.models.dots_ocr.dots_ocr import DotsOCRConfig
from mlx_vlm.models.dots_ocr.processor import DotsOCRProcessor


def test_processor_single_image_shapes():
    cfg = DotsOCRConfig({})
    proc = DotsOCRProcessor(cfg)
    im = Image.fromarray((np.random.rand(333, 517, 3) * 255).astype("uint8"))
    pixels, grid = proc.process_one(im)
    assert pixels.shape[0] == 1 and pixels.shape[1] == 3
    H, W = pixels.shape[-2], pixels.shape[-1]
    assert H % 14 == 0 and W % 14 == 0
    assert grid == [[1, H // 14, W // 14]]
