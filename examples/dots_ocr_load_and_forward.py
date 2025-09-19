from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image
import mlx.core as mx

from mlx_vlm.models.dots_ocr.dots_ocr import DotsOCRConfig
from mlx_vlm.models.dots_ocr.dots_vision import DotsVisionTransformer_MLX
from mlx_vlm.models.dots_ocr.processor import DotsOCRProcessor
from mlx_vlm.models.dots_ocr.weight_loader import load_npz_into_vision


def _resolve_weight_path() -> str:
    here = Path(__file__).resolve().parent.parent
    weight_path = here / "weights" / "dots_ocr_vision.npz"
    return str(weight_path)


def main() -> None:
    cfg = DotsOCRConfig({"vision_config": {"num_layers": 2}})
    model = DotsVisionTransformer_MLX(cfg)

    weight_path = _resolve_weight_path()
    report = load_npz_into_vision(model, weight_path)
    print("[load]", weight_path, report)

    processor = DotsOCRProcessor(cfg)
    image = Image.fromarray((np.random.rand(360, 520, 3) * 255).astype("uint8"))
    pixels, grid = processor.process_one(image)

    mx.random.seed(0)
    output = model(pixels, grid)
    print("[forward] output shape:", tuple(output.shape))


def heavy_forward_demo() -> None:
    cfg = DotsOCRConfig({"vision_config": {"num_layers": 42}})
    model = DotsVisionTransformer_MLX(cfg)

    weight_path = _resolve_weight_path()
    report = load_npz_into_vision(model, weight_path)
    print("[load-42]", weight_path, report)

    processor = DotsOCRProcessor(cfg)
    image = Image.fromarray((np.random.rand(672, 672, 3) * 255).astype("uint8"))
    pixels, grid = processor.process_one(image)

    output = model(pixels, grid)
    print("[heavy-forward] output shape:", tuple(output.shape))


if __name__ == "__main__":
    main()
    # Uncomment to try a heavy pass
    # heavy_forward_demo()
