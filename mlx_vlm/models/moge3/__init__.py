"""MoGe-3 for MLX.

Monocular geometry estimation: metric point maps, depth maps, normal maps,
and camera intrinsics from a single image. Unlike standard VLMs, this model
outputs dense geometry instead of text.

Usage:
    from mlx_vlm import load
    from mlx_vlm.models.moge3.generate import MoGe3Predictor

    model, processor = load("mlx-community/moge-3-vitl-mlx-fp32")
    predictor = MoGe3Predictor(model, processor)
    output = predictor.infer(image)  # image: (H, W, 3) uint8 RGB
"""

from . import processing_moge3  # Install processor patch
from .config import ModelConfig
from .moge3 import Model
