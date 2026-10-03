"""DINOv2 encoder for MoGe-3 (channel-last MLX port).å"""

from typing import Tuple

import mlx.core as mx
import mlx.nn as nn

from ..dinov2.dinov2 import DINOv2Encoder
from .config import EncoderConfig


class MoGe3Encoder(DINOv2Encoder):
    """DINOv2 encoder plus per-layer 1x1 projections summed to dim_out."""

    def __init__(self, config: EncoderConfig):
        super().__init__(config)
        self.output_projections = [
            nn.Conv2d(config.embed_dim, config.dim_out, kernel_size=1)
            for _ in config.intermediate_layers
        ]

    def __call__(
        self, image: mx.array, token_rows: int, token_cols: int
    ) -> Tuple[mx.array, mx.array]:
        """image: (B, H, W, 3) in [0, 1] -> features (B, rows, cols, dim_out), cls."""
        features = super().__call__(image, token_rows, token_cols)
        projected = [
            proj(grid) for proj, (grid, _) in zip(self.output_projections, features)
        ]
        return sum(projected[1:], projected[0]), features[-1][1]
