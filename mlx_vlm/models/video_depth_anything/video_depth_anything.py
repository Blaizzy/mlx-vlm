"""Video Depth Anything main model.

Consistent video depth estimation: DINOv2 backbone + temporal DPT head.
Outputs per-frame depth maps instead of text.
"""

from typing import Dict

import mlx.core as mx
import mlx.nn as nn

from .config import ModelConfig
from .dpt import DPTHeadTemporal, upsample_bilinear
from .vision import DINOv2


class Model(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.pretrained = DINOv2(config)
        self.head = DPTHeadTemporal(config)

    def __call__(self, x: mx.array) -> mx.array:
        """
        Args:
            x: (B, T, H, W, 3) channel-last, ImageNet-normalized frames. H and W
                must be multiples of ``config.patch_size``.
        Returns:
            Depth maps of shape (B, T, H, W), relative (or metric when
            ``config.metric``) disparity-like values.
        """
        B, T, H, W, _ = x.shape
        patch_h, patch_w = H // self.config.patch_size, W // self.config.patch_size
        features = self.pretrained.get_intermediate_layers(
            x.reshape(B * T, H, W, 3), self.config.intermediate_layer_idx
        )
        depth = self.head(features, patch_h, patch_w, T)
        # (B*T, H, W, 1) -> resize to input resolution
        depth = upsample_bilinear(depth, size=(H, W))
        depth = nn.relu(depth)
        return depth.reshape(B, T, H, W)

    def predict_depth(self, frames: mx.array) -> mx.array:
        """Depth for a single clip. frames: (T, H, W, 3) -> (T, H, W)."""
        return self(frames[None])[0]

    @staticmethod
    def sanitize(weights: Dict[str, mx.array]) -> Dict[str, mx.array]:
        """Convert PyTorch checkpoint weights to MLX channel-last layout."""
        sanitized = {}
        for k, v in weights.items():
            if v.ndim == 4:
                # resize_layers.0/.1 are ConvTranspose2d; resize_layers.3 is Conv2d
                if "resize_layers.0" in k or "resize_layers.1" in k:
                    # ConvTranspose2d: (in, out, kh, kw) -> (out, kh, kw, in)
                    v = v.transpose(1, 2, 3, 0)
                else:
                    # Conv2d: (out, in, kh, kw) -> (out, kh, kw, in)
                    v = v.transpose(0, 2, 3, 1)
            sanitized[k] = v
        return sanitized
