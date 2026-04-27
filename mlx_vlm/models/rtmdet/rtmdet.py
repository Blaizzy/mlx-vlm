"""RTMDet top-level module + weight sanitize."""

from typing import Dict

import mlx.core as mx
import mlx.nn as nn

from .backbone import CSPNeXt
from .config import RTMDetConfig
from .head import RTMDetSepBNHead
from .neck import CSPNeXtPAFPN


class Model(nn.Module):
    def __init__(self, config: RTMDetConfig):
        super().__init__()
        self.config = config
        ch = config.channels()
        nb = config.num_blocks()

        self.backbone = CSPNeXt(channels=ch, num_blocks=nb)
        # Backbone returns (C3, C4, C5) at stride (8, 16, 32) with channels (ch[2], ch[3], ch[4]).
        self.neck = CSPNeXtPAFPN(
            in_channels=[ch[2], ch[3], ch[4]],
            out_channels=config.neck_channels,
            num_blocks=nb[0],
        )
        self.bbox_head = RTMDetSepBNHead(
            in_channels=config.neck_channels,
            num_classes=config.num_classes,
            num_levels=len(config.strides),
            stacked_convs=config.head_stacked_convs,
            strides=config.strides,
            exp_on_reg=True,  # mmdet's RTMDet-{m,l,x} use exp_on_reg=True
        )

    def __call__(self, pixel_values: mx.array):
        """pixel_values: (B, H, W, 3) channels-last, letterboxed to image_size."""
        c3, c4, c5 = self.backbone(pixel_values)
        feats = self.neck(c3, c4, c5)
        cls_outs, reg_outs = self.bbox_head(feats)
        return {"cls_outs": cls_outs, "reg_outs": reg_outs}

    @staticmethod
    def sanitize(weights: Dict[str, mx.array]) -> Dict[str, mx.array]:
        """Convert PyTorch mmdet checkpoint keys/shapes to MLX layout.

        - Drop running_mean / running_var tracking counters; BN's running stats
          are kept (eval mode) via BatchNorm's `running_mean` / `running_var`
          parameters, which MLX accepts as-is.
        - Rename list-container attribute paths so integer-indexed children
          match MLX's `self.children_list = [...]` serialization.
        - Transpose 4-D Conv2d weights (PT (out, in, H, W) → MLX (out, H, W, in)).
          Idempotent via kernel-position shape check.
        """
        sanitized: Dict[str, mx.array] = {}
        for k, v in weights.items():
            if "num_batches_tracked" in k:
                continue
            new_k = k

            # 4-D conv weight transpose, with idempotence shape check
            if v.ndim == 4:
                s = v.shape
                mlx_like = s[1] == s[2] and s[1] <= 16
                pt_like = s[2] == s[3] and s[2] <= 16
                if pt_like and not mlx_like:
                    v = v.transpose(0, 2, 3, 1)
                # else: already MLX layout

            sanitized[new_k] = v
        return sanitized


class VisionModel(nn.Module):
    """Expose the backbone under the framework's VisionModel slot (no-op sanitize)."""

    def __init__(self, config: RTMDetConfig):
        super().__init__()
        self.model = CSPNeXt(channels=config.channels(), num_blocks=config.num_blocks())

    def __call__(self, pixel_values: mx.array):
        return self.model(pixel_values)

    @staticmethod
    def sanitize(weights):
        return weights
