"""RTMDetSepBNHead: per-level conv towers + 1x1 cls / reg outputs.

Each of the 3 FPN levels has its OWN cls_convs and reg_convs (separate BN =
`SepBN`), and a per-level 1x1 `rtm_cls` / `rtm_reg` projection.

Forward produces:
  cls_logits : list of 3 tensors, each (B, H_l, W_l, num_classes)
  bbox_preds : list of 3 tensors, each (B, H_l, W_l, 4)  — distance (l, t, r, b)
    expressed in **stride-normalized** units; the decoder multiplies by the
    level stride and adds to anchor-point coords to produce image-space boxes.
"""

from typing import List

import mlx.core as mx
import mlx.nn as nn

from .backbone import ConvBN


class RTMDetSepBNHead(nn.Module):
    def __init__(self, in_channels: int, num_classes: int = 1,
                 num_levels: int = 3, stacked_convs: int = 2,
                 strides=(8, 16, 32), exp_on_reg: bool = True):
        super().__init__()
        self.num_levels = num_levels
        self.num_classes = num_classes
        self.strides = list(strides)
        self.exp_on_reg = exp_on_reg

        # cls / reg towers: per level, a list of `stacked_convs` ConvBN units
        self.cls_convs = [
            [ConvBN(in_channels, in_channels, kernel=3, stride=1)
             for _ in range(stacked_convs)]
            for _ in range(num_levels)
        ]
        self.reg_convs = [
            [ConvBN(in_channels, in_channels, kernel=3, stride=1)
             for _ in range(stacked_convs)]
            for _ in range(num_levels)
        ]
        # 1x1 projections per level — have bias for focal-loss init.
        self.rtm_cls = [
            nn.Conv2d(in_channels, num_classes, kernel_size=1, stride=1, bias=True)
            for _ in range(num_levels)
        ]
        self.rtm_reg = [
            nn.Conv2d(in_channels, 4, kernel_size=1, stride=1, bias=True)
            for _ in range(num_levels)
        ]

    def __call__(self, feats):
        """Per MMDet `RTMDetSepBNHead.forward`:
          cls_score = rtm_cls(cls_tower(x))           # raw logits
          reg_dist  = rtm_reg(reg_tower(x)).exp() * stride   # if exp_on_reg else * stride
        The returned bbox_preds are in IMAGE-space pixels (the decoder just
        adds/subtracts from per-pixel anchor points, no further scaling)."""
        cls_outs = []
        reg_outs = []
        for level, feat in enumerate(feats):
            c = feat
            for conv in self.cls_convs[level]:
                c = conv(c)
            c = self.rtm_cls[level](c)

            r = feat
            for conv in self.reg_convs[level]:
                r = conv(r)
            r = self.rtm_reg[level](r)
            stride = self.strides[level]
            if self.exp_on_reg:
                r = mx.exp(r) * stride
            else:
                r = r * stride

            cls_outs.append(c)
            reg_outs.append(r)
        return cls_outs, reg_outs
