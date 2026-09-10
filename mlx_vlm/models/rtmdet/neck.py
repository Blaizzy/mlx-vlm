"""CSPNeXtPAFPN neck.

Inputs (from CSPNeXt for "m"):
  C3 = 192 channels, stride  8
  C4 = 384 channels, stride 16
  C5 = 768 channels, stride 32

Flow (mmdet naming):
  P5'  = reduce_layers[0](C5)                                            # 768 → 384
  P4_t = top_down_blocks[0](concat(upsample(P5'),  C4))                  # 768 → 384
  P4'  = reduce_layers[1](P4_t)                                          # 384 → 192
  P3_t = top_down_blocks[1](concat(upsample(P4'),  C3))                  # 384 → 192
  P4_b = bottom_up_blocks[0](concat(downsamples[0](P3_t),  P4'))         # 384 → 384
  P5_b = bottom_up_blocks[1](concat(downsamples[1](P4_b),  P5'))         # 768 → 768
  out  = [out_convs[0](P3_t), out_convs[1](P4_b), out_convs[2](P5_b)]    # → 192 each
"""

from typing import List

import mlx.core as mx
import mlx.nn as nn

from .backbone import ConvBN, CSPLayer


def _upsample_nearest(x: mx.array, factor: int = 2) -> mx.array:
    B, H, W, C = x.shape
    # simple nearest upsample: repeat along H and W
    x = mx.broadcast_to(x[:, :, None, :, None, :], (B, H, factor, W, factor, C))
    return x.reshape(B, H * factor, W * factor, C)


class CSPNeXtPAFPN(nn.Module):
    def __init__(self, in_channels: List[int], out_channels: int, num_blocks: int = 2):
        super().__init__()
        c3, c4, c5 = in_channels

        # Top-down
        self.reduce_layers = [
            ConvBN(c5, c4, kernel=1, stride=1),      # P5 → P4-channels
            ConvBN(c4, c3, kernel=1, stride=1),      # P4-out → P3-channels
        ]
        self.top_down_blocks = [
            CSPLayer(c_in=c5, c_out=c4, num_blocks=num_blocks,
                     add_identity=False, use_attention=False),
            CSPLayer(c_in=c4, c_out=c3, num_blocks=num_blocks,
                     add_identity=False, use_attention=False),
        ]

        # Bottom-up
        self.downsamples = [
            ConvBN(c3, c3, kernel=3, stride=2),
            ConvBN(c4, c4, kernel=3, stride=2),
        ]
        self.bottom_up_blocks = [
            CSPLayer(c_in=c4, c_out=c4, num_blocks=num_blocks,
                     add_identity=False, use_attention=False),
            CSPLayer(c_in=c5, c_out=c5, num_blocks=num_blocks,
                     add_identity=False, use_attention=False),
        ]

        # Project each level to the unified channel count consumed by the head.
        self.out_convs = [
            ConvBN(c3, out_channels, kernel=3, stride=1),
            ConvBN(c4, out_channels, kernel=3, stride=1),
            ConvBN(c5, out_channels, kernel=3, stride=1),
        ]

    def __call__(self, c3: mx.array, c4: mx.array, c5: mx.array):
        # Top-down
        p5 = self.reduce_layers[0](c5)                          # (stride 32, c4 channels)
        p4_t = self.top_down_blocks[0](
            mx.concatenate([_upsample_nearest(p5), c4], axis=-1)
        )                                                        # (stride 16, c4)
        p4 = self.reduce_layers[1](p4_t)                         # (stride 16, c3)
        p3_t = self.top_down_blocks[1](
            mx.concatenate([_upsample_nearest(p4), c3], axis=-1)
        )                                                        # (stride 8, c3)

        # Bottom-up
        p4_b = self.bottom_up_blocks[0](
            mx.concatenate([self.downsamples[0](p3_t), p4], axis=-1)
        )                                                        # (stride 16, c4)
        p5_b = self.bottom_up_blocks[1](
            mx.concatenate([self.downsamples[1](p4_b), p5], axis=-1)
        )                                                        # (stride 32, c5)

        return (
            self.out_convs[0](p3_t),  # stride  8, out_channels
            self.out_convs[1](p4_b),  # stride 16, out_channels
            self.out_convs[2](p5_b),  # stride 32, out_channels
        )
