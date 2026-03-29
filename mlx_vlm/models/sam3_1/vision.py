"""SAM 3.1 Vision Encoder: ViT backbone (reused from SAM 3) + TriViTDetNeck.

The TriViTDetNeck has 3 parallel FPN heads:
- convs: detection features
- interactive_convs: interactive prompt features (clicks/boxes)
- propagation_convs: tracking propagation features
"""

from typing import List, Tuple

import mlx.core as mx
import mlx.nn as nn

from ..sam3.vision import FPNLayer, ViTBackbone
from .config import VisionEncoderConfig


class TriViTDetNeck(nn.Module):
    """Triple-head FPN neck for SAM 3.1.

    Three parallel FPN pipelines sharing the same ViT backbone output.
    Only 3 scale factors [4.0, 2.0, 1.0] (no 0.5x downsample).

    Weight keys:
        detector_model.vision_encoder.neck.convs.*
        detector_model.vision_encoder.neck.interactive_convs.*
        detector_model.vision_encoder.neck.propagation_convs.*
    """

    def __init__(self, config: VisionEncoderConfig):
        super().__init__()
        backbone_config = config.backbone_config
        in_channels = backbone_config.hidden_size

        self.convs = [
            FPNLayer(
                in_channels,
                config.fpn_hidden_size,
                sf,
                config.fpn_kernel_size,
                config.fpn_stride,
            )
            for sf in config.scale_factors
        ]
        self.interactive_convs = [
            FPNLayer(
                in_channels,
                config.fpn_hidden_size,
                sf,
                config.fpn_kernel_size,
                config.fpn_stride,
            )
            for sf in config.scale_factors
        ]
        self.propagation_convs = [
            FPNLayer(
                in_channels,
                config.fpn_hidden_size,
                sf,
                config.fpn_kernel_size,
                config.fpn_stride,
            )
            for sf in config.scale_factors
        ]

    def __call__(
        self,
        x: mx.array,
        need_det: bool = True,
        need_interactive: bool = True,
        need_propagation: bool = True,
    ) -> Tuple[List[mx.array], List[mx.array], List[mx.array]]:
        """
        Args:
            x: (B, H, W, C) backbone output
            need_det: compute detection FPN
            need_interactive: compute interactive FPN
            need_propagation: compute propagation FPN
        Returns:
            (det_features, interactive_features, propagation_features)
            Each is a list of (B, H_i, W_i, D) at scales [4x, 2x, 1x]
        """
        det_features = []
        interactive_features = []
        propagation_features = []

        if need_det:
            for layer in self.convs:
                det_features.append(layer(x))

        if need_interactive:
            for layer in self.interactive_convs:
                interactive_features.append(layer(x))

        if need_propagation:
            for layer in self.propagation_convs:
                propagation_features.append(layer(x))

        return det_features, interactive_features, propagation_features


class VisionEncoder(nn.Module):
    """SAM 3.1 vision encoder: ViT backbone + TriViTDetNeck."""

    def __init__(self, config: VisionEncoderConfig):
        super().__init__()
        self.backbone = ViTBackbone(config.backbone_config)
        self.neck = TriViTDetNeck(config)

    def __call__(
        self,
        x: mx.array,
        need_det: bool = True,
        need_interactive: bool = True,
        need_propagation: bool = True,
    ) -> Tuple[List[mx.array], List[mx.array], List[mx.array]]:
        features = self.backbone(x)
        return self.neck(
            features,
            need_det=need_det,
            need_interactive=need_interactive,
            need_propagation=need_propagation,
        )
