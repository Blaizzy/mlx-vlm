"""RF-DETR configuration."""

import inspect
from dataclasses import dataclass, field
from typing import List, Optional

from ..base import BaseModelConfig


@dataclass
class DINOv2Config(BaseModelConfig):
    model_type: str = "rfdetr_dinov2"
    hidden_size: int = 384
    num_hidden_layers: int = 12
    num_attention_heads: int = 6
    intermediate_size: int = 1536
    patch_size: int = 14
    num_channels: int = 3
    image_size: int = 518
    positional_encoding_size: Optional[int] = None  # Overrides image_size for pos embed init
    layer_norm_eps: float = 1e-6
    qkv_bias: bool = True
    out_feature_indexes: List[int] = field(
        default_factory=lambda: [2, 5, 8, 11]
    )


@dataclass
class ProjectorConfig(BaseModelConfig):
    model_type: str = "rfdetr_projector"
    hidden_dim: int = 256
    num_bottlenecks: int = 3
    bottleneck_channels: int = 128


@dataclass
class TransformerConfig(BaseModelConfig):
    model_type: str = "rfdetr_transformer"
    hidden_dim: int = 256
    dec_layers: int = 3
    sa_nheads: int = 8
    ca_nheads: int = 16
    dim_feedforward: int = 2048
    dec_n_points: int = 2
    n_levels: int = 1
    num_queries: int = 300
    group_detr: int = 13
    num_classes: int = 91
    two_stage: bool = True
    bbox_reparam: bool = True
    lite_refpoint_refine: bool = True
    layer_norm_eps: float = 1e-5


@dataclass
class SegmentationConfig(BaseModelConfig):
    model_type: str = "rfdetr_segmentation"
    in_dim: int = 256
    num_blocks: int = 4
    bottleneck_ratio: int = 1
    downsample_ratio: int = 4


@dataclass
class ModelConfig(BaseModelConfig):
    model_type: str = "rf-detr"
    resolution: int = 560
    hidden_dim: int = 256
    num_classes: int = 90
    num_queries: int = 300
    dec_layers: int = 3
    sa_nheads: int = 8
    ca_nheads: int = 16
    dec_n_points: int = 2
    group_detr: int = 13
    two_stage: bool = True
    bbox_reparam: bool = True
    lite_refpoint_refine: bool = True
    layer_norm: bool = True
    encoder: str = "dinov2_windowed_small"
    patch_size: int = 14
    num_windows: int = 4
    projector_scale: List[str] = field(default_factory=lambda: ["P4"])
    out_feature_indexes: List[int] = field(
        default_factory=lambda: [2, 5, 8, 11]
    )
    # Segmentation
    positional_encoding_size: Optional[int] = None  # Override for pos embed grid size
    segmentation: bool = False
    segmentation_config: Optional[dict] = None
    # Sub-configs
    backbone_config: Optional[dict] = None
    projector_config: Optional[dict] = None
    transformer_config: Optional[dict] = None
    # Framework compatibility
    text_config: Optional[dict] = None
    vision_config: Optional[dict] = None

    def __post_init__(self):
        # Build DINOv2 config from encoder type
        dinov2_sizes = {
            "dinov2_windowed_small": {"hidden_size": 384, "num_attention_heads": 6, "intermediate_size": 1536},
            "dinov2_windowed_base": {"hidden_size": 768, "num_attention_heads": 12, "intermediate_size": 3072},
            "dinov2_windowed_large": {"hidden_size": 1024, "num_attention_heads": 16, "intermediate_size": 4096},
        }
        encoder_params = dinov2_sizes.get(self.encoder, dinov2_sizes["dinov2_windowed_small"])

        # Normalize out_feature_indexes to 0-indexed layer indices
        # HF convention uses 1-indexed (stage3=after layer 2), detect uses 0-indexed
        num_layers = 12  # DINOv2 small/base
        if any(idx >= num_layers for idx in self.out_feature_indexes):
            self.out_feature_indexes = [idx - 1 for idx in self.out_feature_indexes]

        if self.backbone_config is None:
            self.backbone_config = DINOv2Config(
                out_feature_indexes=self.out_feature_indexes,
                patch_size=self.patch_size,
                positional_encoding_size=self.positional_encoding_size,
                **encoder_params,
            )
        elif isinstance(self.backbone_config, dict):
            self.backbone_config = DINOv2Config.from_dict(self.backbone_config)

        # Compute projector in_channels from backbone
        n_features = len(self.out_feature_indexes)
        in_channels = encoder_params["hidden_size"] * n_features

        if self.projector_config is None:
            self.projector_config = ProjectorConfig(
                hidden_dim=self.hidden_dim,
            )
        elif isinstance(self.projector_config, dict):
            self.projector_config = ProjectorConfig.from_dict(self.projector_config)

        # Store computed in_channels
        self.projector_config.in_channels = in_channels

        if self.transformer_config is None:
            self.transformer_config = TransformerConfig(
                hidden_dim=self.hidden_dim,
                dec_layers=self.dec_layers,
                sa_nheads=self.sa_nheads,
                ca_nheads=self.ca_nheads,
                dec_n_points=self.dec_n_points,
                num_queries=self.num_queries,
                group_detr=self.group_detr,
                num_classes=self.num_classes + 1,  # +1 for background
                two_stage=self.two_stage,
                bbox_reparam=self.bbox_reparam,
                lite_refpoint_refine=self.lite_refpoint_refine,
            )
        elif isinstance(self.transformer_config, dict):
            self.transformer_config = TransformerConfig.from_dict(self.transformer_config)

        # Auto-detect segmentation from weight keys (set externally)
        if self.segmentation_config is None and self.segmentation:
            self.segmentation_config = SegmentationConfig(in_dim=self.hidden_dim)
        elif isinstance(self.segmentation_config, dict):
            self.segmentation_config = SegmentationConfig.from_dict(self.segmentation_config)
