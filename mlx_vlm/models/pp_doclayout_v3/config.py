"""PP-DocLayoutV3 configuration.

Flat HF-style config like ``rt_detr_v2``: backbone / hybrid-encoder /
decoder fields plus the reading-order head (``decoder_order_head`` +
``decoder_global_pointer``) and the mask-feature path used by the
mask-enhanced query init. Training-only loss/matcher fields are omitted.
"""

from dataclasses import dataclass, field
from typing import List, Optional

from ..base import BaseModelConfig


@dataclass
class HGNetV2Config(BaseModelConfig):
    """HGNetV2-L backbone configuration."""

    model_type: str = "hgnet_v2"
    arch: str = "L"
    depths: List[int] = field(default_factory=lambda: [3, 4, 6, 3])
    hidden_sizes: List[int] = field(default_factory=lambda: [256, 512, 1024, 2048])
    embedding_size: int = 64
    num_channels: int = 3
    hidden_act: str = "relu"
    stem_channels: List[int] = field(default_factory=lambda: [3, 32, 48])
    stem_strides: List[int] = field(default_factory=lambda: [2, 1, 1, 2, 1])
    stage_in_channels: List[int] = field(default_factory=lambda: [48, 128, 512, 1024])
    stage_mid_channels: List[int] = field(default_factory=lambda: [48, 96, 192, 384])
    stage_out_channels: List[int] = field(
        default_factory=lambda: [128, 512, 1024, 2048]
    )
    stage_num_blocks: List[int] = field(default_factory=lambda: [1, 1, 3, 1])
    stage_numb_of_layers: List[int] = field(default_factory=lambda: [6, 6, 6, 6])
    stage_downsample: List[bool] = field(
        default_factory=lambda: [False, True, True, True]
    )
    stage_downsample_strides: List[int] = field(default_factory=lambda: [2, 2, 2, 2])
    stage_light_block: List[bool] = field(
        default_factory=lambda: [False, False, True, True]
    )
    stage_kernel_size: List[int] = field(default_factory=lambda: [3, 3, 5, 5])
    use_learnable_affine_block: bool = False
    out_features: List[str] = field(
        default_factory=lambda: ["stage1", "stage2", "stage3", "stage4"]
    )
    out_indices: List[int] = field(default_factory=lambda: [1, 2, 3, 4])


@dataclass
class ModelConfig(BaseModelConfig):
    model_type: str = "pp_doclayout_v3"
    num_labels: int = 25
    id2label: Optional[dict] = None
    label2id: Optional[dict] = None
    backbone_config: Optional[HGNetV2Config] = None
    # Backbone / encoder / decoder geometry (RT-DETR core)
    d_model: int = 256
    encoder_hidden_dim: int = 256
    encoder_in_channels: List[int] = field(default_factory=lambda: [512, 1024, 2048])
    feat_strides: List[int] = field(default_factory=lambda: [8, 16, 32])
    encoder_layers: int = 1
    encoder_ffn_dim: int = 1024
    encoder_attention_heads: int = 8
    encoder_activation_function: str = "gelu"
    encode_proj_layers: List[int] = field(default_factory=lambda: [2])
    positional_encoding_temperature: int = 10000
    activation_function: str = "silu"
    normalize_before: bool = False
    layer_norm_eps: float = 1e-5
    hidden_expansion: float = 1.0
    batch_norm_eps: float = 1e-5
    eval_size: Optional[List[int]] = None
    decoder_layers: int = 6
    decoder_attention_heads: int = 8
    decoder_ffn_dim: int = 1024
    decoder_in_channels: List[int] = field(default_factory=lambda: [256, 256, 256])
    decoder_activation_function: str = "relu"
    num_feature_levels: int = 3
    decoder_n_points: int = 4
    num_queries: int = 300
    learn_initial_query: bool = False
    use_focal_loss: bool = True
    freeze_backbone_batch_norms: bool = True
    # Reading-order head
    global_pointer_head_size: int = 64
    gp_dropout_value: float = 0.1
    # Mask-enhanced query init path
    mask_enhanced: bool = True
    mask_feature_channels: List[int] = field(default_factory=lambda: [64, 64])
    num_prototypes: int = 32
    x4_feat_dim: int = 128

    def __post_init__(self):
        if self.backbone_config is None:
            self.backbone_config = HGNetV2Config()
        elif isinstance(self.backbone_config, dict):
            self.backbone_config = HGNetV2Config.from_dict(self.backbone_config)

        # JSON object keys arrive as strings; detection indexes by int.
        if isinstance(self.id2label, dict):
            self.id2label = {int(k): v for k, v in self.id2label.items()}
        if isinstance(self.label2id, dict):
            self.label2id = {str(k): int(v) for k, v in self.label2id.items()}
        # Stock configs omit num_labels; infer from the label map.
        if self.id2label and self.num_labels != len(self.id2label):
            self.num_labels = len(self.id2label)

        # Framework compatibility: sanitize_weights accesses these.
        self.text_config = None
        self.vision_config = None


# Backwards-compatible alias used across the detector modules.
LayoutConfig = ModelConfig
