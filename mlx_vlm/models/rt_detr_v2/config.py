"""RT-DETRv2 configuration dataclasses.

Fields mirror the HuggingFace `RTDetrV2Config` schema
(`transformers.models.rt_detr_v2.configuration_rt_detr_v2`). `from_dict`
flattens an HF `config.json` straight into `ModelConfig`; the sub-configs
for backbone / hybrid encoder / decoder are constructed in
`__post_init__`.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Union

from ..base import BaseModelConfig


@dataclass
class RTDetrResNetConfig(BaseModelConfig):
    """ResNet-vd backbone configuration.

    The `vd` variant has a 3-stage stem (3x3 -> 3x3 -> 3x3, stride 2/1/1)
    followed by a 3x3 stride-2 max-pool, and uses `AvgPool2x2 stride 2 +
    1x1 conv` for downsampling shortcuts rather than a stride-2 1x1.
    Depths default to ResNet-50 (`[3, 4, 6, 3]`); ResNet-101 is
    `[3, 4, 23, 3]`. BatchNorms are typically frozen at inference.
    """

    model_type: str = "rt_detr_resnet"
    depths: List[int] = field(default_factory=lambda: [3, 4, 6, 3])
    downsample_in_bottleneck: bool = False
    downsample_in_first_stage: bool = False
    embedding_size: int = 64
    hidden_act: str = "relu"
    hidden_sizes: List[int] = field(default_factory=lambda: [256, 512, 1024, 2048])
    layer_type: str = "bottleneck"
    num_channels: int = 3
    out_features: List[str] = field(
        default_factory=lambda: ["stage2", "stage3", "stage4"]
    )
    out_indices: List[int] = field(default_factory=lambda: [2, 3, 4])
    stage_names: List[str] = field(
        default_factory=lambda: ["stem", "stage1", "stage2", "stage3", "stage4"]
    )


@dataclass
class RTDetrV2HybridEncoderConfig(BaseModelConfig):
    """Hybrid encoder (AIFI + FPN + PAN) configuration."""

    model_type: str = "rt_detr_v2_hybrid_encoder"
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


@dataclass
class RTDetrV2TransformerConfig(BaseModelConfig):
    """Decoder + query-selection configuration."""

    model_type: str = "rt_detr_v2_transformer"
    d_model: int = 256
    decoder_layers: int = 6
    decoder_attention_heads: int = 8
    decoder_ffn_dim: int = 1024
    decoder_in_channels: List[int] = field(default_factory=lambda: [256, 256, 256])
    decoder_activation_function: str = "relu"
    decoder_method: str = "default"
    decoder_n_levels: int = 3
    decoder_n_points: int = 4
    decoder_offset_scale: float = 0.5
    num_feature_levels: int = 3
    num_queries: int = 300
    num_labels: int = 17
    learn_initial_query: bool = False
    layer_norm_eps: float = 1e-5
    with_box_refine: bool = True
    use_focal_loss: bool = True


@dataclass
class ModelConfig(BaseModelConfig):
    """Top-level RT-DETRv2 config. HF stores backbone/encoder/decoder fields
    flat in `config.json`; `__post_init__` rebuilds the sub-configs from
    these flat fields."""

    model_type: str = "rt_detr_v2"
    image_size: int = 640
    num_labels: int = 17
    id2label: Optional[dict] = None
    label2id: Optional[dict] = None
    backbone_config: Optional[Union[dict, RTDetrResNetConfig]] = None
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
    decoder_method: str = "default"
    decoder_n_levels: int = 3
    decoder_n_points: int = 4
    decoder_offset_scale: float = 0.5
    num_feature_levels: int = 3
    num_queries: int = 300
    learn_initial_query: bool = False
    with_box_refine: bool = True
    use_focal_loss: bool = True
    freeze_backbone_batch_norms: bool = True

    def __post_init__(self):
        if self.backbone_config is None:
            self.backbone_config = RTDetrResNetConfig()
        elif isinstance(self.backbone_config, dict):
            self.backbone_config = RTDetrResNetConfig.from_dict(self.backbone_config)

        self._hybrid_encoder_config = RTDetrV2HybridEncoderConfig(
            encoder_hidden_dim=self.encoder_hidden_dim,
            encoder_in_channels=self.encoder_in_channels,
            feat_strides=self.feat_strides,
            encoder_layers=self.encoder_layers,
            encoder_ffn_dim=self.encoder_ffn_dim,
            encoder_attention_heads=self.encoder_attention_heads,
            encoder_activation_function=self.encoder_activation_function,
            encode_proj_layers=self.encode_proj_layers,
            positional_encoding_temperature=self.positional_encoding_temperature,
            activation_function=self.activation_function,
            normalize_before=self.normalize_before,
            layer_norm_eps=self.layer_norm_eps,
            hidden_expansion=self.hidden_expansion,
            batch_norm_eps=self.batch_norm_eps,
            eval_size=self.eval_size,
        )

        self._transformer_config = RTDetrV2TransformerConfig(
            d_model=self.d_model,
            decoder_layers=self.decoder_layers,
            decoder_attention_heads=self.decoder_attention_heads,
            decoder_ffn_dim=self.decoder_ffn_dim,
            decoder_in_channels=self.decoder_in_channels,
            decoder_activation_function=self.decoder_activation_function,
            decoder_method=self.decoder_method,
            decoder_n_levels=self.decoder_n_levels,
            decoder_n_points=self.decoder_n_points,
            decoder_offset_scale=self.decoder_offset_scale,
            num_feature_levels=self.num_feature_levels,
            num_queries=self.num_queries,
            num_labels=self.num_labels,
            learn_initial_query=self.learn_initial_query,
            layer_norm_eps=self.layer_norm_eps,
            with_box_refine=self.with_box_refine,
            use_focal_loss=self.use_focal_loss,
        )

        # Framework compatibility: sanitize_weights accesses these.
        self.text_config = None
        self.vision_config = None
