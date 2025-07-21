from dataclasses import dataclass, field
from typing import List, Optional, Tuple

from ..base import BaseModelConfig


@dataclass
class VisionConfig(BaseModelConfig):
    """Configuration class for Florence2 Vision model (DaViT)."""

    model_type: str = "davit"
    in_chans: int = 3
    num_classes: int = 1000
    depths: List[int] = field(default_factory=lambda: [1, 1, 9, 1])
    dim_embed: List[int] = field(default_factory=lambda: [128, 256, 512, 1024])
    num_heads: List[int] = field(default_factory=lambda: [4, 8, 16, 32])
    num_groups: List[int] = field(default_factory=lambda: [4, 8, 16, 32])
    window_size: int = 12
    mlp_ratio: float = 4.0
    drop_path_rate: float = 0.1
    patch_size: List[int] = field(default_factory=lambda: [7, 3, 3, 3])
    patch_stride: List[int] = field(default_factory=lambda: [4, 2, 2, 2])
    patch_padding: List[int] = field(default_factory=lambda: [3, 1, 1, 1])
    patch_prenorm: List[bool] = field(
        default_factory=lambda: [False, False, False, False]
    )
    qkv_bias: bool = True
    conv_at_attn: bool = True
    conv_at_ffn: bool = True
    hidden_size: int = 768
    image_size: Tuple[int, int] = (768, 768)


@dataclass
class TextConfig(BaseModelConfig):
    d_model: int = 768
    model_type: str = "florence2"
    encoder_attention_heads: int = 8
    decoder_attention_heads: int = 8
    encoder_ffn_dim: int = 3072
    decoder_ffn_dim: int = 3072
    dropout: float = 0.1
    attention_dropout: float = 0.0
    activation_dropout: float = 0.0
    activation_function: str = "gelu"
    init_std: float = 0.02
    encoder_layerdrop: float = 0.0
    decoder_layerdrop: float = 0.0
    scale_embedding: bool = False
    use_cache: bool = True
    max_position_embeddings: int = 1024
    vocab_size: int = 51289
    pad_token_id: int = 1
    bos_token_id: int = 0
    eos_token_id: int = 2
    encoder_layers: int = 6
    decoder_layers: int = 6


@dataclass
class ModelConfig(BaseModelConfig):
    """Configuration class for Florence2."""

    vision_config: VisionConfig
    text_config: TextConfig
    model_type: str = "florence2"
    vocab_size: int = 50265
    max_position_embeddings: int = 1024
    pad_token_id: int = 1
    bos_token_id: int = 0
    eos_token_id: int = 2
    image_token_index: int = 0
    image_feature_source: List[str] = field(
        default_factory=lambda: ["temporal_avg_pool", "spatial_avg_pool"]
    )
    visual_temporal_embedding: Optional[dict] = field(
        default_factory=lambda: {"type": "COSINE", "max_temporal_embeddings": 100}
    )
    image_pos_embed: Optional[dict] = field(
        default_factory=lambda: {"type": "learned_abs_2d", "max_pos_embeddings": 50}
    )
    eos_token_id: Optional[List[int]] = None
