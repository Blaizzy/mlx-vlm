from dataclasses import dataclass
from typing import List, Optional

from ..base import BaseModelConfig


@dataclass
class TextConfig(BaseModelConfig):
    model_type: str
    hidden_size: int
    intermediate_size: int
    num_attention_heads: int
    rms_norm_eps: float
    vocab_size: int
    num_key_value_heads: int
    rope_theta: float = 500000.0
    num_hidden_layers: int = 48
    rope_traditional: bool = False
    rope_scaling: Optional[dict] = None
    tie_word_embeddings: bool = False
    head_dim: int = 128
    hidden_act: str = "silu"
    intermediate_size_mlp: int = 16384
    max_position_embeddings: int = 10485760
    num_experts_per_tok: int = 1
    num_local_experts: int = 16
    attention_dropout: float = 0.0
    use_qk_norm: bool = True
    bos_token_id: int = 200000
    eos_token_id: list = None
    pad_token_id: int = 200018
    attention_chunk_size: int = 8192
    attention_bias: bool = False
    interleave_moe_layer_step: int = 1
    no_rope_layers: list = 4
    output_router_logits: bool = False
    router_aux_loss_coef: float = 0.001
    router_jitter_noise: float = 0.0
    attn_temperature_tuning: int = 4
    floor_scale: float = 8192
    attn_scale: float = 0.1
    moe_layers: list = None

    def __post_init__(self):
        if self.num_key_value_heads is None:
            self.num_key_value_heads = self.num_attention_heads


@dataclass
class VisionConfig(BaseModelConfig):
    model_type: str
    hidden_size: int
    image_size: int
    initializer_range: float
    intermediate_size: int
    norm_eps: float
    num_attention_heads: int
    num_channels: int
    num_hidden_layers: int
    patch_size: int
    pixel_shuffle_ratio: float
    projector_dropout: float
    projector_input_dim: int
    projector_output_dim: int
    rope_theta: float
    vision_feature_layer: int
    vision_feature_select_strategy: str
    vision_output_dim: int


@dataclass
class ModelConfig(BaseModelConfig):
    text_config: TextConfig
    vision_config: VisionConfig
    model_type: str
    ignore_index: int = -100
    image_token_id: int = 200092
    image_token_index: Optional[int] = None
    eos_token_id: Optional[List[int]] = None

    def __post_init__(self):
        if self.image_token_index is None:
            self.image_token_index = self.image_token_id
