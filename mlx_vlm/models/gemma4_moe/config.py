from dataclasses import dataclass, field
from typing import Dict, List, Optional

from ..base import BaseModelConfig


@dataclass
class VisionConfig(BaseModelConfig):
    model_type: str = "gemma4_vision"
    hidden_size: int = 1152
    intermediate_size: int = 4304
    num_hidden_layers: int = 27
    num_attention_heads: int = 16
    num_key_value_heads: int = 16
    head_dim: int = 72
    global_head_dim: int = 72
    hidden_activation: str = "gelu_pytorch_tanh"
    rms_norm_eps: float = 1e-6
    max_position_embeddings: int = 131072
    attention_bias: bool = False
    attention_dropout: float = 0.0
    use_bidirectional_attention: str = "vision"
    layer_types: Optional[List[str]] = None
    rope_parameters: Optional[Dict] = None
    default_output_length: int = 280
    patch_size: int = 16
    position_embedding_size: int = 10240
    pooling_kernel_size: int = 3
    use_clipped_linears: bool = False
    vocab_offset: int = 262144
    vocab_size: int = 128

    def __post_init__(self):
        if self.layer_types is None:
            self.layer_types = ["full_attention"] * self.num_hidden_layers
        if self.rope_parameters is None:
            self.rope_parameters = {
                "full_attention": {"rope_theta": 100.0, "rope_type": "default"}
            }


@dataclass
class TextConfig(BaseModelConfig):
    model_type: str = "gemma4_text"
    hidden_size: int = 2816
    num_hidden_layers: int = 30
    intermediate_size: int = 2112
    num_attention_heads: int = 16
    head_dim: int = 256
    global_head_dim: int = 512
    rms_norm_eps: float = 1e-6
    vocab_size: int = 262144
    num_key_value_heads: int = 8
    num_global_key_value_heads: int = 2
    num_kv_shared_layers: int = 0
    pad_token_id: int = 0
    hidden_activation: str = "gelu_pytorch_tanh"
    hidden_size_per_layer_input: int = 0
    rope_traditional: bool = False
    rope_parameters: Optional[Dict] = None
    query_pre_attn_scalar: float = 256
    sliding_window: int = 1024
    sliding_window_pattern: int = 6
    _sliding_window_pattern: int = 6
    max_position_embeddings: int = 131072
    attention_bias: bool = False
    attention_dropout: float = 0.0
    attention_k_eq_v: bool = True
    use_bidirectional_attention: Optional[str] = None
    attn_logit_softcapping: Optional[float] = None
    final_logit_softcapping: float = 30.0
    # MoE
    enable_moe_block: bool = True
    num_experts: int = 128
    top_k_experts: int = 8
    expert_intermediate_size: int = 704
    layer_types: Optional[List[str]] = None
    tie_word_embeddings: bool = True

    def __post_init__(self):
        if self.rope_parameters is None:
            self.rope_parameters = {
                "full_attention": {
                    "rope_theta": 1000000.0,
                    "rope_type": "proportional",
                },
                "sliding_attention": {
                    "rope_theta": 10000.0,
                    "rope_type": "default",
                },
            }
        if self.layer_types is None:
            pattern = ["sliding_attention"] * (self.sliding_window_pattern - 1) + [
                "full_attention"
            ]
            self.layer_types = (pattern * (self.num_hidden_layers // len(pattern) + 1))[
                : self.num_hidden_layers
            ]


@dataclass
class ModelConfig(BaseModelConfig):
    text_config: TextConfig = field(default_factory=TextConfig)
    vision_config: VisionConfig = field(default_factory=VisionConfig)
    model_type: str = "gemma4"
    vocab_size: int = 262144
    ignore_index: int = -100
    image_token_id: int = 258880
    boi_token_id: int = 255999
    eoi_token_id: int = 258882
    hidden_size: int = 2816
    pad_token_id: int = 0
    vision_soft_tokens_per_image: int = 280
    eos_token_id: Optional[List[int]] = None
    tie_word_embeddings: bool = True
    initializer_range: float = 0.02
