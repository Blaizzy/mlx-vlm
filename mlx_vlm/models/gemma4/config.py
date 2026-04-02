from dataclasses import dataclass, field
from typing import Dict, List, Optional

from ..base import BaseModelConfig


@dataclass
class AudioConfig(BaseModelConfig):
    hidden_size: int = 1024
    num_hidden_layers: int = 12
    num_attention_heads: int = 8
    hidden_act: str = "silu"
    subsampling_conv_channels: tuple = (128, 32)
    conv_kernel_size: int = 5
    residual_weight: float = 0.5
    attention_chunk_size: int = 12
    attention_context_left: int = 13
    attention_context_right: int = 0
    attention_logit_cap: float = 50.0
    attention_invalid_logits_value: float = -1e9
    use_clipped_linears: bool = True
    rms_norm_eps: float = 1e-6
    gradient_clipping: float = 10000000000.0
    output_proj_dims: Optional[int] = 1536


@dataclass
class VisionConfig(BaseModelConfig):
    model_type: str = "gemma4_vision"
    hidden_size: int = 768
    intermediate_size: int = 3072
    num_hidden_layers: int = 16
    num_attention_heads: int = 12
    num_key_value_heads: int = 12
    head_dim: int = 64
    global_head_dim: int = 64
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
    standardize: bool = False

    def __post_init__(self):
        if self.layer_types is None:
            self.layer_types = ["full_attention"] * self.num_hidden_layers
        if self.rope_parameters is None:
            self.rope_parameters = {"rope_theta": 100.0, "rope_type": "default"}


@dataclass
class TextConfig(BaseModelConfig):
    model_type: str = "gemma4_text"
    hidden_size: int = 1536
    num_hidden_layers: int = 35
    intermediate_size: int = 6144
    num_attention_heads: int = 8
    head_dim: int = 256
    global_head_dim: int = 512
    global_partial_rotary_factor: float = 0.25
    rms_norm_eps: float = 1e-6
    vocab_size: int = 262144
    vocab_size_per_layer_input: int = 262144
    num_key_value_heads: int = 1
    num_global_key_value_heads: Optional[int] = None
    num_kv_shared_layers: int = 20
    pad_token_id: int = 0
    hidden_activation: str = "gelu_pytorch_tanh"
    hidden_size_per_layer_input: int = 256
    rope_traditional: bool = False
    partial_rotary_factor: float = 1.0
    rope_parameters: Optional[Dict] = None
    sliding_window: int = 512
    sliding_window_pattern: int = 5
    _sliding_window_pattern: int = 5
    max_position_embeddings: int = 131072
    attention_bias: bool = False
    attention_dropout: float = 0.0
    attention_k_eq_v: bool = False
    use_bidirectional_attention: Optional[str] = None
    final_logit_softcapping: float = 30.0
    use_double_wide_mlp: bool = True
    enable_moe_block: bool = False
    use_second_mlp_block: bool = False
    num_experts: Optional[int] = None
    top_k_experts: Optional[int] = None
    moe_intermediate_size: Optional[int] = None
    layer_types: Optional[List[str]] = None
    tie_word_embeddings: bool = True

    def __post_init__(self):
        if self.rope_parameters is None:
            self.rope_parameters = {
                "full_attention": {
                    "partial_rotary_factor": 1.0,
                    "rope_theta": 1000000.0,
                    "rope_type": "proportional",
                },
                "sliding_attention": {
                    "partial_rotary_factor": 1.0,
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
    audio_config: Optional[AudioConfig] = None
    model_type: str = "gemma4"
    vocab_size: int = 262144
    ignore_index: int = -100
    image_token_id: int = 258880
    audio_token_id: int = 258881
    boi_token_id: int = 255999
    eoi_token_id: int = 258882
    boa_token_id: int = 256000
    eoa_token_id: int = 258883
    hidden_size: int = 1536
    pad_token_id: int = 0
    vision_soft_tokens_per_image: int = 280
    audio_soft_tokens_per_image: int = 750
    audio_ms_per_token: int = 40
    eos_token_id: Optional[List[int]] = None
    tie_word_embeddings: bool = True
    initializer_range: float = 0.02
