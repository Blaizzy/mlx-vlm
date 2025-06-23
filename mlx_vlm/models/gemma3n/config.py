import inspect
from dataclasses import dataclass
from typing import Dict, List, Optional, Union


@dataclass
class AudioConfig:
    input_feat_size: int = 80
    hidden_size: int = 1536
    conf_attention_chunk_size: int = 12
    conf_attention_context_left: int = 13
    conf_attention_context_right: int = 0
    conf_attention_invalid_logits_value: float = -1e9
    conf_attention_logit_cap: float = 50.0
    conf_num_attention_heads: int = 8
    conf_num_hidden_layers: int = 12
    conf_conv_kernel_size: int = 5
    conf_positional_bias_size: int = 256
    conf_reduction_factor: int = 4
    conf_residual_weight: float = 0.5
    sscp_conv_channel_size: tuple[int, int] = (128, 32)
    sscp_conv_group_norm_eps: float = 1e-3
    sscp_conv_kernel_size: tuple[tuple[int, int], tuple[int, int]] = ((3, 3), (3, 3))
    sscp_conv_stride_size: tuple[tuple[int, int], tuple[int, int]] = ((2, 2), (2, 2))
    vocab_size: int = 128
    sscp_conv_eps: float = 1e-3
    rms_norm_eps: float = 1e-6
    gradient_clipping: float = 10000000000.0
    vocab_offset: int = 262_144 + 128  # text vocab size + vision vocab size

    @classmethod
    def from_dict(cls, params):
        return cls(
            **{
                k: v
                for k, v in params.items()
                if k in inspect.signature(cls).parameters
            }
        )


@dataclass
class VisionConfig:
    model_type: str = "gemma3n_vision"
    num_hidden_layers: int = 12
    hidden_size: int = 2048
    intermediate_size: int = 8192
    num_attention_heads: int = 16
    patch_size: int = 16
    image_size: int = 224
    num_channels: int = 3
    rms_norm_eps: float = 1e-6
    vocab_size: int = 128
    vocab_offset: int = 262_144

    @classmethod
    def from_dict(cls, params):
        return cls(
            **{
                k: v
                for k, v in params.items()
                if k in inspect.signature(cls).parameters
            }
        )


@dataclass
class TextConfig:
    model_type: str
    hidden_size: int
    num_hidden_layers: int
    intermediate_size: int
    num_attention_heads: int = 2
    head_dim: int = 256
    rms_norm_eps: float = 1.0e-6
    vocab_size: int = 262400
    vocab_size_per_layer_input: int = 262144
    num_key_value_heads: int = 4
    laurel_rank: int = 64
    frac_shared_layers: float = 0.5
    altup_active_idx: int = 0
    pad_token_id: int = 0
    altup_num_inputs: int = 4
    altup_coef_clip: Optional[float] = None
    altup_correct_scale: bool = True
    hidden_size_per_layer_input: int = 1024
    rope_local_base_freq: float = 10000.0
    rope_traditional: bool = False
    rope_theta: float = 1000000.0
    query_pre_attn_scalar: float = 0.0625
    sliding_window: int = 1024
    rope_scaling: Optional[Dict[str, Union[float, List[float]]]] = None
    mm_tokens_per_image: int = 256
    sliding_window_pattern: int = 5
    activation_sparsity_pattern: Optional[List[float]] = None
    final_logit_softcapping: float = 30.0
    query_rescale_scalar: float = 1.0
    num_kv_shared_layers: int = 0
    max_position_embeddings: int = 32768
    attn_logit_softcapping: float = 0.0
    layer_types: List[str] = None

    @classmethod
    def from_dict(cls, params):
        return cls(
            **{
                k: v
                for k, v in params.items()
                if k in inspect.signature(cls).parameters
            }
        )


@dataclass
class ModelConfig:
    text_config: TextConfig
    vision_config: VisionConfig
    audio_config: AudioConfig
    model_type: str
    vocab_size: int = 257152
    ignore_index: int = -100
    image_token_index: int = 262145
    audio_token_id: int = 262273
    image_token_id: int = 262145
    hidden_size: int = 2048
    pad_token_id: int = 0
    vision_soft_tokens_per_image: int = 256
    audio_soft_tokens_per_image: int = 188
    eos_token_id: Optional[List[int]] = None

    @classmethod
    def from_dict(cls, params):
        return cls(
            **{
                k: v
                for k, v in params.items()
                if k in inspect.signature(cls).parameters
            }
        )
