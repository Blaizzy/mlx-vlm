import inspect
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from ..base import BaseModelConfig


@dataclass
class VisionConfig(BaseModelConfig):
    model_type: str = "siglip2_vision_model"
    hidden_size: int = 1152
    out_hidden_size: int = 2560
    intermediate_size: int = 4304
    num_hidden_layers: int = 27
    num_attention_heads: int = 16
    num_channels: int = 3
    num_patches: int = 4096
    patch_size: int = 16
    hidden_act: str = "gelu_pytorch_tanh"
    layer_norm_eps: float = 1e-6
    attention_dropout: float = 0.0
    spatial_merge_size: int = 2
    window_size: int = 256
    fullatt_block_indexes: list = field(default_factory=lambda: [7, 15, 23, 26])


@dataclass
class TextConfig(BaseModelConfig):
    model_type: str = "youtu_vl"
    hidden_size: int = 2560
    intermediate_size: int = 9728
    num_hidden_layers: int = 40
    num_attention_heads: int = 32
    num_key_value_heads: int = 32
    vocab_size: int = 283386
    rms_norm_eps: float = 1e-6
    max_position_embeddings: int = 32768

    # MLA params
    kv_lora_rank: int = 512
    q_lora_rank: int = 1536
    qk_rope_head_dim: int = 64
    v_head_dim: int = 128
    qk_nope_head_dim: int = 128

    # RoPE params
    rope_theta: float = 500000.0
    rope_interleave: bool = True
    rope_traditional: bool = True
    rope_scaling: Optional[Dict] = None

    # MoE params (optional, for future larger models)
    n_shared_experts: Optional[int] = None
    n_routed_experts: Optional[int] = None
    moe_intermediate_size: Optional[int] = None
    num_experts_per_tok: int = 1
    n_group: int = 1
    topk_group: int = 1
    routed_scaling_factor: float = 1.0
    norm_topk_prob: bool = True
    moe_layer_freq: int = 1
    first_k_dense_replace: int = 0
    topk_method: str = "noaux_tc"
    scoring_func: str = "sigmoid"

    # Other
    tie_word_embeddings: bool = True
    attention_bias: bool = False
    mlp_bias: bool = False

    def __post_init__(self):
        if self.rope_interleave:
            self.rope_traditional = True


@dataclass
class ModelConfig(BaseModelConfig):
    text_config: TextConfig = None
    vision_config: VisionConfig = None
    model_type: str = "youtu_vl"
    image_token_id: int = 128264
    video_token_id: int = 128265
    vision_start_token_id: int = 128262
    vision_end_token_id: int = 128263
    vocab_size: int = 283386
    eos_token_id: Optional[List[int]] = None

    @classmethod
    def from_dict(cls, params):
        # Copy text config parameters from root level (excluding vision-specific keys)
        excluded_keys = {"vision_config"}
        params["text_config"] = dict(
            filter(lambda x: x[0] not in excluded_keys, params.items())
        )

        return cls(
            **{
                k: v
                for k, v in params.items()
                if k in inspect.signature(cls).parameters
            }
        )

    def __post_init__(self):
        if isinstance(self.vision_config, dict):
            self.vision_config = VisionConfig.from_dict(self.vision_config)
        if isinstance(self.text_config, dict):
            self.text_config = TextConfig.from_dict(self.text_config)
