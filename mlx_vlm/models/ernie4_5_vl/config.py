import inspect
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Union

from ..base import BaseModelConfig


@dataclass
class VisionConfig(BaseModelConfig):
    model_type: str = "ernie4_5_vl"
    depth: int = 32
    hidden_size: int = 1280
    intermediate_size: int = 3420
    out_hidden_size: int = 1536
    num_heads: int = 16
    image_size: int = 384
    patch_size: int = 14
    vocab_size: int = 32000
    mlp_ratio: float = 4.0
    in_channels: int = 3
    layer_norm_eps: float = 1e-6
    spatial_patch_size: int = 14
    spatial_merge_size: int = 2
    tokens_per_second: int = 2
    temporal_patch_size: int = 2
    window_size: int = 112
    patch_size: int = 14
    fullatt_block_indexes: list[int] = field(default_factory=lambda: [7, 15, 23, 31])


@dataclass
class TextConfig(BaseModelConfig):
    hidden_size: int
    intermediate_size: int
    model_type: str
    max_position_embeddings: int
    num_attention_heads: int
    num_key_value_heads: int
    num_hidden_layers: int
    rms_norm_eps: float
    vocab_size: int
    rope_theta: float
    use_bias: bool
    tie_word_embeddings: bool
    moe_num_experts: int
    moe_layer_start_index: int = 0
    moe_intermediate_size: int = 0
    moe_capacity: list[int] = field(default_factory=list)
    moe_k: int = 1
    moe_layer_interval: int = 1
    moe_use_aux_free: bool = False
    moe_num_shared_experts: int = 0
    moe_layer_end_index: Optional[int] = None
    head_dim: Optional[int] = None
    moe_gate_act: str = "softmax"

    def __post_init__(self):
        if self.num_key_value_heads is None:
            self.num_key_value_heads = self.num_attention_heads


@dataclass
class ModelConfig(BaseModelConfig):
    text_config: TextConfig
    vision_config: VisionConfig
    model_type: str
    ignore_index: int = -100
    image_token_id: int = 151655
    video_token_id: int = 151656
    vision_start_token_id: int = 151652
    vision_end_token_id: int = 151653
    vision_token_id: int = 151654
    vision_feature_select_strategy: str = "default"
    vision_feature_layer: int = -2
    vocab_size: int = 32000
    eos_token_id: Optional[List[int]] = None

    @classmethod
    def from_dict(cls, params):
        # Copy text config parameters from root level
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
