import inspect
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Union

from ..base import BaseModelConfig


@dataclass
class VisionConfig(BaseModelConfig):
    model_type: str = "qwen3_vl_moe"
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
    num_position_embeddings: int = 2304
    window_size: int = 112
    fullatt_block_indexes: list[int] = field(default_factory=lambda: [7, 15, 23, 31])
    deepstack_visual_indexes: list[int] = field(default_factory=list)


@dataclass
class TextConfig(BaseModelConfig):
    model_type: str
    num_hidden_layers: int
    hidden_size: int
    intermediate_size: int
    num_attention_heads: int
    num_experts: int
    num_experts_per_tok: int
    decoder_sparse_step: int
    mlp_only_layers: List[int]
    moe_intermediate_size: int
    rms_norm_eps: float
    vocab_size: int
    num_key_value_heads: Optional[int]
    head_dim: int
    rope_theta: float
    max_position_embeddings: int
    norm_topk_prob: bool = True
    rope_scaling: Optional[Dict[str, Union[float, str, bool, List[int]]]] = field(
        default_factory=lambda: {"type": "default", "mrope_section": [24, 20, 20]}
    )
    tie_word_embeddings: bool = False
    attention_bias: bool = False
    hidden_act: str = "silu"

    def __post_init__(self):
        if self.num_key_value_heads is None:
            self.num_key_value_heads = self.num_attention_heads

        if self.rope_scaling:
            # Normalize rope_scaling keys (accept both 'rope_type' and 'type')
            if "type" not in self.rope_scaling and "rope_type" in self.rope_scaling:
                self.rope_scaling["type"] = self.rope_scaling.pop("rope_type")

            required_keys = {"mrope_section", "type"}
            if not all(key in self.rope_scaling for key in required_keys):
                raise ValueError(f"rope_scaling must contain keys {required_keys}")

            if not self.rope_scaling["type"] in ["mrope", "default"]:
                raise ValueError(f"rope_scaling type must be 'mrope' or 'default'")


@dataclass
class ModelConfig(BaseModelConfig):
    text_config: TextConfig
    vision_config: VisionConfig
    model_type: str
    ignore_index: int = -100
    image_token_id: int = 151655
    video_token_id: int = 151656
    image_token_index: Optional[int] = None
    video_token_index: Optional[int] = None
    vision_start_token_id: int = 151652
    vision_end_token_id: int = 151653
    vision_token_id: int = 151654
    vision_feature_select_strategy: str = "default"
    vision_feature_layer: int = -2
    vocab_size: int = 32000
    eos_token_id: Optional[List[int]] = None

    def __post_init__(self):
        if self.image_token_index is None:
            self.image_token_index = self.image_token_id
        if self.video_token_index is None:
            self.video_token_index = self.video_token_id

    @classmethod
    def from_dict(cls, params):

        return cls(
            **{
                k: v
                for k, v in params.items()
                if k in inspect.signature(cls).parameters
            }
        )
