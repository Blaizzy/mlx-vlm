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
    hidden_size: int = 2048
    num_hidden_layers: int = 48
    intermediate_size: int = 6144
    num_attention_heads: int = 32
    num_experts: int = 128
    num_experts_per_tok: int = 8
    decoder_sparse_step: int = 1
    mlp_only_layers: List[int] = field(default_factory=list)
    moe_intermediate_size: int = 768
    rms_norm_eps: float = 1e-06
    vocab_size: int = 151936
    num_key_value_heads: Optional[int] = 4
    head_dim: int = 128
    rope_theta: float = 5_000_000.0
    max_position_embeddings: int = 262144
    norm_topk_prob: bool = True
    rope_scaling: Optional[Dict[str, Union[float, str, bool, List[int]]]] = field(
        default_factory=lambda: {"type": "default", "mrope_section": [24, 20, 20]}
    )
    tie_word_embeddings: bool = True
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
