from dataclasses import dataclass, field
from typing import Dict, List, Optional, Union

from ..base import BaseModelConfig


@dataclass
class VisionConfig(BaseModelConfig):
    model_type: str = "qwen3_vl_moe"
    depth: int = 27
    hidden_size: int = 1152
    intermediate_size: int = 4304
    out_hidden_size: int = 2048
    num_heads: int = 16
    patch_size: int = 16
    in_channels: int = 3
    spatial_merge_size: int = 2
    temporal_patch_size: int = 2
    num_position_embeddings: int = 2304
    hidden_act: str = "gelu_pytorch_tanh"
    initializer_range: float = 0.02
    deepstack_visual_indexes: List[int] = field(default_factory=list)


@dataclass
class TextConfig(BaseModelConfig):
    model_type: str = "qwen3_vl_moe_text"
    hidden_size: int = 2048
    num_hidden_layers: int = 48
    intermediate_size: int = 6144
    num_attention_heads: int = 32
    rms_norm_eps: float = 1e-6
    vocab_size: int = 151936
    num_key_value_heads: int = 4
    max_position_embeddings: int = 262144
    rope_theta: float = 5000000.0
    rope_scaling: Optional[Dict[str, Union[float, str, List[int], bool]]] = None
    tie_word_embeddings: bool = True
    attention_bias: bool = False
    attention_dropout: float = 0.0
    hidden_act: str = "silu"
    head_dim: Optional[int] = None
    initializer_range: float = 0.02
    eos_token_id: Optional[int] = None
    use_cache: bool = True
    # MoE-specific config
    num_experts: int = 128
    num_experts_per_tok: int = 8
    moe_intermediate_size: int = 768
    decoder_sparse_step: int = 1
    mlp_only_layers: List[int] = field(default_factory=list)
    router_aux_loss_coef: float = 0.001
    norm_topk_prob: bool = True

    def __post_init__(self):
        if self.rope_scaling:
            rope_type = self.rope_scaling.get(
                "rope_type", self.rope_scaling.get("type")
            )
            if not rope_type:
                raise ValueError("rope_scaling must contain 'rope_type' or 'type'")
            if rope_type not in ["mrope", "default"]:
                raise ValueError(
                    f"rope_scaling type must be 'mrope' or 'default', got '{rope_type}'"
                )
            if "mrope_section" not in self.rope_scaling:
                raise ValueError("rope_scaling with mrope must contain 'mrope_section'")


@dataclass
class ModelConfig(BaseModelConfig):
    text_config: TextConfig
    vision_config: VisionConfig
    model_type: str = "qwen3_vl_moe"
    ignore_index: int = -100
    image_token_id: int = 151655
    video_token_id: int = 151656
    vision_start_token_id: int = 151652
    vision_end_token_id: int = 151653
    vision_token_id: int = 151654
    vocab_size: int = 151936
    tie_word_embeddings: bool = False
    eos_token_id: Optional[Union[int, List[int]]] = None
