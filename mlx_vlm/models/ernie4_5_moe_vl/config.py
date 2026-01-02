import inspect
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Union

from ..base import BaseModelConfig


@dataclass
class VisionConfig(BaseModelConfig):
    """DFNRopeVisionTransformer configuration."""

    model_type: str = "DFNRope_vision_transformer"
    depth: int = 32
    embed_dim: int = 1280
    hidden_size: int = 3584  # This should match embed_dim for DFNRope
    hidden_act: str = "quick_gelu"
    mlp_ratio: float = 4.0
    num_heads: int = 16
    in_channels: int = 3
    patch_size: int = 14
    spatial_merge_size: int = 2
    layer_norm_eps: float = 1e-6

    def __post_init__(self):
        # hidden_size should equal embed_dim for this architecture
        if self.hidden_size != self.embed_dim:
            self.hidden_size = self.embed_dim


@dataclass
class TextConfig(BaseModelConfig):
    hidden_size: int = 3584
    intermediate_size: int = 18944
    model_type: str = "ernie"
    max_position_embeddings: int = 131072
    num_attention_heads: int = 28
    num_key_value_heads: int = 4
    num_hidden_layers: int = 56
    rms_norm_eps: float = 1e-6
    vocab_size: int = 151936
    rope_theta: float = 1000000.0
    use_bias: bool = False
    tie_word_embeddings: bool = False
    compression_ratio: float = 1.0
    # MoE config
    moe_num_experts: Union[int, List[int]] = 128
    moe_layer_start_index: Union[int, List[int]] = 3
    moe_layer_end_index: Optional[Union[int, List[int]]] = 53
    moe_intermediate_size: Union[int, List[int]] = 1408
    moe_capacity: List[float] = field(default_factory=lambda: [1.2, 2.0, 2.0])
    moe_k: int = 2
    moe_layer_interval: int = 1
    moe_use_aux_free: bool = True
    moe_num_shared_experts: int = 0
    moe_gate_act: str = "softmax"
    moe_norm_gate_logits: bool = True
    head_dim: Optional[int] = None
    # 3D RoPE config
    rope_3d: bool = True
    freq_allocation: int = 20

    def __post_init__(self):
        if self.num_key_value_heads is None:
            self.num_key_value_heads = self.num_attention_heads
        if self.head_dim is None:
            self.head_dim = self.hidden_size // self.num_attention_heads


@dataclass
class ModelConfig(BaseModelConfig):
    text_config: TextConfig = None
    vision_config: VisionConfig = None
    model_type: str = "ernie4_5_moe_vl"
    ignore_index: int = -100
    # Token IDs
    im_patch_id: int = 151655
    image_token_id: int = 151655
    image_start_token_id: int = 151652
    image_end_token_id: int = 151653
    video_token_id: int = 151656
    video_start_token_id: int = 151657
    video_end_token_id: int = 151658
    vision_start_token_id: int = 151652
    vision_end_token_id: int = 151653
    vision_token_id: int = 151654
    vocab_size: int = 151936
    eos_token_id: Optional[List[int]] = None
    # Vision-language integration
    pixel_hidden_size: int = 1280
    hidden_size: int = 3584
    # Resampler config
    spatial_conv_size: int = 2
    temporal_conv_size: int = 2
    use_temporal_conv: bool = True
    # 3D RoPE config
    rope_3d: bool = True
    freq_allocation: int = 20

    @classmethod
    def from_dict(cls, params):
        # Copy text config parameters from root level (like qwen2_vl does)
        # This ensures update_module_configs works correctly
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
