import inspect
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from ..base import BaseModelConfig


@dataclass
class VisionConfig(BaseModelConfig):
    """Configuration for the packed native-resolution vision encoder."""

    model_type: str = "cohere_compass_vision"
    depth: int = 27
    hidden_size: int = 1152
    hidden_act: str = "gelu_pytorch_tanh"
    intermediate_size: int = 4304
    num_heads: int = 16
    in_channels: int = 3
    patch_size: int = 16
    spatial_merge_size: int = 2
    temporal_patch_size: int = 2
    out_hidden_size: int = 3584
    num_position_embeddings: int = 2304
    deepstack_visual_indexes: List[int] = field(default_factory=lambda: [8, 16, 24])
    initializer_range: float = 0.02
    use_rope: bool = True


@dataclass
class TextConfig(BaseModelConfig):
    """Configuration for the Command-A+-style text decoder."""

    model_type: str = "cohere_compass_text"
    vocab_size: int = 131072
    hidden_size: int = 4096
    intermediate_size: int = 11008
    num_hidden_layers: int = 32
    num_attention_heads: int = 32
    num_key_value_heads: Optional[int] = None
    head_dim: Optional[int] = None
    hidden_act: str = "silu"
    max_position_embeddings: int = 8192
    initializer_range: float = 0.02
    rms_norm_eps: Optional[float] = 1e-6
    layer_norm_eps: float = 1e-5
    norm_type: str = "rms_norm"
    transformer_block_type: str = "vanilla"
    use_cache: bool = True
    tie_word_embeddings: bool = False
    pad_token_id: Optional[int] = None
    bos_token_id: Optional[int] = None
    eos_token_id: Optional[object] = None
    rope_theta: float = 10000.0
    swa_rope_theta: Optional[float] = None
    rope_parameters: Optional[Dict] = None
    rope_style: str = "split"
    rope_on_all_layers: bool = True
    attention_bias: bool = False
    attention_dropout: float = 0.0
    mlp_bias: bool = False
    is_causal: bool = True
    sliding_window: Optional[int] = None
    layer_types: Optional[List[str]] = None
    logit_scale: Optional[float] = None
    score_shift_a: Optional[float] = None
    score_shift_b: Optional[float] = None
    pooling: Optional[str] = None

    def __post_init__(self):
        if self.num_key_value_heads is None:
            self.num_key_value_heads = self.num_attention_heads
        if self.head_dim is None:
            self.head_dim = self.hidden_size // self.num_attention_heads
        if self.swa_rope_theta is None:
            self.swa_rope_theta = self.rope_theta
        if self.layer_types is None:
            self.layer_types = ["full_attention"] * self.num_hidden_layers


@dataclass
class ModelConfig(BaseModelConfig):
    """Top-level configuration for Cohere Compass multimodal checkpoints."""

    text_config: TextConfig = field(default_factory=TextConfig)
    vision_config: Optional[VisionConfig] = field(default_factory=VisionConfig)
    model_type: str = "cohere_compass"
    image_token_id: Optional[int] = None
    vision_start_token_id: Optional[int] = None
    vision_end_token_id: Optional[int] = None
    tie_word_embeddings: bool = False
    min_pixels: Optional[int] = None
    max_pixels: Optional[int] = None
    bos_token_id: Optional[int] = None
    eos_token_id: Optional[object] = None
    pad_token_id: Optional[int] = None

    @classmethod
    def from_dict(cls, params):
        params = dict(params or {})
        text_params = params.get("text_config")
        vision_params = params.get("vision_config")
        if isinstance(text_params, dict):
            params["text_config"] = TextConfig(
                **{
                    k: v
                    for k, v in text_params.items()
                    if k in inspect.signature(TextConfig).parameters
                }
            )
        elif text_params is None:
            params["text_config"] = TextConfig()
        if isinstance(vision_params, dict):
            params["vision_config"] = VisionConfig(
                **{
                    k: v
                    for k, v in vision_params.items()
                    if k in inspect.signature(VisionConfig).parameters
                }
            )
        elif vision_params is None:
            params["vision_config"] = None
        return cls(
            **{
                k: v
                for k, v in params.items()
                if k in inspect.signature(cls).parameters
            }
        )
