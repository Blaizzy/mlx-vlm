from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union

from ..base import BaseModelConfig
from ..gemma4.config import VisionConfig
from ..qwen3_vl.config import _config_kwargs, _maybe_deserialize_config


@dataclass
class TextConfig(BaseModelConfig):
    model_type: str = "diffusion_gemma_text"
    vocab_size: int = 262144
    hidden_size: int = 2816
    intermediate_size: int = 2112
    moe_intermediate_size: int = 704
    num_hidden_layers: int = 30
    num_attention_heads: int = 16
    num_key_value_heads: int = 8
    num_global_key_value_heads: Optional[int] = 2
    head_dim: int = 256
    global_head_dim: int = 512
    hidden_activation: str = "gelu_pytorch_tanh"
    rms_norm_eps: float = 1e-6
    max_position_embeddings: int = 262144
    pad_token_id: int = 0
    eos_token_id: Optional[Union[int, List[int]]] = 1
    bos_token_id: Optional[int] = 2
    tie_word_embeddings: bool = True
    rope_parameters: Optional[Dict[str, Dict[str, Any]]] = None
    attention_bias: bool = False
    attention_dropout: float = 0.0
    sliding_window: int = 1024
    layer_types: Optional[List[str]] = None
    final_logit_softcapping: float = 30.0
    use_bidirectional_attention: Optional[str] = "vision"
    num_experts: int = 128
    top_k_experts: int = 8

    def __post_init__(self):
        if self.layer_types is None:
            pattern = ["sliding_attention"] * 5 + ["full_attention"]
            self.layer_types = (pattern * (self.num_hidden_layers // len(pattern) + 1))[
                : self.num_hidden_layers
            ]
            if self.layer_types[-1] != "full_attention":
                self.layer_types[-1] = "full_attention"

        if self.rope_parameters is None:
            self.rope_parameters = {
                "sliding_attention": {
                    "rope_type": "default",
                    "rope_theta": 10000.0,
                },
                "full_attention": {
                    "rope_type": "proportional",
                    "partial_rotary_factor": 0.25,
                    "rope_theta": 1000000.0,
                },
            }


@dataclass
class ModelConfig(BaseModelConfig):
    text_config: TextConfig = field(default_factory=TextConfig)
    vision_config: Optional[VisionConfig] = None
    model_type: str = "diffusion_gemma"
    boi_token_id: Optional[int] = 255999
    eoi_token_id: Optional[int] = 258882
    image_token_id: Optional[int] = 258880
    video_token_id: Optional[int] = None
    initializer_range: float = 0.02
    canvas_length: int = 256
    diffusion_generation_kind: str = "block"
    eos_token_id: Optional[Union[int, List[int]]] = None
    generation_config: Optional[Dict[str, Any]] = None
    dtype: Optional[str] = None

    @classmethod
    def from_dict(cls, params):
        params = dict(params or {})
        params["text_config"] = _maybe_deserialize_config(
            TextConfig, params.get("text_config"), require_all_fields=False
        )
        if params.get("vision_config") is not None:
            params["vision_config"] = _maybe_deserialize_config(
                VisionConfig, params.get("vision_config")
            )
        return cls(**_config_kwargs(cls, params))
