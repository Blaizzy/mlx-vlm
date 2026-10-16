import inspect
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Union

from ..base import BaseModelConfig
from ..muse_glimmer.config import VisionConfig as MuseGlimmerVisionConfig


def _config_kwargs(config_cls, params):
    return {
        key: value
        for key, value in params.items()
        if key in inspect.signature(config_cls).parameters
    }


@dataclass
class VisionConfig(MuseGlimmerVisionConfig):
    model_type: str = "muse_spark_vision"


@dataclass
class TextConfig(BaseModelConfig):
    model_type: str = "muse_spark_text"
    vocab_size: int = 202048
    hidden_size: int = 8192
    intermediate_size: int = 16384
    num_hidden_layers: int = 62
    num_attention_heads: int = 128
    num_key_value_heads: int = 16
    head_dim: int = 64
    hidden_act: str = "silu"
    max_position_embeddings: int = 1048576
    rms_norm_eps: float = 1e-5
    post_norm_eps: float = 1e-8
    attention_bias: bool = False
    attention_dropout: float = 0.0
    sliding_window: int = 2048
    rope_parameters: Optional[Dict] = None
    layer_types: Optional[List[str]] = None
    # 0 disables RoPE (NoPE) for that layer.
    layer_rope_theta: Optional[List[Union[int, float]]] = None
    qk_scale_factor: float = 5.4730064863875
    output_multiplier: float = 0.1767766952966369
    final_logit_softcapping: float = 20.0
    num_local_experts: int = 256
    num_experts_per_tok: int = 8
    # The experts run at moe_hidden_size, between two projections.
    moe_hidden_size: int = 4096
    moe_intermediate_size: int = 4096
    routed_scaling_factor: float = 1.0
    # "sparse" (MoE) or "dense" (MLP) per layer.
    mlp_layer_types: Optional[List[str]] = None
    residual_gate_temperature: float = 0.3
    tie_word_embeddings: bool = False
    bos_token_id: Optional[int] = 200000
    eos_token_id: Optional[Union[int, List[int]]] = 200001
    pad_token_id: Optional[int] = None

    def __post_init__(self):
        if self.rope_parameters is None:
            self.rope_parameters = {"rope_theta": 500000.0, "rope_type": "default"}
        if self.mlp_layer_types is None:
            self.mlp_layer_types = ["sparse"] * self.num_hidden_layers
        if self.layer_types is None:
            self.layer_types = [
                (
                    "full_attention"
                    if (self.num_hidden_layers - 1 - idx) % 4 == 0
                    else "sliding_attention"
                )
                for idx in range(self.num_hidden_layers)
            ]
        if self.layer_rope_theta is None:
            theta = self.rope_parameters.get("rope_theta", 500000.0)
            self.layer_rope_theta = [
                0 if (self.num_hidden_layers - 1 - idx) % 4 == 0 else theta
                for idx in range(self.num_hidden_layers)
            ]


@dataclass
class ModelConfig(BaseModelConfig):
    text_config: TextConfig = field(default_factory=TextConfig)
    vision_config: VisionConfig = field(default_factory=VisionConfig)
    model_type: str = "muse_spark"
    image_token_id: int = 200092
    video_token_id: int = 200091
    # The audio tower is not supported.
    audio_token_id: int = 200038
    out_hidden_size: int = 6144
    projector_hidden_size: int = 4096
    projector_hidden_act: str = "gelu"
    eos_token_id: Optional[Union[int, List[int]]] = None
    vocab_size: int = 202048
    thinking_start_token: str = "to=self<|message|>"
    thinking_end_token: str = "<|eom|>"

    def __post_init__(self):
        if self.eos_token_id is None:
            self.eos_token_id = self.text_config.eos_token_id
        self.vocab_size = self.text_config.vocab_size

    @classmethod
    def from_dict(cls, params):
        params = dict(params)
        text_config = params.get("text_config", {})
        vision_config = params.get("vision_config", {})
        if isinstance(text_config, dict):
            params["text_config"] = TextConfig(
                **_config_kwargs(TextConfig, text_config)
            )
        if isinstance(vision_config, dict):
            params["vision_config"] = VisionConfig(
                **_config_kwargs(VisionConfig, vision_config)
            )
        return cls(**_config_kwargs(cls, params))
