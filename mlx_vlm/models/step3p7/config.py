from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union

from ..base import BaseModelConfig
from ..qwen3_vl.config import _config_kwargs, _maybe_deserialize_config


def _remap_language_key(key: str) -> str:
    remappings = [
        (".moe.gate_proj", ".mlp.switch_mlp.gate_proj"),
        (".moe.up_proj", ".mlp.switch_mlp.up_proj"),
        (".moe.down_proj", ".mlp.switch_mlp.down_proj"),
        (".moe.gate", ".mlp.gate.gate"),
        (".moe.router_bias", ".mlp.gate.router_bias"),
        (".share_expert", ".mlp.share_expert"),
    ]
    for src, dst in remappings:
        if src in key and dst not in key:
            return key.replace(src, dst)
    return key


def _sanitize_quantization_config(quantization):
    if not isinstance(quantization, dict):
        return quantization
    return {_remap_language_key(key): value for key, value in quantization.items()}


@dataclass
class VisionConfig(BaseModelConfig):
    model_type: str = "perception_encoder"
    width: int = 1536
    layers: int = 47
    heads: int = 16
    num_channels: int = 3
    image_size: int = 728
    mlp_ratio: float = 8960 / 1536
    patch_size: int = 14
    hidden_act: str = "quick_gelu"
    layer_norm_eps: float = 1e-5
    use_cls_token: bool = False
    ues_cls_token: Optional[bool] = None
    use_ln_pre: bool = True
    use_ln_post: bool = False
    use_abs_posemb: bool = True
    use_rope2d: bool = True
    ls_init_value: Optional[float] = 0.1
    rope_theta: float = 10000.0
    rope_max_freq: int = 10
    rope_num_freqs: int = 1
    rope_theta_rescale_factor: float = 1.0
    rope_freqs_for: str = "lang"

    def __post_init__(self):
        if self.ues_cls_token is not None:
            self.use_cls_token = self.ues_cls_token


@dataclass
class TextConfig(BaseModelConfig):
    model_type: str = "step3p5"
    hidden_size: int = 4096
    intermediate_size: int = 11264
    num_attention_heads: int = 64
    num_attention_groups: int = 8
    num_hidden_layers: int = 45
    max_seq_len: int = 262144
    vocab_size: int = 128896
    rms_norm_eps: float = 1e-5
    moe_intermediate_size: int = 1280
    moe_num_experts: int = 288
    moe_top_k: int = 8
    rope_theta: Union[float, List[float]] = 10000.0
    rope_scaling: Optional[Dict[str, Any]] = None
    max_position_embeddings: int = 262144
    share_expert_dim: int = 1280
    share_expert_dims: Optional[int] = None
    head_dim: int = 128
    norm_expert_weight: bool = True
    layer_types: Optional[List[str]] = None
    sliding_window: Optional[int] = None
    pad_token_id: int = 1
    attention_dropout: float = 0.0
    use_head_wise_attn_gate: bool = False
    use_moe_router_bias: bool = False
    moe_router_activation: str = "softmax"
    moe_router_scaling_factor: float = 1.0
    need_fp32_gate: bool = False
    attention_other_setting: Optional[Dict[str, Any]] = None
    swiglu_limits: Optional[List[Optional[float]]] = None
    swiglu_limits_shared: Optional[List[Optional[float]]] = None
    use_rope_layers: Optional[List[bool]] = None
    yarn_only_types: Optional[List[str]] = None
    partial_rotary_factors: Optional[List[float]] = None
    moe_layers_enum: Union[str, List[int], tuple] = field(
        default_factory=lambda: tuple(range(3, 45))
    )
    eos_token_id: Optional[Union[int, List[int]]] = None
    bos_token_id: Optional[int] = None
    tie_word_embeddings: bool = False

    def __post_init__(self):
        if self.share_expert_dims is not None:
            self.share_expert_dim = self.share_expert_dims
        if self.layer_types is not None:
            self.layer_types = list(self.layer_types)[: self.num_hidden_layers]
        if self.swiglu_limits is not None:
            self.swiglu_limits = list(self.swiglu_limits)[: self.num_hidden_layers]
        if self.swiglu_limits_shared is not None:
            self.swiglu_limits_shared = list(self.swiglu_limits_shared)[
                : self.num_hidden_layers
            ]
        if isinstance(self.rope_scaling, dict) and "type" not in self.rope_scaling:
            self.rope_scaling = dict(self.rope_scaling)
            if "rope_type" in self.rope_scaling:
                self.rope_scaling["type"] = self.rope_scaling["rope_type"]


@dataclass
class ModelConfig(BaseModelConfig):
    text_config: TextConfig
    vision_config: VisionConfig
    model_type: str = "step3p7"
    understand_projector_stride: int = 2
    projector_bias: bool = False
    image_token_id: int = 128001
    image_token_index: Optional[int] = None
    vocab_size: int = 128896
    eos_token_id: Optional[Union[int, List[int]]] = None
    quantization: Optional[Dict] = None
    quantization_config: Optional[Dict] = None

    def __post_init__(self):
        if self.image_token_index is None:
            self.image_token_index = self.image_token_id
        if self.eos_token_id is None:
            self.eos_token_id = self.text_config.eos_token_id
        self.quantization = _sanitize_quantization_config(self.quantization)
        self.quantization_config = _sanitize_quantization_config(
            self.quantization_config
        )

    @classmethod
    def from_dict(cls, params):
        params = dict(params)
        params["vision_config"] = _maybe_deserialize_config(
            VisionConfig, params.get("vision_config")
        )
        params["text_config"] = _maybe_deserialize_config(
            TextConfig, params.get("text_config")
        )
        return cls(**_config_kwargs(cls, params))
