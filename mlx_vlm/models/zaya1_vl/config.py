import inspect
from dataclasses import dataclass
from typing import Optional

from ..base import BaseModelConfig
from ..qwen2_5_vl.config import VisionConfig as Qwen2_5VisionConfig


@dataclass
class VisionConfig(Qwen2_5VisionConfig):
    @classmethod
    def from_dict(cls, params):
        if not params:
            return cls()

        params = dict(params)
        if "in_chans" in params and "in_channels" not in params:
            params["in_channels"] = params["in_chans"]
        if "spatial_patch_size" in params and "patch_size" not in params:
            params["patch_size"] = params["spatial_patch_size"]

        return cls(
            **{
                k: v
                for k, v in params.items()
                if k in inspect.signature(cls).parameters
            }
        )


@dataclass
class TextConfig(BaseModelConfig):
    model_type: str = "zaya1_vl"
    cca: bool = True
    num_query_groups: int = 2
    use_cache: bool = True
    attention_bias: bool = False
    lm_head_bias: bool = False
    vocab_size: int = 262272
    hidden_size: int = 2048
    ffn_hidden_size: int = 4096
    num_hidden_layers: int = 40
    num_experts: int = 16
    num_attention_heads: int = 8
    head_dim: int = 128
    activation_func: str = "swiglu"
    max_position_embeddings: int = 32768
    norm_epsilon: float = 1e-5
    pad_token_id: Optional[int] = 0
    bos_token_id: int = 2
    eos_token_id: int = 262143
    tie_word_embeddings: bool = True
    rope_theta: float = 1000000.0
    rotary_base: Optional[float] = None
    attention_dropout: float = 0.0
    moe_router_topk: int = 1
    normalization: str = "RMSNorm"
    zaya_mlp_expansion: int = 256
    zaya_use_mod: bool = True
    zaya_high_prec: bool = True
    zaya_use_eda: bool = True
    add_bias_linear: bool = False
    gated_linear_unit: bool = True
    scale_residual_merge: bool = True
    fused_add_norm: bool = False
    residual_in_fp32: bool = False
    sliding_window: Optional[int] = None
    rope_scaling: Optional[dict] = None
    rope_parameters: Optional[dict] = None
    partial_rotary_factor: float = 0.5
    rope_pct: Optional[float] = None
    num_key_value_heads: int = 2
    clamp_temp: bool = False
    cca_time0: int = 2
    cca_time1: int = 2
    swa_layers: Optional[list] = None
    swa_rotary_base: Optional[float] = None
    vision_lora: bool = True
    vision_lora_rank_attn: Optional[int] = 8
    vision_lora_rank_mlp: Optional[int] = 32
    use_lora_att: bool = False
    lora_rank: int = 0
    initializer_range: float = 0.02

    def __post_init__(self):
        if self.rotary_base is not None:
            self.rope_theta = self.rotary_base
        if self.rope_pct is not None:
            self.partial_rotary_factor = self.rope_pct
        if self.num_key_value_heads is None:
            self.num_key_value_heads = self.num_query_groups
        if self.num_query_groups != self.num_key_value_heads:
            raise ValueError("num_query_groups must match num_key_value_heads")

        rope_parameters = self.rope_parameters or self.rope_scaling or {}
        rope_parameters = dict(rope_parameters)
        if "type" in rope_parameters and "rope_type" not in rope_parameters:
            rope_parameters["rope_type"] = rope_parameters.pop("type")
        rope_parameters.setdefault("rope_type", "default")
        rope_parameters.setdefault("rope_theta", self.rope_theta)
        rope_parameters.setdefault("partial_rotary_factor", self.partial_rotary_factor)
        self.rope_parameters = rope_parameters


@dataclass
class ModelConfig(BaseModelConfig):
    text_config: TextConfig
    vision_config: VisionConfig
    model_type: str = "zaya1_vl"
    image_token_id: int = 262147
    vision_start_token_id: Optional[int] = 255999
    vision_end_token_id: Optional[int] = 256000
    projector_hidden_act: str = "gelu"
    vocab_size: int = 262272
    eos_token_id: int = 262143
    pad_token_id: int = 0

    @classmethod
    def from_dict(cls, params):
        excluded_keys = {"vision_config"}
        params["text_config"] = dict(
            filter(lambda item: item[0] not in excluded_keys, params.items())
        )

        return cls(
            **{
                k: v
                for k, v in params.items()
                if k in inspect.signature(cls).parameters
            }
        )
