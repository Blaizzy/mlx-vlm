from dataclasses import dataclass, field
from typing import Dict, Optional

from ..qwen3_vl.config import _config_kwargs, _maybe_deserialize_config
from ..step3p7.config import ModelConfig as Step3ModelConfig
from ..step3p7.config import TextConfig as Step3TextConfig
from ..step3p7.config import VisionConfig


@dataclass
class TextConfig(Step3TextConfig):
    model_type: str = "step5"
    intermediate_size: int = 13824
    num_hidden_layers: int = 92
    num_attention_groups: int = 4
    head_dim: int = 192
    moe_intermediate_size: int = 1536
    share_expert_dim: int = 1536
    moe_num_experts: int = 352
    moe_router_scaling_factor: float = 3.0
    sliding_window: int = 512
    use_head_wise_attn_gate: bool = True
    fp32_residual_connection: bool = True
    moe_layers_enum: tuple = field(default_factory=lambda: tuple(range(3, 91)))
    moe_layer_list: Optional[list] = None
    sparse_config: Optional[Dict] = None

    def __post_init__(self):
        super().__post_init__()
        if self.moe_layer_list is not None:
            self.moe_layers_enum = self.moe_layer_list


@dataclass
class ModelConfig(Step3ModelConfig):
    model_type: str = "step5_vl"

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
