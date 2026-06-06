import math
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

from mlx_lm.models.nemotron_h import ModelArgs

from ..base import BaseModelConfig


def _decode_float_sentinel(value):
    if isinstance(value, dict) and value.get("__float__") == "Infinity":
        return math.inf
    return value


@dataclass
class ModelConfig(BaseModelConfig):
    model_type: str = "nemotron_h"
    vocab_size: int = 131072
    hidden_size: int = 8192
    intermediate_size: int = 5120
    num_hidden_layers: int = 0
    max_position_embeddings: int = 262144
    num_attention_heads: int = 64
    num_key_value_heads: int = 2
    attention_bias: bool = False
    mamba_num_heads: int = 256
    mamba_head_dim: int = 64
    mamba_proj_bias: bool = False
    ssm_state_size: int = 128
    conv_kernel: int = 4
    n_groups: int = 8
    mlp_bias: bool = False
    layer_norm_epsilon: float = 1e-5
    use_bias: bool = False
    use_conv_bias: bool = True
    hybrid_override_pattern: Optional[List[str]] = None
    layers_block_type: Optional[List[str]] = None
    head_dim: Optional[int] = 128
    moe_intermediate_size: Optional[int] = 5120
    moe_shared_expert_intermediate_size: Optional[int] = 10240
    moe_latent_size: Optional[int] = 2048
    n_group: Optional[int] = 1
    n_routed_experts: Optional[int] = 512
    n_shared_experts: Optional[int] = 1
    topk_group: Optional[int] = 1
    num_experts_per_tok: Optional[int] = 22
    norm_topk_prob: Optional[bool] = True
    routed_scaling_factor: Optional[float] = 5.0
    time_step_limit: Optional[Tuple[float, float]] = None
    time_step_min: Optional[float] = 0.001
    time_step_max: Optional[float] = 0.1
    bos_token_id: Optional[int] = None
    eos_token_id: Optional[Union[int, List[int]]] = None
    pad_token_id: Optional[int] = None
    dtype: Optional[str] = None
    tie_word_embeddings: bool = False
    quantization: Optional[Dict[str, Any]] = None
    quantization_config: Optional[Dict[str, Any]] = None

    @classmethod
    def from_dict(cls, params):
        params = dict(params or {})
        if "layer_norm_epsilon" not in params and "norm_eps" in params:
            params["layer_norm_epsilon"] = params["norm_eps"]

        layers_block_type = params.get("layers_block_type")
        if params.get("num_hidden_layers") is None and layers_block_type is not None:
            params["num_hidden_layers"] = len(layers_block_type)

        time_step_limit = params.get("time_step_limit")
        if isinstance(time_step_limit, list):
            params["time_step_limit"] = tuple(
                _decode_float_sentinel(v) for v in time_step_limit
            )

        allowed = cls.__dataclass_fields__
        return cls(**{k: v for k, v in params.items() if k in allowed})

    def to_model_args(self) -> ModelArgs:
        values = asdict(self)
        return ModelArgs.from_dict(values)

