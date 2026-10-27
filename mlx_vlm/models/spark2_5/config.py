from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from ..base import BaseModelConfig


@dataclass
class ModelConfig(BaseModelConfig):
    model_type: str = "spark2_5"
    hidden_size: int = 2560
    num_hidden_layers: int = 36
    intermediate_size: int = 10240
    num_attention_heads: int = 16
    num_key_value_heads: int = 4
    head_dim: int = 256
    vocab_size: int = 131072
    rms_norm_eps: float = 1e-6
    max_position_embeddings: int = 1048576
    sliding_window: int = 512
    tie_word_embeddings: bool = True
    attention_bias: bool = False
    mlp_bias: bool = False
    headwise_attn_output_gate: bool = True
    gate_attn_act_mode: str = "sigmoid"
    hidden_act: str = "gelu"
    layer_types: Optional[List[str]] = None
    rope_parameters: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if self.layer_types is None:
            self.layer_types = ["full_attention"] * self.num_hidden_layers

    def rope_for(self, layer_type: str):
        """(rotary_dims, theta) for a layer type. Spark uses a partial rotary
        factor + a distinct theta per full/sliding layer."""
        params = (
            self.rope_parameters.get(layer_type)
            or self.rope_parameters.get("full_attention")
            or {}
        )
        prf = params.get("partial_rotary_factor", 1.0)
        theta = params.get("rope_theta", 10000.0)
        dims = max(2, int(self.head_dim * prf))
        return dims, float(theta)
