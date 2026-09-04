from dataclasses import dataclass
from typing import Any, Dict, Optional

from ..base import BaseModelConfig


@dataclass
class ModelConfig(BaseModelConfig):
    model_type: str = "k2_horizon"
    hidden_size: int = 4096
    num_hidden_layers: int = 36
    intermediate_size: int = 12288
    num_attention_heads: int = 32
    num_key_value_heads: int = 8
    head_dim: int = 128
    vocab_size: int = 250624
    rms_norm_eps: float = 1e-6
    max_position_embeddings: int = 524288
    rope_theta: float = 1e7
    rope_scaling: Optional[Dict[str, Any]] = None
    attention_bias: bool = False
    tie_word_embeddings: bool = False

    @classmethod
    def from_dict(cls, params: Dict[str, Any]):
        params = dict(params or {})
        # K2-Horizon carries RoPE config under `rope_parameters`; translate it to the
        # (rope_theta, rope_scaling) pair that initialize_rope consumes. A "default"
        # type collapses to plain RoPE (rope_scaling=None); "yarn" keeps the scaling
        # dict (factor / beta_fast / beta_slow / original_max_position_embeddings).
        rope = params.pop("rope_parameters", None) or params.get("rope_scaling")
        if rope:
            rope = dict(rope)
            rope_type = rope.get("rope_type") or rope.get("type") or "default"
            params["rope_theta"] = rope.pop("rope_theta", params.get("rope_theta"))
            params["rope_scaling"] = None if rope_type == "default" else rope
        return cls(**{k: v for k, v in params.items() if k in cls.__dataclass_fields__})


K2HorizonConfig = ModelConfig
