from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union

from ..base import BaseModelConfig


@dataclass
class ModelConfig(BaseModelConfig):
    model_type: str
    vocab_size: int
    hidden_size: int
    intermediate_size: int
    num_hidden_layers: int
    num_attention_heads: int
    num_key_value_heads: int
    head_dim: int
    max_position_embeddings: int
    rms_norm_eps: float = 1e-6
    qkv_bias: bool = False
    attention_bias: bool = False
    gating: Union[bool, str] = True
    tie_word_embeddings: bool = False
    rope_theta: float = 500000.0
    rope_parameters: Optional[Dict[str, Any]] = None
    rope_scaling: Optional[Dict[str, Any]] = None
    partial_rotary_factor: Optional[float] = None
    rope_style: str = "rotate-half"
    sliding_window: Optional[int] = None
    layer_types: Optional[List[str]] = None
    num_attention_heads_per_layer: Optional[List[int]] = None
    swa_rope_parameters: Optional[Dict[str, Any]] = None
    swa_attention_sink_enabled: bool = False
    num_experts: int = 0
    num_experts_per_tok: int = 0
    moe_intermediate_size: int = 0
    shared_expert_intermediate_size: int = 0
    norm_topk_prob: bool = True
    decoder_sparse_step: int = 1
    mlp_only_layers: List[int] = field(default_factory=lambda: [0])
    mlp_layer_types: Optional[List[str]] = None
    moe_routed_scaling_factor: float = 1.0
    moe_apply_router_weight_on_input: bool = False
    moe_router_logit_softcapping: float = 0.0
    moe_router_use_sigmoid: bool = True
    bos_token_id: Optional[int] = None
    eos_token_id: Optional[Union[int, List[int]]] = None
    pad_token_id: Optional[int] = None

    def __post_init__(self):
        if self.gating is True:
            self.gating = "per-head"

        if self.layer_types is None:
            self.layer_types = ["full_attention"] * self.num_hidden_layers
        if len(self.layer_types) != self.num_hidden_layers:
            raise ValueError("layer_types must match num_hidden_layers.")

        if self.mlp_layer_types is not None:
            if len(self.mlp_layer_types) != self.num_hidden_layers:
                raise ValueError("mlp_layer_types must match num_hidden_layers.")
            self.mlp_only_layers = [
                idx
                for idx, layer_type in enumerate(self.mlp_layer_types)
                if layer_type == "dense"
            ]

        if self.num_attention_heads_per_layer is None:
            self.num_attention_heads_per_layer = [
                self.num_attention_heads
            ] * self.num_hidden_layers
        if len(self.num_attention_heads_per_layer) != self.num_hidden_layers:
            raise ValueError(
                "num_attention_heads_per_layer must match num_hidden_layers."
            )
        if any(
            h % self.num_key_value_heads for h in self.num_attention_heads_per_layer
        ):
            raise ValueError(
                "Every query-head count must be divisible by num_key_value_heads."
            )

        rope_parameters = (
            dict(self.rope_parameters)
            if self.rope_parameters is not None
            else (
                dict(self.rope_scaling)
                if self.rope_scaling is not None
                else {"rope_type": "default", "rope_theta": self.rope_theta}
            )
        )

        layer_types = set(self.layer_types)
        layer_rope_parameters = {
            k: v
            for k, v in rope_parameters.items()
            if k in layer_types and isinstance(v, dict)
        }
        if layer_rope_parameters:
            top_level_parameters = {
                k: v
                for k, v in rope_parameters.items()
                if k not in layer_types and not isinstance(v, dict)
            }

            def rope_parameters_for(layer_type: str) -> Dict[str, Any]:
                params = dict(layer_rope_parameters.get(layer_type, {}))
                for k, v in top_level_parameters.items():
                    params.setdefault(k, v)
                return params

            default_layer_type = (
                "full_attention"
                if "full_attention" in layer_rope_parameters
                else next(iter(layer_rope_parameters))
            )
            self.rope_parameters = rope_parameters_for(default_layer_type)

            if (
                self.swa_rope_parameters is None
                and "sliding_attention" in layer_rope_parameters
            ):
                self.swa_rope_parameters = rope_parameters_for("sliding_attention")
        else:
            self.rope_parameters = rope_parameters

        if self.swa_rope_parameters is not None:
            self.swa_rope_parameters = dict(self.swa_rope_parameters)

        self.rope_parameters.setdefault("rope_type", "default")
        if self.swa_rope_parameters is not None:
            self.swa_rope_parameters.setdefault("rope_type", "default")

        if self.partial_rotary_factor is not None:
            self.rope_parameters.setdefault(
                "partial_rotary_factor", self.partial_rotary_factor
            )
            if self.swa_rope_parameters is not None:
                self.swa_rope_parameters.setdefault(
                    "partial_rotary_factor", self.partial_rotary_factor
                )
