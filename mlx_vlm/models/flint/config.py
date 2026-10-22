from dataclasses import dataclass, field
from typing import Dict, List, Optional, Union

from ..base import BaseModelConfig


@dataclass
class ModelConfig(BaseModelConfig):
    model_type: str = "flint"
    hidden_size: int = 2048
    intermediate_size: int = 6144
    num_hidden_layers: int = 49
    num_attention_heads: int = 32
    num_key_value_heads: int = 4
    head_dim: int = 128
    vocab_size: int = 150144
    rms_norm_eps: float = 1e-6
    max_position_embeddings: int = 262144
    rope_theta: float = 10000.0
    rope_scaling: Optional[Dict] = None
    rope_parameters: Optional[Dict] = None
    layer_types: Optional[List[str]] = None
    no_rope_layer_indices: List[int] = field(default_factory=list)
    use_sliding_window: bool = True
    sliding_window: Optional[int] = 513
    softmax_type: str = "learnable"
    attention_bias: bool = False
    hidden_act: str = "silu"
    mlp_only_layers: List[int] = field(default_factory=list)
    decoder_sparse_step: int = 1
    num_experts: int = 256
    num_experts_per_tok: int = 6
    moe_intermediate_size: int = 512
    shared_expert_intermediate_size: int = 1024
    router_score_function: str = "sigmoid"
    norm_topk_prob: bool = True
    tie_word_embeddings: bool = False
    eos_token_id: Optional[Union[int, List[int]]] = None

    def __post_init__(self):
        if self.layer_types is None:
            self.layer_types = ["full_attention"] * self.num_hidden_layers
        self.mlp_only_layers = self.mlp_only_layers or []
        self.no_rope_layer_indices = self.no_rope_layer_indices or []
        if len(self.layer_types) != self.num_hidden_layers:
            raise ValueError("layer_types must match num_hidden_layers")
        if set(self.layer_types) - {"full_attention", "sliding_attention"}:
            raise ValueError("Unsupported Flint attention layer type")
        if not self.use_sliding_window:
            self.sliding_window = None
        if self.sliding_window is not None and self.sliding_window < 1:
            raise ValueError("sliding_window must be positive")
        if self.hidden_act != "silu" or self.router_score_function != "sigmoid":
            raise ValueError("Flint requires SiLU experts and sigmoid routing")
        if self.softmax_type != "learnable":
            raise ValueError("Flint requires learned attention sinks")
        if not 1 <= self.num_experts_per_tok <= self.num_experts:
            raise ValueError("num_experts_per_tok must be between 1 and num_experts")
        if self.rope_parameters is not None:
            self.rope_theta = self.rope_parameters.get("rope_theta", self.rope_theta)
            self.rope_scaling = self.rope_parameters
