from dataclasses import dataclass
from typing import Dict, List, Optional, Union

from ..base import BaseModelConfig


@dataclass
class ModelConfig(BaseModelConfig):
    model_type: str
    hidden_size: int
    num_hidden_layers: int
    num_attention_heads: int
    num_key_value_heads: int
    head_dim: int
    vocab_size: int
    max_position_embeddings: int
    rms_norm_eps: float
    num_experts: int
    num_experts_per_tok: int
    moe_intermediate_size: int
    shared_expert_intermediate_size: int
    sliding_window: int
    layer_types: List[str]
    hidden_act: str = "silu"
    norm_topk_prob: bool = False
    rope_theta: float = 10000.0
    rope_scaling: Optional[Dict[str, Union[float, str]]] = None
    rope_parameters: Optional[dict] = None
    tie_word_embeddings: bool = False
    attention_bias: bool = False
    attention_dropout: float = 0.0
    use_cache: bool = True
    use_sliding_window: bool = True
    head_dtype: str = "float32"

    def __post_init__(self):
        if len(self.layer_types) != self.num_hidden_layers:
            raise ValueError("layer_types must contain one entry per hidden layer")
        if self.rope_parameters is not None:
            self.rope_theta = self.rope_parameters.get("rope_theta", self.rope_theta)
