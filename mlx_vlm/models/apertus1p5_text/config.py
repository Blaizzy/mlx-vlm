from dataclasses import dataclass
from typing import Any, Dict, Optional, Union

from ..base import BaseModelConfig


@dataclass
class ModelConfig(BaseModelConfig):
    model_type: str = "apertus1p5_text"
    hidden_size: int = 4096
    num_hidden_layers: int = 32
    intermediate_size: int = 21504
    num_attention_heads: int = 32
    num_key_value_heads: int = 8
    rms_norm_eps: float = 1e-5
    # Extended vocabulary: text + visual + audio code tokens.
    vocab_size: int = 266752
    max_position_embeddings: int = 262144
    post_norm: bool = False
    qk_norm: bool = True
    mlp_bias: bool = False
    attention_bias: bool = False
    tie_word_embeddings: bool = False
    # The head is physically pruned to the text-only prefix of `vocab_size`.
    output_vocab_size: Optional[int] = None
    # Apertus 1.5 carries a single `rope_parameters` dict; the trunk reads
    # `rope_theta` / `rope_scaling`, which are derived from it below.
    rope_parameters: Optional[Dict[str, Any]] = None
    rope_theta: float = 4000000.0
    rope_traditional: bool = False
    rope_scaling: Optional[Dict[str, Union[float, str]]] = None

    def __post_init__(self):
        if self.rope_parameters is None:
            return
        self.rope_theta = self.rope_parameters.get("rope_theta", self.rope_theta)
        scaling = {k: v for k, v in self.rope_parameters.items() if k != "rope_theta"}
        # `initialize_rope` dispatches on `rope_scaling["rope_type"]`.
        scaling.setdefault("rope_type", scaling.pop("type", "llama3"))
        self.rope_scaling = scaling
