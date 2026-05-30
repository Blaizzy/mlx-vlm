from dataclasses import dataclass
from typing import Dict, List, Optional, Union

from ..base import BaseModelConfig
from ..qwen3_5.config import sanitize_quantization_config


@dataclass
class ModelConfig(BaseModelConfig):
    model_type: str = "hrm_text"
    vocab_size: int = 151808
    hidden_size: int = 1536
    intermediate_size: int = 4096
    num_hidden_layers: int = 16
    num_attention_heads: int = 12
    num_key_value_heads: int = 12
    head_dim: int = 128
    hidden_act: str = "silu"
    max_position_embeddings: int = 2048
    initializer_range: float = 0.02
    rms_norm_eps: float = 1e-6
    use_cache: bool = True
    pad_token_id: Optional[int] = None
    bos_token_id: Optional[int] = None
    eos_token_id: Optional[Union[int, List[int]]] = None
    tie_word_embeddings: bool = False
    rope_parameters: Optional[Dict] = None
    rope_theta: float = 10000.0
    attention_bias: bool = False
    attention_dropout: float = 0.0
    mlp_bias: bool = False
    H_cycles: int = 2
    L_cycles: int = 3
    L_bp_cycles: Optional[List[int]] = None
    embedding_scale: Optional[float] = None
    prefix_lm: bool = True
    num_layers_per_stack: Optional[int] = None
    quantization: Optional[Dict] = None
    quantization_config: Optional[Dict] = None

    def __post_init__(self):
        if self.L_bp_cycles is None:
            self.L_bp_cycles = [2]
        if self.embedding_scale is None:
            self.embedding_scale = 1.0 / self.initializer_range
        if self.num_layers_per_stack is None:
            self.num_layers_per_stack = self.num_hidden_layers
            self.num_hidden_layers = (
                self.num_layers_per_stack * self.H_cycles * (self.L_cycles + 1)
            )
        if self.rope_parameters is not None:
            self.rope_theta = self.rope_parameters.get(
                "rope_theta",
                self.rope_parameters.get("theta", self.rope_theta),
            )

        quantization = self.quantization
        self.quantization = sanitize_quantization_config(quantization)
        if self.quantization_config == quantization:
            self.quantization_config = self.quantization
        else:
            self.quantization_config = sanitize_quantization_config(
                self.quantization_config
            )
