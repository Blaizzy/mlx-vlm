from dataclasses import dataclass
from typing import Dict, Optional, Union

from ..base import BaseModelConfig
from ..qwen3_5.config import sanitize_quantization_config


@dataclass
class ModelConfig(BaseModelConfig):
    model_type: str = "llada2_moe"
    vocab_size: int = 157184
    hidden_size: int = 2048
    intermediate_size: int = 5120
    num_hidden_layers: int = 20
    num_attention_heads: int = 16
    num_key_value_heads: int = 4
    head_dim: Optional[int] = None
    hidden_act: str = "silu"
    use_qkv_bias: bool = False
    use_qk_norm: bool = True
    use_bias: bool = False
    rms_norm_eps: float = 1e-6
    norm_head: bool = False
    tie_word_embeddings: bool = False
    attention_dropout: float = 0.0
    embedding_dropout: float = 0.0
    output_dropout: float = 0.0
    max_position_embeddings: int = 32768
    rope_theta: float = 600000
    rope_scaling: Optional[Dict[str, Union[float, str, bool]]] = None
    partial_rotary_factor: float = 0.5
    rotary_dim: Optional[int] = None
    use_cache: bool = False
    use_sliding_window: bool = False
    sliding_window: int = 4096
    max_window_layers: int = 28
    num_experts: int = 256
    num_shared_experts: int = 1
    num_experts_per_tok: int = 8
    n_group: int = 8
    topk_group: int = 4
    routed_scaling_factor: float = 2.5
    moe_intermediate_size: int = 512
    first_k_dense_replace: int = 1
    output_router_logits: bool = False
    pad_token_id: int = 156892
    eos_token_id: Optional[int] = None
    mask_token_id: int = 156895
    norm_topk_prob: bool = True
    score_function: str = "sigmoid"
    router_dtype: str = "fp32"
    moe_router_enable_expert_bias: bool = True
    quantization: Optional[Dict] = None
    quantization_config: Optional[Dict] = None

    def __post_init__(self):
        if self.head_dim is None:
            self.head_dim = self.hidden_size // self.num_attention_heads
        if self.rotary_dim is None:
            self.rotary_dim = int(self.head_dim * self.partial_rotary_factor)
        if self.eos_token_id is None:
            self.eos_token_id = self.pad_token_id

        quantization = self.quantization
        self.quantization = sanitize_quantization_config(quantization)
        if self.quantization_config == quantization:
            self.quantization_config = self.quantization
        else:
            self.quantization_config = sanitize_quantization_config(
                self.quantization_config
            )
