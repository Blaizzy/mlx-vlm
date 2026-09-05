from dataclasses import dataclass
from typing import List, Optional, Union

from ..base import BaseModelConfig


@dataclass
class ModelConfig(BaseModelConfig):
    model_type: str = "z1t"
    vocab_size: int = 50257
    hidden_size: int = 12288
    num_hidden_layers: int = 4
    max_position_embeddings: int = 256
    aft_kind: str = "conv"
    aft_heads: int = 4
    aft_ksize: int = 4
    dyt_alpha: float = 0.5
    linear_fan_in: Optional[int] = None  # shared fixed sparse fan-in; None = dense
    attn_fan_in: Optional[int] = None  # per-site override for attention linears
    mlp_fan_in: Optional[int] = None  # per-site override for MLP linears
    tanh_linear: bool = (
        False  # wrap every internal linear as tanh(Wx + b); lm_head exempt
    )
    tanh_mlp: bool = False  # MLP activation is tanh instead of silu
    tie_word_embeddings: bool = False
    bos_token_id: Optional[int] = None
    eos_token_id: Optional[Union[int, List[int]]] = None

    def fan_in(self, site: str) -> Optional[int]:
        override = getattr(self, f"{site}_fan_in")
        return override if override is not None else self.linear_fan_in
