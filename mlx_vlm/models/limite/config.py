from dataclasses import dataclass, field
from typing import Dict, List, Optional, Union

from ..base import BaseModelConfig


@dataclass
class ModelConfig(BaseModelConfig):
    model_type: str = "limite"
    hidden_size: int = 1280
    intermediate_size: int = 3328
    num_hidden_layers: int = 48
    num_attention_heads: int = 10
    num_key_value_heads: int = 2
    head_dim: int = 128
    vocab_size: int = 151680
    max_position_embeddings: int = 131072
    tie_word_embeddings: bool = True
    mlp_type: str = "swiglu"

    attention_softmax_scale: float = 0.1
    attn_gate_channels: int = 128
    attn_gate_scale: float = 2.0

    ve_dim: int = 128
    ve_gate_channels: int = 12
    ve_gate_scale: float = 2.0
    ve_stored_heads: int = 2
    ve_layers: List[int] = field(default_factory=list)

    xsa: bool = True
    xsa_layers: List[int] = field(default_factory=list)
    xsa_normalize_eps: float = 1e-4

    global_layers: List[int] = field(default_factory=list)
    global_nope: bool = True
    global_window: int = -1
    sliding_window: int = 1024

    rope_base_local: float = 1024.0
    rope_n_pairs: int = 32
    rope_per_layer: bool = False

    mudd: bool = True
    mudd_mlp: bool = True
    mudd_taps: int = 3
    mudd_inter: int = 32
    mudd_layers: List[int] = field(default_factory=list)
    mudd_tap_idx: Dict[str, List[int]] = field(default_factory=dict)

    final_softcap: float = 0.0
    softcap_logits: Optional[Dict[str, Union[str, float]]] = None

    eos_token_id: Optional[Union[int, List[int]]] = None
    quantization: Optional[Dict] = None
    quantization_config: Optional[Dict] = None

    def __post_init__(self):
        if self.rope_per_layer:
            raise NotImplementedError(
                "rope_per_layer=true needs a per-layer frequency table; the "
                "released checkpoint declares false."
            )
        if self.mlp_type not in ("swiglu", "relu2"):
            raise ValueError(f"unsupported mlp_type {self.mlp_type!r}")
        kind = (self.softcap_logits or {}).get("kind")
        if self.softcap_logits and kind != "sigmoid":
            raise ValueError(f"softcap_logits.kind={kind!r}; only sigmoid is supported")
        # mudd_tap_idx arrives from JSON with string keys.
        self.mudd_tap_idx = {
            int(layer): [int(i) for i in taps]
            for layer, taps in (self.mudd_tap_idx or {}).items()
        }
        if self.mudd_layers and set(self.mudd_tap_idx) != {
            int(x) for x in self.mudd_layers
        }:
            raise ValueError(
                f"mudd_tap_idx covers {sorted(self.mudd_tap_idx)} but mudd_layers "
                f"is {sorted(int(x) for x in self.mudd_layers)}"
            )
