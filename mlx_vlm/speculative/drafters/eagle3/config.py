import inspect
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from ....models.base import BaseModelConfig


@dataclass
class TextConfig(BaseModelConfig):
    model_type: str = "llama"
    hidden_size: int = 4096
    intermediate_size: int = 11008
    num_hidden_layers: int = 1
    num_attention_heads: int = 32
    num_key_value_heads: int = 32
    head_dim: Optional[int] = None
    rms_norm_eps: float = 1e-6
    vocab_size: int = 32000
    max_position_embeddings: int = 131072
    rope_theta: float = 10000.0
    rope_parameters: Optional[Dict] = None
    rope_traditional: bool = False
    attention_bias: bool = False
    hidden_act: str = "silu"
    tie_word_embeddings: bool = False

    def __post_init__(self):
        if self.head_dim is None:
            self.head_dim = self.hidden_size // self.num_attention_heads
        if self.rope_parameters is not None:
            self.rope_theta = float(
                self.rope_parameters.get("rope_theta", self.rope_theta)
            )


@dataclass
class Eagle3Config(BaseModelConfig):
    """Config for Speculators/SGLang EAGLE-3 draft checkpoints."""

    model_type: str = "eagle3"
    speculators_model_type: str = "eagle3"
    transformer_layer_config: TextConfig = field(default_factory=TextConfig)
    draft_vocab_size: int = 32000
    target_hidden_size: Optional[int] = None
    tie_word_embeddings: bool = False
    norm_before_residual: bool = False
    norm_before_fc: bool = False
    embed_requires_grad: bool = False
    eagle_aux_hidden_state_layer_ids: Optional[List[int]] = None
    block_size: int = 5
    adaptive_max_block_size: Optional[int] = None
    verify_mode: Optional[str] = None
    target_layer_ids: List[int] = field(default_factory=list)
    capture_layer_ids: List[int] = field(default_factory=list)

    def __post_init__(self):
        if isinstance(self.transformer_layer_config, dict):
            self.transformer_layer_config = TextConfig.from_dict(
                self.transformer_layer_config
            )

        text = self.transformer_layer_config
        if self.target_hidden_size is None:
            self.target_hidden_size = text.hidden_size

        if not self.target_layer_ids:
            if self.eagle_aux_hidden_state_layer_ids is not None:
                self.target_layer_ids = list(self.eagle_aux_hidden_state_layer_ids)
            else:
                n = int(text.num_hidden_layers)
                self.target_layer_ids = [2, n // 2, max(n - 3, 0)]
        if not self.capture_layer_ids:
            self.capture_layer_ids = [max(int(i) - 1, 0) for i in self.target_layer_ids]

    @classmethod
    def from_dict(cls, params: dict) -> "Eagle3Config":
        flat = dict(params)
        spec_cfg = flat.get("speculators_config") or {}
        proposal_methods = spec_cfg.get("proposal_methods") or []
        if proposal_methods:
            speculative_tokens = proposal_methods[0].get("speculative_tokens")
            if speculative_tokens is not None:
                flat.setdefault("block_size", int(speculative_tokens) + 2)

        if "model_type" not in flat:
            flat["model_type"] = flat.get("speculators_model_type", "eagle3")

        sig = inspect.signature(cls).parameters
        return cls(**{k: v for k, v in flat.items() if k in sig})

    from_hf_dict = from_dict
