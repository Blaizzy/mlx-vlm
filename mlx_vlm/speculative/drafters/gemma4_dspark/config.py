import inspect
from dataclasses import dataclass, field
from typing import List, Optional

from ....models.base import BaseModelConfig


@dataclass
class Gemma4DSparkConfig(BaseModelConfig):
    """Config for the DSpark self-speculative drafter over a Gemma 4 target.

    The published checkpoint (``deepseek-ai/dspark_gemma4_12b_block7``) declares
    ``model_type=gemma4_text`` with ``architectures=["Gemma4DSparkModel"]`` and the
    DSpark draft hyper-parameters at the top level. The draft uses Gemma 4's
    *global* attention shape (``global_head_dim`` / ``num_global_key_value_heads``)
    and the proportional (partial) RoPE from ``rope_parameters.full_attention``.
    """

    hidden_size: int = 3840
    intermediate_size: int = 15360
    num_hidden_layers: int = 5
    num_attention_heads: int = 16
    num_key_value_heads: int = 1
    head_dim: int = 512
    rms_norm_eps: float = 1e-6
    vocab_size: int = 262144
    max_position_embeddings: int = 262144
    rope_theta: float = 1000000.0
    partial_rotary_factor: float = 0.25
    attention_k_eq_v: bool = True
    final_logit_softcapping: Optional[float] = 30.0
    tie_word_embeddings: bool = False
    block_size: int = 7
    mask_token_id: int = 4
    markov_rank: int = 256
    enable_confidence_head: bool = True
    confidence_head_with_markov: bool = True
    target_layer_ids: List[int] = field(default_factory=lambda: [5, 17, 29, 41, 46])
    num_target_layers: int = 48
    runtime_block_size: Optional[int] = None

    @property
    def fc_in(self) -> int:
        return self.hidden_size * len(self.target_layer_ids)

    @classmethod
    def from_dict(cls, params: dict) -> "Gemma4DSparkConfig":
        flat = dict(params)
        # Gemma 4 draft runs over the global-attention shape.
        if flat.get("global_head_dim"):
            flat["head_dim"] = flat["global_head_dim"]
        if flat.get("num_global_key_value_heads") is not None:
            flat["num_key_value_heads"] = flat["num_global_key_value_heads"]
        # Proportional RoPE lives under rope_parameters.full_attention.
        rope = (flat.get("rope_parameters") or {}).get("full_attention") or {}
        if rope.get("rope_theta"):
            flat["rope_theta"] = rope["rope_theta"]
        if "partial_rotary_factor" in rope:
            flat["partial_rotary_factor"] = rope["partial_rotary_factor"]
        sig = inspect.signature(cls).parameters
        kwargs = {k: v for k, v in flat.items() if k in sig}
        if "target_layer_ids" in kwargs:
            kwargs["target_layer_ids"] = list(kwargs["target_layer_ids"])
        return cls(**kwargs)

    from_hf_dict = from_dict
