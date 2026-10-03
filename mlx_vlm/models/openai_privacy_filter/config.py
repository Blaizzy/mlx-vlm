from dataclasses import dataclass, field
from typing import Any, Dict, Optional

from ..base import BaseModelConfig

SPAN_LABELS = (
    "account_number",
    "private_address",
    "private_date",
    "private_email",
    "private_person",
    "private_phone",
    "private_url",
    "secret",
)

TOKEN_LABELS = ("O",) + tuple(
    f"{boundary}-{label}" for label in SPAN_LABELS for boundary in ("B", "I", "E", "S")
)


def _default_rope_parameters() -> Dict[str, Any]:
    return {
        "rope_type": "yarn",
        "rope_theta": 150000.0,
        "factor": 32.0,
        "beta_fast": 32.0,
        "beta_slow": 1.0,
        "truncate": False,
        "original_max_position_embeddings": 4096,
    }


@dataclass
class ModelConfig(BaseModelConfig):
    model_type: str = "openai_privacy_filter"
    vocab_size: int = 200064
    hidden_size: int = 640
    intermediate_size: int = 640
    num_hidden_layers: int = 8
    num_local_experts: int = 128
    num_experts_per_tok: int = 4
    head_dim: int = 64
    num_attention_heads: int = 14
    num_key_value_heads: int = 2
    sliding_window: int = 128
    max_position_embeddings: int = 131072
    default_n_ctx: int = 128000
    initial_context_length: int = 4096
    rms_norm_eps: float = 1e-5
    attention_bias: bool = True
    attention_dropout: float = 0.0
    classifier_dropout: float = 0.0
    num_labels: int = len(TOKEN_LABELS)
    pad_token_id: int = 199999
    eos_token_id: int = 199999
    bos_token_id: Optional[int] = None
    rope_parameters: Dict[str, Any] = field(default_factory=_default_rope_parameters)
    id2label: Optional[Dict[Any, str]] = None
    label2id: Optional[Dict[str, int]] = None
    quantization: Optional[Dict[str, Any]] = None
    quantization_config: Optional[Dict[str, Any]] = None
    attention_chunk_size: int = 4096
    moe_chunk_size: int = 2048

    def __post_init__(self):
        rope = _default_rope_parameters()
        rope.update(self.rope_parameters or {})
        self.rope_parameters = rope

        if self.id2label is None:
            self.id2label = dict(enumerate(TOKEN_LABELS))
        else:
            self.id2label = {int(key): value for key, value in self.id2label.items()}
            self.num_labels = len(self.id2label)

        if self.label2id is None:
            self.label2id = {label: idx for idx, label in self.id2label.items()}

        if self.num_attention_heads % self.num_key_value_heads:
            raise ValueError(
                "num_attention_heads must be divisible by num_key_value_heads"
            )
        if self.attention_chunk_size <= 0:
            raise ValueError("attention_chunk_size must be positive")
        if self.moe_chunk_size <= 0:
            raise ValueError("moe_chunk_size must be positive")
