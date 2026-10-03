from dataclasses import dataclass, field
from typing import Any, Dict, List

from ..base import BaseModelConfig


@dataclass
class EncoderConfig(BaseModelConfig):
    model_type: str = "deberta-v2"
    vocab_size: int = 128011
    hidden_size: int = 384
    num_hidden_layers: int = 12
    num_attention_heads: int = 6
    intermediate_size: int = 1536
    hidden_act: str = "gelu"
    hidden_dropout_prob: float = 0.1
    attention_probs_dropout_prob: float = 0.1
    max_position_embeddings: int = 512
    max_relative_positions: int = 512
    position_buckets: int = 256
    relative_attention: bool = True
    pos_att_type: List[str] = field(default_factory=lambda: ["p2c", "c2p"])
    share_att_key: bool = True
    position_biased_input: bool = False
    type_vocab_size: int = 0
    layer_norm_eps: float = 1e-7
    pad_token_id: int = 0


@dataclass
class ModelConfig(BaseModelConfig):
    model_type: str = "gliner2_5"
    architecture: str = "boundary"
    max_len: int = 4096
    token_pooling: str = "first"
    encoder_config: EncoderConfig = field(default_factory=EncoderConfig)
    boundary_head: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if isinstance(self.encoder_config, dict):
            self.encoder_config = EncoderConfig.from_dict(self.encoder_config)

    @classmethod
    def from_dict(cls, params):
        params = dict(params or {})
        params["model_type"] = "gliner2_5"
        return super().from_dict(params)


__all__ = ["EncoderConfig", "ModelConfig"]
