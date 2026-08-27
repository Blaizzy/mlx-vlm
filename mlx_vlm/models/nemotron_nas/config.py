from dataclasses import dataclass, field
from typing import Dict, Optional, Union

from ..base import BaseModelConfig


@dataclass(frozen=True)
class AttentionConfig:
    no_op: bool = False
    replace_with_linear: bool = False
    sparsify: Optional[list] = None
    n_heads_in_group: Optional[int] = None
    window_length: Optional[int] = None
    num_sink_tokens: Optional[int] = None
    use_prefill_window_in_sink_attention: bool = False
    unshifted_sink: bool = False

    def __post_init__(self):
        if self.no_op or self.replace_with_linear:
            object.__setattr__(self, "n_heads_in_group", None)
            object.__setattr__(self, "window_length", None)
            object.__setattr__(self, "num_sink_tokens", None)
        elif not self.no_op:
            if self.n_heads_in_group is None:
                raise ValueError(
                    "n_heads_in_group must be specified for active attention blocks"
                )
            if self.n_heads_in_group <= 0:
                raise ValueError(
                    f"n_heads_in_group must be positive, got {self.n_heads_in_group}"
                )


@dataclass(frozen=True)
class FFNConfig:
    no_op: bool = False
    replace_with_linear: bool = False
    sparsify: Optional[list] = None
    ffn_mult: Optional[float] = None

    def __post_init__(self):
        if self.no_op or self.replace_with_linear:
            object.__setattr__(self, "ffn_mult", None)
        elif not self.no_op:
            if self.ffn_mult is None:
                raise ValueError("ffn_mult must be specified for active FFN blocks")
            object.__setattr__(self, "ffn_mult", round(self.ffn_mult, 6))


@dataclass(frozen=True)
class BlockConfig:
    attention: AttentionConfig
    ffn: FFNConfig

    @classmethod
    def from_dict(cls, data: dict):
        attn_conf = AttentionConfig(**data.get("attention", {}))
        ffn_conf = FFNConfig(**data.get("ffn", {}))
        return cls(attention=attn_conf, ffn=ffn_conf)


@dataclass
class ModelConfig(BaseModelConfig):
    model_type: str = "nemotron-nas"
    hidden_size: int = 8192
    num_hidden_layers: int = 80
    num_attention_heads: int = 64
    rms_norm_eps: float = 1e-5
    vocab_size: int = 128256
    block_configs: list = field(default_factory=list)
    hidden_act: str = "silu"
    attention_bias: bool = False
    mlp_bias: bool = False
    rope_theta: float = 500000.0
    rope_scaling: Optional[Dict[str, Union[float, str]]] = None
    max_position_embeddings: int = 131072
    tie_word_embeddings: bool = False

    def __post_init__(self):
        if self.block_configs and isinstance(self.block_configs[0], dict):
            self.block_configs = [
                BlockConfig.from_dict(conf) for conf in self.block_configs
            ]

        if len(self.block_configs) != self.num_hidden_layers:
            raise ValueError(
                f"Number of block_configs ({len(self.block_configs)}) must match "
                f"num_hidden_layers ({self.num_hidden_layers})"
            )

        if self.rope_scaling:
            if "factor" not in self.rope_scaling:
                raise ValueError("rope_scaling must contain 'factor'")
            rope_type = self.rope_scaling.get("rope_type")
            if rope_type is None:
                raise ValueError("rope_scaling must contain 'rope_type'")

        for i, block_conf in enumerate(self.block_configs):
            attn_conf = block_conf.attention
            if not attn_conf.no_op and not attn_conf.replace_with_linear:
                if self.num_attention_heads % attn_conf.n_heads_in_group != 0:
                    raise ValueError(
                        f"Layer {i}: num_attention_heads ({self.num_attention_heads}) "
                        f"must be divisible by n_heads_in_group "
                        f"({attn_conf.n_heads_in_group})"
                    )
