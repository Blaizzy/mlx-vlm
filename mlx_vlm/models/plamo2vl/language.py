import inspect
from dataclasses import dataclass
from typing import Optional

import mlx.core as mx
import mlx.nn as nn
from mlx_lm.models.cache import ArraysCache, KVCache
from mlx_lm.models.plamo2 import PlamoDecoder, RMSNorm

from ..base import LanguageModelOutput


@dataclass
class TextConfig:
    model_type: str = "plamo2"
    hidden_size: int = 4096
    num_hidden_layers: int = 32
    rms_norm_eps: float = 1e-6
    tie_word_embeddings: bool = True
    num_attention_heads: int = 32
    num_key_value_heads: int = 4
    hidden_size_per_head: int = 128
    max_position_embeddings: int = 2048
    attention_window_size: int = 2048
    full_attention_idx: Optional[list[int]] = None
    rope_theta: float = 10000
    rope_local_theta: float = 10000
    mamba_d_state: int = 64
    mamba_d_conv: int = 4
    mamba_num_heads: int = 64
    mamba_step: int = 2
    mamba_chunk_size: int = 256
    mamba_enabled: bool = True
    intermediate_size: int = 13312
    vocab_size: int = 32000

    @classmethod
    def from_dict(cls, params):
        values = {
            k: v for k, v in params.items() if k in inspect.signature(cls).parameters
        }
        values.setdefault("model_type", "plamo2")
        return cls(**values)

    def __post_init__(self):
        if self.full_attention_idx is None:
            self.full_attention_idx = []


class Plamo2Model(nn.Module):
    def __init__(self, config: TextConfig):
        super().__init__()
        self.config = config
        self.vocab_size = config.vocab_size
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = PlamoDecoder(config)  # type: ignore[arg-type]
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def __call__(
        self,
        inputs: Optional[mx.array] = None,
        cache=None,
        inputs_embeds: Optional[mx.array] = None,
    ):
        if inputs_embeds is None:
            if inputs is None:
                raise ValueError("Either inputs or inputs_embeds must be provided")
            hidden_states = self.embed_tokens(inputs)
        else:
            hidden_states = inputs_embeds

        hidden_states = self.layers(hidden_states, cache)
        return self.norm(hidden_states)


class LanguageModel(nn.Module):
    def __init__(self, config: TextConfig):
        super().__init__()
        self.config = config
        self.model_type = config.model_type
        if self.model_type != "plamo2":
            raise ValueError(f"Unsupported model type: {self.model_type}")

        self.model = Plamo2Model(config)
        vocab_size = ((config.vocab_size + 15) // 16) * 16
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, vocab_size, bias=False)

    def make_cache(self):
        return [ArraysCache(size=2) if l.is_mamba else KVCache() for l in self.layers]

    def __call__(
        self,
        inputs: Optional[mx.array],
        cache=None,
        inputs_embeds: Optional[mx.array] = None,
        mask: Optional[mx.array] = None,
    ):
        del mask
        hidden_states = self.model(inputs, cache=cache, inputs_embeds=inputs_embeds)
        logits = self.lm_head(hidden_states)[..., : self.vocab_size]
        return LanguageModelOutput(logits=logits)

    @property
    def layers(self):
        return self.model.layers.layers
