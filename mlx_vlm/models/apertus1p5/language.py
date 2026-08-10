from typing import Optional

import mlx.core as mx
import mlx.nn as nn

from ..apertus.language import ApertusModel
from ..base import LanguageModelOutput
from ..cache import KVCache
from .config import TextConfig


class LanguageModel(nn.Module):
    def __init__(self, config: TextConfig):
        super().__init__()
        self.args = config
        self.config = config
        self.model_type = config.model_type
        self.model = ApertusModel(config)

        output_vocab_size = config.output_vocab_size or config.vocab_size
        if config.tie_word_embeddings:
            if output_vocab_size != config.vocab_size:
                raise ValueError(
                    "tie_word_embeddings requires output_vocab_size to equal "
                    f"vocab_size, got {output_vocab_size} and {config.vocab_size}."
                )
        else:
            self.lm_head = nn.Linear(config.hidden_size, output_vocab_size, bias=False)

    def __call__(
        self,
        inputs: Optional[mx.array] = None,
        cache=None,
        input_embeddings: Optional[mx.array] = None,
        inputs_embeds: Optional[mx.array] = None,
        **kwargs,
    ) -> LanguageModelOutput:
        if inputs is None:
            inputs = kwargs.get("input_ids")
        if inputs_embeds is None:
            inputs_embeds = input_embeddings

        hidden_states = self.model(inputs, cache, inputs_embeds)
        if self.args.tie_word_embeddings:
            logits = self.model.embed_tokens.as_linear(hidden_states)
        else:
            logits = self.lm_head(hidden_states)
        return LanguageModelOutput(logits=logits)

    def make_cache(self):
        return [KVCache() for _ in self.model.layers]

    @property
    def layers(self):
        return self.model.layers

    @property
    def head_dim(self):
        return self.args.hidden_size // self.args.num_attention_heads

    @property
    def n_kv_heads(self):
        return self.args.num_key_value_heads
