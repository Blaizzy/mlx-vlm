from typing import Optional

import mlx.core as mx
import mlx.nn as nn

from ..base import LanguageModelOutput
from ..cache import ArraysCache, KVCache
from ..lfm2.language import Lfm2Model
from .config import TextConfig


class LanguageModel(nn.Module):
    def __init__(self, config: TextConfig):
        super().__init__()
        self.config = config
        self.model_type = config.model_type
        self.model = Lfm2Model(config)
        if not config.tie_word_embeddings:
            self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

    def __call__(
        self,
        inputs: mx.array,
        mask: mx.array = None,
        cache=None,
        inputs_embeds: Optional[mx.array] = None,
        **kwargs,
    ):
        out = self.model(inputs, cache, inputs_embeds)
        if self.config.tie_word_embeddings:
            out = self.model.embed_tokens.as_linear(out)
        else:
            out = self.lm_head(out)
        return LanguageModelOutput(out)

    def sanitize(self, weights):
        if self.config.tie_word_embeddings:
            weights.pop("language_model.lm_head.weight", None)

        sanitized_weights = {}
        for name, param in weights.items():
            if "conv.weight" in name:
                if param.shape[-1] > param.shape[1]:
                    param = param.transpose(0, 2, 1)

            sanitized_weights[name] = param
        return sanitized_weights

    @property
    def layers(self):
        return self.model.layers

    def make_cache(self):
        return [
            KVCache() if l.is_attention_layer else ArraysCache(size=1)
            for l in self.layers
        ]
