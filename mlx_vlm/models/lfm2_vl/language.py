from typing import Any, Optional

import mlx.core as mx
import mlx.nn as nn
from mlx_lm.models.cache import ArraysCache, KVCache
from mlx_lm.models.lfm2 import Lfm2Model

from ..base import LanguageModelOutput
from .config import TextConfig


class LanguageModel(nn.Module):
    def __init__(self, config: TextConfig):
        super().__init__()
        self.config = config
        self.model_type = config.model_type
        self.model = Lfm2Model(config)

    def __call__(
        self,
        inputs: mx.array,
        mask: mx.array = None,
        cache=None,
        inputs_embeds: Optional[mx.array] = None,
    ):
        out = self.model(inputs, mask, cache, inputs_embeds)
        out = self.model.embed_tokens.as_linear(out)
        return LanguageModelOutput(out)

    def sanitize(self, weights):
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
