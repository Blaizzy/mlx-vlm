from typing import Any, Optional

import mlx.core as mx
import mlx.nn as nn
from mlx_lm.models.cache import ArraysCache, KVCache
from mlx_lm.models.qwen2 import Qwen2Model

from ..base import LanguageModelOutput
from .config import TextConfig


# Copied from mlx_lm.models.qwen2.Model
class LanguageModel(nn.Module):
    def __init__(self, config: TextConfig):
        super().__init__()
        self.config = config
        self.model_type = config.model_type
        self.model = Qwen2Model(config)
        if not config.tie_word_embeddings:
            self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

    # Note: mask is going away in mlx-lm, see https://github.com/ml-explore/mlx-lm/pull/430
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
        if self.config.tie_word_embeddings:
            weights.pop("lm_head.weight", None)
        # Remove unused precomputed rotary freqs
        return {
            k: v for k, v in weights.items() if "self_attn.rotary_emb.inv_freq" not in k
        }

    @property
    def layers(self):
        return self.model.layers

    def make_cache(self):
        return [
            KVCache() if l.is_attention_layer else ArraysCache(size=1)
            for l in self.layers
        ]
