from typing import Optional

import mlx.core as mx
import mlx.nn as nn

from ..base import LanguageModelOutput
from ..llama.language import LlamaModel
from .config import ModelConfig


class NoPE(nn.Module):
    def __call__(self, x, offset: int = 0):
        return x


class LanguageModel(nn.Module):
    def __init__(self, args: ModelConfig):
        super().__init__()
        self.args = args
        self.model_type = args.model_type
        self.model = LlamaModel(args)
        if not args.tie_word_embeddings:
            self.lm_head = nn.Linear(args.hidden_size, args.vocab_size, bias=False)
        for idx, use_rope in enumerate(args.no_rope_layers):
            if not use_rope:
                self.model.layers[idx].self_attn.rope = NoPE()

    def __call__(
        self,
        inputs: mx.array,
        cache=None,
        inputs_embeds: Optional[mx.array] = None,
        mask: Optional[mx.array] = None,
        **kwargs,
    ):
        out = self.model(inputs, cache, inputs_embeds)
        if self.args.tie_word_embeddings:
            out = self.model.embed_tokens.as_linear(out)
        else:
            out = self.lm_head(out)
        return LanguageModelOutput(logits=out)

    def sanitize(self, weights):
        weights = {
            k: v for k, v in weights.items() if "self_attn.rotary_emb.inv_freq" not in k
        }
        if self.args.tie_word_embeddings:
            weights.pop("lm_head.weight", None)
        return weights

    @property
    def layers(self):
        return self.model.layers

    def make_cache(self):
        from ..cache import KVCache

        return [KVCache() for _ in self.layers]
