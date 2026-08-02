from typing import Optional

import mlx.core as mx
import mlx.nn as nn

from ..apertus.language import ApertusModel
from ..base import LanguageModelOutput
from ..cache import KVCache
from .config import ModelConfig

# xIELU parameters are stored as ``(1,)``-shaped tensors in the checkpoint while
# ``XieLU`` holds them as scalars.
XIELU_PARAMS = (".alpha_p", ".alpha_n", ".beta", ".eps")


class LanguageModel(nn.Module):
    def __init__(self, args: ModelConfig):
        super().__init__()
        self.args = args
        self.config = args
        self.model_type = args.model_type
        # The trunk embeds over the extended vocabulary (`vocab_size`).
        self.model = ApertusModel(args)
        if not args.tie_word_embeddings:
            self.lm_head = nn.Linear(
                args.hidden_size,
                args.output_vocab_size or args.vocab_size,
                bias=False,
            )

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
        out = self.model(inputs, cache, inputs_embeds)
        if self.args.tie_word_embeddings:
            # A pruned head cannot be tied; tying only applies to full-width heads.
            out = self.model.embed_tokens.as_linear(out)
        else:
            out = self.lm_head(out)
        return LanguageModelOutput(logits=out)

    def sanitize(self, weights):
        for k, v in weights.items():
            if k.endswith(XIELU_PARAMS):
                weights[k] = v.squeeze()
        if self.args.tie_word_embeddings:
            weights.pop("lm_head.weight", None)
        return weights

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
