from typing import Optional

import mlx.core as mx
import mlx.nn as nn
from mlx_lm.models.qwen2 import Qwen2Model

from ..base import LanguageModelOutput
from .config import TextConfig


class LanguageModel(nn.Module):
    def __init__(self, config: TextConfig):
        super().__init__()
        self.config = config
        self.model_type = config.model_type
        self.model = Qwen2Model(config)
        if not config.tie_word_embeddings:
            self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

    # TODO: mask is going away in mlx-lm, see https://github.com/ml-explore/mlx-lm/pull/430
    def __call__(
        self,
        inputs: mx.array,
        mask: mx.array = None,
        cache=None,
        inputs_embeds: Optional[mx.array] = None,
        **kwargs,
    ):
        out = self.model(inputs, cache=cache, input_embeddings=inputs_embeds)
        out = self.model.embed_tokens.as_linear(out)
        return LanguageModelOutput(out)

    def sanitize(self, weights):
        if self.config.tie_word_embeddings:
            weights.pop("lm_head.weight", None)
        return {
            k: v for k, v in weights.items() if "self_attn.rotary_emb.inv_freq" not in k
        }

    @property
    def layers(self):
        return self.model.layers

    @property
    def head_dim(self):
        return self.args.hidden_size // self.args.num_attention_heads

    @property
    def n_kv_heads(self):
        return self.args.num_key_value_heads
