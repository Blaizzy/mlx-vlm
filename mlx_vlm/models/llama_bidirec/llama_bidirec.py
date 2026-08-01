from typing import Optional

import mlx.core as mx
import mlx.nn as nn

from ..llama.language import LlamaModel
from ..pooling import EmbeddingOutput, mean_pooling, normalize_embeddings
from .config import ModelConfig


class Model(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.model_type = config.model_type
        self.model = LlamaModel(config)

    def _last_hidden_state(
        self, input_ids: mx.array, attention_mask: mx.array
    ) -> mx.array:
        h = self.model.embed_tokens(input_ids)
        mask = attention_mask[:, None, None, :]
        mask = mx.repeat(mask, attention_mask.shape[-1], -2)
        mask = mx.where(mask.astype(mx.bool_), 0.0, -mx.inf).astype(h.dtype)
        for layer in self.model.layers:
            h = layer(h, mask, cache=None)
        return self.model.norm(h)

    def __call__(
        self,
        input_ids: Optional[mx.array] = None,
        attention_mask: Optional[mx.array] = None,
        **kwargs,
    ) -> EmbeddingOutput:
        if attention_mask is None:
            attention_mask = mx.ones(input_ids.shape)
        hidden_states = self._last_hidden_state(input_ids, attention_mask)
        text_embeds = normalize_embeddings(mean_pooling(hidden_states, attention_mask))
        return EmbeddingOutput(last_hidden_state=hidden_states, text_embeds=text_embeds)

    def sanitize(self, weights):
        sanitized = {}
        for k, v in weights.items():
            if "rotary_emb.inv_freq" in k or "lm_head" in k or "position_ids" in k:
                continue
            if k.startswith("language_model."):
                k = k[len("language_model.") :]
            if not k.startswith("model."):
                k = "model." + k
            sanitized[k] = v
        return sanitized

    @property
    def layers(self):
        return self.model.layers
