from typing import Optional

import mlx.core as mx
import mlx.nn as nn

from ..lfm2.language import Lfm2Model
from ..pooling import EmbeddingOutput, normalize_embeddings
from .config import ModelConfig


class Model(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        config.conv_causal = False
        self.config = config
        self.model_type = config.model_type
        self.model = Lfm2Model(config)
        self.projection = nn.Linear(
            config.hidden_size, config.embedding_dim, bias=False
        )

    def __call__(
        self,
        input_ids: mx.array,
        attention_mask: Optional[mx.array] = None,
        **kwargs,
    ) -> EmbeddingOutput:
        batch_size, sequence_length = input_ids.shape
        if attention_mask is None:
            attention_mask = mx.ones((batch_size, sequence_length))
        h = self.model.embed_tokens(input_ids)
        attn_mask = (1.0 - attention_mask[:, None, None, :].astype(h.dtype)) * mx.finfo(
            h.dtype
        ).min
        for layer in self.model.layers:
            mask = attn_mask if layer.is_attention_layer else None
            h = layer(h, mask, None)
        h = self.model.embedding_norm(h)
        text_embeds = normalize_embeddings(self.projection(h).astype(mx.float32))
        return EmbeddingOutput(last_hidden_state=h, text_embeds=text_embeds)

    def sanitize(self, weights):
        out = {}
        for key, value in weights.items():
            if key == "1_Dense.linear.weight":
                key = "projection.weight"
            else:
                if "conv.weight" in key and value.shape[-1] > value.shape[1]:
                    value = value.transpose(0, 2, 1)
                if not key.startswith("model."):
                    key = "model." + key
            out[key] = value
        return out

    @property
    def layers(self):
        return self.model.layers
