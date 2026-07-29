from typing import Optional

import mlx.core as mx
import mlx.nn as nn

from ..lfm2.language import Lfm2Model
from ..pooling import EmbeddingOutput, normalize_embeddings, pool_by_config
from .config import ModelConfig


class Model(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        config.conv_causal = False
        self.config = config
        self.model_type = config.model_type
        self.model = Lfm2Model(config)

    def __call__(
        self,
        input_ids: mx.array,
        attention_mask: Optional[mx.array] = None,
        **kwargs,
    ):
        B, L = input_ids.shape
        if attention_mask is None:
            attention_mask = mx.ones((B, L))
        h = self.model.embed_tokens(input_ids)
        attn_mask = (1.0 - attention_mask[:, None, None, :].astype(h.dtype)) * -1e9
        for layer in self.model.layers:
            mask = attn_mask if layer.is_attention_layer else None
            h = layer(h, mask, None)
        h = self.model.embedding_norm(h)
        pooling_config = getattr(self, "pooling_config", None) or {
            "pooling_mode": "cls"
        }
        text_embeds = normalize_embeddings(
            pool_by_config(h, attention_mask, pooling_config)
        )
        return EmbeddingOutput(last_hidden_state=h, text_embeds=text_embeds)

    def sanitize(self, weights):
        out = {}
        for k, v in weights.items():
            if k.startswith("lm_head."):
                continue
            if "conv.weight" in k and v.shape[-1] > v.shape[1]:
                v = v.transpose(0, 2, 1)
            if not k.startswith("model."):
                k = "model." + k
            out[k] = v
        return out

    @property
    def layers(self):
        return self.model.layers
