from typing import Optional

import mlx.core as mx
import mlx.nn as nn

from ..ministral3.language import Ministral3Decoder, _get_llama_4_attn_scale
from ..pooling import EmbeddingOutput, normalize_embeddings, pool_by_config
from .config import ModelConfig


class Model(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.model_type = config.model_type
        self.model = Ministral3Decoder(config)

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
        attn_mask = (1.0 - attention_mask[:, None, None, :].astype(h.dtype)) * mx.finfo(
            h.dtype
        ).min
        attn_scale = _get_llama_4_attn_scale(
            L,
            0,
            self.config.rope_parameters["llama_4_scaling_beta"],
            self.config.rope_parameters["original_max_position_embeddings"],
        ).astype(h.dtype)
        for layer in self.model.layers:
            h = layer(h, attn_scale, attn_mask, cache=None)
        h = self.model.norm(h)
        pooling_config = getattr(self, "pooling_config", None) or {
            "pooling_mode": "mean"
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
            if not k.startswith("model."):
                k = "model." + k
            out[k] = v
        return out

    @property
    def layers(self):
        return self.model.layers
