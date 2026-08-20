import re
from typing import Optional

import mlx.core as mx
import mlx.nn as nn

from ..gemma3.language import Gemma3Model
from ..pooling import EmbeddingOutput, normalize_embeddings, pool_by_config
from .config import ModelConfig


class Model(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.model_type = config.model_type
        self.model = Gemma3Model(config)
        self.dense = [
            nn.Linear(config.hidden_size, config.hidden_size * 4, bias=False),
            nn.Linear(config.hidden_size * 4, config.hidden_size, bias=False),
        ]

    def _extended_attention_mask(self, attention_mask, dtype):
        if attention_mask.ndim == 3:
            mask = attention_mask[:, None, :, :]
        elif attention_mask.ndim == 2:
            mask = attention_mask[:, None, None, :]
            mask = mx.repeat(mask, attention_mask.shape[-1], -2)
        else:
            raise ValueError(
                f"Wrong shape for attention_mask (shape {attention_mask.shape})"
            )
        return mx.where(mask.astype(mx.bool_), 0.0, -mx.inf).astype(dtype)

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
        mask = self._extended_attention_mask(attention_mask, h.dtype)
        out = self.model(input_ids, inputs_embeds=h, mask=mask)
        pooling_config = getattr(self, "pooling_config", None) or {
            "pooling_mode": "mean"
        }
        text_embeds = pool_by_config(out, attention_mask, pooling_config)
        for dense in self.dense:
            text_embeds = dense(text_embeds)
        text_embeds = normalize_embeddings(text_embeds)
        return EmbeddingOutput(last_hidden_state=out, text_embeds=text_embeds)

    def sanitize(self, weights):
        out = {}
        for k, v in weights.items():
            if k.startswith("lm_head.") or k.startswith("head."):
                continue
            if "linear" in k and "dense" not in k:
                key_id = "0" if v.shape[0] > v.shape[1] else "1"
                out[re.sub(r"\d+_Dense\.linear", f"dense.{key_id}", k)] = v
            elif "dense" in k:
                out[k] = v
            elif not k.startswith("model."):
                out["model." + k] = v
            else:
                out[k] = v
        return out

    @property
    def layers(self):
        return self.model.layers
