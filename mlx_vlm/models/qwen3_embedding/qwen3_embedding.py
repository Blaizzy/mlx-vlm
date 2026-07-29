import mlx.core as mx
import mlx.nn as nn

from ..pooling import EmbeddingOutput, normalize_embeddings, pool_by_config
from ..qwen3.language import Qwen3Model
from .config import ModelConfig


class Model(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.model_type = config.model_type
        self.model = Qwen3Model(config)

    def __call__(self, input_ids, attention_mask=None, **kwargs):
        B, L = input_ids.shape
        h = self.model.embed_tokens(input_ids)
        if attention_mask is None:
            attention_mask = mx.ones((B, L))
        causal = mx.triu(mx.full((L, L), -1e9, dtype=h.dtype), k=1)
        pad = (1.0 - attention_mask[:, None, None, :].astype(h.dtype)) * -1e9
        mask = causal[None, None, :, :] + pad
        for layer in self.model.layers:
            h = layer(h, mask, None)
        h = self.model.norm(h)
        pooling_config = getattr(self, "pooling_config", None) or {
            "pooling_mode": "lasttoken"
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
