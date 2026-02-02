from typing import Any, Optional

import mlx.core as mx
import mlx.nn as nn
from mlx_lm.models.base import create_attention_mask
from mlx_lm.models.deepseek_v3 import DeepseekV3Model
from mlx_lm.models.deepseek_v3 import Model as DeepseekV3LM

from ..base import LanguageModelOutput
from .config import TextConfig


class LanguageModel(nn.Module):
    def __init__(self, config: TextConfig):
        super().__init__()
        self.config = config
        self.args = config
        self.model_type = config.model_type
        self.model = DeepseekV3Model(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

    def __call__(
        self,
        inputs: mx.array,
        inputs_embeds: Optional[mx.array] = None,
        cache: Optional[Any] = None,
        mask: Optional[mx.array] = None,
    ):
        if inputs_embeds is None:
            h = self.model.embed_tokens(inputs)
        else:
            h = inputs_embeds

        if cache is None:
            cache = [None] * len(self.model.pipeline_layers)

        if mask is None:
            mask = create_attention_mask(h, cache[0])

        pipeline_rank = self.model.pipeline_rank
        pipeline_size = self.model.pipeline_size

        # Receive from the previous process in the pipeline
        if pipeline_rank < pipeline_size - 1:
            h = mx.distributed.recv_like(h, (pipeline_rank + 1))

        for layer, c in zip(self.model.pipeline_layers, cache):
            h = layer(h, mask, cache=c)

        # Send to the next process in the pipeline
        if pipeline_rank != 0:
            h = mx.distributed.send(h, (pipeline_rank - 1) % pipeline_size)
            if cache[-1] is not None:
                cache[-1].keys = mx.depends(cache[-1].keys, h)

        # Broadcast h while keeping it in the graph
        if pipeline_size > 1:
            h = mx.distributed.all_gather(h)[: h.shape[0]]

        out = self.lm_head(self.model.norm(h))
        return LanguageModelOutput(logits=out)

    def sanitize(self, weights):
        # Remap the keys so the Deepseek implementation uses the right prefix,
        # then map back the sanitized weights.
        sanitized = {}
        lm_weights = {}
        for k, v in weights.items():
            if k.startswith("language_model."):
                if k.startswith("language_model.model."):
                    lm_weights[k.replace("language_model.model.", "model.", 1)] = v
                else:
                    lm_weights[k.replace("language_model.", "", 1)] = v
            else:
                sanitized[k] = v

        lm_weights = DeepseekV3LM.sanitize(self, lm_weights)
        remapped = {"language_model." + k: v for k, v in lm_weights.items()}
        sanitized.update(remapped)
        return sanitized

    def embed_tokens(self, x):
        return self.model.embed_tokens(x)

    def shard(self, group: Optional[mx.distributed.Group] = None):
        DeepseekV3LM.shard(self, group)

    @property
    def layers(self):
        return self.model.pipeline_layers

    @property
    def n_kv_heads(self):
        return self.config.num_key_value_heads

    @property
    def cast_predicate(self):
        return DeepseekV3LM.cast_predicate(self)
