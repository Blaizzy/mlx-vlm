from typing import Optional

import mlx.core as mx
import mlx.nn as nn

from ..base import create_attention_mask
from ..idefics3.idefics3 import Model as Idefics3Model
from ..pooling import EmbeddingOutput
from .config import ModelConfig


class Model(Idefics3Model):
    def __init__(self, config: ModelConfig):
        super().__init__(config)
        if config.mask_non_image_embeddings:
            raise NotImplementedError(
                "mask_non_image_embeddings is not supported in colidefics3."
            )
        self.embedding_dim = config.embedding_dim
        self.mask_non_image_embeddings = config.mask_non_image_embeddings
        self.linear = nn.Linear(config.text_config.hidden_size, config.embedding_dim)
        self.language_model.lm_head = None

    def _last_hidden_state(self, inputs_embeds: mx.array) -> mx.array:
        lm = self.language_model
        h = inputs_embeds.astype(lm.norm.weight.dtype)
        cache = [None] * len(lm.layers)
        mask = create_attention_mask(h, cache)
        for layer, c in zip(lm.layers, cache):
            h = layer(h, mask, c)
        return lm.norm(h)

    def __call__(
        self,
        input_ids: Optional[mx.array] = None,
        pixel_values: Optional[mx.array] = None,
        **kwargs,
    ) -> EmbeddingOutput:
        features = self.get_input_embeddings(input_ids, pixel_values, **kwargs)
        last_hidden_state = self._last_hidden_state(features.inputs_embeds)
        proj = self.linear(last_hidden_state)
        proj = proj / mx.linalg.norm(proj, axis=-1, keepdims=True)
        return EmbeddingOutput(last_hidden_state=last_hidden_state, text_embeds=proj)
