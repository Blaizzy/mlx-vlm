from typing import Optional

import mlx.core as mx
import mlx.nn as nn

from ..base import InputEmbeddingsFeatures, LanguageModelOutput
from .config import ModelConfig
from .language import LanguageModel


class Model(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.model_type = config.model_type
        self.language_model = LanguageModel(config)

    def get_input_embeddings(
        self,
        input_ids: Optional[mx.array] = None,
        pixel_values: Optional[mx.array] = None,
        **kwargs,
    ) -> InputEmbeddingsFeatures:
        return InputEmbeddingsFeatures(
            inputs_embeds=self.language_model.model.embed_tokens(input_ids)
        )

    def __call__(
        self,
        input_ids: mx.array,
        pixel_values: mx.array = None,
        mask: mx.array = None,
        cache=None,
        **kwargs,
    ) -> LanguageModelOutput:
        features = self.get_input_embeddings(input_ids, pixel_values)
        return self.language_model(
            input_ids,
            cache=cache,
            inputs_embeds=features.inputs_embeds,
            **kwargs,
        )

    def sanitize(self, weights):
        if any(key.startswith("language_model.") for key in weights):
            return weights
        return {f"language_model.{key}": value for key, value in weights.items()}

    @property
    def layers(self):
        return self.language_model.layers

    def make_cache(self):
        return self.language_model.make_cache()

    def shard(self, group: Optional[mx.distributed.Group] = None):
        return self.language_model.shard(group)
