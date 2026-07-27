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
        if pixel_values is not None:
            raise ValueError("GLM-4.7-Flash does not accept image inputs.")
        if input_ids is None:
            raise ValueError("input_ids are required.")
        return InputEmbeddingsFeatures(
            inputs_embeds=self.language_model.model.embed_tokens(input_ids)
        )

    def __call__(
        self,
        input_ids: mx.array,
        pixel_values: Optional[mx.array] = None,
        cache=None,
        inputs_embeds: Optional[mx.array] = None,
        **kwargs,
    ) -> LanguageModelOutput:
        if pixel_values is not None:
            raise ValueError("GLM-4.7-Flash does not accept image inputs.")
        if inputs_embeds is None:
            inputs_embeds = self.get_input_embeddings(input_ids).inputs_embeds
        return self.language_model(
            input_ids,
            cache=cache,
            inputs_embeds=inputs_embeds,
            **kwargs,
        )

    def sanitize(self, weights):
        weights = self.language_model.sanitize(weights)

        def transform_key(key):
            if key.startswith("language_model."):
                return key
            if key.startswith("model.") or key.startswith("lm_head."):
                return f"language_model.{key}"
            return key

        return {transform_key(k): v for k, v in weights.items()}

    def shard(self, group: Optional[mx.distributed.Group] = None):
        return self.language_model.shard(group)

    @property
    def cast_predicate(self):
        return self.language_model.cast_predicate

    @property
    def layers(self):
        return self.language_model.layers

    def make_cache(self):
        return self.language_model.make_cache()
