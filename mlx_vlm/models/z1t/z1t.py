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
            raise ValueError("Z1T is a text-only model.")
        if input_ids is None:
            raise ValueError("input_ids are required for Z1T.")
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
        embeds = self.get_input_embeddings(input_ids, pixel_values).inputs_embeds
        return self.language_model(
            input_ids, cache=cache, mask=mask, inputs_embeds=embeds, **kwargs
        )

    def sanitize(self, weights):
        def transform_key(key: str) -> Optional[str]:
            if key.startswith("language_model."):
                return key
            if key == "pe.pe":
                return "language_model.model.pe"
            if key.startswith("embedding."):
                return "language_model.model.embed_tokens." + key[len("embedding.") :]
            if key.startswith("blocks."):
                return "language_model.model.layers." + key[len("blocks.") :]
            if key.startswith("norm."):
                return "language_model.model.norm." + key[len("norm.") :]
            if key.startswith("clf."):
                return "language_model.lm_head." + key[len("clf.") :]
            return key

        sanitized = {}
        for key, value in weights.items():
            new_key = transform_key(key)
            if new_key is not None:
                sanitized[new_key] = value
        return sanitized

    @property
    def layers(self):
        return self.language_model.layers

    def make_cache(self):
        return self.language_model.make_cache()
