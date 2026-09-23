from typing import Optional

import mlx.core as mx
import mlx.nn as nn

from ..base import InputEmbeddingsFeatures, LanguageModelOutput
from ..qwen3_5.qwen3_5 import (
    NORM_WEIGHT_SUFFIXES,
    should_offset_norm_weight,
    should_shift_norm_weights,
)
from .config import ModelConfig
from .language import LanguageModel


def sanitize_key(key):
    if key.startswith("language_model."):
        return key
    if key.startswith("model.language_model."):
        # Some decoder-only exports keep the multimodal nesting. Match this
        # before the bare model. prefix below, which would otherwise leave the
        # language_model level in place.
        return key.replace("model.language_model.", "language_model.model.", 1)
    if key.startswith("model."):
        return key.replace("model.", "language_model.model.", 1)
    if key.startswith("lm_head"):
        return key.replace("lm_head", "language_model.lm_head", 1)
    return key


class Model(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.model_type = config.model_type
        self.language_model = LanguageModel(config, config)

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
        return self.language_model(input_ids, mask=mask, cache=cache, **kwargs)

    def sanitize(self, weights):
        # The MTP draft shard ships alongside the base model and must not
        # select the base model's RMSNorm loading convention.
        weights = {key: value for key, value in weights.items() if "mtp." not in key}
        shift_norm_weights = should_shift_norm_weights(weights)

        if self.config.tie_word_embeddings:
            weights.pop("lm_head.weight", None)

        sanitized_weights = {}
        for key, value in weights.items():
            original_key = key
            key = sanitize_key(key)

            if "conv1d.weight" in key and value.shape[-1] != 1:
                value = value.moveaxis(2, 1)
            if any(key.endswith(sfx) for sfx in NORM_WEIGHT_SUFFIXES):
                if value.ndim == 1 and should_offset_norm_weight(
                    original_key, shift_norm_weights
                ):
                    value += 1.0

            sanitized_weights[key] = value

        return sanitized_weights

    @property
    def layers(self):
        return self.language_model.layers

    def make_cache(self):
        return self.language_model.make_cache()

    @property
    def quant_predicate(self):
        return self.language_model.quant_predicate

    @property
    def cast_predicate(self):
        return self.language_model.cast_predicate
