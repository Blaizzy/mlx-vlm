from typing import Optional

import mlx.core as mx
import mlx.nn as nn

from ..base import InputEmbeddingsFeatures, LanguageModelOutput
from .config import ModelConfig
from .language import LanguageModel
from .vision import VisionModel


class Model(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.model_type = config.model_type
        self.language_model = LanguageModel(config.text_config)
        self.vision_tower = (
            None if config.skip_vision else VisionModel(config.vision_config)
        )

    def get_input_embeddings(
        self,
        input_ids: Optional[mx.array] = None,
        pixel_values: Optional[mx.array] = None,
        **kwargs,
    ) -> InputEmbeddingsFeatures:
        if pixel_values is not None:
            raise NotImplementedError(
                "MiMo-V2.6 image input is not implemented yet; text-only for now"
            )
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
        inputs_embeds = self.get_input_embeddings(
            input_ids, pixel_values, **kwargs
        ).inputs_embeds
        return self.language_model(
            input_ids, cache=cache, inputs_embeds=inputs_embeds, mask=mask, **kwargs
        )

    def sanitize(self, weights):
        # The vision and audio towers are not ported yet, so their weights are
        # dropped rather than loaded. Remove this once vision.py lands.
        weights = {
            k: v
            for k, v in weights.items()
            if not k.startswith(
                ("visual.", "audio_encoder.", "audio_tokenizer.", "speech_embeddings.")
            )
        }
        weights = self.language_model.sanitize(weights)
        return {f"language_model.{k}": v for k, v in weights.items()}

    @property
    def layers(self):
        return self.language_model.layers

    @property
    def cast_predicate(self):
        return self.language_model.cast_predicate

    def make_cache(self):
        return self.language_model.make_cache()
