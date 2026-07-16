from typing import Optional

import mlx.core as mx
import mlx.nn as nn
import numpy as np

from ..base import InputEmbeddingsFeatures
from .audio import AudioModel
from .config import ModelConfig
from .language import LanguageModel
from .vision import VisionModel


class Model(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.model_type = config.model_type
        self.language_model = LanguageModel(config.text_config)
        self.vision_tower = VisionModel(config.vision_config)
        self.audio_tower = AudioModel(config.audio_config)

    def get_image_features(self, pixel_values: mx.array) -> mx.array:
        return self.vision_tower(pixel_values)

    def get_audio_features(
        self,
        audio_input_ids: mx.array,
        audio_input_ids_mask: Optional[mx.array] = None,
    ) -> mx.array:
        if audio_input_ids_mask is not None:
            audio_input_ids = audio_input_ids[audio_input_ids_mask.astype(mx.bool_)]
        else:
            audio_input_ids = audio_input_ids.reshape(-1, audio_input_ids.shape[-1])
        return self.audio_tower(audio_input_ids)

    def _scatter_placeholder(self, inputs_embeds, input_ids, features, token_id):
        mask = np.array(input_ids == token_id)
        positions = np.where(mask)[1].tolist()
        if len(positions) != features.shape[0]:
            raise ValueError(
                "Number of placeholder tokens does not match extracted features: "
                f"{len(positions)} tokens (id={token_id}) for {features.shape[0]} features."
            )
        inputs_embeds[:, positions, :] = features.astype(inputs_embeds.dtype)
        return inputs_embeds

    def get_input_embeddings(
        self,
        input_ids: Optional[mx.array] = None,
        pixel_values: Optional[mx.array] = None,
        audio_input_ids: Optional[mx.array] = None,
        audio_input_ids_mask: Optional[mx.array] = None,
        **kwargs,
    ):
        inputs_embeds = self.language_model.model.embed_norm(
            self.language_model.model.embed_tokens(input_ids)
        )

        if pixel_values is not None:
            image_features = self.get_image_features(pixel_values)
            inputs_embeds = self._scatter_placeholder(
                inputs_embeds, input_ids, image_features, self.config.image_token_id
            )

        if audio_input_ids is not None:
            audio_features = self.get_audio_features(
                audio_input_ids, audio_input_ids_mask
            )
            inputs_embeds = self._scatter_placeholder(
                inputs_embeds, input_ids, audio_features, self.config.audio_token_id
            )

        return InputEmbeddingsFeatures(inputs_embeds=inputs_embeds)

    def __call__(
        self,
        input_ids: mx.array,
        pixel_values: Optional[mx.array] = None,
        cache=None,
        **kwargs,
    ):
        if pixel_values is not None or kwargs.get("audio_input_ids") is not None:
            inputs_embeds = self.get_input_embeddings(
                input_ids, pixel_values, **kwargs
            ).inputs_embeds
            return self.language_model(inputs_embeds=inputs_embeds, cache=cache)

        return self.language_model(input_ids, cache=cache)

    def make_cache(self):
        return self.language_model.make_cache()

    def sanitize(self, weights):
        # HF nests the transformer under `model.language_model.` and keeps
        # `lm_head` at the very top; we nest the transformer under
        # `language_model.model.` and keep `lm_head` under `language_model.`.
        renames = (
            ("model.language_model.", "language_model.model."),
            ("model.vision_tower.", "vision_tower."),
            ("model.audio_tower.", "audio_tower."),
            ("lm_head.", "language_model.lm_head."),
        )
        new_weights = {}
        for k, v in weights.items():
            for old, new in renames:
                if k.startswith(old):
                    k = new + k[len(old) :]
                    break
            new_weights[k] = v
        return new_weights

    @property
    def layers(self):
        return self.language_model.model.layers
