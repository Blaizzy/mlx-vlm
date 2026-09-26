from typing import Optional

import mlx.core as mx
import mlx.nn as nn
import numpy as np

from ..base import InputEmbeddingsFeatures, pixel_shuffle
from ..qwen3.language import LanguageModel
from .config import ModelConfig
from .vision import VisionModel


class MultiModalProjector(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        vision_size = (
            config.vision_config.hidden_size * int(1 / config.downsample_ratio) ** 2
        )
        self.layer_norm = nn.LayerNorm(vision_size)
        self.linear_1 = nn.Linear(vision_size, config.text_config.hidden_size)
        self.activation = nn.GELU(approx="precise")
        self.linear_2 = nn.Linear(
            config.text_config.hidden_size, config.text_config.hidden_size
        )

    def __call__(self, image_features):
        image_features = self.layer_norm(image_features)
        image_features = self.activation(self.linear_1(image_features))
        return self.linear_2(image_features)


class Model(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.vision_tower = VisionModel(config.vision_config)
        self.multi_modal_projector = MultiModalProjector(config)
        self.language_model = LanguageModel(config.text_config)

    def get_input_embeddings(
        self,
        input_ids: mx.array,
        pixel_values: Optional[mx.array] = None,
        **kwargs,
    ):
        inputs_embeds = self.language_model.model.embed_tokens(input_ids)
        if pixel_values is None:
            return InputEmbeddingsFeatures(inputs_embeds=inputs_embeds)

        if pixel_values.ndim == 5:
            pixel_values = pixel_values[0]
        pixel_values = pixel_values.transpose(0, 2, 3, 1).astype(
            self.vision_tower.embeddings.patch_embeddings.projection.weight.dtype
        )
        image_features, _ = self.vision_tower(pixel_values)
        image_features = image_features[:, 1:]
        image_features = pixel_shuffle(image_features, self.config.downsample_ratio)
        image_features = self.multi_modal_projector(image_features).reshape(
            -1, inputs_embeds.shape[-1]
        )

        image_positions = np.where(input_ids == self.config.image_token_id)[1].tolist()
        if len(image_positions) != image_features.shape[0]:
            raise ValueError(
                f"Image features and image tokens do not match: "
                f"{image_features.shape[0]} features and {len(image_positions)} tokens."
            )
        inputs_embeds[:, image_positions, :] = image_features
        return InputEmbeddingsFeatures(inputs_embeds=inputs_embeds)

    def __call__(self, input_ids, pixel_values=None, cache=None, **kwargs):
        features = self.get_input_embeddings(input_ids, pixel_values, **kwargs)
        return self.language_model(inputs_embeds=features.inputs_embeds, cache=cache)

    def sanitize(self, weights):
        sanitized = {}
        for key, value in weights.items():
            if key.startswith("language_model.model.model."):
                key = (
                    "language_model.model." + key[len("language_model.model.model.") :]
                )
            elif key.startswith("language_model.model.lm_head."):
                key = (
                    "language_model.lm_head."
                    + key[len("language_model.model.lm_head.") :]
                )
            sanitized[key] = value
        return sanitized

    @property
    def layers(self):
        return self.language_model.layers
