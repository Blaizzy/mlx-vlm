from typing import Optional

import mlx.core as mx
import mlx.nn as nn
import numpy as np

from ..base import InputEmbeddingsFeatures
from . import processing_cohere_compass  # noqa: F401
from .config import ModelConfig
from .language import LanguageModel
from .processing_cohere_compass import CohereCompassImageProcessor
from .vision import VisionModel


def masked_scatter(inputs_embeds, image_mask, image_features):
    shape = inputs_embeds.shape
    flat_embeds = inputs_embeds.reshape(-1)
    flat_mask = image_mask.reshape(-1)
    positions = mx.array(np.where(flat_mask)[0], dtype=mx.uint32)
    flat_embeds[positions] = image_features.reshape(-1)
    return flat_embeds.reshape(shape)


class Model(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.language_model = LanguageModel(config.text_config)
        self.language_model._vision_config = config
        self.vision_tower = (
            VisionModel(config.vision_config)
            if config.vision_config is not None
            else None
        )

    def get_input_embeddings(
        self,
        input_ids: Optional[mx.array] = None,
        pixel_values: Optional[mx.array] = None,
        **kwargs,
    ):
        if pixel_values is None or self.vision_tower is None:
            return InputEmbeddingsFeatures(
                inputs_embeds=self.language_model.model.embed_tokens(input_ids),
                attention_mask=kwargs.get("mask"),
            )

        image_grid_thw = kwargs.get("image_grid_thw")
        if image_grid_thw is None:
            raise ValueError("image_grid_thw must be provided with pixel_values")

        inputs_embeds = self.language_model.model.embed_tokens(input_ids)
        cached = kwargs.get("cached_image_features")
        if cached is not None:
            image_features = cached
            deepstack_visual_embeds = None
        else:
            dtype = self.vision_tower.patch_embed.proj.weight.dtype
            image_features, deepstack_visual_embeds = self.vision_tower(
                pixel_values.astype(dtype), image_grid_thw
            )

        image_mask = input_ids == self.config.image_token_id
        image_mask = mx.broadcast_to(image_mask[..., None], inputs_embeds.shape)
        if int(image_mask.sum().item()) != image_features.size:
            raise ValueError(
                "Image features and image tokens do not match: "
                f"tokens: {int(image_mask.sum().item()) // inputs_embeds.shape[-1]}, "
                f"features: {image_features.shape[0]}"
            )
        inputs_embeds = masked_scatter(inputs_embeds, image_mask, image_features)
        visual_pos_masks = image_mask[..., 0]

        position_ids, rope_deltas = self.language_model.get_rope_index(
            input_ids,
            image_grid_thw=image_grid_thw,
            attention_mask=kwargs.get("mask"),
        )
        return InputEmbeddingsFeatures(
            inputs_embeds=inputs_embeds,
            visual_pos_masks=visual_pos_masks,
            deepstack_visual_embeds=deepstack_visual_embeds,
            position_ids=position_ids,
            rope_deltas=rope_deltas,
            attention_mask=kwargs.get("mask"),
        )

    @property
    def layers(self):
        return self.language_model.model.layers

    def __call__(
        self,
        input_ids: mx.array,
        pixel_values: Optional[mx.array] = None,
        mask: Optional[mx.array] = None,
        cache=None,
        **kwargs,
    ):
        features = self.get_input_embeddings(input_ids, pixel_values, **kwargs)
        kwargs.update(features.to_dict())
        kwargs["pixel_values"] = pixel_values
        return self.language_model(
            input_ids,
            mask=mask,
            cache=cache,
            **kwargs,
        )

    def sanitize(self, weights):
        if self.config.text_config.tie_word_embeddings:
            weights = {
                key: value for key, value in weights.items() if key != "lm_head.weight"
            }
        sanitized = {}
        for key, value in weights.items():
            if key.startswith("model.language_model."):
                key = key.replace("model.language_model", "language_model.model", 1)
            elif key.startswith("model.visual."):
                key = key.replace("model.visual", "vision_tower", 1)
            elif key.startswith("lm_head."):
                key = "language_model." + key
            sanitized[key] = value
        return sanitized


ImageProcessor = CohereCompassImageProcessor
