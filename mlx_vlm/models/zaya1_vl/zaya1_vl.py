from typing import Optional

import mlx.core as mx
import mlx.nn as nn

from ..base import InputEmbeddingsFeatures
from . import processing_zaya1_vl  # noqa: F401
from .config import ModelConfig
from .language import LanguageModel
from .vision import VisionModel


class Model(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.model_type = config.model_type
        self.vision_tower = VisionModel(config.vision_config)
        self.language_model = LanguageModel(config.text_config, config)

    def get_input_embeddings(
        self,
        input_ids: Optional[mx.array] = None,
        pixel_values: Optional[mx.array] = None,
        **kwargs,
    ):
        inputs_embeds = self.language_model.model.embed_tokens(input_ids)
        image_mask = None

        if pixel_values is not None:
            image_grid_thw = kwargs.get("image_grid_thw", None)
            if image_grid_thw is None:
                raise ValueError(
                    "image_grid_thw is required when pixel_values are provided"
                )

            cached = kwargs.get("cached_image_features", None)
            if cached is not None:
                image_features = cached
            else:
                dtype = self.vision_tower.patch_embed.proj.weight.dtype
                image_features = self.vision_tower(
                    pixel_values.astype(dtype),
                    image_grid_thw,
                    output_hidden_states=False,
                )

            inputs_embeds, image_mask = self.merge_input_ids_with_image_features(
                image_features,
                inputs_embeds,
                input_ids,
                self.config.image_token_id,
            )

        return InputEmbeddingsFeatures(
            inputs_embeds=inputs_embeds,
            visual_pos_masks=image_mask,
        )

    @staticmethod
    def merge_input_ids_with_image_features(
        image_features,
        inputs_embeds,
        input_ids,
        image_token_id,
    ):
        image_positions = input_ids == image_token_id
        batch_outputs = []
        feature_start_idx = 0

        for batch_idx in range(input_ids.shape[0]):
            image_mask = image_positions[batch_idx]
            num_positions = mx.sum(image_mask).item()

            if num_positions > 0:
                batch_features = image_features[
                    feature_start_idx : feature_start_idx + num_positions
                ]
                if batch_features.shape[0] != num_positions:
                    raise ValueError(
                        "Image features and image tokens do not match: "
                        f"tokens: {num_positions}, features {batch_features.shape[0]}"
                    )

                cumsum = mx.cumsum(image_mask.astype(mx.int32))
                feature_indices = mx.where(image_mask, cumsum - 1, 0)
                gathered_features = batch_features[feature_indices]
                batch_output = mx.where(
                    image_mask[..., None], gathered_features, inputs_embeds[batch_idx]
                )
                feature_start_idx += num_positions
            else:
                batch_output = inputs_embeds[batch_idx]

            batch_outputs.append(batch_output)

        return mx.stack(batch_outputs, axis=0), image_positions

    @property
    def layers(self):
        return self.language_model.model.layers

    def make_cache(self):
        return self.language_model.make_cache()

    def __call__(
        self,
        input_ids: mx.array,
        pixel_values: Optional[mx.array] = None,
        mask: Optional[mx.array] = None,
        cache=None,
        **kwargs,
    ):
        input_embeddings_features = self.get_input_embeddings(
            input_ids,
            pixel_values,
            **kwargs,
        )

        return self.language_model(
            input_ids,
            input_embeddings_features.inputs_embeds,
            mask=mask,
            image_mask=input_embeddings_features.visual_pos_masks,
            cache=cache,
            **kwargs,
        )

    def sanitize(self, weights):
        sanitized = {}
        for key, value in weights.items():
            if key == "lm_head.weight" and self.config.text_config.tie_word_embeddings:
                continue
            if key.startswith("model."):
                key = key.replace("model.", "language_model.model.", 1)
            elif key.startswith("lm_head."):
                key = key.replace("lm_head.", "language_model.lm_head.", 1)
            sanitized[key] = value
        return sanitized
