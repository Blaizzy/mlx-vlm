from typing import Optional

import mlx.core as mx
import mlx.nn as nn
import numpy as np
from transformers.image_processing_utils import select_best_resolution

from ..base import InputEmbeddingsFeatures
from . import processing_llava_next  # noqa: F401
from .config import ModelConfig
from .image_features import pack_image_features
from .language import LanguageModel
from .vision import VisionModel


class LlavaMultiModalProjector(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.linear_1 = nn.Linear(
            config.vision_config.hidden_size, config.text_config.hidden_size, bias=True
        )
        self.gelu = nn.GELU()
        self.linear_2 = nn.Linear(
            config.text_config.hidden_size, config.text_config.hidden_size, bias=True
        )

    def __call__(self, x: mx.array) -> mx.array:
        x = self.linear_1(x)
        x = self.gelu(x)
        x = self.linear_2(x)
        return x


class Model(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.vision_tower = VisionModel(config.vision_config)
        self.language_model = LanguageModel(config.text_config)
        embed_std = 1 / mx.sqrt(config.text_config.hidden_size)
        self.image_newline = (
            mx.random.normal((config.text_config.hidden_size,)) * embed_std
        )

        self.multi_modal_projector = LlavaMultiModalProjector(config)
        self.vision_feature_layer = config.vision_feature_layer
        self.vision_feature_select_strategy = config.vision_feature_select_strategy

    def get_input_embeddings(
        self,
        input_ids: Optional[mx.array] = None,
        pixel_values: Optional[mx.array] = None,
        **kwargs,
    ):
        if pixel_values is None:
            return InputEmbeddingsFeatures(
                inputs_embeds=self.language_model.model.embed_tokens(input_ids)
            )

        # Get the input embeddings from the language model
        inputs_embeds = self.language_model.model.embed_tokens(input_ids)

        cached = kwargs.get("cached_image_features", None)
        if cached is not None:
            image_features = cached.astype(inputs_embeds.dtype)
        else:
            image_sizes = kwargs.get("image_sizes")
            if image_sizes is None:
                raise ValueError(
                    "LLaVA-NeXT requires original image_sizes for spatial packing"
                )
            image_sizes = np.asarray(image_sizes).tolist()
            patch_counts = []
            for size in image_sizes:
                height, width = select_best_resolution(
                    size, self.config.image_grid_pinpoints
                )
                tile_size = self.config.vision_config.image_size
                patch_counts.append(1 + (height // tile_size) * (width // tile_size))
            if pixel_values.ndim == 5:
                if pixel_values.shape[0] != len(patch_counts):
                    raise ValueError("Image sizes and pixel batch do not match")
                pixel_values = mx.concatenate(
                    [
                        pixels[:count]
                        for pixels, count in zip(pixel_values, patch_counts)
                    ]
                )
            if pixel_values.ndim != 4 or pixel_values.shape[0] != sum(patch_counts):
                raise ValueError("Image crop count does not match original image sizes")
            *_, hidden_states = self.vision_tower(
                pixel_values.transpose(0, 2, 3, 1), output_hidden_states=True
            )

            # Select the hidden states from the desired layer
            selected_image_feature = hidden_states[self.vision_feature_layer]

            if self.vision_feature_select_strategy == "default":
                selected_image_feature = selected_image_feature[:, 1:]
            elif self.vision_feature_select_strategy == "full":
                selected_image_feature = selected_image_feature
            else:
                raise ValueError(
                    "Unexpected feature selection strategy: "
                    f"{self.vision_feature_select_strategy}"
                )

            # Pass image features through the multi-modal projector
            image_features = self.multi_modal_projector(selected_image_feature)

            boundaries = np.cumsum(patch_counts)[:-1].tolist()
            features = mx.split(image_features, boundaries, axis=0)
            image_features = mx.concatenate(
                pack_image_features(
                    features,
                    image_sizes,
                    self.config.vision_config.image_size,
                    self.config.vision_config.patch_size,
                    self.config.image_grid_pinpoints,
                    self.image_newline,
                ),
                axis=0,
            ).astype(inputs_embeds.dtype)

        # Insert special image tokens in the input_ids
        final_inputs_embeds = self._merge_input_ids_with_image_features(
            image_features, inputs_embeds, input_ids
        )
        return InputEmbeddingsFeatures(inputs_embeds=final_inputs_embeds)

    def _merge_input_ids_with_image_features(
        self, image_features, inputs_embeds, input_ids
    ):
        positions = np.where(np.asarray(input_ids) == self.config.image_token_index)
        image_features = image_features.reshape(-1, inputs_embeds.shape[-1])
        if len(positions[0]) != image_features.shape[0]:
            raise ValueError(
                f"Image features and image tokens do not match: "
                f"{image_features.shape[0]} features, {len(positions[0])} tokens"
            )
        result = mx.array(inputs_embeds)
        result[mx.array(positions[0]), mx.array(positions[1])] = image_features
        return result

    @property
    def layers(self):
        return self.language_model.model.layers

    def __call__(
        self,
        input_ids: mx.array,
        pixel_values: mx.array,
        mask: mx.array,
        cache=None,
        **kwargs,
    ):

        input_embeddings_features = self.get_input_embeddings(
            input_ids, pixel_values, **kwargs
        )
        logits = self.language_model(
            input_ids,
            cache=cache,
            inputs_embeds=input_embeddings_features.inputs_embeds,
        )
        return logits
