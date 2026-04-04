from typing import Optional

import mlx.core as mx
import mlx.nn as nn
import numpy as np

from ..base import InputEmbeddingsFeatures
from . import processing_granite_vision  # noqa: F401
from .config import ModelConfig
from .language import LanguageModel
from .vision import VisionModel


class LlavaMultiModalProjector(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        # When using multiple vision feature layers, features are concatenated
        num_feature_layers = (
            1
            if isinstance(config.vision_feature_layer, int)
            else len(config.vision_feature_layer)
        )
        projector_input_size = config.vision_config.hidden_size * num_feature_layers

        self.linear_1 = nn.Linear(
            projector_input_size, config.text_config.hidden_size, bias=True
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
        embed_std = 1 / mx.sqrt(
            mx.array(config.text_config.hidden_size, dtype=mx.float32)
        )
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
            selected_image_feature = cached
        else:
            # Get the output hidden states from the vision model
            *_, hidden_states = self.vision_tower(
                pixel_values[0].transpose(0, 2, 3, 1), output_hidden_states=True
            )

            # Select the hidden states from the desired layer(s)
            if isinstance(self.vision_feature_layer, int):
                selected_image_feature = hidden_states[self.vision_feature_layer]

                if self.vision_feature_select_strategy == "default":
                    selected_image_feature = selected_image_feature[:, 1:]
                elif self.vision_feature_select_strategy != "full":
                    raise ValueError(
                        "Unexpected feature selection strategy: "
                        f"{self.vision_feature_select_strategy}"
                    )
            else:
                # Multi-layer: concatenate features from multiple layers
                hs_pool = [
                    hidden_states[layer_idx] for layer_idx in self.vision_feature_layer
                ]
                if self.vision_feature_select_strategy == "default":
                    hs_pool = [hs[:, 1:] for hs in hs_pool]
                selected_image_feature = mx.concatenate(hs_pool, axis=-1)

        # Pass image features through the multi-modal projector
        image_features = self.multi_modal_projector(selected_image_feature)

        # Add a newline token to the image features
        if self.image_newline is not None:
            newline = mx.broadcast_to(
                self.image_newline[None, None, :], image_features.shape
            )
            image_features = mx.concatenate([image_features, newline], axis=0)

        image_features = image_features.astype(inputs_embeds.dtype)

        # Insert special image tokens in the input_ids
        final_inputs_embeds = self._merge_input_ids_with_image_features(
            image_features, inputs_embeds, input_ids
        )
        return InputEmbeddingsFeatures(inputs_embeds=final_inputs_embeds)

    def _merge_input_ids_with_image_features(
        self, image_features, inputs_embeds, input_ids
    ):
        image_token_index = self.config.image_token_index
        num_images, num_image_patches, embed_dim = image_features.shape

        image_positions = np.where(input_ids == image_token_index)[1].tolist()

        text_segments = []
        start_idx = 0

        for position in image_positions:
            text_segments.append(inputs_embeds[:, start_idx:position])
            start_idx = position + 1

        image_embeddings = mx.split(image_features, image_features.shape[0])
        final_embeddings = [v for p in zip(text_segments, image_embeddings) for v in p]
        final_embeddings += [inputs_embeds[:, start_idx:]]

        # Create a final embedding of shape
        # (1, num_image_patches*num_images + sequence_len, embed_dim)
        return mx.concatenate(final_embeddings, axis=1)

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
        input_embeddings_features = self.get_input_embeddings(input_ids, pixel_values)
        logits = self.language_model(
            input_ids,
            cache=cache,
            inputs_embeds=input_embeddings_features.inputs_embeds,
        )
        return logits

    @staticmethod
    def sanitize(weights):
        # Handle tied embeddings: copy embed_tokens weight to lm_head if missing
        lm_head_key = "language_model.lm_head.weight"
        embed_key = "language_model.model.embed_tokens.weight"
        if lm_head_key not in weights and embed_key in weights:
            weights[lm_head_key] = weights[embed_key]
        return weights
