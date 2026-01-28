from typing import Optional

import mlx.core as mx
import mlx.nn as nn
import numpy as np

from ..base import InputEmbeddingsFeatures
from .config import ModelConfig
from .language import LanguageModel
from .vision import VisionModel


class AyaVisionMultiModalProjector(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.downsample_factor = config.downsample_factor
        self.alignment_intermediate_size = getattr(
            config, "alignment_intermediate_size", config.text_config.hidden_size
        )
        if config.model_type == "aya_vision":
            self.layernorm = nn.LayerNorm(
                config.vision_config.hidden_size * (config.downsample_factor**2),
                eps=config.adapter_layer_norm_eps,
            )

        self.linear_1 = nn.Linear(
            config.vision_config.hidden_size * (config.downsample_factor**2),
            self.alignment_intermediate_size,
            bias=True,
        )

        self.act = nn.SiLU()  # SwiGLU uses SiLU activation

        # For SwiGLU, project down to half size since we split intermediate dim
        self.linear_2 = nn.Linear(
            self.alignment_intermediate_size // 2,
            config.text_config.hidden_size,
            bias=True,
        )

    def __call__(self, image_features):
        image_features = self.pixel_shuffle(image_features)
        if self.config.model_type == "aya_vision":
            image_features = self.layernorm(image_features)
        hidden_states = self.linear_1(image_features)

        # Split along last dimension and apply SwiGLU
        x, gate = mx.split(hidden_states, 2, axis=-1)
        hidden_states = self.act(gate) * x

        hidden_states = self.linear_2(hidden_states)
        return hidden_states

    def pixel_shuffle(self, image_features):  # B, S, D
        batch_size, seq_length, feature_dim = image_features.shape
        height = width = int(seq_length**0.5)
        image_features = image_features.reshape(
            image_features.shape[0], width, height, -1
        )
        channels = image_features.shape[-1]
        image_features = image_features.reshape(
            batch_size,
            width,
            int(height / self.downsample_factor),
            int(channels * self.downsample_factor),
        )
        image_features = image_features.transpose(0, 2, 1, 3)
        image_features = image_features.reshape(
            batch_size,
            int(height / self.downsample_factor),
            int(width / self.downsample_factor),
            -1,
        )
        image_features = image_features.transpose(0, 2, 1, 3)
        return image_features


class Model(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.vision_tower = VisionModel(config.vision_config)
        self.language_model = LanguageModel(config.text_config)
        self.multi_modal_projector = AyaVisionMultiModalProjector(config)
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

        spatial_shapes = kwargs.get("spatial_shapes", None)
        # Get the ouptut hidden states from the vision model
        *_, hidden_states = self.vision_tower(
            pixel_values.transpose(0, 2, 3, 1),
            spatial_shapes=spatial_shapes,
            output_hidden_states=True,
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

        # Insert special image tokens in the input_ids
        final_inputs_embeds = self._merge_input_ids_with_image_features(
            image_features, inputs_embeds, input_ids
        )
        return InputEmbeddingsFeatures(inputs_embeds=final_inputs_embeds)

    def _merge_input_ids_with_image_features(
        self, image_features, inputs_embeds, input_ids
    ):
        image_token_index = self.config.image_token_index

        # Positions of <image> tokens in input_ids, assuming batch size is 1
        image_positions = np.where(input_ids[0] == image_token_index)[0].tolist()
        num_images, _, _, vision_hidden_size = image_features.shape

        reshaped_image_hidden_states = image_features.reshape(-1, vision_hidden_size)

        # cast to the dtype of the input_embeds to support quantized models
        reshaped_image_hidden_states = reshaped_image_hidden_states.astype(
            inputs_embeds.dtype
        )
        inputs_embeds[:, image_positions, :] = reshaped_image_hidden_states
        return inputs_embeds

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

    def sanitize(self, weights):
        def transform_key(key):
            if "model.vision_tower" in key:
                key = key.replace("model.vision_tower", "vision_tower")
            if "model.multi_modal_projector" in key:
                key = key.replace(
                    "model.multi_modal_projector", "multi_modal_projector"
                )
            if "model.language_model" in key:
                key = key.replace("model.language_model", "language_model.model")
            if "lm_head" in key and not key.startswith("language_model"):
                key = key.replace("lm_head", "language_model.lm_head")
            return key

        return {transform_key(k): v for k, v in weights.items()}
