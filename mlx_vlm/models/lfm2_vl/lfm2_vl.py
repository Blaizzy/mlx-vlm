from typing import Optional

import mlx.core as mx
import mlx.nn as nn
import numpy as np

from .config import ModelConfig
from .language import LanguageModel
from .vision import VisionModel


class Lfm2VlMultiModalProjector(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        in_channels = config.vision_config.hidden_size * (config.downsample_factor**2)
        self.layer_norm = nn.LayerNorm(in_channels)
        self.linear_1 = nn.Linear(
            in_channels,
            config.projector_hidden_size,
            bias=config.projector_bias,
        )

        self.linear_2 = nn.Linear(
            config.projector_hidden_size,
            config.text_config.hidden_size,
            bias=config.projector_bias,
        )

    def __call__(self, x):
        x = self.linear_1(self.layer_norm(x))
        x = self.linear_2(nn.gelu(x))
        return x


class PixelUnshuffleBlock(nn.Module):
    def __init__(self, factor: int):
        super().__init__()
        self.factor = factor

    def __call__(self, x):
        n, w, h, c = x.shape
        if w % self.factor != 0:
            x = mx.concatenate(
                [
                    x,
                    mx.zeros((n, self.factor - (w % self.factor), h, c), dtype=x.dtype),
                ],
                axis=1,
            )
            n, w, h, c = x.shape

        if h % self.factor != 0:
            x = mx.concatenate(
                [
                    x,
                    mx.zeros((n, w, self.factor - (h % self.factor), c), dtype=x.dtype),
                ],
                axis=2,
            )
            n, w, h, c = x.shape
        x = x.reshape(n, w, int(h / self.factor), int(c * self.factor))
        x = x.transpose(0, 2, 1, 3)
        x = x.reshape(
            n, int(h / self.factor), int(w / self.factor), int(c * self.factor**2)
        )
        x = x.transpose(0, 2, 1, 3)
        return x


class Model(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.model_type = config.model_type
        self.config = config
        self.vision_tower = VisionModel(config.vision_config)

        if config.vision_feature_layer != -1:
            self.vision_tower.vision_model.encoder.layers = (
                self.vision_tower.vision_model.encoder.layers[
                    : config.vision_feature_layer + 1
                ]
            )
        if config.downsample_factor > 1:
            self.pixel_unshuffle = PixelUnshuffleBlock(config.downsample_factor)
        else:
            self.pixel_unshuffle = nn.Identity()

        self.multi_modal_projector = Lfm2VlMultiModalProjector(config)
        self.language_model = LanguageModel(config.text_config)

    def get_input_embeddings(
        self,
        input_ids: Optional[mx.array] = None,
        pixel_values: Optional[mx.array] = None,
    ):
        if pixel_values is None:
            return self.language_model.model.embed_tokens(input_ids)

        # Get the input embeddings from the language model
        inputs_embeds = self.language_model.model.embed_tokens(input_ids)

        # Get the ouptut hidden states from the vision model
        *_, hidden_states = self.vision_tower(
            pixel_values.transpose(0, 2, 3, 1), output_hidden_states=True
        )

        img_feature_lengths = pixel_attention_mask.sum(dim=1)
        image_features = []

        for img_idx in range(hidden_states.shape[0]):
            feature = hidden_states[img_idx]
            # unpad the image representation
            feature = feature[: img_feature_lengths[img_idx], :][None, ...]

            feature_org_h, feature_org_w = spatial_shapes[img_idx]
            feature = feature.reshape(1, feature_org_h, feature_org_w, -1)
            feature = self.pixel_unshuffle(feature)

            # project the image representation
            img_embedding = self.multi_modal_projector(feature)

            # flatten here to handle variable length in naflex
            img_embedding = img_embedding.reshape(-1, img_embedding.size(-1))
            image_features.append(img_embedding)

        # Insert special image tokens in the input_ids
        final_inputs_embeds = self._merge_input_ids_with_image_features(
            image_features, inputs_embeds, input_ids
        )
        return final_inputs_embeds

    def _merge_input_ids_with_image_features(
        self, image_features, inputs_embeds, input_ids
    ):
        image_token_index = self.config.image_token_index

        # Positions of <image> tokens in input_ids, assuming batch size is 1
        image_positions = np.where(input_ids == image_token_index)[1].tolist()
        num_images, _, vision_hidden_size = image_features.shape

        reshaped_image_hidden_states = image_features.reshape(-1, vision_hidden_size)

        # cast to the dtype of the input_embeds to support quantized models
        reshaped_image_hidden_states = reshaped_image_hidden_states.astype(
            inputs_embeds.dtype
        )

        # Pad image_positions to match the length of reshaped_image_hidden_states
        num_positions_needed = len(image_positions)

        if reshaped_image_hidden_states.shape[0] > num_positions_needed:
            # TODO: Think about how to handle this case
            raise ValueError(
                "Llava model supports only one image per input. Please check your input_ids and pixel_values."
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
        input_embddings = self.get_input_embeddings(input_ids, pixel_values)
        logits = self.language_model(
            input_ids, mask=mask, cache=cache, inputs_embeds=input_embddings
        )
        return logits

    def sanitize(self, weights):
        def transform_key(key):
            if "vision_tower" in key:
                key = (
                    key.replace("model.", "")
                    .replace("vision_encoder", "encoder")
                    .replace("vision_embeddings", "embeddings")
                    .replace("vision_post_layernorm", "post_layernorm")
                )

            if "language_model" in key:
                key = key.replace("model.language_model", "language_model.model")

            if "multi_modal_projector" in key:
                key = key.replace(
                    "model.multi_modal_projector", "multi_modal_projector"
                )

            return key

        return {transform_key(k): v for k, v in weights.items()}
