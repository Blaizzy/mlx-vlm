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


def masked_scatter(
    final_embedding: mx.array,
    image_mask_expanded: mx.array,
    scaled_image_features: mx.array,
):
    # Reshape the tensors to 1D
    final_embedding_shape = final_embedding.shape
    scaled_image_features_flattened = mx.flatten(scaled_image_features)
    final_embedding_flattened = mx.flatten(final_embedding)
    image_mask_expanded_flattened = mx.flatten(image_mask_expanded)

    # Scatter the scaled image features into the special image token positions
    image_positions = mx.array(np.where(image_mask_expanded_flattened)[0], mx.uint32)
    final_embedding_flattened[image_positions] = scaled_image_features_flattened

    # Reshape back to the original shape
    final_embedding = mx.reshape(final_embedding_flattened, final_embedding_shape)

    return final_embedding


class Model(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.model_type = config.model_type
        self.config = config
        self.vision_tower = VisionModel(config.vision_config)

        if config.vision_feature_layer != -1:
            self.vision_tower.encoder.layers = self.vision_tower.encoder.layers[
                : config.vision_feature_layer + 1
            ]
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
        spatial_shapes: Optional[mx.array] = None,
        pixel_attention_mask: Optional[mx.array] = None,
    ):

        inputs_embeds = self.language_model.model.embed_tokens(input_ids)

        if pixel_values is None:
            return inputs_embeds

        # Get the ouptut hidden states from the vision model
        *_, hidden_states = self.vision_tower(
            pixel_values, output_hidden_states=True, spatial_shapes=spatial_shapes
        )

        img_feature_lengths = pixel_attention_mask.sum(axis=1).tolist()
        image_features = []

        for img_idx in range(hidden_states.shape[0]):
            feature = hidden_states[img_idx]

            feature = feature[: img_feature_lengths[img_idx], :][None, ...]

            feature_org_h, feature_org_w = spatial_shapes[img_idx]
            feature = feature.reshape(1, feature_org_h, feature_org_w, -1)
            feature = self.pixel_unshuffle(feature)

            img_embedding = self.multi_modal_projector(feature)

            img_embedding = img_embedding.reshape(-1, img_embedding.shape[-1])
            image_features.append(img_embedding)

        image_features = mx.concatenate(image_features, axis=0)

        final_inputs_embeds = self.merge_input_ids_with_image_features(
            image_features, inputs_embeds, input_ids, self.config.image_token_index
        )
        return final_inputs_embeds

    @staticmethod
    def merge_input_ids_with_image_features(
        image_features, inputs_embeds, input_ids, image_token_index
    ):
        special_image_mask = input_ids == image_token_index
        n_image_tokens = special_image_mask.sum()
        special_image_mask = special_image_mask[..., None]
        special_image_mask = mx.broadcast_to(special_image_mask, inputs_embeds.shape)

        n_image_features = image_features.shape[0]
        n_image_mask_elements = special_image_mask.sum()
        if n_image_mask_elements != image_features.size:
            raise ValueError(
                f"Image features and image tokens do not match: tokens: {n_image_tokens}, features {n_image_features}"
            )

        inputs_embeds = masked_scatter(
            inputs_embeds, special_image_mask, image_features
        )

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
        spatial_shapes = kwargs.get("spatial_shapes", None)
        pixel_attention_mask = kwargs.get("pixel_attention_mask", None)
        input_embeddings = self.get_input_embeddings(
            input_ids, pixel_values, spatial_shapes, pixel_attention_mask
        )

        logits = self.language_model(
            input_ids, mask=None, cache=cache, inputs_embeds=input_embeddings
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
