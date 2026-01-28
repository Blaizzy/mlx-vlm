import re
from typing import Optional

import mlx.core as mx
import mlx.nn as nn
import numpy as np

from ..base import InputEmbeddingsFeatures
from .config import ModelConfig
from .language import LanguageModel
from .vision import VisionModel


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


class MLP(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        input_size = config.vision_config.hidden_size * (config.scale_factor**2)
        output_size = config.text_config.hidden_size
        self.proj = nn.Linear(input_size, output_size, bias=False)

    def __call__(self, x):
        return self.proj(x)


class Idefics3Connector(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.scale_factor = config.scale_factor
        self.modality_projection = MLP(config)

    def pixel_shuffle(self, x, scale_factor=2):
        bsz, seq, embed_dim = x.shape
        height = width = int(seq**0.5)
        x = x.reshape(bsz, height, width, embed_dim)
        x = x.reshape(bsz, height, int(width / scale_factor), embed_dim * scale_factor)
        x = x.transpose(0, 2, 1, 3)
        x = x.reshape(
            bsz,
            int(width / scale_factor),
            int(height / scale_factor),
            embed_dim * (scale_factor**2),
        )
        x = x.transpose(0, 2, 1, 3)
        x = x.reshape(bsz, int(seq / (scale_factor**2)), embed_dim * (scale_factor**2))
        return x

    def __call__(self, image_hidden_states):
        image_hidden_states = self.pixel_shuffle(image_hidden_states, self.scale_factor)
        image_hidden_states = self.modality_projection(image_hidden_states)
        return image_hidden_states


class Model(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.model_type = config.model_type
        self.config = config

        self.vision_model = VisionModel(config.vision_config)
        self.language_model = LanguageModel(config.text_config)
        self.connector = Idefics3Connector(config)

    def get_input_embeddings(
        self,
        input_ids: Optional[mx.array] = None,
        pixel_values: Optional[mx.array] = None,
        **kwargs,
    ):
        pixel_attention_mask = kwargs.get("pixel_attention_mask", None)

        if pixel_values is None:
            return InputEmbeddingsFeatures(
                inputs_embeds=self.language_model.embed_tokens(input_ids)
            )

        inputs_embeds = self.language_model.embed_tokens(input_ids)

        batch_size, num_images, num_channels, height, width = pixel_values.shape
        pixel_values = pixel_values.reshape(
            batch_size * num_images, num_channels, height, width
        )

        # Remove padding images - padding image are full 0.
        nb_values_per_image = np.prod(pixel_values.shape[1:])
        real_images_mask = (pixel_values == 0.0).sum(
            axis=(-1, -2, -3)
        ) != nb_values_per_image
        real_images_inds = np.where(real_images_mask)[0].tolist()
        pixel_values = pixel_values[real_images_inds, ...]

        if pixel_attention_mask is None:
            pixel_attention_mask = mx.ones(
                (pixel_values.size(0), pixel_values.size(2), pixel_values.size(3)),
                dtype=mx.bool,
            )
        else:
            # Remove padding images from the mask
            pixel_attention_mask = pixel_attention_mask.reshape(
                batch_size * num_images, height, width
            )
            pixel_attention_mask = pixel_attention_mask[real_images_inds]

        patch_size = self.config.vision_config.patch_size
        batch_size, height, width = pixel_attention_mask.shape

        # Calculate number of patches
        patches_h = height // patch_size
        patches_w = width // patch_size

        # Reshape to extract patches
        reshaped = pixel_attention_mask[
            :, : patches_h * patch_size, : patches_w * patch_size
        ]
        reshaped = reshaped.reshape(
            batch_size, patches_h, patch_size, patches_w, patch_size
        )
        reshaped = reshaped.transpose(
            0, 1, 3, 2, 4
        )  # (batch, patches_h, patches_w, patch_size, patch_size)

        # Sum over patch dimensions and check if any pixels are active
        patch_attention_mask = reshaped.sum(axis=(-1, -2)) > 0

        pooler_output, *_ = self.vision_model(
            pixel_values.transpose(0, 2, 3, 1),
            patch_attention_mask=patch_attention_mask,
            output_hidden_states=True,
        )

        image_features = pooler_output.astype(pixel_values.dtype)
        image_features = self.connector(image_features)

        final_inputs_embeds = self._prepare_inputs_for_multimodal(
            image_features, inputs_embeds, input_ids
        )
        return InputEmbeddingsFeatures(inputs_embeds=final_inputs_embeds)

    def _prepare_inputs_for_multimodal(self, image_features, inputs_embeds, input_ids):
        special_image_mask = input_ids == self.config.image_token_index
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
        return self.language_model.layers

    def __call__(
        self,
        input_ids: mx.array,
        pixel_values: mx.array,
        cache=None,
        **kwargs,
    ):
        input_embeddings_features = self.get_input_embeddings(
            input_ids, pixel_values, **kwargs
        )
        logits = self.language_model(
            inputs=input_ids,
            cache=cache,
            inputs_embeds=input_embeddings_features.inputs_embeds,
        )
        return logits

    def sanitize(self, weights):
        weights = {
            (
                f"{k.split('.', 1)[1]}"
                if re.match(r"^model\.", k)
                else (f"language_model.{k}" if re.match(r"^lm_head\.", k) else k)
            ): v
            for k, v in weights.items()
        }

        weights = {
            (
                f"language_model.{k.split('.', 1)[1]}"
                if re.match(
                    r"^text_model\.",
                    k,
                )
                else k
            ): v
            for k, v in weights.items()
        }

        return weights
