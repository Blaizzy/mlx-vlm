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


class Model(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.vision_tower = VisionModel(config.vision_config)
        self.language_model = LanguageModel(config.text_config, config)

    def get_input_embeddings(
        self,
        input_ids: Optional[mx.array] = None,
        pixel_values: Optional[mx.array] = None,
        **kwargs,
    ):
        image_grid_thw = kwargs.get("image_grid_thw", None)
        video_grid_thw = kwargs.get("video_grid_thw", None)
        mask = kwargs.get("mask", None)
        grid_thw = image_grid_thw if image_grid_thw is not None else video_grid_thw

        if pixel_values is None:
            # Reset position state for text-only generation
            self.language_model._position_ids = None
            self.language_model._rope_deltas = None
            return InputEmbeddingsFeatures(
                inputs_embeds=self.language_model.model.embed_tokens(input_ids),
                visual_pos_masks=None,
                deepstack_visual_embeds=None,
            )

        dtype = self.vision_tower.patch_embed.proj.weight.dtype
        pixel_values = pixel_values.astype(dtype)

        # Get the input embeddings from the language model
        inputs_embeds = self.language_model.model.embed_tokens(input_ids)

        # Get the ouptut hidden states from the vision model
        hidden_states, deepstack_image_embeds = self.vision_tower(
            pixel_values, grid_thw
        )

        visual_pos_masks = None
        deepstack_visual_embeds = None

        # Insert special image tokens in the input_ids
        inputs_embeds, image_mask = self.merge_input_ids_with_image_features(
            hidden_states,
            inputs_embeds,
            input_ids,
            self.config.image_token_index,
            self.config.video_token_index,
        )

        image_mask = image_mask[..., 0]
        visual_pos_masks = image_mask
        deepstack_visual_embeds = deepstack_image_embeds

        # Pre-calculate position_ids for chunked prefill
        if image_grid_thw is not None or video_grid_thw is not None:
            position_ids, rope_deltas = self.language_model.get_rope_index(
                input_ids, image_grid_thw, video_grid_thw, mask
            )
            self.language_model._position_ids = position_ids
            self.language_model._rope_deltas = rope_deltas

        return InputEmbeddingsFeatures(
            inputs_embeds=inputs_embeds,
            visual_pos_masks=visual_pos_masks,
            deepstack_visual_embeds=deepstack_visual_embeds,
        )

    @staticmethod
    def merge_input_ids_with_image_features(
        image_features, inputs_embeds, input_ids, image_token_index, video_token_index
    ):
        special_image_mask = input_ids == image_token_index
        special_video_mask = input_ids == video_token_index
        special_image_mask = special_image_mask | special_video_mask
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

        return inputs_embeds, special_image_mask

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
        input_embeddings_features = self.get_input_embeddings(
            input_ids, pixel_values, **kwargs
        )

        kwargs = {
            "pixel_values": pixel_values,
            **input_embeddings_features.to_dict(),
            **kwargs,
        }

        logits = self.language_model(input_ids, mask=mask, cache=cache, **kwargs)

        return logits

    def sanitize(self, weights):
        sanitized_weights = {}
        for key, value in weights.items():
            if "model" in key:
                if "model.language_model" in key:
                    key = key.replace("model.language_model", "language_model.model")
                    if "gate_up_proj" in key:
                        gate_key = key.replace("gate_up_proj", "gate_proj")
                        up_key = key.replace("gate_up_proj", "up_proj")

                        mid_dim = value.shape[-1] // 2
                        sanitized_weights[gate_key] = value[..., :mid_dim].transpose(
                            0, 2, 1
                        )
                        sanitized_weights[up_key] = value[..., mid_dim:].transpose(
                            0, 2, 1
                        )

                        continue
                    if "down_proj" in key:
                        sanitized_weights[key] = value.transpose(0, 2, 1)
                        continue

                elif "model.visual" in key:
                    key = key.replace("model.visual", "vision_tower")
            elif "lm_head" in key:
                key = key.replace("lm_head", "language_model.lm_head")

            sanitized_weights[key] = value

        return sanitized_weights
