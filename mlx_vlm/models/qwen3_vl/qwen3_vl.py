from typing import Optional

import mlx.core as mx
import mlx.nn as nn
import numpy as np

from ..base import InputEmbeddingsFeatures
from . import processing_qwen3_vl  # noqa: F401
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
        self.vision_tower = (
            None if config.skip_vision else VisionModel(config.vision_config)
        )
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
        pixel_values_videos = kwargs.get("pixel_values_videos", None)

        # Video inputs flow in via pixel_values_videos from the generic
        # prepare_inputs path; alias to pixel_values for the unified encoder.
        if pixel_values is None and pixel_values_videos is None:
            position_ids, rope_deltas = self.language_model.get_rope_index(
                input_ids, attention_mask=mask
            )
            return InputEmbeddingsFeatures(
                inputs_embeds=self.language_model.model.embed_tokens(input_ids),
                position_ids=position_ids,
                rope_deltas=rope_deltas,
            )

        # Get the input embeddings from the language model
        inputs_embeds = self.language_model.model.embed_tokens(input_ids)

        cached = kwargs.get("cached_image_features", None)
        if cached is not None:
            inputs_embeds, _ = self.merge_input_ids_with_image_features(
                cached,
                inputs_embeds,
                input_ids,
                self.config.image_token_index,
                self.config.video_token_index,
            )
            deepstack_visual_embeds = None
        else:
            if self.vision_tower is None:
                raise ValueError(
                    "this Qwen3-VL model was loaded without a vision tower"
                )
            dtype = self.vision_tower.patch_embed.proj.weight.dtype
            image_features = image_deepstack = None
            video_features = video_deepstack = None
            if pixel_values is not None:
                image_features, image_deepstack = self.vision_tower(
                    pixel_values.astype(dtype), image_grid_thw
                )
            if pixel_values_videos is not None:
                video_features, video_deepstack = self.vision_tower(
                    pixel_values_videos.astype(dtype), video_grid_thw
                )

            image_mask = input_ids == self.config.image_token_index
            video_mask = input_ids == self.config.video_token_index
            if image_features is not None:
                inputs_embeds = masked_scatter(
                    inputs_embeds,
                    mx.broadcast_to(image_mask[..., None], inputs_embeds.shape),
                    image_features,
                )
            if video_features is not None:
                inputs_embeds = masked_scatter(
                    inputs_embeds,
                    mx.broadcast_to(video_mask[..., None], inputs_embeds.shape),
                    video_features,
                )
            deepstack_visual_embeds = self._merge_deepstack_features(
                input_ids,
                image_deepstack,
                video_deepstack,
            )

        visual_pos_masks = (input_ids == self.config.image_token_index) | (
            input_ids == self.config.video_token_index
        )
        if deepstack_visual_embeds is not None:
            mx.eval(deepstack_visual_embeds)

        position_ids, rope_deltas = self.language_model.get_rope_index(
            input_ids, image_grid_thw, video_grid_thw, mask
        )

        return InputEmbeddingsFeatures(
            inputs_embeds=inputs_embeds,
            visual_pos_masks=visual_pos_masks,
            deepstack_visual_embeds=deepstack_visual_embeds,
            position_ids=position_ids,
            rope_deltas=rope_deltas,
        )

    def _merge_deepstack_features(
        self,
        input_ids: mx.array,
        image_features: Optional[mx.array],
        video_features: Optional[mx.array],
    ) -> Optional[mx.array]:
        if image_features is None:
            return video_features
        if video_features is None:
            return image_features
        if len(image_features) != len(video_features):
            raise ValueError("image and video deepstack feature counts must match")

        token_rows = input_ids.tolist()
        merged_layers = []
        for image_layer, video_layer in zip(image_features, video_features):
            rows = []
            image_index = video_index = 0
            for token_row in token_rows:
                for token in token_row:
                    if token == self.config.image_token_index:
                        rows.append(image_layer[image_index])
                        image_index += 1
                    elif token == self.config.video_token_index:
                        rows.append(video_layer[video_index])
                        video_index += 1
            merged_layers.append(mx.stack(rows))
        return mx.stack(merged_layers)

    def hidden_state_at_layer(
        self,
        input_ids: mx.array,
        layer: int,
        pixel_values: Optional[mx.array] = None,
        mask: Optional[mx.array] = None,
        **kwargs,
    ) -> mx.array:
        """Return the pre-final-norm hidden state after ``layer`` decoder layers."""
        features = self.get_input_embeddings(
            input_ids, pixel_values, mask=mask, **kwargs
        )
        return self.language_model.model(
            input_ids,
            inputs_embeds=features.inputs_embeds,
            mask=mask,
            position_ids=features.position_ids,
            visual_pos_masks=features.visual_pos_masks,
            deepstack_visual_embeds=features.deepstack_visual_embeds,
            stop_after_layer=layer,
            apply_final_norm=False,
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

        kwargs.update(
            {
                "pixel_values": pixel_values,
                **input_embeddings_features.to_dict(),
            }
        )

        logits = self.language_model(input_ids, mask=mask, cache=cache, **kwargs)
        return logits

    def sanitize(self, weights):
        sanitized_weights = {}
        for key, value in weights.items():
            if "model" in key:
                if "model.language_model" in key:
                    key = key.replace("model.language_model", "language_model.model")

                elif "model.visual" in key:
                    key = key.replace("model.visual", "vision_tower")
            elif "lm_head" in key:
                key = key.replace("lm_head", "language_model.lm_head")

            sanitized_weights[key] = value

        return sanitized_weights
