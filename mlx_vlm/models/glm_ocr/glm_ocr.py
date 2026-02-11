from typing import Optional

import mlx.core as mx
import mlx.nn as nn

from ..base import InputEmbeddingsFeatures
from .config import ModelConfig
from .language import LanguageModel
from .vision import VisionModel


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
                inputs_embeds=self.language_model.model.embed_tokens(input_ids)
            )

        dtype = self.vision_tower.patch_embed.proj.weight.dtype
        pixel_values = pixel_values.astype(dtype)

        inputs_embeds = self.language_model.model.embed_tokens(input_ids)

        hidden_states = self.vision_tower(
            pixel_values, grid_thw, output_hidden_states=False
        )

        split_sizes = (
            image_grid_thw.prod(-1) // self.vision_tower.spatial_merge_size**2
        ).tolist()

        if len(split_sizes) > 1:
            split_indices = []
            cumsum = 0
            for size in split_sizes[:-1]:
                cumsum += size
                split_indices.append(cumsum)
            hidden_states = mx.split(hidden_states, split_indices, axis=0)
            hidden_states = mx.concatenate(hidden_states, axis=0).astype(
                hidden_states[0].dtype
            )

        final_inputs_embeds = self.merge_input_ids_with_image_features(
            self.config.image_token_id,
            self.config.video_token_id,
            hidden_states,
            inputs_embeds,
            input_ids,
        )

        # Pre-calculate position_ids for chunked prefill (like Qwen3-VL)
        if image_grid_thw is not None or video_grid_thw is not None:
            position_ids, rope_deltas = self.language_model.get_rope_index(
                input_ids, image_grid_thw, video_grid_thw, mask
            )
            self.language_model._position_ids = position_ids
            self.language_model._rope_deltas = rope_deltas

        return InputEmbeddingsFeatures(inputs_embeds=final_inputs_embeds)

    @staticmethod
    def merge_input_ids_with_image_features(
        image_token_id,
        video_token_id,
        image_features,
        inputs_embeds,
        input_ids,
    ):
        """Merge image features into input embeddings at image token positions."""
        image_positions = input_ids == image_token_id
        if mx.sum(image_positions) == 0:
            image_positions = input_ids == video_token_id

        batch_size, seq_len = input_ids.shape

        batch_outputs = []
        feature_start_idx = 0

        for batch_idx in range(batch_size):
            image_mask = image_positions[batch_idx]
            num_positions = mx.sum(image_mask).item()

            if num_positions > 0:
                batch_features = image_features[
                    feature_start_idx : feature_start_idx + num_positions
                ]

                if batch_features.shape[0] != num_positions:
                    raise ValueError(
                        f"Number of image token positions ({num_positions}) does not match "
                        f"number of image features ({batch_features.shape[0]}) for batch {batch_idx}"
                    )

                cumsum = mx.cumsum(image_mask.astype(mx.int32))
                feature_indices = mx.where(image_mask, cumsum - 1, 0)

                gathered_features = batch_features[feature_indices]

                image_mask_expanded = mx.expand_dims(image_mask, axis=-1)
                batch_output = mx.where(
                    image_mask_expanded, gathered_features, inputs_embeds[batch_idx]
                )

                feature_start_idx += num_positions
            else:
                batch_output = inputs_embeds[batch_idx]

            batch_outputs.append(batch_output)

        return mx.stack(batch_outputs, axis=0)

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

        logits = self.language_model(
            input_ids,
            input_embeddings_features.inputs_embeds,
            mask=mask,
            cache=cache,
            **kwargs,
        )

        return logits

    def sanitize(self, weights):
        def transform_key(key):
            if "visual" in key:
                if "vision_tower" not in key:
                    key = key.replace("model.", "").replace("visual", "vision_tower")
            if "model.language_model" in key:
                key = key.replace("model.language_model", "language_model.model")
            if "lm_head" in key and not key.startswith("language_model"):
                key = key.replace("lm_head", "language_model.lm_head")
            return key

        sanitized = {}
        for k, v in weights.items():
            new_key = transform_key(k)

            if "layers.16" in new_key:
                continue

            sanitized[new_key] = v

        sanitized = self.vision_tower.sanitize(sanitized)

        return sanitized
