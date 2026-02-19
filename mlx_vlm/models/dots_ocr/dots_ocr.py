from typing import Optional

import mlx.core as mx
import mlx.nn as nn

from ..base import InputEmbeddingsFeatures
from ..llava_bunny.language import LanguageModel
from .config import ModelConfig
from .vision import VisionModel


class Model(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.vision_tower = VisionModel(config.vision_config)
        self.language_model = LanguageModel(config.text_config)

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

        image_grid_thw = kwargs.get("image_grid_thw", None)
        if image_grid_thw is None:
            raise ValueError("image_grid_thw is required when pixel_values is provided")

        dtype = self.vision_tower.patch_embed.patchifier.proj.weight.dtype
        pixel_values = pixel_values.astype(dtype)

        inputs_embeds = self.language_model.model.embed_tokens(input_ids)
        hidden_states = self.vision_tower(
            pixel_values, image_grid_thw, output_hidden_states=False
        )

        final_inputs_embeds = self.merge_input_ids_with_image_features(
            self.config.image_token_id,
            self.config.video_token_id,
            hidden_states,
            inputs_embeds,
            input_ids,
        )

        return InputEmbeddingsFeatures(inputs_embeds=final_inputs_embeds)

    @staticmethod
    def merge_input_ids_with_image_features(
        image_token_id,
        video_token_id,
        image_features,
        inputs_embeds,
        input_ids,
    ):
        image_positions = input_ids == image_token_id
        if mx.sum(image_positions) == 0:
            image_positions = input_ids == video_token_id

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
        return self.language_model.layers

    def __call__(
        self,
        input_ids: mx.array,
        pixel_values: Optional[mx.array] = None,
        mask: Optional[mx.array] = None,
        cache=None,
        **kwargs,
    ):
        input_embeddings_features = self.get_input_embeddings(
            input_ids=input_ids,
            pixel_values=pixel_values,
            **kwargs,
        )

        logits = self.language_model(
            input_ids,
            inputs_embeds=input_embeddings_features.inputs_embeds,
            mask=mask,
            cache=cache,
        )
        return logits

    def sanitize(self, weights):
        def transform_key(key):
            if key.startswith("model.vision_tower."):
                key = key.replace("model.vision_tower.", "vision_tower.", 1)
            elif key.startswith("model."):
                key = key.replace("model.", "language_model.model.", 1)
            elif key.startswith("lm_head."):
                key = key.replace("lm_head.", "language_model.model.lm_head.", 1)
            return key

        return {transform_key(k): v for k, v in weights.items()}
