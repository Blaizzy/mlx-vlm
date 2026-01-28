from typing import List, Optional, Union

import mlx.core as mx
import mlx.nn as nn

from ..base import InputEmbeddingsFeatures
from .config import ModelConfig
from .language import LanguageModel
from .vision import Llama4MultiModalProjector, VisionModel


class Model(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.vision_model = VisionModel(config.vision_config)
        self.multi_modal_projector = Llama4MultiModalProjector(config)
        self.language_model = LanguageModel(config.text_config)
        self.vocab_size = config.text_config.vocab_size

    def set_input_embeddings(self, value):
        self.language_model.set_input_embeddings(value)

    def get_output_embeddings(self):
        return self.language_model.get_output_embeddings()

    def set_output_embeddings(self, new_embeddings):
        self.language_model.set_output_embeddings(new_embeddings)

    def set_decoder(self, decoder):
        self.language_model.set_decoder(decoder)

    def get_decoder(self):
        return self.language_model.get_decoder()

    def get_image_features(
        self,
        pixel_values: mx.array,
        vision_feature_layer: Union[int, List[int]],
        vision_feature_select_strategy: str,
        **kwargs,
    ):
        if vision_feature_select_strategy not in ["default", "full"]:
            raise ValueError(
                f"Unexpected select feature strategy: {self.vision_feature_select_strategy}"
            )
        kwargs = {k: v for k, v in kwargs.items() if v is not None}
        hidden_state = self.vision_model(
            pixel_values, output_hidden_states=False, **kwargs
        )
        return hidden_state

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

        image_features = self.get_image_features(
            pixel_values=pixel_values,
            vision_feature_layer=kwargs.get("vision_feature_layer", -1),
            vision_feature_select_strategy=kwargs.get(
                "vision_feature_select_strategy", "default"
            ),
        )

        vision_flat = image_features.reshape(-1, image_features.shape[-1])
        projected_vision_flat = self.multi_modal_projector(vision_flat)

        # Insert special image tokens in the input_ids
        final_inputs_embeds = self._prepare_inputs_for_multimodal(
            projected_vision_flat, inputs_embeds, input_ids
        )
        return InputEmbeddingsFeatures(inputs_embeds=final_inputs_embeds)

    def _prepare_inputs_for_multimodal(self, image_features, inputs_embeds, input_ids):
        image_token_index = self.config.image_token_index

        # Find positions of <image> tokens
        image_mask = input_ids == image_token_index

        batch_size, seq_len = input_ids.shape

        # Process each batch item
        batch_outputs = []
        feature_start_idx = 0

        for batch_idx in range(batch_size):
            batch_mask = image_mask[batch_idx]
            num_positions = mx.sum(batch_mask).item()

            if num_positions > 0:
                batch_features = image_features[
                    feature_start_idx : feature_start_idx + num_positions
                ]

                # Create indices for gathering
                cumsum = mx.cumsum(batch_mask.astype(mx.int32))
                feature_indices = mx.where(batch_mask, cumsum - 1, 0)

                # Gather features
                gathered_features = batch_features[feature_indices]

                # Combine with original embeddings
                batch_mask_expanded = mx.expand_dims(batch_mask, axis=-1)
                batch_output = mx.where(
                    batch_mask_expanded, gathered_features, inputs_embeds[batch_idx]
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
        pixel_values: mx.array,
        cache=None,
        **kwargs,
    ):

        input_embeddings_features = self.get_input_embeddings(
            input_ids, pixel_values, **kwargs
        )
        logits = self.language_model(
            inputs=input_ids,
            inputs_embeds=input_embeddings_features.inputs_embeds,
            cache=cache,
        )
        return logits
