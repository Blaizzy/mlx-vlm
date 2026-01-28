from typing import Optional

import mlx.core as mx
import mlx.nn as nn

from ..base import InputEmbeddingsFeatures
from .config import ModelConfig
from .language import LanguageModel
from .vision import VisionModel


class LlavaMultiModalProjector(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.linear_1 = nn.Linear(
            config.vision_config.hidden_size, config.text_config.hidden_size, bias=True
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

        # Get the output hidden states from the vision model
        if isinstance(pixel_values, list):
            pixel_values = mx.concatenate(
                [mx.array(pv)[None, ...] for pv in pixel_values], axis=0
            )
        if pixel_values.ndim == 3:
            pixel_values = pixel_values[None, ...]

        # Pass pixel_values as list of images, as each image is individually run through conv2d and position encoding
        # Reference code from transformers: https://github.com/huggingface/transformers/blob/main/src/transformers/models/pixtral/modeling_pixtral.py#L479C9-L479C21
        # and mistral_inference: https://github.com/mistralai/mistral-inference/blob/main/src/mistral_inference/vision_encoder.py#L85
        *_, hidden_states = self.vision_tower(
            pixel_values.transpose(0, 2, 3, 1),
            output_hidden_states=True,
        )
        # Select the hidden states from the desired layer
        selected_image_feature = hidden_states[self.vision_feature_layer]

        # Pass image features through the multi-modal projector
        image_features = self.multi_modal_projector(selected_image_feature)

        # Insert special image tokens in the input_ids
        final_inputs_embeds = self.merge_input_ids_with_image_features(
            self.config.image_token_index, image_features, inputs_embeds, input_ids
        )
        return InputEmbeddingsFeatures(inputs_embeds=final_inputs_embeds)

    @staticmethod
    def merge_input_ids_with_image_features(
        image_token_index, image_features, inputs_embeds, input_ids
    ):
        """Merge image features into input embeddings at image token positions.

        Args:
            image_token_index: Token ID for image placeholder
            image_features: Vision features from the projector [1, num_features, hidden_dim]
            inputs_embeds: Input embeddings [batch_size, seq_len, hidden_dim]
            input_ids: Input token IDs [batch_size, seq_len]

        Returns:
            Updated input embeddings with image features inserted
        """
        # Remove the extra batch dimension from image_features if present
        if image_features.ndim == 3 and image_features.shape[0] == 1:
            image_features = image_features.squeeze(0)  # [num_features, hidden_dim]

        # Positions of <image> tokens in input_ids
        image_positions = input_ids == image_token_index

        # Get dimensions
        batch_size, seq_len = input_ids.shape

        # Process each batch item
        batch_outputs = []
        feature_start_idx = 0

        for batch_idx in range(batch_size):
            # Get mask for this batch
            image_mask = image_positions[batch_idx]
            num_positions = mx.sum(image_mask).item()

            if num_positions > 0:
                # Extract features for this batch
                batch_features = image_features[
                    feature_start_idx : feature_start_idx + num_positions
                ]

                # Validate we have the right number of features
                if batch_features.shape[0] != num_positions:
                    raise ValueError(
                        f"Number of image token positions ({num_positions}) does not match "
                        f"number of image features ({batch_features.shape[0]}) for batch {batch_idx}"
                    )

                # Create indices for gathering
                cumsum = mx.cumsum(image_mask.astype(mx.int32))
                feature_indices = mx.where(image_mask, cumsum - 1, 0)

                # Gather features
                gathered_features = batch_features[feature_indices]

                # Combine with original embeddings
                image_mask_expanded = mx.expand_dims(image_mask, axis=-1)
                batch_output = mx.where(
                    image_mask_expanded, gathered_features, inputs_embeds[batch_idx]
                )

                feature_start_idx += num_positions
            else:
                # No image tokens in this batch item
                batch_output = inputs_embeds[batch_idx]

            batch_outputs.append(batch_output)

        # Stack all batch outputs
        return mx.stack(batch_outputs, axis=0)

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
            if "vision_tower" in key and "vision_model" not in key:
                if "transformer" in key:
                    key = key.replace("vision_tower", "vision_tower.vision_model")
                if "patch_conv" in key:
                    key = key.replace("vision_tower", "vision_tower.vision_model")
                if "ln_pre" in key:
                    key = key.replace("vision_tower", "vision_tower.vision_model")

            elif "vision_encoder" in key and "vision_tower" not in key:
                if "transformer" in key:
                    key = key.replace(
                        "model.vision_encoder", "vision_tower.vision_model"
                    )
                if "patch_conv" in key:
                    key = key.replace(
                        "model.vision_encoder", "vision_tower.vision_model"
                    )
                if "ln_pre" in key:
                    key = key.replace(
                        "model.vision_encoder", "vision_tower.vision_model"
                    )

            elif "model.language_model" in key and "language_model.model" not in key:
                key = key.replace("model.language_model", "language_model.model")

            elif "lm_head" in key and "language_model" not in key:
                key = key.replace("lm_head", "language_model.lm_head")

            elif "model.vision_projection" in key:
                key = key.replace("model.vision_projection", "multi_modal_projector")

            return key

        return {transform_key(k): v for k, v in weights.items()}
