from typing import Optional

import mlx.core as mx
import mlx.nn as nn

from .config import ModelConfig
from .language import LanguageModel
from .processing import Glm46VProcessor
from .vision import VisionModel

# Register the processor with the name expected by the model config
try:
    from transformers import AutoProcessor

    # The model's preprocessor_config.json specifies "processor_class": "Glm46VProcessor"
    AutoProcessor.register("Glm46VProcessor", Glm46VProcessor)
except Exception as e:
    print(f"Error registering glm4v processor: {e}")


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
        image_grid_thw: Optional[mx.array] = None,
    ):
        if pixel_values is None:
            return self.language_model.model.embed_tokens(input_ids)

        dtype = self.vision_tower.patch_embed.proj.weight.dtype
        pixel_values = pixel_values.astype(dtype)

        # Get the input embeddings from the language model
        inputs_embeds = self.language_model.model.embed_tokens(input_ids)

        # Get the ouptut hidden states from the vision model
        hidden_states = self.vision_tower(
            pixel_values, image_grid_thw, output_hidden_states=False
        )

        split_sizes = (
            image_grid_thw.prod(-1) // self.vision_tower.spatial_merge_size**2
        ).tolist()
        hidden_states = mx.split(
            hidden_states, [split_sizes[0], sum(split_sizes[:2])], axis=0
        )

        hidden_states = mx.concatenate(hidden_states, axis=0).astype(
            hidden_states[0].dtype
        )

        # Insert special image tokens in the input_ids
        final_inputs_embeds = self.merge_input_ids_with_image_features(
            self.config.image_token_id,
            self.config.video_token_id,
            hidden_states,
            inputs_embeds,
            input_ids,
        )
        return final_inputs_embeds

    @staticmethod
    def merge_input_ids_with_image_features(
        image_token_id,
        video_token_id,
        image_features,
        inputs_embeds,
        input_ids,
    ):
        """Merge image features into input embeddings at image token positions.

        Args:
            image_token_id: The token ID for image placeholders
            video_token_id: The token ID for video placeholders (fallback)
            image_features: Vision features from the vision tower [num_features, hidden_dim]
            inputs_embeds: Input embeddings [batch_size, seq_len, hidden_dim]
            input_ids: Input token IDs [batch_size, seq_len]
            grid_thw: Grid dimensions for each image (optional, not used in simple case)

        Returns:
            Updated input embeddings with image features inserted
        """
        # Find positions of image tokens
        image_positions = input_ids == image_token_id
        if mx.sum(image_positions) == 0:
            image_positions = input_ids == video_token_id

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
        pixel_values: Optional[mx.array] = None,
        mask: Optional[mx.array] = None,
        cache=None,
        **kwargs,
    ):
        image_grid_thw = kwargs.pop("image_grid_thw", None)
        video_grid_thw = kwargs.pop("video_grid_thw", None)
        grid_thw = image_grid_thw if image_grid_thw is not None else video_grid_thw

        inputs_embeds = self.get_input_embeddings(input_ids, pixel_values, grid_thw)

        kwargs = {
            "pixel_values": pixel_values,
            "image_grid_thw": image_grid_thw,
            "video_grid_thw": video_grid_thw,
            **kwargs,
        }

        logits = self.language_model(
            input_ids, inputs_embeds, mask=mask, cache=cache, **kwargs
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

        return {transform_key(k): v for k, v in weights.items()}
