from typing import Optional

import mlx.core as mx
import mlx.nn as nn
from transformers import AutoProcessor

from ..base import InputEmbeddingsFeatures
from .config import ModelConfig
from .language import LanguageModel
from .processing_paddleocr_vl import PaddleOCRVLProcessor
from .vision import VisionModel

AutoProcessor.register("paddleocr_vl", PaddleOCRVLProcessor)


class Model(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.visual = VisionModel(config.vision_config)
        self.language_model = LanguageModel(config.text_config, config)

    def get_input_embeddings(
        self,
        input_ids: Optional[mx.array] = None,
        pixel_values: Optional[mx.array] = None,
        **kwargs,
    ):
        image_grid_thw = kwargs.pop("image_grid_thw", None)
        video_grid_thw = kwargs.pop("video_grid_thw", None)
        mask = kwargs.pop("mask", None)
        grid_thw = image_grid_thw if image_grid_thw is not None else video_grid_thw

        if pixel_values is None:
            # Reset position state for text-only generation
            self.language_model._position_ids = None
            self.language_model._rope_deltas = None
            return InputEmbeddingsFeatures(
                inputs_embeds=self.language_model.model.embed_tokens(input_ids)
            )

        dtype = self.visual.embeddings.patch_embedding.weight.dtype
        pixel_values = mx.array(pixel_values, dtype=dtype)

        # Get the input embeddings from the language model
        inputs_embeds = self.language_model.model.embed_tokens(input_ids)

        # Get the ouptut hidden states from the vision model
        hidden_states = self.visual(pixel_values, grid_thw, output_hidden_states=False)

        # Insert special image tokens in the input_ids
        final_inputs_embeds = self.merge_input_ids_with_image_features(
            self.config.image_token_id,
            hidden_states,
            inputs_embeds,
            input_ids,
        )

        # Pre-calculate position_ids for chunked prefill
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
        image_features,
        inputs_embeds,
        input_ids,
    ):
        """Merge image features into input embeddings at image token positions.

        Args:
            image_features: Vision features from the vision tower [num_features, hidden_dim]
            inputs_embeds: Input embeddings [batch_size, seq_len, hidden_dim]
            input_ids: Input token IDs [batch_size, seq_len]

        Returns:
            Updated input embeddings with image features inserted
        """

        # Positions of <image> tokens in input_ids
        image_positions = input_ids == image_token_id

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

        input_embeddings_features = self.get_input_embeddings(
            input_ids, pixel_values, **kwargs
        )
        kwargs = {
            "pixel_values": pixel_values,
            **kwargs,
        }
        logits = self.language_model(
            input_ids,
            input_embeddings_features.inputs_embeds,
            mask=mask,
            cache=cache,
            **kwargs,
        )
        return logits

    def sanitize(self, weights):
        _keys_to_ignore_on_load_unexpected = [
            "packing_position_embedding",
            "vision_model.head",
        ]

        def transform_key(key):
            if "visual.vision_model" in key:
                if "embeddings" in key or "post_layernorm" in key:
                    key = key.replace("visual.vision_model", "visual")
                elif "encoder" in key:
                    key = key.replace("visual.vision_model.encoder", "visual")
            elif "mlp_AR" in key:
                key = key.replace("mlp_AR", "visual.projector")
            elif "model" in key:
                key = key.replace("model", "language_model.model")
            elif "lm_head" in key:
                key = key.replace("lm_head", "language_model.lm_head")

            return key

        new_weights = {}
        for k, v in weights.items():
            if (
                "packing_position_embedding" in k
                or "vision_model.head" in k
                or ("visual" in k and "k_proj" in k)
                or ("visual" in k and "v_proj" in k)
            ):
                continue
            elif "visual" in k and "q_proj" in k:
                new_key = transform_key(k)
                k_proj = weights.get(k.replace("q_proj", "k_proj"), None)
                v_proj = weights.get(k.replace("q_proj", "v_proj"), None)
                if k_proj is not None and v_proj is not None:
                    merged_tensor = mx.concatenate([v, k_proj, v_proj], axis=0)
                    merged_key = new_key.replace("q_proj", "qkv")
                    new_weights[merged_key] = merged_tensor
            else:
                new_weights[transform_key(k)] = v

        return new_weights
