from typing import Dict, Optional

import mlx.core as mx
import mlx.nn as nn

from ..base import InputEmbeddingsFeatures, check_array_shape
from .config import ModelConfig
from .language import LanguageModel
from .vision import VisionModel

try:
    from transformers import AutoImageProcessor, AutoProcessor

    from .processing_hunyuan_vl import HunYuanVLImageProcessor, HunYuanVLProcessor

    MODEL_TYPE = "hunyuan_vl"

    AutoImageProcessor.register(
        MODEL_TYPE, slow_image_processor_class=HunYuanVLImageProcessor
    )
    AutoProcessor.register(MODEL_TYPE, HunYuanVLProcessor)

except Exception as e:
    raise e


class Model(nn.Module):

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.model_type = config.model_type
        self.vision_tower = VisionModel(config.vision_config)
        self.language_model = LanguageModel(config)

    def get_input_embeddings(
        self,
        input_ids: Optional[mx.array] = None,
        pixel_values: Optional[mx.array] = None,
        **kwargs,
    ) -> mx.array:

        image_grid_thw = kwargs.pop("image_grid_thw", None)

        position_ids_from_processor = kwargs.pop("position_ids", None)

        # Get text embeddings
        inputs_embeds = self.language_model.model.embed_tokens(input_ids)

        # If no image, return text embeddings
        if pixel_values is None:
            # Reset stored position_ids when no image
            self.language_model._position_ids = None
            return InputEmbeddingsFeatures(inputs_embeds=inputs_embeds)

        # Get vision features
        vision_features = self.vision_tower(pixel_values, image_grid_thw)

        # Find image token positions and replace with vision features
        image_token_id = self.config.image_token_id
        image_mask = input_ids == image_token_id

        # Get number of image tokens expected
        num_image_tokens = image_mask.sum().item()
        num_vision_tokens = vision_features.shape[1]

        if num_image_tokens != num_vision_tokens:
            raise ValueError(
                f"Number of image placeholders ({num_image_tokens}) does not match "
                f"number of vision tokens ({num_vision_tokens}). "
                f"Expected token count based on grid: {num_vision_tokens}"
            )

        B, L, _ = inputs_embeds.shape

        output_parts = []

        for b in range(B):
            mask_b = image_mask[b]  # (L,) boolean mask
            text_embeds_b = inputs_embeds[b]  # (L, D)
            vis_feats_b = vision_features[b]  # (num_vis_tokens, D)

            # Build sequence for this batch
            vis_idx = 0
            seq_parts = []
            for pos in range(L):
                if mask_b[pos].item():
                    # Use vision feature
                    seq_parts.append(vis_feats_b[vis_idx : vis_idx + 1])
                    vis_idx += 1
                else:
                    # Use text embedding
                    seq_parts.append(text_embeds_b[pos : pos + 1])

            # Concatenate all parts for this batch
            batch_embeds = mx.concatenate(seq_parts, axis=0)  # (L, D)
            output_parts.append(batch_embeds[None, :, :])  # (1, L, D)

        # Stack batches
        inputs_embeds = mx.concatenate(output_parts, axis=0)  # (B, L, D)

        # Pre-calculate position_ids for chunked prefill
        if position_ids_from_processor is not None:
            self.language_model._position_ids = position_ids_from_processor
        elif image_grid_thw is not None:
            position_ids = self.language_model.get_xdrope_input_positions(
                input_tokens=input_ids[0].tolist(),
                image_grid_thw=image_grid_thw,
                image_token_id=self.config.image_token_id,
                spatial_merge_size=self.config.vision_config.spatial_merge_size,
            )[None, ...]
            self.language_model._position_ids = position_ids

        return InputEmbeddingsFeatures(inputs_embeds=inputs_embeds)

    @property
    def layers(self):
        return self.language_model.model.layers

    @property
    def head_dim(self):
        return self.config.text_config.head_dim

    @property
    def n_kv_heads(self):
        return self.config.text_config.num_key_value_heads

    def __call__(
        self,
        input_ids: mx.array,
        pixel_values: Optional[mx.array] = None,
        mask: Optional[mx.array] = None,
        cache=None,
        **kwargs,
    ):

        # Get embeddings (with vision features merged if image provided)
        input_embeddings_features = self.get_input_embeddings(
            input_ids=input_ids,
            pixel_values=pixel_values,
            **kwargs,
        )

        # Forward through language model
        return self.language_model(
            input_ids=input_ids,
            inputs_embeds=input_embeddings_features.inputs_embeds,
            mask=mask,
            cache=cache,
            image_grid_thw=image_grid_thw,
        )

    def sanitize(self, weights: Dict[str, mx.array]) -> Dict[str, mx.array]:

        sanitized = {}

        for key, value in weights.items():
            new_key = key

            # Language model mappings
            if key.startswith("model."):
                new_key = "language_model." + key

            # Vision tower mappings
            elif key.startswith("vit."):
                new_key = key.replace("vit.", "vision_tower.", 1)

            # Handle Conv2d weight transposition for MLX
            # PyTorch Conv2d: [out_channels, in_channels, kH, kW]
            # MLX Conv2d: [out_channels, kH, kW, in_channels]
            if (
                "patch_embedding.weight" in new_key
                or "proj.0.weight" in new_key
                or "proj.2.weight" in new_key
            ):
                if not check_array_shape(value):
                    value = value.transpose(0, 2, 3, 1)

            sanitized[new_key] = value

        return sanitized
