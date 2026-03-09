from typing import Optional

import mlx.core as mx
import mlx.nn as nn

from ..base import InputEmbeddingsFeatures
from .config import ModelConfig
from .language import LanguageModel, RMSNormNoScale
from .vision import VisionModel


def masked_scatter(input_tensor, mask, source):
    """MLX implementation of PyTorch's masked_scatter."""
    mask = mask.astype(mx.bool_)

    if not mask.any():
        return mx.broadcast_to(input_tensor, mask.shape)

    input_shape = mask.shape
    result_flat = mx.broadcast_to(input_tensor, input_shape).flatten()
    mask_flat = mask.flatten()
    source_flat = source.flatten()

    selection_mask = mx.cumsum(mask_flat.astype(mx.int32)) - 1
    source_len = len(source_flat)
    bounded_indices = selection_mask % source_len
    selected_values = source_flat[bounded_indices]
    result_flat = mx.where(mask_flat, selected_values, result_flat)

    return result_flat.reshape(input_shape)


class MultimodalEmbedder(nn.Module):
    """Projects soft tokens from vision/audio into language model space."""

    def __init__(self, embedding_dim: int, text_hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.embedding_projection = nn.Linear(
            embedding_dim, text_hidden_size, bias=False
        )
        self.embedding_post_projection_norm = RMSNormNoScale(text_hidden_size, eps=eps)

    def __call__(self, inputs_embeds: mx.array) -> mx.array:
        proj = self.embedding_projection(inputs_embeds)
        return self.embedding_post_projection_norm(proj)


class Model(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.model_type = config.model_type
        self.config = config

        # Text
        self.language_model = LanguageModel(config.text_config)
        self.vocab_size = config.text_config.vocab_size

        # Vision
        self.vision_tower = VisionModel(config.vision_config)
        self.embed_vision = MultimodalEmbedder(
            embedding_dim=config.vision_config.hidden_size,
            text_hidden_size=config.text_config.hidden_size,
            eps=config.vision_config.rms_norm_eps,
        )

    def get_input_embeddings(
        self,
        input_ids: Optional[mx.array] = None,
        pixel_values: Optional[mx.array] = None,
        **kwargs,
    ):
        inputs_embeds = self.language_model.model.embed_tokens(input_ids)

        # Compute per-layer inputs for text tokens only (exclude image/audio tokens)
        per_layer_inputs = None
        if self.language_model.model.hidden_size_per_layer_input:
            image_mask = input_ids == self.config.image_token_id
            audio_mask = input_ids == self.config.audio_token_id
            text_mask = ~(image_mask | audio_mask)
            per_layer_inputs_tokens = mx.where(
                text_mask, input_ids, mx.zeros_like(input_ids)
            )
            per_layer_inputs = self.language_model.model.get_per_layer_inputs(
                per_layer_inputs_tokens
            )

        if pixel_values is None:
            return InputEmbeddingsFeatures(
                inputs_embeds=inputs_embeds, per_layer_inputs=per_layer_inputs
            )

        # Get vision features
        image_features = self.vision_tower(pixel_values)
        image_features = self.embed_vision(image_features)
        image_features = image_features.astype(inputs_embeds.dtype)

        # Create image mask and scatter
        image_mask = input_ids == self.config.image_token_id
        image_mask_expanded = mx.expand_dims(image_mask, -1)
        image_mask_expanded = mx.broadcast_to(image_mask_expanded, inputs_embeds.shape)

        inputs_embeds = masked_scatter(
            inputs_embeds, image_mask_expanded, image_features
        )

        return InputEmbeddingsFeatures(
            inputs_embeds=inputs_embeds, per_layer_inputs=per_layer_inputs
        )

    def __call__(
        self,
        input_ids: mx.array,
        pixel_values: Optional[mx.array] = None,
        mask: Optional[mx.array] = None,
        cache: Optional[mx.array] = None,
        **kwargs,
    ):
        input_embeddings_features = self.get_input_embeddings(
            input_ids=input_ids,
            pixel_values=pixel_values,
            **kwargs,
        )

        logits = self.language_model(
            input_ids=None,
            cache=cache,
            inputs_embeds=input_embeddings_features.inputs_embeds,
            per_layer_inputs=input_embeddings_features.per_layer_inputs,
        )
        return logits

    def sanitize(self, weights):
        sanitized = {}
        for k, v in weights.items():
            # Skip clipping parameters for non-vision weights (language model doesn't use them)
            if any(
                s in k for s in ["input_max", "input_min", "output_max", "output_min"]
            ):
                if "vision_tower" not in k:
                    continue
            # Skip rotary embedding inv_freq
            if "rotary_emb.inv_freq" in k or "rotary_emb" in k:
                continue
            # Skip audio tower weights until audio encoder is implemented
            if "audio_tower" in k or "embed_audio" in k:
                continue
            # Strip model. prefix
            if k.startswith("model."):
                new_key = k[len("model.") :]
            else:
                new_key = k

            # Remap language_model.xxx -> language_model.model.xxx
            # Weight keys: language_model.layers.0.xxx
            # Model paths: language_model.model.layers.0.xxx
            if new_key.startswith("language_model."):
                rest = new_key[len("language_model.") :]
                new_key = "language_model.model." + rest

            # Handle Conv2d weight transposition for audio SSCP
            if (
                "subsample_conv_projection" in new_key
                and "conv.weight" in new_key
                and v.ndim == 4
            ):
                v = v.transpose(0, 2, 3, 1)
            # Handle depthwise Conv1d for audio
            if "depthwise_conv1d.weight" in new_key and v.ndim == 3:
                v = v.transpose(0, 2, 1)

            sanitized[new_key] = v
        return sanitized

    @property
    def layers(self):
        return self.language_model.model.layers
