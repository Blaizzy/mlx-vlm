from typing import Optional

import mlx.core as mx
import mlx.nn as nn
import numpy as np

from ..base import InputEmbeddingsFeatures
from . import processing_moondream3  # noqa: F401
from .config import ModelConfig
from .language import LanguageModel
from .vision import VisionModel


class Model(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.model_type = config.model_type
        self.vision = VisionModel(config.vision_config)
        self.text = LanguageModel(config.text_config)

    def __call__(
        self,
        inputs: mx.array,
        inputs_embeds: Optional[mx.array] = None,
        pixel_values: Optional[mx.array] = None,
        mask: Optional[mx.array] = None,
        cache=None,
        **kwargs,
    ):
        if inputs_embeds is None:
            input_embeddings_features = self.get_input_embeddings(
                inputs, pixel_values, **kwargs
            )
            inputs_embeds = input_embeddings_features.inputs_embeds
            if input_embeddings_features.attention_mask_4d is not None:
                mask = input_embeddings_features.attention_mask_4d

        return self.text(inputs, inputs_embeds=inputs_embeds, mask=mask, cache=cache)

    def get_input_embeddings(
        self,
        inputs: mx.array,
        pixel_values: Optional[mx.array] = None,
        **kwargs,
    ):
        inputs_embeds = self.text.model.wte(inputs)

        if pixel_values is None:
            return InputEmbeddingsFeatures(inputs_embeds=inputs_embeds)

        num_crops = kwargs.get("num_crops", None)
        crop_layouts = kwargs.get("crop_layouts", None)

        # Process images through vision encoder
        image_features = self.vision(
            pixel_values,
            num_crops=num_crops,
            crop_layouts=crop_layouts,
        )  # (B, 729, text_dim)

        # Build combined embeddings: [BOS] [vision_tokens] [text_tokens]
        # The BOS token is at position 0, vision tokens follow
        B = inputs.shape[0]
        bos_embed = inputs_embeds[:, :1, :]  # (B, 1, D)

        # Find where text tokens start (after the image placeholder tokens)
        # Vision tokens = 729 per image, preceded by BOS
        num_vision_tokens = image_features.shape[1]  # 729
        text_start = 1 + num_vision_tokens

        if inputs_embeds.shape[1] > text_start:
            text_embeds = inputs_embeds[:, text_start:, :]
            final_embeds = mx.concatenate(
                [bos_embed, image_features, text_embeds], axis=1
            )
        else:
            final_embeds = mx.concatenate([bos_embed, image_features], axis=1)

        # Create prefix attention mask if needed
        # First 730 tokens (1 BOS + 729 vision) get bidirectional attention
        prefix_len = 1 + num_vision_tokens
        seq_len = final_embeds.shape[1]
        attention_mask_4d = self._create_prefix_attention_mask(
            seq_len, prefix_len, cache=None
        )

        return InputEmbeddingsFeatures(
            inputs_embeds=final_embeds,
            attention_mask_4d=attention_mask_4d,
        )

    def _create_prefix_attention_mask(
        self, seq_len: int, prefix_len: int, cache=None
    ) -> Optional[mx.array]:
        """Create attention mask with bidirectional attention for prefix tokens.

        Tokens 0..prefix_len-1 can attend to each other (bidirectional).
        Tokens prefix_len+ use causal masking but can attend to prefix tokens.
        """
        if cache is not None:
            # During generation, we're past the prefix - use standard causal mask
            return None

        # Build the mask: (1, 1, seq_len, seq_len)
        # Start with causal mask
        causal = mx.triu(
            mx.full((seq_len, seq_len), -mx.inf), k=1
        )

        # Make prefix tokens bidirectional: allow all prefix tokens to attend
        # to each other (zero out the upper triangle in the prefix block)
        causal[:prefix_len, :prefix_len] = 0.0

        return causal.reshape(1, 1, seq_len, seq_len)

    @property
    def layers(self):
        return self.text.model.blocks

    @property
    def head_dim(self):
        return self.config.text_config.head_dim

    @property
    def n_kv_heads(self):
        return self.config.text_config.num_key_value_heads

    @property
    def language_model(self):
        return self.text

    @property
    def vision_model(self):
        return self.vision

    def sanitize(self, weights):
        sanitized = {}
        for k, v in weights.items():
            new_key = k

            # Strip 'model.' prefix from original moondream weights
            if new_key.startswith("model."):
                new_key = new_key[len("model."):]

            # Skip region model weights (not needed for basic generation)
            if new_key.startswith("region."):
                continue

            # Skip position_ids
            if "position_ids" in new_key:
                continue

            # Handle text model key mapping
            if new_key == "text.wte":
                # Raw embedding parameter -> nn.Embedding weight
                new_key = "text.model.wte.weight"
            elif new_key.startswith("text.lm_head"):
                # lm_head stays at text.lm_head level
                pass
            elif new_key.startswith("text."):
                # All other text keys go under text.model.*
                new_key = "text.model." + new_key[len("text."):]

            # Handle vision model key mapping
            # vision.proj_mlp.* stays as vision.proj_mlp.*
            # All other vision.* goes to vision.encoder.*
            if new_key.startswith("vision.") and not new_key.startswith(
                "vision.proj_mlp"
            ):
                new_key = "vision.encoder." + new_key[len("vision."):]

            sanitized[new_key] = v

        return sanitized
