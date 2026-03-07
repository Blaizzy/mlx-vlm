from typing import Optional

import mlx.core as mx
import mlx.nn as nn

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

        dtype = inputs_embeds.dtype
        pixel_values = pixel_values.astype(dtype)

        image_features = self.vision(
            pixel_values,
            num_crops=num_crops,
            crop_layouts=crop_layouts,
        )

        B = inputs.shape[0]
        bos_embed = inputs_embeds[:, :1, :]

        num_vision_tokens = image_features.shape[1]
        text_start = 1 + num_vision_tokens

        if inputs_embeds.shape[1] > text_start:
            text_embeds = inputs_embeds[:, text_start:, :]
            final_embeds = mx.concatenate(
                [bos_embed, image_features, text_embeds], axis=1
            )
        else:
            final_embeds = mx.concatenate([bos_embed, image_features], axis=1)

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
        if cache is not None:
            return None

        causal = mx.triu(mx.full((seq_len, seq_len), -mx.inf), k=1)
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

            if new_key.startswith("model."):
                new_key = new_key[len("model.") :]

            if new_key.startswith("region."):
                continue

            if "position_ids" in new_key:
                continue

            if new_key == "text.wte":
                new_key = "text.model.wte.weight"
            elif new_key.startswith("text.lm_head"):
                pass
            elif new_key.startswith("text."):
                new_key = "text.model." + new_key[len("text.") :]

            if new_key.startswith("vision.") and not new_key.startswith(
                "vision.proj_mlp"
            ):
                new_key = "vision.encoder." + new_key[len("vision.") :]

            sanitized[new_key] = v

        return sanitized
