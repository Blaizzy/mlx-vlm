from typing import Optional

import mlx.core as mx
import mlx.nn as nn
import numpy as np

from ..base import pixel_shuffle
from ..granite_vision.vision import VisionModel
from ..llama.language import LanguageModel
from ..pooling import EmbeddingOutput, mean_pooling, normalize_embeddings
from .config import ModelConfig


class Model(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.model_type = config.model_type
        self.downsample_ratio = config.downsample_ratio
        self.img_context_token_id = config.img_context_token_id
        self.normalize = config.normalize

        self.vision_model = VisionModel(config.vision_config)
        self.language_model = LanguageModel(config.text_config)

        vit_hidden = config.vision_config.hidden_size
        llm_hidden = config.text_config.hidden_size
        mlp_input = int(vit_hidden * (1 / config.downsample_ratio) ** 2)
        self.mlp1 = [
            nn.LayerNorm(mlp_input),
            nn.Linear(mlp_input, llm_hidden),
            nn.GELU(),
            nn.Linear(llm_hidden, llm_hidden),
        ]

    def extract_feature(self, pixel_values: mx.array) -> mx.array:
        vit_embeds = self.vision_model(pixel_values)[0]
        vit_embeds = pixel_shuffle(vit_embeds, shuffle_ratio=self.downsample_ratio)
        for layer in self.mlp1:
            vit_embeds = layer(vit_embeds)
        return vit_embeds

    def _merge_input_ids_with_image_features(
        self, image_features, inputs_embeds, input_ids
    ):
        B, N, C = inputs_embeds.shape
        image_positions = input_ids == self.img_context_token_id
        image_indices = np.where(image_positions)[1].tolist()
        image_features = image_features.reshape(-1, image_features.shape[-1])
        inputs_embeds[:, image_indices, :] = image_features
        return inputs_embeds.reshape(B, N, C)

    def _last_hidden_state(
        self,
        input_ids: mx.array,
        attention_mask: mx.array,
        pixel_values: Optional[mx.array] = None,
    ) -> mx.array:
        lm = self.language_model.model
        h = lm.embed_tokens(input_ids)

        if pixel_values is not None:
            if pixel_values.ndim == 5:
                pixel_values = pixel_values[0]
            dtype = (
                self.vision_model.vision_model.embeddings.patch_embedding.weight.dtype
            )
            vit_embeds = self.extract_feature(pixel_values.astype(dtype))
            h = self._merge_input_ids_with_image_features(vit_embeds, h, input_ids)

        mask = attention_mask[:, None, None, :]
        mask = mx.repeat(mask, attention_mask.shape[-1], -2)
        mask = mx.where(mask.astype(mx.bool_), 0.0, -mx.inf).astype(h.dtype)

        for layer in lm.layers:
            h = layer(h, mask, cache=None)
        return lm.norm(h)

    def __call__(
        self,
        input_ids: Optional[mx.array] = None,
        attention_mask: Optional[mx.array] = None,
        pixel_values: Optional[mx.array] = None,
        **kwargs,
    ) -> EmbeddingOutput:
        if attention_mask is None:
            attention_mask = mx.ones(input_ids.shape)

        hidden_states = self._last_hidden_state(input_ids, attention_mask, pixel_values)
        pooled = mean_pooling(hidden_states, attention_mask)
        text_embeds = normalize_embeddings(pooled) if self.normalize else pooled
        return EmbeddingOutput(last_hidden_state=hidden_states, text_embeds=text_embeds)

    def sanitize(self, weights):
        sanitized = {}
        for k, v in weights.items():
            if "rotary_emb.inv_freq" in k:
                continue
            if "lm_head" in k:
                continue
            if "position_ids" in k:
                continue
            if "head.attention.in_proj_weight" in k:
                sanitized[k.replace("in_proj_weight", "in_proj.weight")] = v
                continue
            if "head.attention.in_proj_bias" in k:
                sanitized[k.replace("in_proj_bias", "in_proj.bias")] = v
                continue
            if (
                "patch_embedding.weight" in k
                and v.ndim == 4
                and v.shape[1] < v.shape[2]
            ):
                v = v.transpose(0, 2, 3, 1)
            if k.startswith("language_model.") and not k.startswith(
                "language_model.model."
            ):
                k = "language_model.model." + k[len("language_model.") :]
            sanitized[k] = v
        return sanitized

    @property
    def layers(self):
        return self.language_model.layers
