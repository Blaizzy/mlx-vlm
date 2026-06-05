from typing import Optional

import mlx.core as mx
import mlx.nn as nn

from ..base import InputEmbeddingsFeatures, LanguageModelOutput
from .config import ModelConfig
from .language import LanguageModel


class Model(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.model_type = config.model_type
        self.language_model = LanguageModel(config)

    def get_input_embeddings(
        self,
        input_ids: Optional[mx.array] = None,
        pixel_values: Optional[mx.array] = None,
        **kwargs,
    ) -> InputEmbeddingsFeatures:
        if pixel_values is not None:
            raise ValueError("HRM-Text is a text-only model.")
        if input_ids is None:
            raise ValueError("input_ids are required for HRM-Text.")
        return InputEmbeddingsFeatures(
            inputs_embeds=self.language_model.model.embed_tokens(input_ids)
        )

    def __call__(
        self,
        input_ids: mx.array,
        pixel_values: mx.array = None,
        mask: mx.array = None,
        cache=None,
        **kwargs,
    ) -> LanguageModelOutput:
        input_embeddings_features = self.get_input_embeddings(input_ids, pixel_values)
        return self.language_model(
            input_ids,
            cache=cache,
            mask=mask,
            inputs_embeds=input_embeddings_features.inputs_embeds,
            **kwargs,
        )

    def sanitize(self, weights):
        config = self.config

        def transform_key(key):
            if key.startswith("language_model."):
                return key
            if key.startswith("model."):
                key = key.replace("model.", "language_model.model.", 1)
            if key.startswith("lm_head."):
                key = key.replace("lm_head.", "language_model.lm_head.", 1)
            return key.replace(".attn.", ".self_attn.")

        sanitized = {}
        q_size = config.num_attention_heads * config.head_dim
        kv_size = config.num_key_value_heads * config.head_dim
        for key, value in weights.items():
            key = transform_key(key)
            if key.endswith(".self_attn.gqkv_proj.weight"):
                prefix = key[: -len("gqkv_proj.weight")]
                gate, query, key_weight, value_weight = mx.split(
                    value,
                    [q_size, 2 * q_size, 2 * q_size + kv_size],
                    axis=0,
                )
                sanitized[f"{prefix}gate_proj.weight"] = gate
                sanitized[f"{prefix}q_proj.weight"] = query
                sanitized[f"{prefix}k_proj.weight"] = key_weight
                sanitized[f"{prefix}v_proj.weight"] = value_weight
                continue
            if key.endswith(".mlp.gate_up_proj.weight"):
                prefix = key[: -len("gate_up_proj.weight")]
                gate, up = mx.split(value, 2, axis=0)
                sanitized[f"{prefix}gate_proj.weight"] = gate
                sanitized[f"{prefix}up_proj.weight"] = up
                continue
            if key.endswith(".self_attn.gqkv_proj.bias"):
                prefix = key[: -len("gqkv_proj.bias")]
                gate, query, key_bias, value_bias = mx.split(
                    value,
                    [q_size, 2 * q_size, 2 * q_size + kv_size],
                    axis=0,
                )
                sanitized[f"{prefix}gate_proj.bias"] = gate
                sanitized[f"{prefix}q_proj.bias"] = query
                sanitized[f"{prefix}k_proj.bias"] = key_bias
                sanitized[f"{prefix}v_proj.bias"] = value_bias
                continue
            if key.endswith(".mlp.gate_up_proj.bias"):
                prefix = key[: -len("gate_up_proj.bias")]
                gate, up = mx.split(value, 2, axis=0)
                sanitized[f"{prefix}gate_proj.bias"] = gate
                sanitized[f"{prefix}up_proj.bias"] = up
                continue
            sanitized[key] = value

        return sanitized

    @property
    def layers(self):
        return self.language_model.layers

    def make_cache(self):
        return self.language_model.make_cache()
