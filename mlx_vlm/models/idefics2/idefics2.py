import glob
import inspect
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import mlx.core as mx
import mlx.nn as nn
import numpy as np
from huggingface_hub import snapshot_download
from transformers import AutoConfig

from .language import LanguageModel, TextConfig
from .vision import VisionConfig, VisionModel


@dataclass
class PerceiverConfig:
    model_type: str
    num_key_value_heads: int = 4
    resampler_depth: int = 3
    resampler_head_dim: int = 96
    resampler_n_heads: int = 16
    resampler_n_latents: int = 64

    @classmethod
    def from_dict(cls, params):
        return cls(
            **{
                k: v
                for k, v in params.items()
                if k in inspect.signature(cls).parameters
            }
        )


@dataclass
class ModelConfig:
    text_config: TextConfig
    vision_config: VisionConfig
    perceiver_config: PerceiverConfig
    model_type: str
    ignore_index: int = -100
    image_token_id: int = 32001
    vocab_size: int = 151936
    image_token_index: Optional[int] = None
    eos_token_id: Optional[List[int]] = None

    @classmethod
    def from_dict(cls, params):
        return cls(
            **{
                k: v
                for k, v in params.items()
                if k in inspect.signature(cls).parameters
            }
        )

    def __post_init__(self):
        if self.image_token_index is None:
            self.image_token_index = self.image_token_id


class Idefics2PerceiverAttention(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()

        dim = config.text_config.hidden_size
        self.n_heads = n_heads = config.perceiver_config.resampler_n_heads
        self.n_kv_heads = n_kv_heads = config.perceiver_config.num_key_value_heads

        head_dim = config.perceiver_config.resampler_head_dim
        self.scale = head_dim**-0.5

        self.q_proj = nn.Linear(dim, n_heads * head_dim, bias=False)
        self.k_proj = nn.Linear(dim, n_kv_heads * head_dim, bias=False)
        self.v_proj = nn.Linear(dim, n_kv_heads * head_dim, bias=False)
        self.o_proj = nn.Linear(n_heads * head_dim, dim, bias=False)

    def __call__(
        self,
        x: mx.array,
        kv: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional[Tuple[mx.array, mx.array]] = None,
    ) -> mx.array:
        B, L, D = x.shape
        kv_seq_len = L + kv.shape[1]
        hidden_states = mx.concatenate([kv, x], axis=-2)

        queries = self.q_proj(x)
        keys = self.k_proj(hidden_states)
        values = self.v_proj(hidden_states)

        # Prepare the queries, keys and values for the attention computation
        queries = queries.reshape(B, L, self.n_heads, -1).transpose(0, 2, 1, 3)
        keys = keys.reshape(B, kv_seq_len, self.n_kv_heads, -1).transpose(0, 2, 1, 3)
        values = values.reshape(B, kv_seq_len, self.n_kv_heads, -1).transpose(
            0, 2, 1, 3
        )

        if cache is not None:
            key_cache, value_cache = cache
            keys = mx.concatenate([key_cache, keys], axis=2)
            values = mx.concatenate([value_cache, values], axis=2)

        output = mx.fast.scaled_dot_product_attention(
            queries, keys, values, scale=self.scale
        )
        output = output.transpose(0, 2, 1, 3).reshape(B, L, -1)
        return self.o_proj(output)


class Idefics2PerceiverLayer(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.hidden_size = config.text_config.hidden_size
        self.n_latents = config.perceiver_config.resampler_n_latents
        self.depth = config.perceiver_config.resampler_depth
        self.rms_norm_eps = config.text_config.rms_norm_eps

        self.input_latents_norm = nn.RMSNorm(self.hidden_size, eps=self.rms_norm_eps)
        self.input_context_norm = nn.RMSNorm(self.hidden_size, eps=self.rms_norm_eps)
        self.self_attn = Idefics2PerceiverAttention(config)
        self.post_attention_layernorm = nn.RMSNorm(
            self.hidden_size, eps=self.rms_norm_eps
        )
        self.mlp = MLP(self.hidden_size, self.hidden_size * 4, self.hidden_size)

    def __call__(
        self,
        x: mx.array,
        hidden_states: mx.array,
        mask: Optional[mx.array] = None,
    ) -> mx.array:
        latents = self.input_latents_norm(x)
        context = self.input_context_norm(hidden_states)

        latents = self.self_attn(latents, context, mask=mask)

        latents = x + latents
        r = latents

        latents = self.post_attention_layernorm(latents)
        latents = self.mlp(latents)
        latents = r + latents
        return latents


class Idefics2PerceiverResampler(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.hidden_size = config.text_config.hidden_size
        self.n_latents = config.perceiver_config.resampler_n_latents

        self.latents = mx.ones((self.n_latents, self.hidden_size))
        self.layers = [
            Idefics2PerceiverLayer(config)
            for _ in range(config.perceiver_config.resampler_depth)
        ]
        self.norm = nn.RMSNorm(self.hidden_size, eps=config.text_config.rms_norm_eps)

    def __call__(self, x: mx.array, mask: Optional[mx.array] = None):

        h = mx.expand_dims(self.latents, axis=0)
        h = mx.repeat(h, x.shape[0], axis=0)

        for layer in self.layers:
            h = layer(h, x, mask=mask)

        return self.norm(h)


class MLP(nn.Module):
    def __init__(self, dim, hidden_dim, output_size):
        super().__init__()
        self.gate_proj = nn.Linear(dim, hidden_dim, bias=False)
        self.down_proj = nn.Linear(hidden_dim, output_size, bias=False)
        self.up_proj = nn.Linear(dim, hidden_dim, bias=False)

    def __call__(self, x) -> mx.array:
        return self.down_proj(nn.silu(self.gate_proj(x)) * self.up_proj(x))


class Idefics2Connector(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.modality_projection = MLP(
            config.vision_config.hidden_size,
            config.text_config.intermediate_size,
            config.text_config.hidden_size,
        )

        self.perceiver_resampler = Idefics2PerceiverResampler(config)

    def __call__(self, x: mx.array, mask=None) -> mx.array:
        x = self.modality_projection(x)
        x = self.perceiver_resampler(x, mask=mask)
        return x


class Model(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.model_type = config.model_type
        self.config = config

        self.vision_model = VisionModel(config.vision_config)
        self.language_model = LanguageModel(config.text_config)
        self.connector = Idefics2Connector(config)

    def get_input_embeddings(
        self,
        input_ids: Optional[mx.array] = None,
        pixel_values: Optional[mx.array] = None,
        pixel_attention_mask: Optional[mx.array] = None,
    ):
        if pixel_values is None:
            return self.language_model.embed_tokens(input_ids)

        inputs_embeds = self.language_model.embed_tokens(input_ids)

        pooler_output, embeddings, hidden_state = self.vision_model(
            pixel_values[0].transpose(0, 2, 3, 1), output_hidden_states=True
        )
        image_features = pooler_output.astype(pixel_values.dtype)
        image_features = self.connector(image_features, mask=None)

        final_inputs_embeds = self._prepare_inputs_for_multimodal(
            image_features, inputs_embeds, input_ids
        )
        return final_inputs_embeds

    def _prepare_inputs_for_multimodal(self, image_features, inputs_embeds, input_ids):
        image_token_index = self.config.image_token_index

        # Positions of <image> tokens in input_ids, assuming batch size is 1
        image_positions = np.where(input_ids == image_token_index)[1].tolist()
        num_images, _, vision_hidden_size = image_features.shape

        reshaped_image_hidden_states = image_features.reshape(-1, vision_hidden_size)

        # cast to the dtype of the input_embeds to support quantized models
        reshaped_image_hidden_states = reshaped_image_hidden_states.astype(
            inputs_embeds.dtype
        )

        inputs_embeds[:, image_positions, :] = reshaped_image_hidden_states

        return inputs_embeds

    def __call__(
        self,
        input_ids: mx.array,
        pixel_values: mx.array,
        mask: mx.array,
        cache=None,
        **kwargs,
    ):
        input_embeddings = self.get_input_embeddings(input_ids, pixel_values)
        logits = self.language_model(
            inputs=input_ids, cache=cache, inputs_embeds=input_embeddings
        )
        return logits

    def sanitize(self, weights):
        weights = {
            (
                f"{k.split('.', 1)[1]}"
                if re.match(r"^model\.", k)
                else (f"language_model.{k}" if re.match(r"^lm_head\.", k) else k)
            ): v
            for k, v in weights.items()
        }

        weights = {
            (
                f"language_model.{k.split('.', 1)[1]}"
                if re.match(
                    r"^text_model\.",
                    k,
                )
                else k
            ): v
            for k, v in weights.items()
        }

        return weights
