from typing import Optional

import mlx.core as mx
import mlx.nn as nn
import numpy as np

from ..base import InputEmbeddingsFeatures
from ..llava.vision import VisionModel as SigLipVisionModel
from ..qwen3_omni_moe.language import LanguageModel
from .config import ModelConfig

VisionModel = SigLipVisionModel


def masked_scatter(
    final_embedding: mx.array,
    image_mask_expanded: mx.array,
    scaled_image_features: mx.array,
):
    final_embedding_shape = final_embedding.shape
    scaled_image_features_flattened = mx.flatten(scaled_image_features)
    final_embedding_flattened = mx.flatten(final_embedding)
    image_mask_expanded_flattened = mx.flatten(image_mask_expanded)

    image_positions = mx.array(np.where(image_mask_expanded_flattened)[0], mx.uint32)
    final_embedding_flattened[image_positions] = scaled_image_features_flattened

    final_embedding = mx.reshape(final_embedding_flattened, final_embedding_shape)
    return final_embedding


class ResamplerAttention(nn.Module):
    def __init__(self, hidden_size: int):
        super().__init__()
        self.q_proj = nn.Linear(hidden_size, hidden_size, bias=True)
        self.k_proj = nn.Linear(hidden_size, hidden_size, bias=True)
        self.v_proj = nn.Linear(hidden_size, hidden_size, bias=True)
        self.out_proj = nn.Linear(hidden_size, hidden_size, bias=True)
        self.scale = (hidden_size // 8) ** -0.5

    def __call__(self, q: mx.array, kv: mx.array) -> mx.array:
        q = self.q_proj(q)
        k = self.k_proj(kv)
        v = self.v_proj(kv)

        attn = (q @ k.transpose(0, 2, 1)) * self.scale
        attn = mx.softmax(attn, axis=-1)
        out = attn @ v
        return self.out_proj(out)


class Resampler(nn.Module):
    def __init__(self, hidden_size: int, vision_hidden_size: int, query_num: int):
        super().__init__()
        self.query = mx.zeros((query_num, hidden_size))
        self.kv_proj = nn.Linear(vision_hidden_size, hidden_size, bias=False)
        self.ln_kv = nn.LayerNorm(hidden_size)
        self.ln_q = nn.LayerNorm(hidden_size)
        self.attn = ResamplerAttention(hidden_size)
        self.ln_post = nn.LayerNorm(hidden_size)
        self.proj = mx.eye(hidden_size)

    def __call__(self, vision_tokens: mx.array) -> mx.array:
        kv = self.ln_kv(self.kv_proj(vision_tokens))
        q = mx.broadcast_to(self.query[None, :, :], (kv.shape[0], self.query.shape[0], self.query.shape[1]))
        q = self.ln_q(q)
        out = self.attn(q, kv)
        out = self.ln_post(out)
        return out @ self.proj


class Model(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.model_type = config.model_type

        self.language_model = LanguageModel(config.text_config, config=config)
        self.vision_tower = SigLipVisionModel(config.vision_config)
        self.resampler = Resampler(
            hidden_size=config.text_config.hidden_size,
            vision_hidden_size=config.vision_config.hidden_size,
            query_num=config.query_num,
        )

    def get_image_features(self, pixel_values: mx.array) -> mx.array:
        if pixel_values.ndim == 4 and pixel_values.shape[1] == self.config.vision_config.num_channels:
            pixel_values = pixel_values.transpose(0, 2, 3, 1)
        _, vision_tokens, _ = self.vision_tower(pixel_values, output_hidden_states=False)
        return self.resampler(vision_tokens)

    def get_input_embeddings(
        self,
        input_ids: Optional[mx.array] = None,
        pixel_values: Optional[mx.array] = None,
        **kwargs,
    ):
        inputs_embeds = self.language_model.model.embed_tokens(input_ids)

        if pixel_values is None:
            return InputEmbeddingsFeatures(inputs_embeds=inputs_embeds)

        image_features = self.get_image_features(pixel_values).astype(inputs_embeds.dtype)

        image_mask = input_ids == self.config.image_token_id
        num_image_tokens = int(mx.sum(image_mask))

        flat_features = image_features.reshape(-1, image_features.shape[-1])
        if num_image_tokens <= 0:
            return InputEmbeddingsFeatures(inputs_embeds=inputs_embeds)

        if flat_features.shape[0] < num_image_tokens:
            repeats = int(np.ceil(num_image_tokens / flat_features.shape[0]))
            flat_features = mx.concatenate([flat_features] * repeats, axis=0)
        flat_features = flat_features[:num_image_tokens]

        image_mask = mx.expand_dims(image_mask, axis=-1)
        image_mask = mx.broadcast_to(image_mask, inputs_embeds.shape)

        if flat_features.ndim == 2:
            # Expand to [num_tokens, hidden] -> [1, num_tokens, hidden] to align masking helpers
            flat_features = flat_features.reshape(-1, inputs_embeds.shape[-1])

        inputs_embeds = masked_scatter(inputs_embeds, image_mask, flat_features)
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
        input_embedding_features = self.get_input_embeddings(
            input_ids=input_ids,
            pixel_values=pixel_values,
        )
        return self.language_model(
            input_ids=input_ids,
            inputs_embeds=input_embedding_features.inputs_embeds,
            mask=mask,
            cache=cache,
        )

    def sanitize(self, weights):
        sanitized = {}
        for key, value in weights.items():
            # Drop unsupported omni modules
            if key.startswith("audio_encoder.") or key.startswith("tts.") or key.startswith("audio_projection_layer."):
                continue

            # Vision key layout in checkpoint omits intermediate `vision_model`
            if key.startswith("vision_tower.") and not key.startswith("vision_tower.vision_model."):
                key = key.replace("vision_tower.", "vision_tower.vision_model.", 1)

            if "vision_tower.vision_model.encoder_layers." in key:
                key = key.replace(
                    "vision_tower.vision_model.encoder_layers.",
                    "vision_tower.vision_model.encoder.layers.",
                )

            # Conv2d layout fix for patch embedding
            if "vision_tower.vision_model.embeddings.patch_embedding.weight" in key:
                if value.ndim == 4 and value.shape[-1] != self.config.vision_config.num_channels:
                    value = value.transpose(0, 2, 3, 1)

            sanitized[key] = value

        return sanitized
