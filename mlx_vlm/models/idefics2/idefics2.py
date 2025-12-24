import re
from typing import Optional, Tuple

import mlx.core as mx
import mlx.nn as nn
import numpy as np

from .config import ModelConfig
from .language import LanguageModel
from .vision import VisionModel


def masked_scatter(
    final_embedding: mx.array,
    image_mask_expanded: mx.array,
    scaled_image_features: mx.array,
):
    # Reshape the tensors to 1D
    final_embedding_shape = final_embedding.shape
    scaled_image_features_flattened = mx.flatten(scaled_image_features)
    final_embedding_flattened = mx.flatten(final_embedding)
    image_mask_expanded_flattened = mx.flatten(image_mask_expanded)

    # Scatter the scaled image features into the special image token positions
    image_positions = mx.array(np.where(image_mask_expanded_flattened)[0], mx.uint32)
    final_embedding_flattened[image_positions] = scaled_image_features_flattened

    # Reshape back to the original shape
    final_embedding = mx.reshape(final_embedding_flattened, final_embedding_shape)

    return final_embedding


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
        **kwargs,
    ):
        pixel_attention_mask = kwargs.get("pixel_attention_mask", None)

        if pixel_values is None:
            return self.language_model.embed_tokens(input_ids)

        inputs_embeds = self.language_model.embed_tokens(input_ids)

        batch_size, num_images, num_channels, height, width = pixel_values.shape
        pixel_values = pixel_values.reshape(
            batch_size * num_images, num_channels, height, width
        )

        # Remove padding images - padding image are full 0.
        nb_values_per_image = np.prod(pixel_values.shape[1:])
        real_images_mask = (pixel_values == 0.0).sum(
            axis=(-1, -2, -3)
        ) != nb_values_per_image
        real_images_inds = np.where(real_images_mask)[0].tolist()
        pixel_values = pixel_values[real_images_inds, ...]

        if pixel_attention_mask is None:
            pixel_attention_mask = mx.ones(
                (pixel_values.size(0), pixel_values.size(2), pixel_values.size(3)),
                dtype=mx.bool,
            )
        else:
            # Remove padding images from the mask
            pixel_attention_mask = pixel_attention_mask.reshape(
                batch_size * num_images, height, width
            )
            pixel_attention_mask = pixel_attention_mask[real_images_inds]

        patch_size = self.config.vision_config.patch_size
        batch_size, height, width = pixel_attention_mask.shape

        # Calculate number of patches
        patches_h = height // patch_size
        patches_w = width // patch_size

        # Reshape to extract patches
        reshaped = pixel_attention_mask[
            :, : patches_h * patch_size, : patches_w * patch_size
        ]
        reshaped = reshaped.reshape(
            batch_size, patches_h, patch_size, patches_w, patch_size
        )
        reshaped = reshaped.transpose(
            0, 1, 3, 2, 4
        )  # (batch, patches_h, patches_w, patch_size, patch_size)

        # Sum over patch dimensions and check if any pixels are active
        patch_attention_mask = reshaped.sum(axis=(-1, -2)) > 0

        pooler_output, *_ = self.vision_model(
            pixel_values.transpose(0, 2, 3, 1),
            patch_attention_mask=patch_attention_mask,
            output_hidden_states=True,
        )

        image_features = pooler_output.astype(pixel_values.dtype)
        image_features = self.connector(image_features)

        final_inputs_embeds = self._prepare_inputs_for_multimodal(
            image_features, inputs_embeds, input_ids
        )
        return final_inputs_embeds

    def _prepare_inputs_for_multimodal(self, image_features, inputs_embeds, input_ids):
        special_image_mask = input_ids == self.config.image_token_index
        n_image_tokens = special_image_mask.sum()
        special_image_mask = special_image_mask[..., None]
        special_image_mask = mx.broadcast_to(special_image_mask, inputs_embeds.shape)

        n_image_features = image_features.shape[0]
        n_image_mask_elements = special_image_mask.sum()
        if n_image_mask_elements != image_features.size:
            raise ValueError(
                f"Image features and image tokens do not match: tokens: {n_image_tokens}, features {n_image_features}"
            )

        inputs_embeds = masked_scatter(
            inputs_embeds, special_image_mask, image_features
        )

        return inputs_embeds

    @property
    def layers(self):
        return self.language_model.model.layers

    def __call__(
        self,
        input_ids: mx.array,
        pixel_values: mx.array,
        mask: mx.array,
        cache=None,
        **kwargs,
    ):
        input_embeddings = self.get_input_embeddings(input_ids, pixel_values, **kwargs)
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
