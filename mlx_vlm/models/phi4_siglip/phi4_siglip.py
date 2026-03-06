import re
from typing import Optional

import mlx.core as mx
import mlx.nn as nn
import numpy as np

from ..base import InputEmbeddingsFeatures, LanguageModelOutput, create_attention_mask
from ..cache import KVCache
from .config import ModelConfig, TextConfig, VisionConfig
from .vision import VisionModel

# Import processor to register it with AutoProcessor
from . import processing_phi4_siglip  # noqa: F401

# Sentinel value used by the PyTorch model to mark image token positions
IMAGE_TOKEN_INDEX = -200


# =============================================================================
# Language Model Components (Phi3 architecture)
# =============================================================================


class Attention(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()

        dim = config.hidden_size
        self.n_heads = n_heads = config.num_attention_heads
        self.n_kv_heads = n_kv_heads = config.num_key_value_heads
        self.head_dim = head_dim = config.hidden_size // n_heads
        self.scale = head_dim**-0.5

        op_size = n_heads * head_dim + 2 * (n_kv_heads * head_dim)
        self.qkv_proj = nn.Linear(dim, op_size, bias=False)
        self.o_proj = nn.Linear(n_heads * head_dim, dim, bias=False)

        rope_dim = int(head_dim * config.partial_rotary_factor)

        rope_scale = 1.0
        if config.rope_scaling:
            rope_type = config.rope_scaling.get("type", None)
            if rope_type == "linear":
                rope_scale = 1 / config.rope_scaling["factor"]

        self.rope = nn.RoPE(
            rope_dim,
            traditional=config.rope_traditional,
            base=config.rope_theta,
            scale=rope_scale,
        )

    def __call__(
        self,
        x: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional[KVCache] = None,
    ) -> mx.array:
        B, L, _ = x.shape

        qkv = self.qkv_proj(x)
        query_pos = self.n_heads * self.head_dim
        queries, keys, values = mx.split(
            qkv, [query_pos, query_pos + self.n_kv_heads * self.head_dim], axis=-1
        )

        queries = queries.reshape(B, L, self.n_heads, -1).transpose(0, 2, 1, 3)
        keys = keys.reshape(B, L, self.n_kv_heads, -1).transpose(0, 2, 1, 3)
        values = values.reshape(B, L, self.n_kv_heads, -1).transpose(0, 2, 1, 3)

        if cache is not None:
            queries = self.rope(queries, offset=cache.offset)
            keys = self.rope(keys, offset=cache.offset)
            keys, values = cache.update_and_fetch(keys, values)
        else:
            queries = self.rope(queries)
            keys = self.rope(keys)

        output = mx.fast.scaled_dot_product_attention(
            queries, keys, values, scale=self.scale, mask=mask
        )
        output = output.transpose(0, 2, 1, 3).reshape(B, L, -1)
        return self.o_proj(output)


class MLP(nn.Module):
    def __init__(self, dim, hidden_dim):
        super().__init__()
        self.gate_up_proj = nn.Linear(dim, 2 * hidden_dim, bias=False)
        self.down_proj = nn.Linear(hidden_dim, dim, bias=False)

    def __call__(self, x) -> mx.array:
        x = self.gate_up_proj(x)
        gate, x = mx.split(x, 2, axis=-1)
        return self.down_proj(nn.silu(gate) * x)


class TransformerBlock(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.self_attn = Attention(config)
        self.mlp = MLP(config.hidden_size, config.intermediate_size)
        self.input_layernorm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = nn.RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )

    def __call__(
        self,
        x: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional[KVCache] = None,
    ) -> mx.array:
        r = self.self_attn(self.input_layernorm(x), mask, cache)
        h = x + r
        r = self.mlp(self.post_attention_layernorm(h))
        out = h + r
        return out


# =============================================================================
# Multimodal Projector
# =============================================================================


class MultiModalProjector(nn.Module):
    """MLP 2x GELU projector: Linear → GELU → Linear"""

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.linear_1 = nn.Linear(
            config.mm_hidden_size, config.hidden_size, bias=True
        )
        self.gelu = nn.GELU()
        self.linear_2 = nn.Linear(
            config.hidden_size, config.hidden_size, bias=True
        )

    def __call__(self, x: mx.array) -> mx.array:
        x = self.linear_1(x)
        x = self.gelu(x)
        x = self.linear_2(x)
        return x


# =============================================================================
# Vision Tower Wrapper (matches PyTorch weight path nesting)
# =============================================================================


class VisionTower(nn.Module):
    """Wrapper matching model.vision_tower.vision_tower.* weight paths."""

    def __init__(self, config: VisionConfig):
        super().__init__()
        self.vision_tower = VisionModel(config)

    def __call__(self, *args, **kwargs):
        return self.vision_tower(*args, **kwargs)


# =============================================================================
# Backbone Model
# =============================================================================


class Phi4SigLipModel(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.vocab_size = config.vocab_size
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.vision_tower = VisionTower(config.vision_config)
        self.mm_projector = MultiModalProjector(config)
        self.layers = [
            TransformerBlock(config=config) for _ in range(config.num_hidden_layers)
        ]
        self.norm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def __call__(
        self,
        inputs: mx.array,
        inputs_embeds: Optional[mx.array] = None,
        mask: Optional[mx.array] = None,
        cache=None,
    ):
        if inputs_embeds is None:
            h = self.embed_tokens(inputs)
        else:
            h = inputs_embeds

        if cache is None:
            cache = [None] * len(self.layers)

        if mask is None:
            mask = create_attention_mask(h, cache[0])

        for layer, c in zip(self.layers, cache):
            h = layer(h, mask, c)

        return self.norm(h)


# =============================================================================
# Main Model
# =============================================================================


class Model(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.model_type = config.model_type
        self.config = config
        self.model = Phi4SigLipModel(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

    def __call__(
        self,
        inputs: mx.array,
        inputs_embeds: Optional[mx.array] = None,
        pixel_values=None,
        mask=None,
        cache=None,
        **kwargs,
    ):
        if inputs_embeds is None:
            input_embeddings_features = self.get_input_embeddings(
                inputs, pixel_values, **kwargs
            )
            inputs_embeds = input_embeddings_features.inputs_embeds

        out = self.model(inputs, inputs_embeds, mask=mask, cache=cache)
        logits = self.lm_head(out)
        return LanguageModelOutput(logits=logits)

    def get_input_embeddings(
        self,
        inputs: mx.array,
        pixel_values: Optional[mx.array] = None,
        **kwargs,
    ):
        spatial_shapes = kwargs.get("spatial_shapes", None)
        pixel_attention_mask = kwargs.get("pixel_attention_mask", None)

        inputs_embeds = self.model.embed_tokens(inputs)

        if pixel_values is None:
            return InputEmbeddingsFeatures(inputs_embeds=inputs_embeds)

        # Select vision feature layer (second to last hidden state)
        select_layer = -2
        encoder_outputs, _, _ = self.model.vision_tower(
            pixel_values, output_hidden_states=True, spatial_shapes=spatial_shapes
        )
        # encoder_outputs is a tuple of hidden states
        hidden_states = encoder_outputs[select_layer]

        # Remove padding tokens using attention mask (NaFlex) and project
        image_features_list = []
        if pixel_attention_mask is not None:
            for img_idx in range(hidden_states.shape[0]):
                valid_len = int(pixel_attention_mask[img_idx].sum().item())
                feature = hidden_states[img_idx, :valid_len, :]
                projected = self.model.mm_projector(feature)
                image_features_list.append(projected)
        else:
            for img_idx in range(hidden_states.shape[0]):
                feature = hidden_states[img_idx]
                projected = self.model.mm_projector(feature)
                image_features_list.append(projected)

        # Merge: replace each IMAGE_TOKEN_INDEX with variable-length image features
        final_inputs_embeds = self._prepare_inputs_for_multimodal(
            image_features_list, inputs_embeds, inputs
        )
        return InputEmbeddingsFeatures(inputs_embeds=final_inputs_embeds)

    @staticmethod
    def _prepare_inputs_for_multimodal(
        image_features_list, inputs_embeds, input_ids
    ):
        """
        Replace IMAGE_TOKEN_INDEX positions with image features.

        Each IMAGE_TOKEN_INDEX in input_ids maps to one entry in
        image_features_list which may have many tokens (NaFlex variable length).
        The output sequence will be longer than the input.
        """
        # Process each batch item
        batch_size = input_ids.shape[0]
        new_embeds_list = []
        cur_image_idx = 0

        for b in range(batch_size):
            cur_input_ids = input_ids[b]
            cur_embeds = inputs_embeds[b]

            # Find image token positions
            image_positions = np.where(np.array(cur_input_ids) == IMAGE_TOKEN_INDEX)[0]
            num_images = len(image_positions)

            if num_images == 0:
                new_embeds_list.append(cur_embeds)
                continue

            # Split embeddings around image token positions
            segments = []
            prev_pos = 0
            for i, pos in enumerate(image_positions):
                pos = int(pos)
                # Text segment before this image token
                if pos > prev_pos:
                    segments.append(cur_embeds[prev_pos:pos])
                # Image features for this image
                segments.append(image_features_list[cur_image_idx])
                cur_image_idx += 1
                prev_pos = pos + 1  # skip the image token itself

            # Remaining text after the last image token
            seq_len = int(cur_input_ids.shape[0])
            if prev_pos < seq_len:
                segments.append(cur_embeds[prev_pos:])

            new_embeds_list.append(mx.concatenate(segments, axis=0))

        # Stack batch (for batch_size=1, just add batch dim)
        if batch_size == 1:
            return new_embeds_list[0][None, :]
        else:
            # Pad to max length
            max_len = max(e.shape[0] for e in new_embeds_list)
            embed_dim = new_embeds_list[0].shape[-1]
            padded = mx.zeros((batch_size, max_len, embed_dim))
            for i, emb in enumerate(new_embeds_list):
                padded[i, : emb.shape[0]] = emb
            return padded

    @property
    def layers(self):
        return self.model.layers

    @property
    def head_dim(self):
        return self.config.hidden_size // self.config.num_attention_heads

    @property
    def n_kv_heads(self):
        return self.config.num_key_value_heads

    @property
    def language_model(self):
        return self

    @property
    def vision_model(self):
        return self.model.vision_tower

    def sanitize(self, weights):
        sanitized = {}
        for k, v in weights.items():
            if "position_ids" in k:
                continue

            # Skip vision pooling head weights (not used, model uses select_layer=-2)
            if "vision_model.head." in k:
                continue

            new_key = k

            # Remap projector keys from Sequential indices to named attributes
            # model.mm_projector.0.* -> model.mm_projector.linear_1.*
            new_key = re.sub(
                r"mm_projector\.0\.", "mm_projector.linear_1.", new_key
            )
            # model.mm_projector.2.* -> model.mm_projector.linear_2.*
            new_key = re.sub(
                r"mm_projector\.2\.", "mm_projector.linear_2.", new_key
            )

            sanitized[new_key] = v

        return sanitized
