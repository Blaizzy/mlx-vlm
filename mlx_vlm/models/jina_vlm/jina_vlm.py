"""Main Jina VLM model for MLX."""

from typing import Optional

import mlx.core as mx
import mlx.nn as nn
from transformers import AutoProcessor

from .config import ModelConfig, VisionConfig
from .language import LanguageModel
from .processing_jinavlm import JinaVLMProcessor
from .vision import VisionModel

AutoProcessor.register("jvlm", JinaVLMProcessor)


class CrossAttention(nn.Module):
    """Cross-attention for pooling - matches weight naming: pooling.q, pooling.kv, pooling.out"""

    def __init__(self, config: VisionConfig):
        super().__init__()
        input_size = config.hidden_size * len(config.vit_layers)
        n_heads = config.num_attention_heads
        head_dim = config.head_dim

        self.num_heads = n_heads
        self.head_dim = head_dim
        self.scale = head_dim**-0.5

        # Named to match weights: pooling.q, pooling.kv, pooling.out
        self.q = nn.Linear(input_size, n_heads * head_dim, bias=True)
        self.kv = nn.Linear(input_size, 2 * n_heads * head_dim, bias=True)
        self.out = nn.Linear(n_heads * head_dim, config.hidden_size, bias=True)

    def __call__(self, query: mx.array, key_value: mx.array) -> mx.array:
        B, Lq, _ = query.shape
        _, Lkv, _ = key_value.shape

        q = self.q(query)
        kv = self.kv(key_value)

        q = q.reshape(B, Lq, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)

        # Split KV
        kv = kv.reshape(B, Lkv, 2, self.num_heads, self.head_dim)
        kv = kv.transpose(2, 0, 3, 1, 4)  # (2, B, n_heads, Lkv, head_dim)
        k, v = kv[0], kv[1]

        attn = (q @ k.transpose(0, 1, 3, 2)) * self.scale
        attn = mx.softmax(attn, axis=-1)
        x = attn @ v

        x = x.transpose(0, 2, 1, 3).reshape(B, Lq, -1)
        x = self.out(x)
        return x


class ConnectorMLP(nn.Module):
    """MLP projector with SwiGLU - matches weight naming: projector.gate_up, projector.down"""

    def __init__(self, config: VisionConfig):
        super().__init__()
        input_size = config.hidden_size
        hidden_size = config.connector_hidden_size
        output_size = config.output_size

        # Named to match weights: projector.gate_up, projector.down
        self.gate_up = nn.Linear(input_size, 2 * hidden_size, bias=False)
        self.down = nn.Linear(hidden_size, output_size, bias=False)

    def __call__(self, x: mx.array) -> mx.array:
        gate_up = self.gate_up(x)
        # Jina VLM convention: first half is value, second half is gate (activated)
        up, gate = mx.split(gate_up, 2, axis=-1)
        return self.down(nn.silu(gate) * up)


class VisionLanguageConnector(nn.Module):
    """Vision-Language Connector - matches weight naming: vl_connector.pooling, vl_connector.projector"""

    def __init__(self, config: VisionConfig):
        super().__init__()
        self.config = config

        self.pooling_h = config.pooling_h
        self.pooling_w = config.pooling_w

        self.crop_patches = config.image_size // config.patch_size
        self.token_length_h = (
            self.crop_patches + config.pooling_h - 1
        ) // config.pooling_h
        self.token_length_w = (
            self.crop_patches + config.pooling_w - 1
        ) // config.pooling_w
        self.tokens_per_image = self.token_length_h * self.token_length_w

        input_size = config.hidden_size * len(config.vit_layers)
        # Named to match weights: vl_connector.pad_embed
        self.pad_embed = mx.zeros((2, input_size))

        # Named to match weights: vl_connector.pooling
        self.pooling = CrossAttention(config)

        # Named to match weights: vl_connector.projector
        self.projector = ConnectorMLP(config)

    def __call__(
        self, image_features: mx.array, image_masks: Optional[mx.array] = None
    ) -> mx.array:
        B, n_crops = image_features.shape[:2]
        n_patch_h = n_patch_w = self.crop_patches

        if image_masks is not None:
            all_pad = (image_masks == 0).astype(mx.float32)
            partial_pad = mx.logical_and(
                image_masks < 1, mx.logical_not(image_masks == 0)
            ).astype(mx.float32)

            pad_embed_0 = self.pad_embed[0][None, None, None, :]
            pad_embed_1 = self.pad_embed[1][None, None, None, :]

            image_features = image_features + pad_embed_0 * all_pad[..., None]
            image_features = image_features + pad_embed_1 * partial_pad[..., None]

        image_features = image_features.reshape(B, n_crops, n_patch_h, n_patch_w, -1)

        pad_h = n_patch_h % self.pooling_h
        pad_w = n_patch_w % self.pooling_w
        if pad_h != 0 or pad_w != 0:
            pad_h = self.pooling_h - pad_h if pad_h != 0 else 0
            pad_w = self.pooling_w - pad_w if pad_w != 0 else 0
            image_features = mx.pad(
                image_features, [(0, 0), (0, 0), (0, pad_h), (0, pad_w), (0, 0)]
            )

        _, _, H, W, C = image_features.shape
        new_h, new_w = H // self.pooling_h, W // self.pooling_w

        image_features = image_features.reshape(
            B, n_crops, new_h, self.pooling_h, new_w, self.pooling_w, C
        )
        image_features = image_features.transpose(0, 1, 2, 4, 3, 5, 6)
        image_features = image_features.reshape(
            B * n_crops * new_h * new_w, self.pooling_h * self.pooling_w, C
        )

        query = image_features.mean(axis=1, keepdims=True)
        pooled = self.pooling(query, image_features)

        pooled = pooled.reshape(B, n_crops, new_h * new_w, -1)
        output = self.projector(pooled)

        return output


class Model(nn.Module):
    """Jina Vision-Language Model - matches weight naming structure"""

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config

        # Named to match weights: vision_model
        self.vision_model = VisionModel(config.vision_config)

        # Named to match weights: vl_connector
        self.vl_connector = VisionLanguageConnector(config.vision_config)

        # Named to match weights: language_model
        self.language_model = LanguageModel(config.text_config)

        # lm_head is now inside language_model (weights will be mapped in sanitize)
        self.language_model.lm_head = nn.Linear(
            config.text_config.hidden_size, config.text_config.vocab_size, bias=False
        )

    @property
    def layers(self):
        return self.language_model.layers

    def get_image_features(
        self,
        images: mx.array,
        image_masks: Optional[mx.array] = None,
    ) -> mx.array:
        B, n_crops, n_patches, patch_dim = images.shape
        dtype = self.vision_model.patch_embed.proj.weight.dtype

        images_flat = images.reshape(B * n_crops, n_patches, patch_dim).astype(dtype)
        valid_mask = ~mx.all(
            images_flat.reshape(B * n_crops, -1) == -1, axis=-1, keepdims=True
        )
        valid_mask = valid_mask[:, :, None]

        image_features = self.vision_model.get_features(images_flat)
        image_features = image_features * valid_mask

        n_output_patches = image_features.shape[1]
        image_features = image_features.reshape(B, n_crops, n_output_patches, -1)
        image_features = self.vl_connector(image_features, image_masks)

        return image_features

    def __call__(
        self,
        input_ids: mx.array,
        pixel_values: Optional[mx.array] = None,
        mask: Optional[mx.array] = None,
        cache=None,
        **kwargs,
    ) -> mx.array:
        if input_ids.ndim == 1:
            input_ids = input_ids[None, :]

        batch_size, seq_len = input_ids.shape

        image_masks = kwargs.get("image_masks", None)
        image_input_idx = kwargs.get("image_input_idx", None)

        input_embeddings = self.language_model.embedding(input_ids)

        if pixel_values is not None and image_input_idx is not None:
            if pixel_values.ndim == 3:
                pixel_values = mx.expand_dims(pixel_values, 0)
                image_masks = (
                    mx.expand_dims(image_masks, 0) if image_masks is not None else None
                )
                image_input_idx = (
                    mx.expand_dims(image_input_idx, 0)
                    if image_input_idx is not None
                    else None
                )

            image_features = self.get_image_features(pixel_values, image_masks)

            num_image, num_patch = image_features.shape[1:3]

            image_features = image_features.reshape(
                batch_size, num_image * num_patch, -1
            )
            image_input_idx = image_input_idx.reshape(batch_size, num_image * num_patch)

            for b in range(batch_size):
                idx = image_input_idx[b]
                features = image_features[b]

                for i in range(idx.shape[0]):
                    pos = int(idx[i].item())
                    if pos >= 0 and pos < seq_len:
                        input_embeddings = input_embeddings.at[b, pos].add(features[i])

        return self.language_model(
            input_ids,
            inputs_embeds=input_embeddings,
            mask=mask,
            cache=cache,
        )

    def sanitize(self, weights):
        """Sanitize weight names for loading."""
        new_weights = {}
        for k, v in weights.items():
            # Map lm_head to language_model.lm_head since language_model now has lm_head
            if k.startswith("lm_head."):
                new_k = "language_model." + k
            else:
                new_k = k
            new_weights[new_k] = v

        return new_weights
