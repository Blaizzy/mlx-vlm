from typing import Optional

import mlx.core as mx
import mlx.nn as nn

from ..attention import VisionAttention as Attention
from ..mlp import GELUMLP as MLP
from .config import VisionConfig


def check_array_shape(arr):
    shape = arr.shape

    # Check if the shape has 4 dimensions
    if len(shape) != 4:
        return False

    out_channels, kH, KW, _ = shape

    # Check if out_channels is the largest, and kH and KW are the same
    if (out_channels >= kH) and (out_channels >= KW) and (kH == KW):
        return True
    else:
        return False


class EncoderLayer(nn.Module):
    def __init__(self, config: VisionConfig):
        super().__init__()
        self.embed_dim = config.hidden_size
        self.self_attn = Attention(
            config.hidden_size, config.num_attention_heads, bias=True
        )
        self.layer_norm1 = nn.LayerNorm(self.embed_dim, eps=config.layer_norm_eps)
        self.mlp = MLP(config)
        self.layer_norm2 = nn.LayerNorm(self.embed_dim, eps=config.layer_norm_eps)

    def __call__(self, x: mx.array, mask: Optional[mx.array] = None) -> mx.array:
        r = self.self_attn(self.layer_norm1(x), mask)
        h = x + r
        r = self.mlp(self.layer_norm2(h))
        return h + r


class Encoder(nn.Module):
    def __init__(self, config: VisionConfig):
        super().__init__()
        self.layers = [EncoderLayer(config) for _ in range(config.num_hidden_layers)]

    def __call__(
        self,
        x: mx.array,
        output_hidden_states: Optional[bool] = None,
        mask: Optional[mx.array] = None,
    ) -> mx.array:
        encoder_states = (x,) if output_hidden_states else None
        h = x
        for l in self.layers:
            x = l(x, mask=mask)
            if output_hidden_states:
                encoder_states = encoder_states + (x,)

            h = x

        return (h, encoder_states)


class VisionEmbeddings(nn.Module):
    def __init__(self, config: VisionConfig):
        super().__init__()
        self.config = config
        self.embed_dim = config.hidden_size
        self.image_size = config.image_size
        self.patch_size = config.patch_size

        self.patch_embedding = nn.Conv2d(
            in_channels=config.num_channels,
            out_channels=self.embed_dim,
            kernel_size=self.patch_size,
            stride=self.patch_size,
        )

        self.num_patches_per_side = self.image_size // self.patch_size
        self.num_patches = self.num_patches_per_side**2
        self.num_positions = self.num_patches
        self.position_embedding = nn.Embedding(self.num_positions, self.embed_dim)

    def __call__(self, x: mx.array, patch_attention_mask: mx.array = None) -> mx.array:
        batch_size, max_im_h, max_im_w, _ = x.shape
        patch_embeds = self.patch_embedding(x)
        embeddings = mx.flatten(patch_embeds, start_axis=1, end_axis=2)

        seq_len = embeddings.shape[1]

        if patch_attention_mask is None:
            position_ids = mx.tile(mx.arange(seq_len), (batch_size, 1))
        else:
            boundaries = mx.arange(
                1 / self.num_patches_per_side, 1.0, 1 / self.num_patches_per_side
            )

            # Flatten mask to match sequence length (handles both (B,H,W) and (B,H,W,1))
            if patch_attention_mask.ndim == 4:
                flat_mask = patch_attention_mask.squeeze(-1).reshape(batch_size, -1)[
                    :, :seq_len
                ]
            else:
                flat_mask = patch_attention_mask.reshape(batch_size, -1)[:, :seq_len]

            # Compute valid patches per image (channels-last indexing)
            nb_patches_h = mx.maximum(patch_attention_mask[:, :, 0].sum(axis=1), 1)
            nb_patches_w = mx.maximum(patch_attention_mask[:, 0, :].sum(axis=1), 1)

            position_ids = mx.zeros((batch_size, seq_len), dtype=mx.int32)

            for batch_idx in range(batch_size):
                nb_h = int(nb_patches_h[batch_idx])
                nb_w = int(nb_patches_w[batch_idx])

                # Compute fractional coordinates
                fractional_h = mx.arange(nb_h, dtype=mx.float32) / nb_h
                fractional_w = mx.arange(nb_w, dtype=mx.float32) / nb_w
                fractional_h = mx.clip(fractional_h, a_min=0.0, a_max=1.0 - 1e-6)
                fractional_w = mx.clip(fractional_w, a_min=0.0, a_max=1.0 - 1e-6)

                # Bucket into position IDs
                bucket_h = mx.sum(fractional_h[:, None] >= boundaries[None, :], axis=1)
                bucket_w = mx.sum(fractional_w[:, None] >= boundaries[None, :], axis=1)

                # Create 2D grid: iterate over height, then width (row-major)
                pos_ids = (
                    bucket_h[:, None] * self.num_patches_per_side + bucket_w[None, :]
                ).reshape(-1)

                valid_len = min(pos_ids.shape[0], seq_len)
                position_ids[batch_idx, :valid_len] = pos_ids[:valid_len]

            # Zero out position embeddings for padding
            mask_expanded = flat_mask[:, :, None]  # (batch, seq_len, 1)

        pos_embeddings = self.position_embedding(position_ids)

        # Apply mask to zero out padding position embeddings
        if patch_attention_mask is not None:
            pos_embeddings = pos_embeddings * mask_expanded

        embeddings = embeddings + pos_embeddings
        return embeddings


class VisionModel(nn.Module):
    def __init__(self, config: VisionConfig):
        super().__init__()
        self.model_type = config.model_type
        if self.model_type not in [
            "siglip_vision_model",
            "idefics3",
            "idefics3_vision",
            "smolvlm_vision",
        ]:
            raise ValueError(f"Unsupported model type: {self.model_type}")

        self.embeddings = VisionEmbeddings(config)
        self.encoder = Encoder(config)
        self.post_layernorm = nn.LayerNorm(config.hidden_size)

    def __call__(
        self,
        x: mx.array,
        patch_attention_mask: Optional[mx.array] = None,
        output_hidden_states: Optional[bool] = None,
    ) -> mx.array:
        x = self.embeddings(x, patch_attention_mask)
        x = x.astype(self.embeddings.patch_embedding.weight.dtype)
        encoder_outputs = self.encoder(
            x=x, output_hidden_states=output_hidden_states, mask=None
        )
        pooler_output = self.post_layernorm(encoder_outputs[0])
        return pooler_output, x, encoder_outputs[-1]

    def sanitize(self, weights):
        sanitized_weights = {}
        for k, v in weights.items():
            if "position_ids" in k:
                # Remove unused position_ids
                continue
            elif "patch_embedding.weight" in k:
                # PyTorch conv2d weight tensors have shape:
                #   [out_channels, in_channels, kH, KW]
                # MLX conv2d expects the weight be of shape:
                #   [out_channels, kH, KW, in_channels]
                if check_array_shape(v):
                    sanitized_weights[k] = v
                else:
                    sanitized_weights[k] = v.transpose(0, 2, 3, 1)
            else:
                sanitized_weights[k] = v

        return sanitized_weights
