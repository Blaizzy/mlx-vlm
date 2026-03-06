from typing import Optional

import mlx.core as mx
import mlx.nn as nn
import numpy as np

from ..interpolate import resize_bilinear
from .config import VisionConfig


class Attention(nn.Module):
    def __init__(
        self,
        dims: int,
        num_heads: int,
        bias: bool = True,
    ):
        super().__init__()

        if (dims % num_heads) != 0:
            raise ValueError(
                "The input feature dimensions should be divisible by the "
                f"number of heads ({dims} % {num_heads}) != 0"
            )

        self.num_heads = num_heads
        head_dim = dims // num_heads
        self.scale = head_dim**-0.5

        self.q_proj = nn.Linear(dims, dims, bias=bias)
        self.k_proj = nn.Linear(dims, dims, bias=bias)
        self.v_proj = nn.Linear(dims, dims, bias=bias)
        self.out_proj = nn.Linear(dims, dims, bias=bias)

    def __call__(self, x, mask=None):
        queries = self.q_proj(x)
        keys = self.k_proj(x)
        values = self.v_proj(x)

        num_heads = self.num_heads
        B, L, D = queries.shape
        _, S, _ = keys.shape
        queries = queries.reshape(B, L, num_heads, -1).transpose(0, 2, 1, 3)
        keys = keys.reshape(B, S, num_heads, -1).transpose(0, 2, 1, 3)
        values = values.reshape(B, S, num_heads, -1).transpose(0, 2, 1, 3)

        output = mx.fast.scaled_dot_product_attention(
            queries, keys, values, scale=self.scale, mask=mask
        )
        output = output.transpose(0, 2, 1, 3).reshape(B, L, -1)
        return self.out_proj(output)


class MLP(nn.Module):
    def __init__(self, config: VisionConfig):
        super().__init__()
        self.activation_fn = nn.GELU(approx="precise")
        self.fc1 = nn.Linear(config.hidden_size, config.intermediate_size, bias=True)
        self.fc2 = nn.Linear(config.intermediate_size, config.hidden_size, bias=True)

    def __call__(self, x: mx.array) -> mx.array:
        x = self.fc1(x)
        x = self.activation_fn(x)
        x = self.fc2(x)
        return x


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
        output_hidden_states: bool = False,
        mask: Optional[mx.array] = None,
    ):
        encoder_states = (x,) if output_hidden_states else None
        h = x
        for layer in self.layers:
            h = layer(h, mask=mask)
            if output_hidden_states:
                encoder_states = encoder_states + (h,)
        return h, encoder_states


class VisionEmbeddings(nn.Module):
    def __init__(self, config: VisionConfig):
        super().__init__()
        self.config = config
        self.embed_dim = config.hidden_size
        self.patch_size = config.patch_size

        # NaFlex: patches are already extracted, so use Linear projection
        self.patch_embedding = nn.Linear(
            config.patch_size * config.patch_size * config.num_channels,
            self.embed_dim,
            bias=True,
        )

        # 2D positional embeddings
        self.position_embedding_size = config.image_size // config.patch_size
        self.num_positions = self.position_embedding_size**2
        self.position_embedding = nn.Embedding(self.num_positions, self.embed_dim)

    @staticmethod
    def resize_positional_embeddings(
        positional_embeddings: mx.array,
        spatial_shapes: mx.array,
        max_length: int,
    ) -> mx.array:
        """
        Resize 2D positional embeddings per image and pad to fixed size.

        Args:
            positional_embeddings: (pos_size, pos_size, embed_dim)
            spatial_shapes: (batch_size, 2) - (h_patches, w_patches) per image
            max_length: max number of patches to pad to
        """
        batch_size = spatial_shapes.shape[0]
        embed_dim = positional_embeddings.shape[-1]
        source_dtype = positional_embeddings.dtype

        pos_size = positional_embeddings.shape[0]
        # (pos_size, pos_size, embed_dim) -> (1, embed_dim, pos_size, pos_size)
        pos_emb = positional_embeddings.transpose(2, 0, 1).reshape(
            1, embed_dim, pos_size, pos_size
        )

        batch_results = []
        for i in range(batch_size):
            height = int(spatial_shapes[i, 0].item())
            width = int(spatial_shapes[i, 1].item())

            resized = resize_bilinear(
                pos_emb,
                (height, width),
                align_corners=False,
                antialias=True,
            )

            # (1, embed_dim, h, w) -> (h*w, embed_dim)
            resized = resized.reshape(embed_dim, height * width).T
            resized = resized.astype(source_dtype)

            num_patches = height * width
            if num_patches < max_length:
                # Pad with first embedding repeated
                pad_count = max_length - num_patches
                padding = mx.broadcast_to(resized[0:1], (pad_count, embed_dim))
                resized = mx.concatenate([resized, padding], axis=0)

            batch_results.append(resized[:max_length])

        return mx.stack(batch_results)

    def __call__(
        self, x: mx.array, spatial_shapes: Optional[mx.array] = None
    ) -> mx.array:
        # x: (B, max_patches, patch_dim) for NaFlex
        patch_embeddings = self.patch_embedding(x)

        if spatial_shapes is not None:
            # NaFlex path: resize 2D positional embeddings per image
            positional_embeddings = self.position_embedding.weight.reshape(
                self.position_embedding_size, self.position_embedding_size, -1
            )
            resized_pos_embeddings = self.resize_positional_embeddings(
                positional_embeddings, spatial_shapes, max_length=x.shape[1]
            )
            embeddings = patch_embeddings + resized_pos_embeddings
        else:
            # Standard fixed-size path
            position_ids = mx.array(np.arange(self.num_positions)[None, :])
            embeddings = patch_embeddings + self.position_embedding(position_ids)

        return embeddings


class SigLip2VisionModel(nn.Module):
    """SigLIP2 vision model with NaFlex support."""

    def __init__(self, config: VisionConfig):
        super().__init__()
        self.embeddings = VisionEmbeddings(config)
        self.encoder = Encoder(config)
        self.post_layernorm = nn.LayerNorm(
            config.hidden_size, eps=config.layer_norm_eps
        )

    def __call__(
        self,
        x: mx.array,
        spatial_shapes: Optional[mx.array] = None,
        pixel_attention_mask: Optional[mx.array] = None,
        output_hidden_states: bool = False,
    ):
        x = self.embeddings(x, spatial_shapes=spatial_shapes)
        x = x.astype(self.embeddings.patch_embedding.weight.dtype)

        # Create attention mask from pixel_attention_mask
        mask = None
        if pixel_attention_mask is not None:
            # Block attention to padding tokens
            # mask shape: (B, 1, 1, max_patches)
            mask = mx.where(
                pixel_attention_mask[:, None, None, :].astype(mx.bool_),
                mx.array(0.0),
                mx.array(float("-inf")),
            ).astype(x.dtype)

        final_output, hidden_states = self.encoder(
            x, output_hidden_states=output_hidden_states, mask=mask
        )

        pooler_output = self.post_layernorm(final_output)
        return pooler_output, x, hidden_states


class VisionTower(nn.Module):
    """Wrapper matching the HF weight structure: model.vision_tower.vision_tower.*"""

    def __init__(self, config: VisionConfig):
        super().__init__()
        self.vision_tower = SigLip2VisionModel(config)
        self.select_layer = -2

    def __call__(
        self,
        pixel_values: mx.array,
        pixel_attention_mask: Optional[mx.array] = None,
        spatial_shapes: Optional[mx.array] = None,
    ):
        """
        Returns list of image features (one per image, variable length for NaFlex).
        """
        _, _, hidden_states = self.vision_tower(
            pixel_values,
            spatial_shapes=spatial_shapes,
            pixel_attention_mask=pixel_attention_mask,
            output_hidden_states=True,
        )

        # Select features from second-to-last layer
        selected = hidden_states[self.select_layer]

        # Remove padding tokens using spatial_shapes
        if spatial_shapes is not None:
            features = []
            for i in range(selected.shape[0]):
                h = int(spatial_shapes[i, 0].item())
                w = int(spatial_shapes[i, 1].item())
                num_valid = h * w
                feat = selected[i, :num_valid]
                features.append(feat)
            return features

        return selected

    def sanitize(self, weights):
        sanitized_weights = {}
        for k, v in weights.items():
            if "position_ids" in k:
                continue
            # Convert Conv2d patch_embedding weight to Linear format
            # Conv2d: (out_ch, in_ch, kH, kW) -> Linear: (out_ch, kH*kW*in_ch)
            if "patch_embedding.weight" in k and v.ndim == 4:
                # (out_ch, C, H, W) -> (out_ch, H, W, C) -> (out_ch, H*W*C)
                v = v.transpose(0, 2, 3, 1).reshape(v.shape[0], -1)
            sanitized_weights[k] = v
        return sanitized_weights
