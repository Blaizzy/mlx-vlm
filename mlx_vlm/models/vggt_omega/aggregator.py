"""VGGT-Omega aggregator: a DINOv3 patch encoder, then alternating frame and
cross-frame attention over all tokens of a sequence."""

from typing import List, Tuple

import mlx.core as mx
import mlx.nn as nn

from ..dinov2.dinov2 import PatchEmbed
from ..sapiens2.backbone import RopePositionEmbedding
from .config import ModelConfig
from .layers import Block, norm32

IMAGENET_MEAN = mx.array([0.485, 0.456, 0.406])
IMAGENET_STD = mx.array([0.229, 0.224, 0.225])


class PatchEncoder(nn.Module):
    """DINOv3 ViT-L/16 with 4 storage tokens; returns normed patch tokens."""

    def __init__(self, config: ModelConfig):
        super().__init__()
        dim = config.embed_dim
        self.patch_size = config.patch_size
        self.num_prefix_tokens = 1 + config.num_storage_tokens
        self.patch_embed = PatchEmbed(224, config.patch_size, 3, dim)
        self.cls_token = mx.zeros((1, 1, dim))
        self.storage_tokens = mx.zeros((1, config.num_storage_tokens, dim))
        self.rope_embed = RopePositionEmbedding(
            dim, config.num_heads, config.rope_base, normalize_coords="max"
        )
        self.blocks = [
            Block(dim, config.num_heads, config.mlp_ratio, eps=config.layer_norm_eps)
            for _ in range(config.encoder_depth)
        ]
        self.norm = nn.LayerNorm(dim, eps=config.layer_norm_eps)

    def __call__(self, x: mx.array) -> mx.array:
        """x: (B, H, W, 3) normalized images -> (B, h * w, D) float32."""
        B, H, W, _ = x.shape
        x = self.patch_embed(x.astype(self.patch_embed.proj.weight.dtype))
        prefix = mx.concatenate([self.cls_token, self.storage_tokens], axis=1)
        prefix = mx.broadcast_to(prefix, (B, *prefix.shape[1:]))
        x = mx.concatenate([prefix, x], axis=1).astype(mx.float32)

        p = self.patch_size
        rope = self.rope_embed(H // p, W // p, prefix=self.num_prefix_tokens)
        for block in self.blocks:
            x = block(x, rope)
        # Norm the contiguous rows, then drop the prefix: the caller's
        # concatenate reads the strided view without another copy.
        return norm32(self.norm, x)[:, self.num_prefix_tokens :]


class Aggregator(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        dim, heads = config.embed_dim, config.num_heads
        eps = config.layer_norm_eps
        self.patch_size = config.patch_size
        self.patch_token_start = 1 + config.num_register_tokens
        self.cached_layer_indices = frozenset(config.cached_layer_indices)
        self.register_attention_blocks = frozenset(
            config.register_attention_block_indices
        )

        self.patch_embed = PatchEncoder(config)
        self.rope_embed = RopePositionEmbedding(
            dim, heads, config.rope_base, normalize_coords="max"
        )
        self.frame_blocks = [
            Block(dim, heads, config.mlp_ratio, qk_norm=True, eps=eps)
            for _ in range(config.depth)
        ]
        self.inter_frame_blocks = [
            Block(dim, heads, config.mlp_ratio, qk_norm=True, eps=eps)
            for _ in range(config.depth)
        ]
        # Index 0 is for the first (reference) frame, index 1 for the others.
        self.camera_token = mx.zeros((1, 2, 1, dim))
        self.register_token = mx.zeros((1, 2, config.num_register_tokens, dim))

    def __call__(self, images: mx.array) -> Tuple[mx.array, List[mx.array]]:
        """images: (B, S, H, W, 3) in [0, 1].

        Returns float32 frame-attention outputs concatenated with the
        cross-frame outputs: the camera and register tokens of the last
        layer (B, S, P, 2D), and the patch tokens (B, S, h * w, 2D) of each
        cached layer.
        """
        B, S, H, W, _ = images.shape
        x = (images.reshape(B * S, H, W, 3) - IMAGENET_MEAN) / IMAGENET_STD
        patches = self.patch_embed(x)
        _, _, D = patches.shape

        P = self.patch_token_start
        special = mx.concatenate([self.camera_token, self.register_token], axis=2)
        special = mx.take(special, mx.array([0] + [1] * (S - 1)), axis=1)
        special = mx.broadcast_to(special, (B, S, P, D)).reshape(B * S, P, D)
        tokens = mx.concatenate([special.astype(mx.float32), patches], axis=1)
        N = tokens.shape[1]

        p = self.patch_size
        rope = self.rope_embed(H // p, W // p, prefix=P)

        def cat(frame_tokens, tokens):
            return mx.concatenate([frame_tokens, tokens], axis=-1).reshape(
                B, S, -1, 2 * D
            )

        layers = []
        for i, (frame_block, cross_block) in enumerate(
            zip(self.frame_blocks, self.inter_frame_blocks)
        ):
            tokens = frame_block(tokens, rope)
            frame_tokens = tokens
            if i in self.register_attention_blocks:
                # Only camera and register tokens attend across frames.
                special = cross_block(tokens[:, :P].reshape(B, S * P, D))
                tokens = mx.concatenate(
                    [special.reshape(B * S, P, D), tokens[:, P:]], axis=1
                )
            else:
                tokens = cross_block(tokens.reshape(B, S * N, D))
                tokens = tokens.reshape(B * S, N, D)
            if i in self.cached_layer_indices:
                layers.append(cat(frame_tokens[:, P:], tokens[:, P:]))
        return cat(frame_tokens[:, :P], tokens[:, :P]), layers
