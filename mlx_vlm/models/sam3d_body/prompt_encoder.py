"""Prompt encoder for SAM 3D Body: keypoint/box prompts to embeddings."""

import math

import mlx.core as mx
import mlx.nn as nn


class PositionalEncodingGaussian(nn.Module):
    """Gaussian random Fourier features for 2D positional encoding."""

    def __init__(self, num_feats: int = 640, scale: float = 1.0):
        super().__init__()
        # Loaded from prompt_encoder.pe_layer.positional_encoding_gaussian_matrix
        self.positional_encoding_gaussian_matrix = mx.zeros((2, num_feats))
        self.scale = scale

    def __call__(self, coords: mx.array) -> mx.array:
        """
        Args:
            coords: (B, N, 2) normalized coordinates in [0, 1]
        Returns:
            (B, N, num_feats*2) positional encoding
        """
        coords = coords * 2 - 1  # map to [-1, 1]
        coords = coords @ (
            self.positional_encoding_gaussian_matrix * self.scale * 2 * math.pi
        )
        # (B, N, num_feats) -> concat sin and cos -> (B, N, num_feats*2)
        return mx.concatenate([mx.sin(coords), mx.cos(coords)], axis=-1)


class PromptEncoder(nn.Module):
    """Encodes point/box prompts into token and positional embeddings.

    Weight keys:
        pe_layer.positional_encoding_gaussian_matrix: (2, 640)
        point_embeddings.{0-69}.weight: (1, 1280)
        not_a_point_embed.weight: (1, 1280)
        invalid_point_embed.weight: (1, 1280)
        no_mask_embed.weight: (1, 1280)
        mask_downscaling.{0,3,6,9,10,12}: Conv2d / LayerNorm2d weights
    """

    def __init__(self, embed_dim: int = 1280, num_point_embeddings: int = 70):
        super().__init__()
        self.embed_dim = embed_dim
        self.pe_layer = PositionalEncodingGaussian(num_feats=embed_dim // 2)

        # Per-keypoint type embeddings
        self.point_embeddings = [
            nn.Embedding(1, embed_dim) for _ in range(num_point_embeddings)
        ]
        self.not_a_point_embed = nn.Embedding(1, embed_dim)
        self.invalid_point_embed = nn.Embedding(1, embed_dim)
        self.no_mask_embed = nn.Embedding(1, embed_dim)

    def encode_points(
        self,
        points: mx.array,
        labels: mx.array,
    ) -> tuple[mx.array, mx.array]:
        """Encode point prompts.

        Args:
            points: (B, N, 2) point coordinates normalized to [0, 1]
            labels: (B, N) integer labels indicating point type (0-69, or -1 for padding)
        Returns:
            embeddings: (B, N, embed_dim) point type embeddings
            pe: (B, N, embed_dim) positional encodings
        """
        pe = self.pe_layer(points)

        B, N = labels.shape
        embeddings = mx.zeros((B, N, self.embed_dim))

        # Apply per-label type embedding
        for i in range(len(self.point_embeddings)):
            mask = labels == i  # (B, N)
            if mx.any(mask):
                embed = self.point_embeddings[i].weight[0]  # (embed_dim,)
                embeddings = embeddings + mask[..., None] * embed

        # Invalid/padding points (label == -1)
        invalid_mask = labels == -1
        if mx.any(invalid_mask):
            embeddings = (
                embeddings
                + invalid_mask[..., None] * self.invalid_point_embed.weight[0]
            )
            pe = pe * (1 - invalid_mask[..., None].astype(pe.dtype))

        return embeddings, pe

    def get_dense_pe(self, h: int, w: int) -> mx.array:
        """Get dense positional encoding for an image grid.

        Args:
            h: grid height
            w: grid width
        Returns:
            (1, h, w, embed_dim) positional encoding
        """
        grid_y = mx.arange(h, dtype=mx.float32)
        grid_x = mx.arange(w, dtype=mx.float32)
        grid_y = (grid_y + 0.5) / h
        grid_x = (grid_x + 0.5) / w

        # Create meshgrid
        gy = mx.broadcast_to(grid_y[:, None], (h, w))
        gx = mx.broadcast_to(grid_x[None, :], (h, w))

        coords = mx.stack([gx, gy], axis=-1)  # (h, w, 2)
        coords = coords.reshape(1, h * w, 2)  # (1, h*w, 2)
        pe = self.pe_layer(coords)  # (1, h*w, embed_dim)
        pe = pe.reshape(1, h, w, self.embed_dim)  # (1, h, w, embed_dim)
        return pe
