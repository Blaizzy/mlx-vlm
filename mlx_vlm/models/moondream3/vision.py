import math
from typing import Optional

import mlx.core as mx
import mlx.nn as nn

from .config import VisionConfig


class Attention(nn.Module):
    def __init__(self, config: VisionConfig):
        super().__init__()
        self.num_heads = config.num_attention_heads
        self.head_dim = config.hidden_size // config.num_attention_heads
        self.scale = self.head_dim**-0.5
        self.qkv = nn.Linear(
            config.hidden_size, 3 * config.hidden_size, bias=config.attention_bias
        )
        self.proj = nn.Linear(
            config.hidden_size, config.hidden_size, bias=config.attention_bias
        )

    def __call__(self, x: mx.array, mask: Optional[mx.array] = None) -> mx.array:
        B, L, _ = x.shape
        qkv = self.qkv(x)
        q, k, v = mx.split(qkv, 3, axis=-1)

        q = q.reshape(B, L, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)
        k = k.reshape(B, L, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)
        v = v.reshape(B, L, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)

        output = mx.fast.scaled_dot_product_attention(
            q, k, v, scale=self.scale, mask=mask
        )
        output = output.transpose(0, 2, 1, 3).reshape(B, L, -1)
        return self.proj(output)


class MLP(nn.Module):
    def __init__(self, config: VisionConfig):
        super().__init__()
        self.fc1 = nn.Linear(
            config.hidden_size, config.intermediate_size, bias=True
        )
        self.fc2 = nn.Linear(
            config.intermediate_size, config.hidden_size, bias=True
        )
        self.act = nn.GELU(approx="tanh")

    def __call__(self, x: mx.array) -> mx.array:
        return self.fc2(self.act(self.fc1(x)))


class EncoderBlock(nn.Module):
    def __init__(self, config: VisionConfig):
        super().__init__()
        self.ln1 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.attn = Attention(config)
        self.ln2 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.mlp = MLP(config)

    def __call__(self, x: mx.array, mask: Optional[mx.array] = None) -> mx.array:
        x = x + self.attn(self.ln1(x), mask)
        x = x + self.mlp(self.ln2(x))
        return x


class VisionEncoder(nn.Module):
    def __init__(self, config: VisionConfig):
        super().__init__()
        self.config = config
        patch_dim = config.patch_size * config.patch_size * config.in_channels
        num_patches = (config.crop_size // config.patch_size) ** 2

        self.patch_emb = nn.Linear(patch_dim, config.hidden_size, bias=True)
        self.pos_emb = mx.zeros((1, num_patches, config.hidden_size))
        self.blocks = [EncoderBlock(config) for _ in range(config.num_hidden_layers)]
        self.post_ln = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

    def _patchify(self, x: mx.array) -> mx.array:
        """Convert image to patch embeddings.

        Args:
            x: (B, H, W, C) in MLX channel-last format
        Returns:
            (B, num_patches, patch_dim)
        """
        B, H, W, C = x.shape
        P = self.config.patch_size
        pH = H // P
        pW = W // P
        # Reshape into patches (must match PyTorch's channel-first patch ordering)
        x = x.reshape(B, pH, P, pW, P, C)
        x = x.transpose(0, 1, 3, 5, 2, 4)  # (B, pH, pW, C, P, P)
        x = x.reshape(B, pH * pW, C * P * P)  # (B, num_patches, patch_dim)
        return x

    def __call__(self, x: mx.array) -> mx.array:
        """Encode a single crop.

        Args:
            x: (B, H, W, C) image tensor
        Returns:
            (B, num_patches, hidden_size)
        """
        x = self._patchify(x)
        x = self.patch_emb(x)
        x = x + self.pos_emb
        for block in self.blocks:
            x = block(x)
        x = self.post_ln(x)
        return x


class VisionProjection(nn.Module):
    """Projects vision features from encoder space to text model space."""

    def __init__(self, config: VisionConfig):
        super().__init__()
        self.fc1 = nn.Linear(
            2 * config.hidden_size, config.proj_inner_dim, bias=True
        )
        self.fc2 = nn.Linear(
            config.proj_inner_dim, config.proj_out_dim, bias=True
        )
        self.act = nn.GELU(approx="tanh")

    def __call__(self, x: mx.array) -> mx.array:
        return self.fc2(self.act(self.fc1(x)))


class VisionModel(nn.Module):
    def __init__(self, config: VisionConfig):
        super().__init__()
        self.config = config
        self.encoder = VisionEncoder(config)
        self.proj_mlp = VisionProjection(config)

    def _reconstruct_local_features(
        self, local_features: list, layout: tuple
    ) -> mx.array:
        """Reconstruct local crop features into a spatial grid and pool to match
        the global feature resolution.

        Args:
            local_features: list of (num_patches, hidden_size) for each local crop
            layout: (rows, cols) grid layout of crops
        Returns:
            (num_patches_global, hidden_size)
        """
        rows, cols = layout
        grid_size = self.config.crop_size // self.config.patch_size  # 27
        margin = self.config.overlap_margin

        # Reconstruct spatial grid from local crops
        crop_rows = []
        idx = 0
        for r in range(rows):
            row_features = []
            for c in range(cols):
                feat = local_features[idx]  # (num_patches, D)
                feat = feat.reshape(grid_size, grid_size, -1)  # (27, 27, D)

                # Trim overlap margins
                top = margin if r > 0 else 0
                bottom = grid_size - (margin if r < rows - 1 else 0)
                left = margin if c > 0 else 0
                right = grid_size - (margin if c < cols - 1 else 0)
                feat = feat[top:bottom, left:right, :]
                row_features.append(feat)
                idx += 1
            crop_rows.append(mx.concatenate(row_features, axis=1))  # concat along W

        full_grid = mx.concatenate(crop_rows, axis=0)  # (H_total, W_total, D)

        # Adaptive average pool to global resolution (grid_size x grid_size)
        H, W, D = full_grid.shape
        # Use reshape-based pooling
        pool_h = H / grid_size
        pool_w = W / grid_size

        # Simple approach: use strided slicing + averaging via reshape when possible
        # For arbitrary sizes, use a loop-based approach
        pooled = mx.zeros((grid_size, grid_size, D))
        for i in range(grid_size):
            h_start = int(round(i * pool_h))
            h_end = int(round((i + 1) * pool_h))
            h_end = max(h_end, h_start + 1)
            for j in range(grid_size):
                w_start = int(round(j * pool_w))
                w_end = int(round((j + 1) * pool_w))
                w_end = max(w_end, w_start + 1)
                pooled[i, j] = full_grid[h_start:h_end, w_start:w_end].mean(
                    axis=(0, 1)
                )

        return pooled.reshape(-1, D)  # (num_patches, D)

    def __call__(
        self,
        pixel_values: mx.array,
        num_crops: Optional[list] = None,
        crop_layouts: Optional[list] = None,
    ) -> mx.array:
        """Process multi-crop images through vision encoder and projection.

        Args:
            pixel_values: (total_crops, H, W, C) all crops stacked
            num_crops: list of number of crops per image in the batch
            crop_layouts: list of (rows, cols) tuples for local crop arrangement
        Returns:
            (B, num_vision_tokens, proj_out_dim) projected features
        """
        # Encode all crops at once
        all_features = self.encoder(pixel_values)  # (total_crops, num_patches, D)

        if num_crops is None:
            # Single crop per image - just project directly
            # Concatenate with itself (global only, no local)
            global_feats = all_features  # (B, 729, 1152)
            combined = mx.concatenate(
                [global_feats, global_feats], axis=-1
            )  # (B, 729, 2304)
            return self.proj_mlp(combined)

        # Multi-crop: process each image
        batch_features = []
        crop_idx = 0
        for i, nc in enumerate(num_crops):
            # First crop is always the global crop
            global_feats = all_features[crop_idx]  # (729, 1152)

            if nc > 1:
                # Local crops
                local_feats = [
                    all_features[crop_idx + j] for j in range(1, nc)
                ]
                layout = crop_layouts[i] if crop_layouts else (1, nc - 1)
                reconstructed = self._reconstruct_local_features(
                    local_feats, layout
                )  # (729, 1152)
            else:
                reconstructed = global_feats

            # Concatenate global + local: (729, 2304)
            combined = mx.concatenate(
                [global_feats, reconstructed], axis=-1
            )
            # Project: (729, 2048)
            projected = self.proj_mlp(combined)
            batch_features.append(projected)
            crop_idx += nc

        return mx.stack(batch_features)  # (B, 729, 2048)
