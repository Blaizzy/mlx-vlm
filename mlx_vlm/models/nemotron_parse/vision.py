import mlx.core as mx
import mlx.nn as nn

from .config import VisionConfig


class Attention(nn.Module):
    """Standard multi-head self-attention used in C-RADIO ViT blocks."""

    def __init__(self, config: VisionConfig):
        super().__init__()
        self.num_heads = config.num_heads
        self.head_dim = config.hidden_size // config.num_heads
        self.scale = self.head_dim**-0.5

        self.qkv = nn.Linear(config.hidden_size, config.hidden_size * 3)
        self.proj = nn.Linear(config.hidden_size, config.hidden_size)

    def __call__(self, x):
        batch, n_tokens, dim = x.shape

        qkv = self.qkv(x)
        qkv = qkv.reshape(batch, n_tokens, 3, self.num_heads, self.head_dim)
        qkv = qkv.transpose(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        q = q * self.scale
        attn = mx.matmul(q, k.transpose(0, 1, 3, 2))
        attn = mx.softmax(attn, axis=-1)

        out = mx.matmul(attn, v)
        out = out.transpose(0, 2, 1, 3).reshape(batch, n_tokens, dim)
        out = self.proj(out)
        return out


class Mlp(nn.Module):
    """MLP used in C-RADIO ViT blocks."""

    def __init__(self, config: VisionConfig):
        super().__init__()
        hidden_features = int(config.hidden_size * config.mlp_ratio)
        self.fc1 = nn.Linear(config.hidden_size, hidden_features)
        self.fc2 = nn.Linear(hidden_features, config.hidden_size)
        self.gelu = nn.GELU()

    def __call__(self, x):
        return self.fc2(self.gelu(self.fc1(x)))


class ViTBlock(nn.Module):
    """Pre-norm ViT block."""

    def __init__(self, config: VisionConfig):
        super().__init__()
        self.norm1 = nn.LayerNorm(config.hidden_size, eps=1e-6)
        self.attn = Attention(config)
        self.norm2 = nn.LayerNorm(config.hidden_size, eps=1e-6)
        self.mlp = Mlp(config)

    def __call__(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class Neck(nn.Module):
    """Projection/compression neck between the vision encoder and decoder."""

    def __init__(self, config: VisionConfig):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.neck_dim = config.neck_dim
        self.patch_size = config.patch_size

        # 1x1 Conv1d over the feature dimension; equivalent to a Linear projection.
        self.conv1 = nn.Conv1d(
            self.hidden_size,
            self.neck_dim,
            kernel_size=1,
        )
        self.layer_norm1 = nn.LayerNorm(self.neck_dim, eps=1e-6)

        self.conv2 = nn.Conv2d(
            self.neck_dim,
            self.neck_dim,
            kernel_size=(1, 4),
            stride=(1, 4),
            bias=False,
        )
        self.layer_norm2 = nn.LayerNorm(self.neck_dim, eps=1e-6)

        self.sum_proj = nn.Linear(
            len(config.summary_idxs) * self.hidden_size, self.neck_dim
        )
        self.layer_norm3 = nn.LayerNorm(self.neck_dim, eps=1e-6)

    def __call__(self, features, summary, input_size):
        # features: [B, num_patches, hidden_size]
        # summary: [B, num_cls_tokens * hidden_size]
        batch = features.shape[0]
        h, w = input_size
        h_patches = h // self.patch_size
        w_patches = w // self.patch_size

        # Conv1d over the last dimension.
        x = self.conv1(features)
        x = self.layer_norm1(x)

        # Compress horizontally by 4x.
        x = x.reshape(batch, h_patches, w_patches, self.neck_dim)
        x = self.conv2(x)
        x = x.reshape(batch, -1, self.neck_dim)
        x = self.layer_norm2(x)

        summary = self.sum_proj(summary)
        summary = self.layer_norm3(summary)

        x = mx.concatenate([x, summary[:, None, :]], axis=1)
        return x


class VisionModel(nn.Module):
    """C-RADIO ViT-Huge + neck for Nemotron-Parse."""

    def __init__(self, config: VisionConfig):
        super().__init__()
        self.config = config
        patch_dim = 3 * config.patch_size * config.patch_size

        self.patch_embed = nn.Linear(patch_dim, config.hidden_size, bias=False)
        self.pos_embed = mx.zeros((1, 128 * 128, config.hidden_size))
        self.cls_token = mx.zeros(
            (config.num_cls_tokens + config.num_register_tokens, config.hidden_size)
        )

        self.blocks = [ViTBlock(config) for _ in range(config.num_layers)]
        self.neck = Neck(config)

    def _bilinear_interp(self, grid, m):
        """Bilinear 2D interpolation with align_corners=True (matches F.interpolate)."""
        h, w, c = grid.shape
        if (h, w) == (m, m):
            return grid
        ys = mx.arange(m, dtype=mx.float32) * ((h - 1) / (m - 1))
        xs = mx.arange(m, dtype=mx.float32) * ((w - 1) / (m - 1))
        y0 = mx.floor(ys).astype(mx.int32)
        x0 = mx.floor(xs).astype(mx.int32)
        y1 = mx.minimum(y0 + 1, h - 1)
        x1 = mx.minimum(x0 + 1, w - 1)
        wy = (ys - y0.astype(mx.float32))[:, None, None]
        wx = (xs - x0.astype(mx.float32))[None, :, None]
        g00 = grid[y0][:, x0]
        g01 = grid[y0][:, x1]
        g10 = grid[y1][:, x0]
        g11 = grid[y1][:, x1]
        top = g00 * (1 - wx) + g01 * wx
        bot = g10 * (1 - wx) + g11 * wx
        return top * (1 - wy) + bot * wy

    def _get_pos_embed(self, h_patches, w_patches):
        # Learned 128x128 grid; the processor always emits 2048x1664 -> (128, 104)
        # patches, so the grid is a pure window crop (the source's interpolate to
        # (128, 128) is identity). For smaller grids, interpolate to the square
        # (max_dim, max_dim) with bilinear align_corners, then crop first rows/cols.
        pos = self.pos_embed.reshape(128, 128, self.config.hidden_size)
        if (128, 128) == (h_patches, w_patches):
            return pos.reshape(-1, self.config.hidden_size)
        max_dim = max(h_patches, w_patches)
        pos = self._bilinear_interp(pos, max_dim)
        pos = pos[:h_patches, :w_patches, :]
        return pos.reshape(-1, self.config.hidden_size)

    def _patchify(self, x):
        batch, _, h, w = x.shape
        h_patches = h // self.config.patch_size
        w_patches = w // self.config.patch_size
        x = x.reshape(
            batch,
            3,
            h_patches,
            self.config.patch_size,
            w_patches,
            self.config.patch_size,
        )
        x = x.transpose(0, 2, 4, 1, 3, 5).reshape(batch, h_patches * w_patches, -1)
        return x, (h, w), (h_patches, w_patches)

    def __call__(self, pixel_values):
        patches, input_size, (h_patches, w_patches) = self._patchify(pixel_values)
        x = self.patch_embed(patches)

        pos = self._get_pos_embed(h_patches, w_patches)
        x = x + pos[None, :, :]

        cls = mx.expand_dims(self.cls_token, 0)
        x = mx.concatenate(
            [mx.broadcast_to(cls, (x.shape[0], cls.shape[1], cls.shape[2])), x], axis=1
        )

        for block in self.blocks:
            x = block(x)

        # Extract summary tokens (first cls tokens) and features (after registers).
        num_skip = self.config.num_cls_tokens + self.config.num_register_tokens
        all_summary = x[:, : self.config.num_cls_tokens]
        summary = all_summary[:, self.config.summary_idxs]
        summary = summary.reshape(x.shape[0], -1)
        features = x[:, num_skip:]

        x = self.neck(features, summary, input_size)
        return x

    @staticmethod
    def sanitize(weights):
        sanitized_weights = {}

        for k, v in weights.items():
            if "summary_idxs" in k:
                # Hardcoded in the model; drop the checkpoint buffer.
                continue
            if "input_conditioner" in k:
                # The processor normalizes externally; skip the HF conditioner.
                continue

            # Layout transposes are idempotent: apply them only when the
            # weight is still in the PyTorch layout (the loader pipeline runs
            # this on both HF checkpoints and already-converted MLX files).
            if "vision_tower.neck.conv1.weight" in k:
                # Torch Conv1d: [out, in, 1] -> MLX Conv1d: [out, 1, in]
                if v.shape[-1] == 1 and v.shape[1] != 1:
                    sanitized_weights[k] = v.transpose(0, 2, 1)
                else:
                    sanitized_weights[k] = v
            elif "vision_tower.neck.conv2.weight" in k:
                # Torch Conv2d: [out, in, 1, 4] -> MLX Conv2d: [out, 1, 4, in]
                if v.shape[1] == v.shape[0] and v.shape[-1] != v.shape[0]:
                    sanitized_weights[k] = v.transpose(0, 2, 3, 1)
                else:
                    sanitized_weights[k] = v
            else:
                sanitized_weights[k] = v

        return sanitized_weights
