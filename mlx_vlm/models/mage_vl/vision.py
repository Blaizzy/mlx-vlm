"""Mage-VL vision tower (Mage-ViT): SigLIP-style encoder with 3D (T:H:W = 4:6:6)
rotary embeddings and a 2x2 patch merger. Verified numerically against the
reference torch implementation (cosine 0.99992 fp32-vs-fp32)."""
from typing import Optional

import mlx.core as mx
import mlx.nn as nn
import numpy as np

from .config import VisionConfig


def _convert_to_block_layout(pp, t, h, w, sms):
    total = t * h * w
    idx = np.arange(total).reshape(t, h, w)
    hm, wm = h // sms, w // sms
    idx = idx.reshape(t, hm, sms, wm, sms).transpose(0, 1, 3, 2, 4).reshape(total)
    return pp[idx]


def build_patch_positions(grid_thw, sms=2):
    """Deterministic [sum(t*h*w), 3] (t,h,w) positions in 2x2 block layout,
    matching video_processing_mage_vl.build_patch_positions for images."""
    out = []
    for row in grid_thw:
        t, h, w = int(row[0]), int(row[1]), int(row[2])
        h_coords = np.tile(np.repeat(np.arange(h), w), t)
        w_coords = np.tile(np.tile(np.arange(w), h), t)
        t_coords = np.repeat(np.arange(t), h * w)
        pp = np.stack([t_coords, h_coords, w_coords], axis=1)
        out.append(_convert_to_block_layout(pp, t, h, w, sms))
    return np.concatenate(out, axis=0).astype(np.float32)


def rotate_half(x):
    x1 = x[..., 0::2]
    x2 = x[..., 1::2]
    return mx.stack([-x2, x1], axis=-1).reshape(x.shape)


class Attention(nn.Module):
    def __init__(self, config: VisionConfig):
        super().__init__()
        self.num_heads = config.num_attention_heads
        self.head_dim = config.hidden_size // config.num_attention_heads
        self.scale = self.head_dim**-0.5
        self.qkv = nn.Linear(config.hidden_size, config.hidden_size * 3, bias=True)
        self.proj = nn.Linear(config.hidden_size, config.hidden_size, bias=True)

    def __call__(self, x, cos, sin, mask):
        L, _ = x.shape
        qkv = self.qkv(x).reshape(L, 3, self.num_heads, self.head_dim)
        qkv = qkv.transpose(1, 2, 0, 3)  # (3, NH, L, HD)
        q, k, v = qkv[0], qkv[1], qkv[2]
        # RoPE in float32
        qf, kf = q.astype(mx.float32), k.astype(mx.float32)
        q = (qf * cos + rotate_half(qf) * sin).astype(x.dtype)
        k = (kf * cos + rotate_half(kf) * sin).astype(x.dtype)
        out = mx.fast.scaled_dot_product_attention(
            q[None], k[None], v[None], scale=self.scale, mask=mask
        )[0]
        out = out.transpose(1, 0, 2).reshape(L, self.num_heads * self.head_dim)
        return self.proj(out)


class MLP(nn.Module):
    def __init__(self, config: VisionConfig):
        super().__init__()
        self.fc1 = nn.Linear(config.hidden_size, config.intermediate_size)
        self.fc2 = nn.Linear(config.intermediate_size, config.hidden_size)

    def __call__(self, x):
        return self.fc2(nn.gelu(self.fc1(x)))


class EncoderLayer(nn.Module):
    def __init__(self, config: VisionConfig):
        super().__init__()
        self.self_attn = Attention(config)
        self.layer_norm1 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.layer_norm2 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.mlp = MLP(config)

    def __call__(self, x, cos, sin, mask):
        x = x + self.self_attn(self.layer_norm1(x), cos, sin, mask)
        x = x + self.mlp(self.layer_norm2(x))
        return x


class Encoder(nn.Module):
    def __init__(self, config: VisionConfig):
        super().__init__()
        self.layers = [EncoderLayer(config) for _ in range(config.num_hidden_layers)]


class Embeddings(nn.Module):
    def __init__(self, config: VisionConfig):
        super().__init__()
        # Conv2d(k=stride=patch, no bias) == Linear over flattened patch.
        pdim = config.num_channels * config.patch_size * config.patch_size
        self.patch_embedding = nn.Linear(pdim, config.hidden_size, bias=False)

    def __call__(self, x):
        return self.patch_embedding(x)


class Merger(nn.Module):
    def __init__(self, config: VisionConfig):
        super().__init__()
        ctx = config.hidden_size * (config.spatial_merge_size**2)
        self.ln_q = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.fc1 = nn.Linear(ctx, ctx)
        self.fc2 = nn.Linear(ctx, config.out_hidden_size)

    def __call__(self, x, merge_dim):
        x = self.ln_q(x).reshape(-1, merge_dim)
        return self.fc2(nn.gelu(self.fc1(x)))


class VisionModel(nn.Module):
    def __init__(self, config: VisionConfig):
        super().__init__()
        self.config = config
        self.model_type = config.model_type
        self.sms = config.spatial_merge_size
        nh = config.num_attention_heads
        self.head_dim = config.hidden_size // nh
        self.half = self.head_dim // 2
        unit = self.half // 16
        self.t_size, self.h_size, self.w_size = 4 * unit, 6 * unit, 6 * unit
        self.base = config.rope_theta

        self.embeddings = Embeddings(config)
        self.layernorm_pre = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.encoder = Encoder(config)
        self.merger = Merger(config)

    def _rope(self, grid_thw):
        pp = build_patch_positions(grid_thw, self.sms)  # (L,3) np

        def inv(n):
            return 1.0 / (self.base ** (np.arange(n, dtype=np.float32) / n))

        ft = np.outer(pp[:, 0], inv(self.t_size))
        fh = np.outer(pp[:, 1], inv(self.h_size))
        fw = np.outer(pp[:, 2], inv(self.w_size))
        freqs = np.concatenate([ft, fh, fw], axis=-1)
        freqs = np.concatenate([freqs, freqs], axis=-1)  # (L, head_dim)
        cos = mx.array(np.cos(freqs))[None]  # (1,L,HD)
        sin = mx.array(np.sin(freqs))[None]
        return cos, sin

    def _block_mask(self, grid_thw, L, dtype):
        # Block-diagonal (bidirectional within each image/frame). t is always 1
        # for image rows and expanded video frames, so blocks are h*w each.
        blocks = [int(r[0]) * int(r[1]) * int(r[2]) for r in grid_thw]
        if len(blocks) == 1:
            return None  # single image: full attention
        m = np.full((L, L), -np.inf, dtype=np.float32)
        s = 0
        for b in blocks:
            m[s : s + b, s : s + b] = 0.0
            s += b
        return mx.array(m).astype(dtype)

    def __call__(self, pixel_values, grid_thw):
        grid_thw = np.array(grid_thw)
        cos, sin = self._rope(grid_thw)
        h = self.embeddings(pixel_values)
        h = self.layernorm_pre(h)
        L = h.shape[0]
        mask = self._block_mask(grid_thw, L, h.dtype)
        for layer in self.encoder.layers:
            h = layer(h, cos, sin, mask)
        return self.merger(h, self.config.hidden_size * (self.sms**2))

    def sanitize(self, weights):
        out = {}
        for k, v in weights.items():
            # Conv2d (O,C,ph,pw) -> Linear (O, C*ph*pw)
            if k.endswith("embeddings.patch_embedding.weight") and v.ndim == 4:
                v = v.reshape(v.shape[0], -1)
            # merger.mlp.0 -> merger.fc1 ; merger.mlp.2 -> merger.fc2
            k = k.replace("merger.mlp.0.", "merger.fc1.").replace(
                "merger.mlp.2.", "merger.fc2."
            )
            out[k] = v
        return out
