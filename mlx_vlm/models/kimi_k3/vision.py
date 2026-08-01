from typing import List, Optional

import mlx.core as mx
import mlx.nn as nn

from .config import VisionConfig


def check_array_shape(arr):
    if arr.ndim != 4:
        return False
    out_channels, kH, kW, _ = arr.shape
    return (out_channels >= kH) and (out_channels >= kW) and (kH == kW)


def _interp_indices(in_size: int, out_size: int):
    src = (mx.arange(out_size, dtype=mx.float32) + 0.5) * (in_size / out_size) - 0.5
    src = mx.clip(src, 0, in_size - 1)
    i0 = mx.floor(src).astype(mx.int32)
    i1 = mx.minimum(i0 + 1, in_size - 1)
    frac = src - i0.astype(mx.float32)
    return i0, i1, frac


def bilinear_interpolate(weight: mx.array, h: int, w: int) -> mx.array:
    H, W, _ = weight.shape
    if (H, W) == (h, w):
        return weight
    weight = weight.astype(mx.float32)
    r0, r1, rf = _interp_indices(H, h)
    rows = weight[r0] * (1 - rf)[:, None, None] + weight[r1] * rf[:, None, None]
    c0, c1, cf = _interp_indices(W, w)
    return rows[:, c0] * (1 - cf)[None, :, None] + rows[:, c1] * cf[None, :, None]


def sincos_time_embedding(dim: int, t_size: int) -> mx.array:
    omega = mx.arange(dim // 2, dtype=mx.float32) / (dim / 2.0)
    omega = 1.0 / (10000**omega)
    pos = mx.arange(t_size, dtype=mx.float32)
    out = pos[:, None] * omega[None, :]
    return mx.concatenate([mx.sin(out), mx.cos(out)], axis=1)


def rope_2d_angles(head_dim: int, t: int, h: int, w: int) -> mx.array:
    freqs = 1.0 / (
        10000
        ** (mx.arange(0, head_dim, 4, dtype=mx.float32)[: head_dim // 4] / head_dim)
    )
    y = mx.repeat(mx.arange(h, dtype=mx.float32), w)
    x = mx.tile(mx.arange(w, dtype=mx.float32), h)
    x_freqs = x[:, None] * freqs[None, :]
    y_freqs = y[:, None] * freqs[None, :]
    angles = mx.stack([x_freqs, y_freqs], axis=-1).reshape(h * w, -1)
    if t > 1:
        angles = mx.tile(angles, (t, 1))
    return angles


def apply_rope_2d(x: mx.array, cos: mx.array, sin: mx.array) -> mx.array:
    dtype = x.dtype
    x = x.astype(mx.float32)
    xe = x[..., 0::2]
    xo = x[..., 1::2]
    cos = cos[:, None, :]
    sin = sin[:, None, :]
    oe = xe * cos - xo * sin
    oo = xe * sin + xo * cos
    return mx.stack([oe, oo], axis=-1).flatten(-2).astype(dtype)


class Learnable2DInterpPosEmb(nn.Module):
    def __init__(self, config: VisionConfig):
        super().__init__()
        self.height = config.init_pos_emb_height
        self.width = config.init_pos_emb_width
        self.num_frames = config.init_pos_emb_time
        self.weight = mx.zeros((self.height, self.width, config.vt_hidden_size))
        self._time_weight = sincos_time_embedding(
            config.vt_hidden_size, self.num_frames
        )

    def __call__(self, x: mx.array, grid_thws: List[List[int]]) -> mx.array:
        pos_embs = []
        for t, h, w in grid_thws:
            pos_2d = bilinear_interpolate(self.weight, h, w).reshape(h * w, -1)
            if t == 1:
                pos_3d = pos_2d
            else:
                pos_3d = pos_2d[None] + self._time_weight[:t, None, :]
                pos_3d = pos_3d.reshape(t * h * w, -1)
            pos_embs.append(pos_3d)
        return x + mx.concatenate(pos_embs).astype(x.dtype)


class KimiK3PatchEmbed(nn.Module):
    def __init__(self, config: VisionConfig):
        super().__init__()
        self.proj = nn.Conv2d(
            3,
            config.vt_hidden_size,
            kernel_size=config.patch_size,
            stride=config.patch_size,
            bias=config.patch_embed_proj_bias,
        )
        self.pos_emb = Learnable2DInterpPosEmb(config)

    def __call__(self, patches: mx.array, grid_thws: List[List[int]]) -> mx.array:
        x = self.proj(patches).reshape(patches.shape[0], -1)
        return self.pos_emb(x, grid_thws)


class KimiK3VisionMLP(nn.Module):
    def __init__(self, config: VisionConfig):
        super().__init__()
        bias = config.linear_bias
        self.fc0 = nn.Linear(
            config.vt_hidden_size, config.vt_intermediate_size, bias=bias
        )
        self.fc1 = nn.Linear(
            config.vt_intermediate_size, config.vt_hidden_size, bias=bias
        )

    def __call__(self, x: mx.array) -> mx.array:
        return self.fc1(nn.gelu_approx(self.fc0(x)))


class KimiK3VisionBlock(nn.Module):
    def __init__(self, config: VisionConfig):
        super().__init__()
        self.num_heads = config.vt_num_attention_heads
        self.qkv_hidden_size = config.qkv_hidden_size or config.vt_hidden_size
        self.head_dim = self.qkv_hidden_size // self.num_heads
        self.scale = self.head_dim**-0.5

        self.norm0 = nn.RMSNorm(config.vt_hidden_size, eps=2**-7)
        self.norm1 = nn.RMSNorm(config.vt_hidden_size, eps=2**-7)
        self.wqkv = nn.Linear(
            config.vt_hidden_size, 3 * self.qkv_hidden_size, bias=config.attn_bias
        )
        self.wo = nn.Linear(
            self.qkv_hidden_size, config.vt_hidden_size, bias=config.attn_bias
        )
        self.mlp = KimiK3VisionMLP(config)

    def __call__(self, x: mx.array, cos: mx.array, sin: mx.array) -> mx.array:
        L = x.shape[0]
        qkv = self.wqkv(self.norm0(x)).reshape(L, 3, self.num_heads, self.head_dim)
        q, k, v = qkv[:, 0], qkv[:, 1], qkv[:, 2]
        q = apply_rope_2d(q, cos, sin)
        k = apply_rope_2d(k, cos, sin)
        q = q.transpose(1, 0, 2)[None]
        k = k.transpose(1, 0, 2)[None]
        v = v.transpose(1, 0, 2)[None]
        out = mx.fast.scaled_dot_product_attention(q, k, v, scale=self.scale)
        out = out[0].transpose(1, 0, 2).reshape(L, -1)
        x = x + self.wo(out)
        return x + self.mlp(self.norm1(x))


class KimiK3VisionEncoder(nn.Module):
    def __init__(self, config: VisionConfig):
        super().__init__()
        self.head_dim = (
            config.qkv_hidden_size or config.vt_hidden_size
        ) // config.vt_num_attention_heads
        self.blocks = [
            KimiK3VisionBlock(config) for _ in range(config.vt_num_hidden_layers)
        ]
        self.final_layernorm = nn.RMSNorm(config.vt_hidden_size, eps=2**-7)

    def __call__(self, x: mx.array, t: int, h: int, w: int) -> mx.array:
        angles = rope_2d_angles(self.head_dim, t, h, w)
        cos = mx.cos(angles)
        sin = mx.sin(angles)
        for block in self.blocks:
            x = block(x, cos, sin)
        return self.final_layernorm(x)


def tpool_patch_merge(x: mx.array, t: int, h: int, w: int, kernel) -> mx.array:
    kh, kw = kernel
    d = x.shape[-1]
    nh, nw = h // kh, w // kw
    x = x.reshape(t, nh, kh, nw, kw, d)
    x = x.transpose(0, 1, 3, 2, 4, 5).mean(axis=0)
    return x.reshape(nh * nw, kh * kw, d)


class VisionModel(nn.Module):
    def __init__(self, config: VisionConfig):
        super().__init__()
        self.config = config
        self.model_type = config.model_type
        if self.model_type != "moonvit3d":
            raise ValueError(f"Unsupported model type: {self.model_type}")
        self.merge_kernel_size = config.merge_kernel_size
        self.patch_embed = KimiK3PatchEmbed(config)
        self.encoder = KimiK3VisionEncoder(config)

    def __call__(
        self,
        patches: mx.array,
        grid_thws=None,
        output_hidden_states: Optional[bool] = None,
        **kwargs,
    ) -> mx.array:
        if hasattr(grid_thws, "tolist"):
            grid_thws = grid_thws.tolist()
        grid_thws = [[1, *thw] if len(thw) == 2 else list(thw) for thw in grid_thws]
        h_all = self.patch_embed(patches, grid_thws)
        outputs = []
        start = 0
        for t, h, w in grid_thws:
            n = t * h * w
            x = self.encoder(h_all[start : start + n], t, h, w)
            outputs.append(tpool_patch_merge(x, t, h, w, self.merge_kernel_size))
            start += n
        return mx.concatenate(outputs, axis=0)

    def sanitize(self, weights):
        sanitized = {}
        for k, v in weights.items():
            if k.startswith("vision_tower") and "patch_embed.proj.weight" in k:
                if v.ndim == 4 and not check_array_shape(v):
                    v = v.transpose(0, 2, 3, 1)
                sanitized[k] = v
            else:
                sanitized[k] = v
        return sanitized
