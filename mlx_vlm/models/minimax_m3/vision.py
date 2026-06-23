"""MiniMax-M3 vision tower for mlx-vlm.

CLIP-style encoder (q/k/v/out + biases, layer_norm1/2, mlp.fc1/fc2, pre-layernorm) with a
Conv3d patch embedding (Qwen2.5-VL style) and Qwen-style 3D RoPE over the (T,H,W) patch grid,
followed by the patch-merge MLP connector that projects merged patches to the text hidden size.
"""
import mlx.core as mx
import mlx.nn as nn

from .config import VisionConfig


def rotate_half(x):
    x1, x2 = mx.split(x, 2, axis=-1)
    return mx.concatenate([-x2, x1], axis=-1)


def apply_rope_vision(q, k, cos, sin):
    rot = cos.shape[-1]
    cos = cos[None, :, None, :]
    sin = sin[None, :, None, :]
    q_rot, q_pass = q[..., :rot], q[..., rot:]
    k_rot, k_pass = k[..., :rot], k[..., rot:]
    q_rot = q_rot * cos + rotate_half(q_rot) * sin
    k_rot = k_rot * cos + rotate_half(k_rot) * sin
    return mx.concatenate([q_rot, q_pass], axis=-1), mx.concatenate([k_rot, k_pass], axis=-1)


class Vision3DRoPE:
    """cos/sin for the (T,H,W) patch grid; rotary dims split T|H|W, tail passes through."""

    def __init__(self, head_dim, theta=10000.0, spatial_merge_size=2):
        rope_dims = 2 * (head_dim // 2)
        self.axis_dim = 2 * ((rope_dims // 3) // 2)
        self.theta = theta
        self.m = spatial_merge_size

    def __call__(self, grid_thw):
        m = self.m
        all_coords = []
        for (t, h, w) in grid_thw:
            t, h, w = int(t), int(h), int(w)
            hi = mx.broadcast_to(mx.arange(h)[:, None], (h, w))
            hi = hi.reshape(h // m, m, w // m, m).transpose(0, 2, 1, 3).reshape(-1)
            wi = mx.broadcast_to(mx.arange(w)[None, :], (h, w))
            wi = wi.reshape(h // m, m, w // m, m).transpose(0, 2, 1, 3).reshape(-1)
            ti = mx.repeat(mx.arange(t), h * w)
            all_coords.append(mx.stack([ti, mx.tile(hi, (t,)), mx.tile(wi, (t,))], axis=-1))
        coords = mx.concatenate(all_coords, axis=0).astype(mx.float32)
        inv_freq = 1.0 / (self.theta ** (mx.arange(0, self.axis_dim, 2).astype(mx.float32) / self.axis_dim))
        freqs = mx.concatenate([coords[:, i:i + 1] * inv_freq for i in range(3)], axis=-1)
        emb = mx.concatenate([freqs, freqs], axis=-1)
        return mx.cos(emb), mx.sin(emb)


class VisionEmbeddings(nn.Module):
    def __init__(self, config: VisionConfig):
        super().__init__()
        self.patch_size = config.patch_size
        self.temporal_patch_size = config.temporal_patch_size
        self.in_channels = config.num_channels
        ks = (config.temporal_patch_size, config.patch_size, config.patch_size)
        self.patch_embedding = nn.Conv3d(config.num_channels, config.hidden_size, kernel_size=ks, stride=ks, bias=False)

    def __call__(self, x):
        # pixel_values feature order from the image processor is (C, T, ph, pw)
        # (matches the reference torch Conv3d input layout). MLX Conv3d wants
        # channels-last (N, T, ph, pw, C), so reshape to the TRUE order first,
        # then move channels to the last axis. A bare reshape to channels-last
        # scrambles pixels (scan-lines) and shifts channels (cyan/magenta tinge).
        x = x.reshape(-1, self.in_channels, self.temporal_patch_size, self.patch_size, self.patch_size)
        x = x.transpose(0, 2, 3, 4, 1)
        x = self.patch_embedding(x)
        return x.reshape(-1, x.shape[-1])


class VisionAttention(nn.Module):
    def __init__(self, config: VisionConfig):
        super().__init__()
        d = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = d // self.num_heads
        self.scale = self.head_dim ** -0.5
        self.q_proj = nn.Linear(d, d)
        self.k_proj = nn.Linear(d, d)
        self.v_proj = nn.Linear(d, d)
        self.out_proj = nn.Linear(d, d)

    def __call__(self, x, cos, sin, mask=None):
        L, _ = x.shape
        q = self.q_proj(x).reshape(1, L, self.num_heads, self.head_dim)
        k = self.k_proj(x).reshape(1, L, self.num_heads, self.head_dim)
        v = self.v_proj(x).reshape(1, L, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)
        q, k = apply_rope_vision(q, k, cos, sin)
        q, k = q.transpose(0, 2, 1, 3), k.transpose(0, 2, 1, 3)
        out = mx.fast.scaled_dot_product_attention(q, k, v, scale=self.scale, mask=mask)
        out = out.transpose(0, 2, 1, 3).reshape(L, -1)
        return self.out_proj(out)


class VisionMLP(nn.Module):
    def __init__(self, config: VisionConfig):
        super().__init__()
        self.fc1 = nn.Linear(config.hidden_size, config.intermediate_size)
        self.fc2 = nn.Linear(config.intermediate_size, config.hidden_size)

    def __call__(self, x):
        return self.fc2(nn.gelu(self.fc1(x)))


class VisionEncoderLayer(nn.Module):
    def __init__(self, config: VisionConfig):
        super().__init__()
        self.layer_norm1 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.layer_norm2 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.self_attn = VisionAttention(config)
        self.mlp = VisionMLP(config)

    def __call__(self, x, cos, sin, mask=None):
        x = x + self.self_attn(self.layer_norm1(x), cos, sin, mask)
        x = x + self.mlp(self.layer_norm2(x))
        return x


class VisionModel(nn.Module):
    def __init__(self, config: VisionConfig):
        super().__init__()
        self.embeddings = VisionEmbeddings(config)
        self.pre_layrnorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.encoder_layers = [VisionEncoderLayer(config) for _ in range(config.num_hidden_layers)]
        self.rope = Vision3DRoPE(config.hidden_size // config.num_attention_heads,
                                 theta=config.rope_theta, spatial_merge_size=config.spatial_merge_size)

    def __call__(self, pixel_values, grid_thw):
        h = self.embeddings(pixel_values)
        h = self.pre_layrnorm(h)
        cos, sin = self.rope(grid_thw)
        for layer in self.encoder_layers:
            h = layer(h, cos, sin)
        return h
