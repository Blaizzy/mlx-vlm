from typing import Optional

import mlx.core as mx
import mlx.nn as nn

from .config import VisionConfig


def _rotate_half(x: mx.array) -> mx.array:
    x = x.reshape(*x.shape[:-1], -1, 2)
    x1, x2 = x[..., 0], x[..., 1]
    return mx.stack([-x2, x1], axis=-1).reshape(*x.shape[:-2], -1)


def _quick_gelu(x: mx.array) -> mx.array:
    return x * mx.sigmoid(1.702 * x)


class EncoderRope2D(nn.Module):
    def __init__(
        self, dim: int, max_grid_height: int, max_grid_width: int, theta=10000
    ):
        super().__init__()
        self.dim = dim
        self.max_grid_height = max_grid_height
        self.max_grid_width = max_grid_width
        self.theta = theta

    def _freqs(self, gh: int, gw: int):
        inv_freq = 1.0 / (
            self.theta
            ** (mx.arange(0, self.dim // 2, 2, dtype=mx.float32) / (self.dim // 2))
        )
        rows = mx.arange(gh, dtype=mx.float32)
        cols = mx.arange(gw, dtype=mx.float32)
        freqs_h = rows[:, None] * inv_freq[None, :]
        freqs_w = cols[:, None] * inv_freq[None, :]
        freqs_h = mx.broadcast_to(freqs_h[:, None, :], (gh, gw, freqs_h.shape[-1]))
        freqs_w = mx.broadcast_to(freqs_w[None, :, :], (gh, gw, freqs_w.shape[-1]))
        return mx.concatenate([freqs_w, freqs_h], axis=-1).reshape(gh * gw, -1)

    def __call__(self, q: mx.array, k: mx.array, grid_hw: tuple[int, int]):
        gh, gw = grid_hw
        freqs = self._freqs(gh, gw)
        freqs = freqs[None, None, :, :]
        cos = mx.repeat(mx.cos(freqs), 2, axis=-1)
        sin = mx.repeat(mx.sin(freqs), 2, axis=-1)
        return (q * cos) + (_rotate_half(q) * sin), (k * cos) + (_rotate_half(k) * sin)


class EncoderMLP(nn.Module):
    def __init__(self, dim: int, hidden_dim: int, hidden_act: str):
        super().__init__()
        self.c_fc = nn.Linear(dim, hidden_dim, bias=True)
        self.c_proj = nn.Linear(hidden_dim, dim, bias=True)
        self.hidden_act = hidden_act

    def __call__(self, x: mx.array) -> mx.array:
        x = self.c_fc(x)
        x = _quick_gelu(x) if self.hidden_act == "quick_gelu" else nn.gelu(x)
        return self.c_proj(x)


class EncoderVisionAttention(nn.Module):
    def __init__(self, config: VisionConfig):
        super().__init__()
        self.num_heads = config.heads
        self.head_dim = config.width // config.heads
        self.scale = self.head_dim**-0.5
        self.in_proj = nn.Linear(config.width, config.width * 3, bias=True)
        self.out_proj = nn.Linear(config.width, config.width, bias=True)
        self.rope = (
            EncoderRope2D(
                self.head_dim,
                config.image_size // config.patch_size,
                config.image_size // config.patch_size,
                theta=config.rope_theta,
            )
            if config.use_rope2d
            else None
        )

    def __call__(self, x: mx.array, grid_hw: tuple[int, int]) -> mx.array:
        b, l, _ = x.shape
        qkv = self.in_proj(x).reshape(b, l, 3, self.num_heads, self.head_dim)
        q, k, v = qkv[:, :, 0], qkv[:, :, 1], qkv[:, :, 2]
        q = q.transpose(0, 2, 1, 3)
        k = k.transpose(0, 2, 1, 3)
        v = v.transpose(0, 2, 1, 3)
        if self.rope is not None:
            q, k = self.rope(q, k, grid_hw)
        y = mx.fast.scaled_dot_product_attention(q, k, v, scale=self.scale)
        y = y.transpose(0, 2, 1, 3).reshape(b, l, -1)
        return self.out_proj(y)


class EncoderLayerScale(nn.Module):
    def __init__(self, dim: int, init_value: Optional[float]):
        super().__init__()
        self.gamma = mx.ones(dim) * (1.0 if init_value is None else init_value)

    def __call__(self, x: mx.array) -> mx.array:
        return x * self.gamma


class EncoderVisionBlock(nn.Module):
    def __init__(self, config: VisionConfig):
        super().__init__()
        hidden = config.width
        self.ln_1 = nn.LayerNorm(hidden, eps=config.layer_norm_eps)
        self.ln_2 = nn.LayerNorm(hidden, eps=config.layer_norm_eps)
        self.attn = EncoderVisionAttention(config)
        self.mlp = EncoderMLP(hidden, int(hidden * config.mlp_ratio), config.hidden_act)
        self.ls_1 = EncoderLayerScale(hidden, config.ls_init_value)
        self.ls_2 = EncoderLayerScale(hidden, config.ls_init_value)

    def __call__(self, x: mx.array, grid_hw: tuple[int, int]) -> mx.array:
        x = x + self.ls_1(self.attn(self.ln_1(x), grid_hw))
        return x + self.ls_2(self.mlp(self.ln_2(x)))


class VisionModel(nn.Module):
    def __init__(self, config: VisionConfig):
        super().__init__()
        self.config = config
        self.hidden_size = config.width
        self.patch_size = config.patch_size
        self.use_cls_token = config.use_cls_token
        self.use_abs_posemb = config.use_abs_posemb
        self.use_ln_post = config.use_ln_post
        self.conv1 = nn.Conv2d(
            config.num_channels,
            config.width,
            config.patch_size,
            stride=config.patch_size,
            bias=False,
        )
        self.ln_pre = (
            nn.LayerNorm(config.width, eps=config.layer_norm_eps)
            if config.use_ln_pre
            else None
        )
        self.ln_post = (
            nn.LayerNorm(config.width, eps=config.layer_norm_eps)
            if config.use_ln_post
            else None
        )
        grid = config.image_size // config.patch_size
        if self.use_cls_token:
            self.class_embedding = mx.random.normal((config.width,)) * (
                config.width**-0.5
            )
        if self.use_abs_posemb:
            self.posemb_grid_size = grid
            self.positional_embedding = mx.random.normal(
                (int(self.use_cls_token) + grid * grid, config.width)
            ) * (config.width**-0.5)
        self.transformer = [EncoderVisionBlock(config) for _ in range(config.layers)]
        self.vit_downsampler1 = nn.Conv2d(
            config.width, config.width * 2, 3, stride=2, padding=1
        )
        self.vit_downsampler2 = nn.Conv2d(
            config.width * 2, config.width * 4, 3, stride=2, padding=1
        )

    def _pos_embed(self, gh: int, gw: int):
        if self.posemb_grid_size == gh and self.posemb_grid_size == gw:
            return self.positional_embedding[None, ...]
        pos = self.positional_embedding
        cls = None
        if self.use_cls_token:
            cls, pos = pos[:1], pos[1:]
        pos = pos.reshape(1, self.posemb_grid_size, self.posemb_grid_size, -1)
        scale = (gh / self.posemb_grid_size, gw / self.posemb_grid_size)
        pos = nn.Upsample(scale_factor=scale, mode="linear")(pos)
        pos = pos.reshape(gh * gw, self.hidden_size)
        if cls is not None:
            pos = mx.concatenate([cls, pos], axis=0)
        return pos[None, ...]

    def __call__(self, pixel_values: mx.array) -> mx.array:
        if pixel_values.ndim == 3:
            pixel_values = pixel_values[None, ...]
        if pixel_values.shape[1] == 3:
            pixel_values = pixel_values.transpose(0, 2, 3, 1)
        b, h, w, _ = pixel_values.shape
        gh, gw = h // self.patch_size, w // self.patch_size
        x = self.conv1(pixel_values).reshape(b, gh * gw, self.hidden_size)
        if self.use_cls_token:
            cls = mx.broadcast_to(
                self.class_embedding.reshape(1, 1, -1), (b, 1, self.hidden_size)
            )
            x = mx.concatenate([cls, x], axis=1)
        if self.use_abs_posemb:
            x = x + self._pos_embed(gh, gw)
        if self.ln_pre is not None:
            x = self.ln_pre(x)
        for block in self.transformer:
            x = block(x, (gh, gw))
        if self.ln_post is not None:
            x = self.ln_post(x)
        if self.use_cls_token:
            x = x[:, 1:, :]
        return x

    def sanitize(self, weights):
        sanitized = {}
        for key, value in weights.items():
            prefix = ""
            if key.startswith("vision_model."):
                prefix = "vision_model."
                key = key.replace("vision_model.", "", 1)
            if key.startswith("transformer.resblocks."):
                key = key.replace("transformer.resblocks.", "transformer.", 1)
            if key.endswith("attn.in_proj_weight"):
                key = key.replace("attn.in_proj_weight", "attn.in_proj.weight")
            elif key.endswith("attn.in_proj_bias"):
                key = key.replace("attn.in_proj_bias", "attn.in_proj.bias")
            if (
                key.endswith("conv1.weight")
                and value.ndim == 4
                and value.shape[-1] != self.config.num_channels
            ):
                value = value.transpose(0, 2, 3, 1)
            elif (
                "vit_downsampler" in key and key.endswith(".weight") and value.ndim == 4
            ):
                expected_in = self.config.width * (
                    2 if "vit_downsampler2" in key else 1
                )
                if value.shape[-1] != expected_in:
                    value = value.transpose(0, 2, 3, 1)
            sanitized[f"{prefix}{key}"] = value
        return sanitized
