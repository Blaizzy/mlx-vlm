from dataclasses import dataclass

import mlx.core as mx
import mlx.nn as nn

from ..interpolate import resize_bilinear
from .config import VisionConfig


@dataclass
class RadioOutput:
    summary: mx.array
    features: mx.array


class InputConditioner(nn.Module):
    def __init__(self):
        super().__init__()
        self.norm_mean = mx.zeros((3, 1, 1))
        self.norm_std = mx.ones((3, 1, 1))

    def __call__(self, x):
        return (x - self.norm_mean) / self.norm_std


class ClsToken(nn.Module):
    def __init__(self, embed_dim: int, num_tokens: int, register_multiple: int | None):
        super().__init__()
        self.num_tokens = num_tokens
        self.num_registers = 0
        if register_multiple:
            self.num_registers = register_multiple - (num_tokens % register_multiple)
        self.token = mx.zeros((self.num_tokens + self.num_registers, embed_dim))

    @property
    def num_patches(self):
        return self.num_tokens + self.num_registers

    def __call__(self, x):
        token = mx.broadcast_to(
            self.token[None, :, :],
            (x.shape[0], self.token.shape[0], self.token.shape[1]),
        ).astype(x.dtype)
        return mx.concatenate([token, x], axis=1)


class ViTPatchGenerator(nn.Module):
    def __init__(self, config: VisionConfig):
        super().__init__()
        args = config.args or {}
        embed_dim = config.hidden_size
        input_dims = (config.image_size, config.image_size)
        max_input_dims = int(args.get("cpe_max_size") or config.max_resolution)
        patch_size = config.patch_size

        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.num_rows = max_input_dims // patch_size
        self.num_cols = max_input_dims // patch_size
        self.input_dims = tuple(d // patch_size for d in input_dims)
        self.num_patches = self.num_rows * self.num_cols
        self.cpe_mode = (self.num_rows, self.num_cols) != self.input_dims

        teachers = args.get("teachers", [])
        if args.get("cls_token_per_teacher", True) and teachers:
            num_cls_tokens = len({teacher["name"] for teacher in teachers})
        else:
            num_cls_tokens = 1
        self.cls_token = ClsToken(
            embed_dim,
            num_tokens=num_cls_tokens,
            register_multiple=args.get("register_multiple", None),
        )
        self.embedder = nn.Linear(3 * patch_size * patch_size, embed_dim, bias=False)
        self.video_embedder = nn.Linear(
            config.video_temporal_patch_size * 3 * patch_size * patch_size,
            embed_dim,
            bias=False,
        )
        self.pos_embed = mx.zeros((1, self.num_patches, embed_dim))

    @property
    def num_cls_tokens(self):
        return self.cls_token.num_tokens

    @property
    def num_registers(self):
        return self.cls_token.num_registers

    @property
    def num_skip(self):
        return self.num_cls_tokens + self.num_registers

    def _im_to_patches(self, x):
        batch, channels, height, width = x.shape
        patch = self.patch_size
        patch_h = height // patch
        patch_w = width // patch
        x = x.reshape(batch, channels, patch_h, patch, patch_w, patch)
        x = x.transpose(0, 2, 4, 1, 3, 5)
        return x.reshape(batch, patch_h * patch_w, channels * patch * patch)

    def _get_pos_embeddings(self, batch_size, input_dims):
        if (self.num_rows, self.num_cols) == input_dims:
            pos_embed = self.pos_embed
        else:
            pos_embed = self.pos_embed.reshape(
                1, self.num_rows, self.num_cols, self.embed_dim
            )[0]

            def window_select(pe):
                if input_dims[0] < pe.shape[0]:
                    pe = pe[: input_dims[0], :, :]
                if input_dims[1] < pe.shape[1]:
                    pe = pe[:, : input_dims[1], :]
                return pe

            if self.cpe_mode:
                max_dim = max(input_dims)
                pos_embed = resize_bilinear(
                    pos_embed,
                    (max_dim, max_dim),
                    align_corners=False,
                    antialias=False,
                )
                pos_embed = window_select(pos_embed)
            else:
                pos_embed = window_select(pos_embed)

            if pos_embed.shape[:2] != input_dims:
                pos_embed = resize_bilinear(
                    pos_embed,
                    input_dims,
                    align_corners=False,
                    antialias=False,
                )
            pos_embed = pos_embed.reshape(1, input_dims[0] * input_dims[1], -1)

        return mx.broadcast_to(
            pos_embed, (batch_size, pos_embed.shape[1], pos_embed.shape[2])
        )

    def __call__(self, x, use_video_embedder=False):
        patches = self._im_to_patches(x)
        patches = (
            self.video_embedder(patches)
            if use_video_embedder
            else self.embedder(patches)
        )
        input_dims = (x.shape[-2] // self.patch_size, x.shape[-1] // self.patch_size)
        patches = patches + self._get_pos_embeddings(x.shape[0], input_dims).astype(
            patches.dtype
        )
        return self.cls_token(patches)


class Attention(nn.Module):
    def __init__(self, dim: int, num_heads: int):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim**-0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=True)
        self.proj = nn.Linear(dim, dim, bias=True)

    def __call__(self, x):
        batch, length, dim = x.shape
        qkv = self.qkv(x).reshape(batch, length, 3, self.num_heads, self.head_dim)
        qkv = qkv.transpose(2, 0, 3, 1, 4)
        queries, keys, values = qkv[0], qkv[1], qkv[2]
        out = mx.fast.scaled_dot_product_attention(
            queries, keys, values, scale=self.scale
        )
        out = out.transpose(0, 2, 1, 3).reshape(batch, length, dim)
        return self.proj(out)


class MLP(nn.Module):
    def __init__(self, dim: int, hidden_dim: int):
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden_dim, bias=True)
        self.fc2 = nn.Linear(hidden_dim, dim, bias=True)

    def __call__(self, x):
        return self.fc2(nn.gelu(self.fc1(x)))


class Block(nn.Module):
    def __init__(self, dim: int, num_heads: int, mlp_hidden_dim: int):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, eps=1e-6)
        self.attn = Attention(dim, num_heads)
        self.norm2 = nn.LayerNorm(dim, eps=1e-6)
        self.mlp = MLP(dim, mlp_hidden_dim)

    def __call__(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class RadioBackbone(nn.Module):
    def __init__(self, config: VisionConfig):
        super().__init__()
        self.embed_dim = config.hidden_size
        self.patch_generator = ViTPatchGenerator(config)
        self.blocks = [
            Block(
                self.embed_dim,
                num_heads=config.num_attention_heads,
                mlp_hidden_dim=config.intermediate_size,
            )
            for _ in range(config.num_hidden_layers)
        ]

    def forward_features(self, x, use_video_embedder=False):
        x = self.patch_generator(x, use_video_embedder=use_video_embedder)
        for block in self.blocks:
            x = block(x)
        return x


class RadioModel(nn.Module):
    def __init__(self, config: VisionConfig):
        super().__init__()
        self.input_conditioner = InputConditioner()
        self.model = RadioBackbone(config)
        self.patch_size = config.patch_size

    def __call__(self, x, use_video_embedder=False):
        y = self.model.forward_features(x, use_video_embedder=use_video_embedder)
        patch_generator = self.model.patch_generator
        all_summary = y[:, : patch_generator.num_cls_tokens]
        all_features = y[:, patch_generator.num_skip :]
        return RadioOutput(all_summary.reshape(all_summary.shape[0], -1), all_features)


class VisionModel(nn.Module):
    def __init__(self, config: VisionConfig):
        super().__init__()
        self.config = config
        self.radio_model = RadioModel(config)

    def __call__(self, pixel_values, use_video_embedder=False):
        return self.radio_model(pixel_values, use_video_embedder=use_video_embedder)
