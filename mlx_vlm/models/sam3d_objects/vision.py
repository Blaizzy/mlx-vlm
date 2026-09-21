"""Image and point-map condition encoders on the shared DINOv2 backbone."""

import mlx.core as mx
import mlx.nn as nn

from ..dinov2 import DINOv2
from ..dinov2 import ModelConfig as DINOv2Config
from ..dinov2.dinov2 import Attention, Mlp
from ..interpolate import resize_bilinear_nhwc, resize_nearest_nhwc
from .layers import LayerNorm, Sequential, layer_norm


def dinov2_backbone(hidden_size, layers, heads, image_size, *, register_tokens=0):
    backbone = DINOv2(
        DINOv2Config(
            hidden_size=hidden_size,
            num_hidden_layers=layers,
            num_attention_heads=heads,
            image_size=image_size,
            patch_size=14,
            num_register_tokens=register_tokens,
            interpolate_offset=0.1,
        )
    )
    for module in (backbone, *backbone.blocks):
        for name in ("norm", "norm1", "norm2"):
            if name in module:
                norm = module[name]
                setattr(module, name, LayerNorm(norm.weight.shape[0], eps=norm.eps))
    return backbone


class Dino(nn.Module):
    def __init__(self, config, prenorm=False):
        super().__init__()
        self.backbone = dinov2_backbone(
            config.dino_hidden_size,
            config.dino_layers,
            config.dino_heads,
            config.dino_image_size,
            register_tokens=4,
        )
        self._size = config.image_size
        self._prenorm = prenorm

    def trunk(self, x):
        x = self.backbone.prepare_tokens(x)
        for block in self.backbone.blocks:
            x = block(x)
        return x

    def head(self, x):
        if self._prenorm:
            return layer_norm(x)
        x = self.backbone.norm(x)
        registers = self.backbone.num_register_tokens
        return mx.concatenate([x[:, :1], x[:, 1 + registers :]], axis=1)

    def __call__(self, image, cache=None, key=None):
        if cache is not None and key in cache:
            return self.head(cache[key])
        x = resize_bilinear_nhwc(image, (self._size, self._size))
        if x.shape[-1] == 1:
            x = mx.repeat(x, 3, axis=-1)
        x = (x - mx.array([0.485, 0.456, 0.406])) / mx.array([0.229, 0.224, 0.225])
        x = self.trunk(x.astype(self.backbone.cls_token.dtype))
        if cache is not None:
            cache[key] = x
        return self.head(x)


class PointBlock(nn.Module):
    def __init__(self, channels, heads):
        super().__init__()
        self.norm1 = LayerNorm(channels, eps=1e-6)
        self.attn = Attention(channels, heads)
        self.norm2 = LayerNorm(channels, eps=1e-6)
        self.mlp = Mlp(channels, 2 * channels)

    def __call__(self, x):
        x = x + self.attn(self.norm1(x))
        return x + self.mlp(self.norm2(x))


class PointPatchEmbed(nn.Module):
    def __init__(self, config):
        super().__init__()
        c, p, s = config.point_channels, config.point_patch_size, config.point_size
        self._patch, self._size = p, s
        self.point_proj = nn.Linear(3, c)
        self.invalid_xyz_token = mx.zeros(c)
        self.pos_embed = mx.zeros((1, c, s // p, s // p))
        self.pos_embed_window = mx.zeros((1, 1 + p * p, c))
        self.cls_token = mx.zeros((1, 1, c))
        self.blocks = [PointBlock(c, config.point_heads)]

    def __call__(self, points):
        points = resize_nearest_nhwc(points, (self._size, self._size))
        valid = mx.all(mx.isfinite(points), axis=-1, keepdims=True)
        x = self.point_proj(
            mx.where(valid, points, 0).astype(self.point_proj.weight.dtype)
        )
        x = mx.where(valid, x, self.invalid_xyz_token)
        b, h, w, c = x.shape
        p = self._patch
        x = (
            x.reshape(b, h // p, p, w // p, p, c)
            .transpose(0, 1, 3, 2, 4, 5)
            .reshape(-1, p * p, c)
        )
        x = (
            mx.concatenate(
                [mx.broadcast_to(self.cls_token, (x.shape[0], 1, c)), x], axis=1
            )
            + self.pos_embed_window
        )
        for block in self.blocks:
            x = block(x)
        return x[:, 0].reshape(b, -1, c) + self.pos_embed.transpose(0, 2, 3, 1).reshape(
            1, -1, c
        )


class ProjectionFFN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        hidden = ((int(8 * output_dim / 3) + 255) // 256) * 256
        self.w1 = nn.Linear(input_dim, hidden, bias=False)
        self.w2 = nn.Linear(hidden, output_dim, bias=False)
        self.w3 = nn.Linear(input_dim, hidden, bias=False)

    def __call__(self, x):
        return self.w2(nn.silu(self.w1(x)) * self.w3(x))


class ConditionEncoder(nn.Module):
    def __init__(self, config, *, pointmap=False):
        super().__init__()
        self.module_list = [
            Dino(config, prenorm=not pointmap),
            Dino(config, prenorm=not pointmap),
        ]
        dims = [config.dino_hidden_size] * 2
        if pointmap:
            self.module_list.append(PointPatchEmbed(config))
            dims.append(config.point_channels)
        self.idx_emb = mx.zeros((3, config.cond_channels))
        self.projection_nets = [
            Sequential(LayerNorm(d), ProjectionFFN(d, config.cond_channels))
            for d in dims
        ]

    def __call__(self, inputs, cache=None):
        groups = [
            ("image", "rgb_image"),
            ("mask", "rgb_image_mask"),
            ("pointmap", "rgb_pointmap"),
        ]
        tokens = []
        for i, module in enumerate(self.module_list):
            for pos, name in enumerate(groups[i]):
                if i == 2 and inputs.get(name) is None:
                    # Match the trained condition dropout: zero complete tokens,
                    # including their learned positional embedding.
                    count = (module._size // module._patch) ** 2
                    tokens.append(
                        mx.zeros(
                            (inputs["image"].shape[0], count, self.idx_emb.shape[-1]),
                            self.idx_emb.dtype,
                        )
                    )
                    continue
                if i < 2:
                    x = module(inputs[name], cache=cache, key=name)
                else:
                    x = module(inputs[name])
                x = self.projection_nets[i](x)
                tokens.append(x + self.idx_emb[pos : pos + 1, None])
        return mx.concatenate(tokens, axis=1)
