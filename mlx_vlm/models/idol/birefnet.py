"""Native MLX BiRefNet, adapted from speridlabs/idol-sdk (Apache-2.0).

All spatial tensors are NHWC. Deformable convolution uses bilinear gathers,
so background removal does not require torchvision or a CUDA extension.
"""

import math

import mlx.core as mx
import mlx.nn as nn
import numpy as np
from PIL import Image

from .layers import sanitize_weights


def resize(x, size):
    """Bilinear resize with PyTorch's align_corners=True convention."""
    h, w = size
    if x.shape[1:3] == (h, w):
        return x
    ys = mx.linspace(0, x.shape[1] - 1, h) if h > 1 else mx.zeros((1,))
    xs = mx.linspace(0, x.shape[2] - 1, w) if w > 1 else mx.zeros((1,))
    y0, x0 = mx.floor(ys).astype(mx.int32), mx.floor(xs).astype(mx.int32)
    y1, x1 = mx.minimum(y0 + 1, x.shape[1] - 1), mx.minimum(x0 + 1, x.shape[2] - 1)
    wy, wx = (ys - y0).astype(x.dtype)[None, :, None, None], (xs - x0).astype(x.dtype)[
        None, None, :, None
    ]
    out = x[:, y0] * (1 - wy) + x[:, y1] * wy
    return out[:, :, x0] * (1 - wx) + out[:, :, x1] * wx


def _window_partition(x, size):
    b, h, w, c = x.shape
    return (
        x.reshape(b, h // size, size, w // size, size, c)
        .transpose(0, 1, 3, 2, 4, 5)
        .reshape(-1, size, size, c)
    )


def _window_reverse(x, size, h, w):
    return (
        x.reshape(-1, h // size, w // size, size, size, x.shape[-1])
        .transpose(0, 1, 3, 2, 4, 5)
        .reshape(-1, h, w, x.shape[-1])
    )


class _SwinMlp(nn.Module):
    def __init__(self, channels, hidden):
        super().__init__()
        self.fc1 = nn.Linear(channels, hidden)
        self.fc2 = nn.Linear(hidden, channels)

    def __call__(self, x):
        return self.fc2(nn.gelu(self.fc1(x)))


class _WindowAttention(nn.Module):
    def __init__(self, dim, window_size, num_heads):
        super().__init__()
        self.num_heads, self.window_size = num_heads, window_size
        self.relative_position_bias_table = mx.zeros(
            ((2 * window_size - 1) ** 2, num_heads)
        )
        y, x = np.meshgrid(
            np.arange(window_size), np.arange(window_size), indexing="ij"
        )
        coords = np.stack([y.ravel(), x.ravel()], axis=-1)
        relative = coords[:, None] - coords[None] + window_size - 1
        self.relative_position_index = mx.array(
            relative[..., 0] * (2 * window_size - 1) + relative[..., 1], dtype=mx.int32
        )
        self.qkv = nn.Linear(dim, 3 * dim)
        self.proj = nn.Linear(dim, dim)

    def __call__(self, x, mask=None):
        b, n, c = x.shape
        qkv = self.qkv(x).reshape(b, n, 3, self.num_heads, -1).transpose(2, 0, 3, 1, 4)
        bias = (
            self.relative_position_bias_table[self.relative_position_index.reshape(-1)]
            .reshape(n, n, self.num_heads)
            .transpose(2, 0, 1)
        )
        if mask is not None:
            nw = mask.shape[0]
            bias = (bias[None, None] + mask[None, :, None]).astype(x.dtype)
            bias = mx.broadcast_to(bias, (b // nw, nw, self.num_heads, n, n)).reshape(
                b, self.num_heads, n, n
            )
        else:
            bias = bias.astype(x.dtype)
        h = mx.fast.scaled_dot_product_attention(
            qkv[0], qkv[1], qkv[2], scale=(c // self.num_heads) ** -0.5, mask=bias
        )
        return self.proj(h.transpose(0, 2, 1, 3).reshape(b, n, c))


class _SwinBlock(nn.Module):
    def __init__(self, dim, num_heads, window_size, shift_size):
        super().__init__()
        self.window_size, self.shift_size = window_size, shift_size
        self.norm1, self.norm2 = nn.LayerNorm(dim), nn.LayerNorm(dim)
        self.attn = _WindowAttention(dim, window_size, num_heads)
        self.mlp = _SwinMlp(dim, 4 * dim)

    def __call__(self, x, mask):
        b, h, w, c = x.shape
        shortcut = x
        size, shift = self.window_size, self.shift_size
        x = mx.pad(self.norm1(x), [(0, 0), (0, -h % size), (0, -w % size), (0, 0)])
        hp, wp = x.shape[1:3]
        if shift:
            x = mx.roll(x, (-shift, -shift), axis=(1, 2))
        windows = _window_partition(x, size).reshape(-1, size * size, c)
        x = self.attn(windows, mask if shift else None)
        x = _window_reverse(x.reshape(-1, size, size, c), size, hp, wp)
        if shift:
            x = mx.roll(x, (shift, shift), axis=(1, 2))
        x = shortcut + x[:, :h, :w]
        return x + self.mlp(self.norm2(x))


class _PatchMerging(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.norm = nn.LayerNorm(4 * dim)
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)

    def __call__(self, x):
        x = mx.pad(x, [(0, 0), (0, x.shape[1] % 2), (0, x.shape[2] % 2), (0, 0)])
        return self.reduction(
            self.norm(
                mx.concatenate(
                    [
                        x[:, 0::2, 0::2],
                        x[:, 1::2, 0::2],
                        x[:, 0::2, 1::2],
                        x[:, 1::2, 1::2],
                    ],
                    axis=-1,
                )
            )
        )


class _SwinBasicLayer(nn.Module):
    def __init__(self, dim, depth, heads, size=12, downsample=True):
        super().__init__()
        self.size = size
        self.blocks = [
            _SwinBlock(dim, heads, size, size // 2 if i % 2 else 0)
            for i in range(depth)
        ]
        self.downsample = _PatchMerging(dim) if downsample else None

    def __call__(self, x):
        h, w = x.shape[1:3]
        s, shift = self.size, self.size // 2
        hp, wp = math.ceil(h / s) * s, math.ceil(w / s) * s
        labels = np.zeros((1, hp, wp, 1), dtype=np.float32)
        for i, sy in enumerate((slice(0, -s), slice(-s, -shift), slice(-shift, None))):
            for j, sx in enumerate(
                (slice(0, -s), slice(-s, -shift), slice(-shift, None))
            ):
                labels[:, sy, sx] = 3 * i + j
        windows = _window_partition(mx.array(labels), s).reshape(-1, s * s)
        mask = mx.where(windows[:, :, None] == windows[:, None, :], 0.0, -100.0).astype(
            x.dtype
        )
        for block in self.blocks:
            x = block(x, mask)
        return x, self.downsample(x) if self.downsample else x


class _SwinPatchEmbed(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.proj = nn.Conv2d(3, dim, 4, stride=4)
        self.norm = nn.LayerNorm(dim)

    def __call__(self, x):
        return self.norm(self.proj(x))


class _SwinLarge(nn.Module):
    def __init__(self):
        super().__init__()
        self.patch_embed = _SwinPatchEmbed(192)
        self.layers = [
            _SwinBasicLayer(192 * 2**i, depth, 6 * 2**i, downsample=i < 3)
            for i, depth in enumerate((2, 2, 18, 2))
        ]
        for i in range(4):
            setattr(self, f"norm{i}", nn.LayerNorm(192 * 2**i))

    def __call__(self, x):
        x = self.patch_embed(x)
        outputs = []
        for i, layer in enumerate(self.layers):
            out, x = layer(x)
            outputs.append(getattr(self, f"norm{i}")(out))
            mx.eval(x, outputs[-1])
        return outputs


def bilinear_sample(x, y, xx):
    """Sample NHWC with zero padding, including partial out-of-bounds pixels."""
    b, h, w, c = x.shape
    y0, x0 = mx.floor(y).astype(mx.int32), mx.floor(xx).astype(mx.int32)
    dy, dx = (y - y0).astype(x.dtype), (xx - x0).astype(x.dtype)
    batch = mx.arange(b).reshape(b, *([1] * (y.ndim - 1))) * (h * w)

    def gather(yi, xi):
        valid = (yi >= 0) & (yi < h) & (xi >= 0) & (xi < w)
        indices = batch + mx.clip(yi, 0, h - 1) * w + mx.clip(xi, 0, w - 1)
        return x.reshape(-1, c)[indices] * valid[..., None]

    return (
        gather(y0, x0) * ((1 - dy) * (1 - dx))[..., None]
        + gather(y0 + 1, x0) * (dy * (1 - dx))[..., None]
        + gather(y0, x0 + 1) * ((1 - dy) * dx)[..., None]
        + gather(y0 + 1, x0 + 1) * (dy * dx)[..., None]
    )


class _DeformableConv2d(nn.Module):
    def __init__(
        self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False
    ):
        super().__init__()
        self.stride, self.padding, self.kernel_size = stride, padding, kernel_size
        args = dict(kernel_size=kernel_size, stride=stride, padding=padding)
        self.offset_conv = nn.Conv2d(in_channels, 2 * kernel_size**2, **args)
        self.modulator_conv = nn.Conv2d(in_channels, kernel_size**2, **args)
        self.regular_conv = nn.Conv2d(in_channels, out_channels, bias=bias, **args)

    def __call__(self, x):
        offset = self.offset_conv(x)
        modulator = 2 * mx.sigmoid(self.modulator_conv(x))
        b, h, w, _ = offset.shape
        k = self.kernel_size
        offset = offset.reshape(b, h, w, k, k, 2)
        modulator = modulator.reshape(b, h, w, k, k)
        y = (mx.arange(h) * self.stride - self.padding)[None, :, None, None]
        xx = (mx.arange(w) * self.stride - self.padding)[
            None, None, :, None
        ] + mx.arange(k)
        out = mx.zeros((b, h, w, self.regular_conv.weight.shape[0]), dtype=x.dtype)
        # A kernel row at a time bounds temporary memory for the 7x7 branches.
        for row in range(k):
            samples = bilinear_sample(
                x, y + row + offset[..., row, :, 0], xx + offset[..., row, :, 1]
            )
            samples = samples * modulator[..., row, :, None]
            weights = self.regular_conv.weight[:, row].reshape(out.shape[-1], -1)
            out = out + samples.reshape(b, h, w, -1) @ weights.T
            mx.eval(out)
        if "bias" in self.regular_conv:
            out = out + self.regular_conv.bias
        return out


class _ASPPModuleDeformable(nn.Module):
    def __init__(self, channels, planes, kernel_size, padding):
        super().__init__()
        self.atrous_conv = _DeformableConv2d(
            channels, planes, kernel_size, padding=padding
        )
        self.bn = nn.BatchNorm(planes)

    def __call__(self, x):
        return nn.relu(self.bn(self.atrous_conv(x)))


class _GlobalAvgPool(nn.Module):
    def __call__(self, x):
        return mx.mean(x, axis=(1, 2), keepdims=True)


class _ASPPDeformable(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.aspp1 = _ASPPModuleDeformable(channels, 256, 1, 0)
        self.aspp_deforms = [
            _ASPPModuleDeformable(channels, 256, k, k // 2) for k in (1, 3, 7)
        ]
        self.global_avg_pool = nn.Sequential(
            _GlobalAvgPool(),
            nn.Conv2d(channels, 256, 1, bias=False),
            nn.BatchNorm(256),
            nn.ReLU(),
        )
        self.conv1 = nn.Conv2d(256 * 5, channels, 1, bias=False)
        self.bn1 = nn.BatchNorm(channels)

    def __call__(self, x):
        parts = (
            [self.aspp1(x)]
            + [m(x) for m in self.aspp_deforms]
            + [resize(self.global_avg_pool(x), x.shape[1:3])]
        )
        return nn.relu(self.bn1(self.conv1(mx.concatenate(parts, axis=-1))))


class _BasicDecBlk(nn.Module):
    def __init__(self, in_channels, out_channels, inter_channels=64):
        super().__init__()
        self.conv_in = nn.Conv2d(in_channels, inter_channels, 3, padding=1)
        self.bn_in = nn.BatchNorm(inter_channels)
        self.dec_att = _ASPPDeformable(inter_channels)
        self.conv_out = nn.Conv2d(inter_channels, out_channels, 3, padding=1)
        self.bn_out = nn.BatchNorm(out_channels)

    def __call__(self, x):
        return self.bn_out(
            self.conv_out(self.dec_att(nn.relu(self.bn_in(self.conv_in(x)))))
        )


class _BasicLatBlk(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, 1)

    def __call__(self, x):
        return self.conv(x)


class _SimpleConvs(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, 64, 3, padding=1)
        self.conv_out = nn.Conv2d(64, out_channels, 3, padding=1)

    def __call__(self, x):
        return self.conv_out(self.conv1(x))


def _image2patches(image, ref):
    b, hf, wf, c = image.shape
    h, w = ref.shape[1:3]
    hg, wg = hf // h, wf // w
    return (
        image.reshape(b, hg, h, wg, w, c)
        .transpose(0, 2, 4, 5, 1, 3)
        .reshape(b, h, w, c * hg * wg)
    )


class _BiRefNetDecoder(nn.Module):
    def __init__(self):
        super().__init__()
        c = (3072, 1536, 768, 384)
        for i, (inputs, outputs) in enumerate(
            (
                (3, c[3] // 8),
                (48, c[2] // 8),
                (192, c[1] // 8),
                (768, c[0] // 8),
                (3072, c[0] // 8),
            ),
            1,
        ):
            setattr(self, f"ipt_blk{i}", _SimpleConvs(inputs, outputs))
        for i, (inputs, outputs) in enumerate(
            (
                (c[3] + c[2] // 8, c[3] // 2),
                (c[2] + c[1] // 8, c[3]),
                (c[1] + c[0] // 8, c[2]),
                (c[0] + c[0] // 8, c[1]),
            ),
            1,
        ):
            setattr(self, f"decoder_block{i}", _BasicDecBlk(inputs, outputs))
        self.conv_out1 = nn.Sequential(nn.Conv2d(c[3] // 2 + c[3] // 8, 1, 1))
        for i, channels in ((2, c[3]), (3, c[2]), (4, c[1])):
            setattr(self, f"lateral_block{i}", _BasicLatBlk(channels))
            setattr(
                self,
                f"gdt_convs_{i}",
                nn.Sequential(
                    nn.Conv2d(channels, 16, 3, padding=1), nn.BatchNorm(16), nn.ReLU()
                ),
            )
            setattr(self, f"gdt_convs_attn_{i}", nn.Sequential(nn.Conv2d(16, 1, 1)))

    def __call__(self, x, x1, x2, x3, x4):
        h = x4
        for i, skip in ((4, x3), (3, x2), (2, x1)):
            h = mx.concatenate(
                [h, getattr(self, f"ipt_blk{i+1}")(_image2patches(x, h))], axis=-1
            )
            h = getattr(self, f"decoder_block{i}")(h)
            gate = getattr(self, f"gdt_convs_attn_{i}")(
                getattr(self, f"gdt_convs_{i}")(h)
            )
            h = resize(h * mx.sigmoid(gate), skip.shape[1:3]) + getattr(
                self, f"lateral_block{i}"
            )(skip)
            mx.eval(h)
        h = self.decoder_block1(
            mx.concatenate([h, self.ipt_blk2(_image2patches(x, h))], axis=-1)
        )
        h = resize(h, x.shape[1:3])
        return self.conv_out1(mx.concatenate([h, self.ipt_blk1(x)], axis=-1))


class BiRefNet(nn.Module):
    INPUT_SIZE = (1024, 1024)

    def __init__(self):
        super().__init__()
        self.bb = _SwinLarge()
        self.squeeze_module = nn.Sequential(_BasicDecBlk(3072 + 384 + 768 + 1536, 3072))
        self.decoder = _BiRefNetDecoder()

    def sanitize(self, weights):
        return sanitize_weights(self, weights)

    def __call__(self, x):
        full = self.bb(x)
        half = self.bb(resize(x, (x.shape[1] // 2, x.shape[2] // 2)))
        x1, x2, x3, x4 = [
            mx.concatenate([a, resize(b, a.shape[1:3])], axis=-1)
            for a, b in zip(full, half)
        ]
        context = mx.concatenate(
            [resize(a, x4.shape[1:3]) for a in (x1, x2, x3)] + [x4], axis=-1
        )
        return mx.sigmoid(self.decoder(x, x1, x2, x3, self.squeeze_module(context)))

    def remove_background(self, image):
        pixels = mx.array(np.asarray(image.convert("RGB"), dtype=np.float32) / 255)[
            None
        ]
        pixels = resize(pixels, self.INPUT_SIZE)
        pixels = (pixels - mx.array([0.485, 0.456, 0.406])) / mx.array(
            [0.229, 0.224, 0.225]
        )
        alpha = self(pixels.astype(self.bb.patch_embed.proj.weight.dtype))
        alpha = resize(alpha.astype(mx.float32), (image.height, image.width))[
            0, :, :, 0
        ]
        alpha = np.array((mx.clip(alpha, 0, 1) * 255).astype(mx.uint8))
        rgba = image.convert("RGBA")
        rgba.putalpha(Image.fromarray(alpha))
        return rgba
