"""Images to prefix embeddings: LFM2-VL tiling, the SigLIP2 NaFlex tower and a
2x2 pixel-unshuffle projector, as in the reference ``vision.py``."""

import math

import mlx.core as mx
import mlx.nn as nn
import numpy as np

from ..lfm2_vl.lfm2_vl import PixelUnshuffleBlock
from ..lfm2_vl.vision import VisionModel

TILE, PATCH, MAX_PATCHES = 512, 16, 1024


def layout(width, height):
    """LFM2-VL's smart resize, tile grid and thumbnail (64 to 256 tokens per crop)."""
    if min(width, height) < 1:
        raise ValueError("empty image")
    factor, maximum, minimum = 32, 256 * 1024, 64 * 1024
    h = max(factor, round(height / factor) * factor)
    w = max(factor, round(width / factor) * factor)
    if h * w > maximum:
        beta = math.sqrt(height * width / maximum)
        h = max(factor, math.floor(height / beta / factor) * factor)
        w = max(factor, math.floor(width / beta / factor) * factor)
    elif h * w < minimum:
        beta = math.sqrt(minimum / (height * width))
        h = math.ceil(height * beta / factor) * factor
        w = math.ceil(width * beta / factor) * factor
    large = (
        max(16, round(height / factor) * factor)
        * max(16, round(width / factor) * factor)
        > maximum * 2
    )
    grid = (1, 1)
    if large:
        ratios = sorted(
            {
                (x, y)
                for n in range(2, 11)
                for x in range(1, n + 1)
                for y in range(1, n + 1)
                if 2 <= x * y <= 10
            },
            key=lambda r: r[0] * r[1],
        )
        best = float("inf")
        for ratio in ratios:
            diff = abs(width / height - ratio[0] / ratio[1])
            if diff < best or (
                diff == best
                and width * height > 0.5 * TILE * TILE * ratio[0] * ratio[1]
            ):
                grid, best = ratio, diff
    return {"grid": grid, "thumbnail": (h, w), "tiled": large}


def _taps(n_in, n_out):
    """torch's uint8 antialiased bilinear taps: triangle filter, int16 fixed point."""
    scale = n_in / n_out
    support = max(scale, 1.0)
    center = scale * (np.arange(n_out) + 0.5)
    lo = np.maximum((center - support + 0.5).astype(np.int64), 0)
    size = np.minimum((center + support + 0.5).astype(np.int64), n_in) - lo
    j = np.arange(size.max())[None]
    x = (j + lo[:, None] - center[:, None] + 0.5) * (1.0 / support)
    w = np.maximum(0.0, 1.0 - np.abs(x))
    w = np.where(j < size[:, None], w, 0.0)
    total = w[:, 0].copy()
    for k in range(1, w.shape[1]):
        total += w[:, k]
    w /= total[:, None]
    bits = 0
    while bits < 22 and int(0.5 + w.max() * (1 << (bits + 1))) < 1 << 15:
        bits += 1
    weights = (0.5 + w * (1 << bits)).astype(np.int64)
    return np.minimum(lo[:, None] + j, n_in - 1), weights, bits


def resize(x, height, width):
    """torchvision ``resize(uint8, BILINEAR, antialias=True)``: width pass, then height."""
    for axis, n in ((1, width), (0, height)):
        if x.shape[axis] == n:
            continue
        index, weights, bits = _taps(x.shape[axis], n)
        shape = (1, -1, 1) if axis == 1 else (-1, 1, 1)
        acc = np.int64(1 << (bits - 1))
        for k in range(index.shape[1]):
            src = np.take(x, index[:, k], axis=axis).astype(np.int64)
            acc = acc + src * weights[:, k].reshape(shape)
        x = np.clip(acc >> bits, 0, 255).astype(np.uint8)
    return x


def preprocess(image):
    """A PIL image -> SigLIP2 NaFlex inputs for its tiles and thumbnail."""
    image = np.asarray(image.convert("RGB"))
    plan = layout(image.shape[1], image.shape[0])
    crops = []
    if plan["tiled"]:
        gw, gh = plan["grid"]
        big = resize(image, gh * TILE, gw * TILE)
        crops = [
            big[r * TILE : (r + 1) * TILE, c * TILE : (c + 1) * TILE]
            for r in range(gh)
            for c in range(gw)
        ]
    crops.append(resize(image, *plan["thumbnail"]))
    pixels = np.zeros((len(crops), MAX_PATCHES, 3 * PATCH * PATCH), np.float32)
    shapes, masks = [], np.zeros((len(crops), MAX_PATCHES), np.int32)
    for i, crop in enumerate(crops):
        crop = (crop.astype(np.float32) - 127.5) / 127.5
        ph, pw = crop.shape[0] // PATCH, crop.shape[1] // PATCH
        patches = crop.reshape(ph, PATCH, pw, PATCH, 3).transpose(0, 2, 1, 3, 4)
        pixels[i, : ph * pw] = patches.reshape(ph * pw, -1)
        masks[i, : ph * pw] = 1
        shapes.append([ph, pw])
    return {
        "pixel_values": pixels,
        "spatial_shapes": np.array(shapes, np.int32),
        "pixel_attention_mask": masks,
    }


class Projector(nn.Module):
    def __init__(self, vision_dim, hidden, out, factor=2):
        super().__init__()
        self.pixel_unshuffle = PixelUnshuffleBlock(factor)
        self.linear_1 = nn.Linear(vision_dim * factor**2, hidden)
        self.linear_2 = nn.Linear(hidden, out)

    def __call__(self, x):
        x = self.linear_2(nn.gelu(self.linear_1(self.pixel_unshuffle(x))))
        return x.reshape(-1, x.shape[-1])


class Vision(nn.Module):
    def __init__(self, config, projector_hidden, out):
        super().__init__()
        self.tower = VisionModel(config)
        self.projector = Projector(config.hidden_size, projector_hidden, out)

    def __call__(self, images):
        """PIL images -> (P, D) prefix: every image's tiles then thumbnail, in order."""
        out = []
        for image in images:
            inputs = preprocess(image)
            shapes = inputs["spatial_shapes"].tolist()
            *_, hidden = self.tower(
                mx.array(inputs["pixel_values"]),
                spatial_shapes=mx.array(inputs["spatial_shapes"]),
                pixel_attention_mask=mx.array(inputs["pixel_attention_mask"]),
            )
            for i, (h, w) in enumerate(shapes):
                out.append(
                    self.projector(hidden[i : i + 1, : h * w].reshape(1, h, w, -1))
                )
        return mx.concatenate(out)
