# Copyright 2026 The HuggingFace Team. All rights reserved.
# Adapted from inclusionAI/LLaDA-Image for MLX.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from __future__ import annotations

import mlx.core as mx
from mlx import nn

from .conditioning import attention


def resize_positions(embedding: mx.array, height: int, width: int) -> mx.array:
    """Bilinear grid_sample with half-pixel coordinates and border padding."""
    size = int(embedding.shape[0] ** 0.5)
    grid = embedding.reshape(size, size, -1).astype(mx.float32)
    y = mx.clip(
        (mx.arange(height, dtype=mx.float32) + 0.5) * size / height - 0.5, 0, size - 1
    )
    x = mx.clip(
        (mx.arange(width, dtype=mx.float32) + 0.5) * size / width - 0.5, 0, size - 1
    )
    y0, x0 = y.astype(mx.int32), x.astype(mx.int32)
    y1, x1 = mx.minimum(y0 + 1, size - 1), mx.minimum(x0 + 1, size - 1)
    dy, dx = (y - y0)[:, None, None], (x - x0)[None, :, None]
    top = (
        grid[y0[:, None], x0[None, :]] * (1 - dx) + grid[y0[:, None], x1[None, :]] * dx
    )
    bottom = (
        grid[y1[:, None], x0[None, :]] * (1 - dx) + grid[y1[:, None], x1[None, :]] * dx
    )
    return ((1 - dy) * top + dy * bottom).reshape(1, height * width, -1)


class SigVQAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        dim = config["hidden_size"]
        self.heads = config["num_attention_heads"]
        self.qkv = nn.Linear(dim, 3 * dim, bias=config.get("attention_bias", True))
        self.proj = nn.Linear(dim, dim, bias=config.get("attention_bias", True))

    def __call__(self, x):
        q, k, v = mx.split(self.qkv(x), 3, axis=-1)
        return self.proj(attention(q, k, v, self.heads))


class SigVQMLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.fc1 = nn.Linear(config["hidden_size"], config["intermediate_size"])
        self.fc2 = nn.Linear(config["intermediate_size"], config["hidden_size"])

    def __call__(self, x):
        return self.fc2(nn.gelu(self.fc1(x)))


class SigVQBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.norm1 = nn.LayerNorm(
            config["hidden_size"], eps=config.get("norm_eps", 1e-6)
        )
        self.norm2 = nn.LayerNorm(
            config["hidden_size"], eps=config.get("norm_eps", 1e-6)
        )
        self.attn = SigVQAttention(config)
        self.mlp = SigVQMLP(config)

    def __call__(self, x):
        x = x + self.attn(self.norm1(x))
        return x + self.mlp(self.norm2(x))


class SigVQ(nn.Module):
    """Image-to-codebook encoder and discrete-token semantic projector."""

    def __init__(self, config: dict, *, include_encoder: bool = True):
        super().__init__()
        self.config = config
        self.include_encoder = include_encoder
        self.patch_size = config["patch_size"]
        if include_encoder:
            self.visual = nn.Module()
            self.visual.patch_embed = nn.Module()
            self.visual.patch_embed.proj = nn.Conv2d(
                config.get("in_channels", 3),
                config["hidden_size"],
                self.patch_size,
                stride=self.patch_size,
            )
            self.visual.embeddings = nn.Module()
            self.visual.embeddings.position_embedding = nn.Embedding(
                (config["image_size"] // self.patch_size) ** 2, config["hidden_size"]
            )
            self.visual.blocks = [
                SigVQBlock(config) for _ in range(config["num_hidden_layers"])
            ]
            self.vqmodel = nn.Module()
            self.vqmodel.quant_conv = nn.Conv2d(
                config["hidden_size"], config["codebook_embed_dim"], 1
            )
            self.vqmodel.quantize = nn.Module()
            self.vqmodel.quantize.embedding = nn.Embedding(
                config["codebook_size"], config["codebook_embed_dim"]
            )
        self.prior_token_embedding = nn.Embedding(
            config["codebook_size"], config["semantic_embed_dim"]
        )
        self.prior_projector = [
            nn.Linear(config["semantic_embed_dim"], config["semantic_embed_dim"])
            for _ in range(2)
        ]

    def encode(self, pixels: mx.array) -> mx.array:
        """Encode RGB NHWC pixels normalized to [-1, 1] into codebook IDs."""
        if not self.include_encoder:
            raise ValueError("SigVQ was loaded without its image encoder")
        if pixels.ndim != 4 or pixels.shape[-1] != 3:
            raise ValueError("SigVQ expects an NHWC RGB image batch")
        if pixels.shape[1] % self.patch_size or pixels.shape[2] % self.patch_size:
            raise ValueError("SigVQ dimensions must be divisible by its patch size")
        hidden = self.visual.patch_embed.proj(pixels)
        batch, height, width, dim = hidden.shape
        hidden = hidden.reshape(batch, height * width, dim)
        positions = resize_positions(
            self.visual.embeddings.position_embedding.weight, height, width
        )
        hidden = hidden + positions.astype(hidden.dtype)
        for block in self.visual.blocks:
            hidden = block(hidden)
        hidden = self.vqmodel.quant_conv(hidden.reshape(batch, height, width, dim))
        hidden = hidden.reshape(batch, height * width, -1)
        codebook = self.vqmodel.quantize.embedding.weight

        def normalize(x):
            norm = mx.sqrt(
                mx.sum(mx.square(x.astype(mx.float32)), axis=-1, keepdims=True)
            ).astype(x.dtype)
            return x / mx.maximum(norm, 1e-12)

        hidden, codebook = normalize(hidden), normalize(codebook)
        distances = (
            mx.sum(hidden**2, axis=-1, keepdims=True)
            + mx.sum(codebook**2, axis=-1)
            - 2 * (hidden @ codebook.T)
        )
        return mx.argmin(distances, axis=-1)

    def __call__(self, *, pixels=None, token_ids=None):
        if (pixels is None) == (token_ids is None):
            raise ValueError("Provide exactly one of pixels or token_ids")
        if pixels is not None:
            token_ids = self.encode(pixels)
        features = self.prior_token_embedding(token_ids)
        return self.prior_projector[1](nn.silu(self.prior_projector[0](features)))


def sanitize_sigvq_weights(weights, *, include_encoder=True):
    result = {}
    for key, value in weights.items():
        if not include_encoder and not key.startswith("prior_"):
            continue
        key = key.replace("prior_projector.net.0.proj.", "prior_projector.0.")
        key = key.replace("prior_projector.net.2.", "prior_projector.1.")
        if value.ndim == 4:
            value = value.transpose(0, 2, 3, 1)
        result[key] = value
    return result
