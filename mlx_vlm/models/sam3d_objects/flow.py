"""Structure/pose and sparse-latent flow networks."""

import mlx.core as mx
import mlx.nn as nn

from .layers import (
    LayerNorm,
    ModulatedBlock,
    Sequential,
    TimestepEmbedder,
    layer_norm,
    modulate,
    position_embedding,
)
from .sparse import SparseConv, pool

LATENTS = {
    "6drotation_normalized": 6,
    "scale": 3,
    "shape": 8,
    "translation": 3,
    "translation_scale": 1,
}
POSE_NAMES = ("6drotation_normalized", "translation", "scale", "translation_scale")


class Latent(nn.Module):
    def __init__(self, input_channels, channels, length):
        super().__init__()
        self.input_layer = nn.Linear(input_channels, channels)
        self.out_layer = nn.Linear(channels, input_channels)
        self.pos_emb = mx.zeros((length, channels))

    def encode(self, x):
        return self.input_layer(x) + self.pos_emb[None]

    def decode(self, x):
        return self.out_layer(layer_norm(x))


class StructureFlow(nn.Module):
    def __init__(self, config):
        super().__init__()
        c = config.hidden_size
        self.d_embedder = TimestepEmbedder(c)
        self.t_embedder = TimestepEmbedder(c)
        self.latent_mapping = {
            n: Latent(ch, c, config.latent_resolution**3 if n == "shape" else 1)
            for n, ch in LATENTS.items()
        }
        self.blocks = [
            ModulatedBlock(
                c,
                config.num_heads,
                config.cond_channels,
                names=("shape", "6drotation_normalized"),
            )
            for _ in range(config.num_blocks)
        ]

    def prepare_condition(self, tokens):
        return [
            {name: attn.prepare(tokens) for name, attn in block.cross_attn.items()}
            for block in self.blocks
        ]

    def __call__(self, latents, time, condition, delta=None):
        return self._forward(latents, time, [condition], delta)[0]

    def guided(self, latents, time, condition, uncondition, delta=None):
        return self._forward(latents, time, [condition, uncondition], delta)

    def _forward(self, latents, time, conditions, delta):
        emb = self.t_embedder(time)
        if delta is not None:
            emb = emb + self.d_embedder(delta)
        x = {
            n: m.encode(latents[n].astype(m.input_layer.weight.dtype))
            for n, m in self.latent_mapping.items()
        }
        encoded = {
            "6drotation_normalized": mx.concatenate([x[n] for n in POSE_NAMES], axis=1),
            "shape": x["shape"],
        }
        outputs = []
        for condition in conditions:
            if isinstance(condition, mx.array):
                condition = [condition] * len(self.blocks)
            h = encoded
            for block, context in zip(self.blocks, condition):
                h = block(h, emb, context)
            velocity = {
                n: self.latent_mapping[n].decode(
                    h["6drotation_normalized"][:, i : i + 1]
                )
                for i, n in enumerate(POSE_NAMES)
            }
            velocity["shape"] = self.latent_mapping["shape"].decode(h["shape"])
            outputs.append(velocity)
        return outputs


class SparseResBlock(nn.Module):
    def __init__(self, input_channels, channels, time_channels):
        super().__init__()
        self.norm1 = LayerNorm(input_channels, eps=1e-6)
        self.conv1 = SparseConv(input_channels, channels)
        self.conv2 = SparseConv(channels, channels)
        self.emb_layers = Sequential(nn.SiLU(), nn.Linear(time_channels, 2 * channels))
        self.skip_connection = (
            nn.Linear(input_channels, channels)
            if channels != input_channels
            else nn.Identity()
        )

    def __call__(self, x, grid, time, parent=None):
        scale, shift = mx.split(self.emb_layers(time), 2, axis=-1)
        h = nn.silu(self.norm1(x))
        if parent is None:
            h = self.conv1(h, grid)
        else:
            h = self.conv1.from_parents(h, parent, grid)
        h = modulate(layer_norm(h, 1e-6), scale, shift)
        skip = self.skip_connection(x)
        if parent is not None:
            skip = skip[parent]
        return self.conv2(nn.silu(h), grid) + skip


class LatentFlow(nn.Module):
    def __init__(self, config):
        super().__init__()
        c, io = config.hidden_size, config.io_channels
        self.t_embedder = TimestepEmbedder(c)
        self.input_layer = nn.Linear(config.latent_channels, io)
        self.input_blocks = [SparseResBlock(io, io, c), SparseResBlock(io, c, c)]
        self.blocks = [
            ModulatedBlock(c, config.num_heads, config.cond_channels)
            for _ in range(config.num_blocks)
        ]
        self.out_blocks = [SparseResBlock(2 * c, io, c), SparseResBlock(2 * io, io, c)]
        self.out_layer = nn.Linear(io, config.latent_channels)

    def prepare_condition(self, tokens):
        return [block.cross_attn.prepare(tokens) for block in self.blocks]

    def __call__(self, x, grid, time, condition):
        return self._forward(x, grid, time, [condition])[0]

    def guided(self, x, grid, time, condition, uncondition):
        return self._forward(x, grid, time, [condition, uncondition])

    def _forward(self, x, grid, time, conditions):
        x = x.astype(self.input_layer.weight.dtype)
        h = self.input_layer(x)
        emb = self.t_embedder(time)
        skip = self.input_blocks[0](h, grid, emb)
        h, child, parent = pool(skip, grid)
        h = self.input_blocks[1](h, child, emb)
        skip_low = h
        stem = (
            h + position_embedding(child.coords[:, 1:], h.shape[-1]).astype(h.dtype)
        )[None]
        outputs = []
        for condition in conditions:
            if isinstance(condition, mx.array):
                condition = [condition] * len(self.blocks)
            h = stem
            for block, context in zip(self.blocks, condition):
                h = block(h, emb, context)
            h = mx.concatenate([h[0], skip_low], axis=-1)
            h = self.out_blocks[0](h, grid, emb, parent=parent)
            h = self.out_blocks[1](mx.concatenate([h, skip], axis=-1), grid, emb)
            outputs.append(self.out_layer(layer_norm(h)))
        return outputs
