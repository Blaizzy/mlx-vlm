"""FLUX 3 Image Klein's independent early streams and joint single stream."""

import math

import mlx.core as mx
from mlx import nn

from ..flux2.transformer.pos_embed import Flux2PosEmbed
from .config import Flux3Config

MODES = ("image", "image_cond", "text")


def timestep_embedding(t: mx.array, dim: int = 256) -> mx.array:
    half = dim // 2
    freqs = mx.exp(-math.log(10000) * mx.arange(half, dtype=mx.float32) / half)
    # BFL multiplies in the timestep dtype before forming float32 phases.
    # In particular BF16 timesteps must round the scaled value first.
    scaled_t = (1000 * t).astype(t.dtype).astype(mx.float32)
    args = scaled_t[..., None] * freqs
    return mx.concatenate([mx.cos(args), mx.sin(args)], axis=-1).astype(t.dtype)


def modulate(x, scale, shift):
    # Reference layer norm and modulation run in the stream's activation dtype.
    return (1 + scale) * mx.fast.layer_norm(x, None, None, 1e-6) + shift


class MLPEmbedder(nn.Module):
    def __init__(self, hidden):
        super().__init__()
        self.in_layer = nn.Linear(256, hidden, bias=False)
        self.out_layer = nn.Linear(hidden, hidden, bias=False)

    def __call__(self, x):
        return self.out_layer(nn.silu(self.in_layer(x)))


class ModeModulation(nn.Module):
    def __init__(self, hidden):
        super().__init__()
        self.lin = nn.Linear(hidden, 3 * hidden, bias=False)

    def __call__(self, vec):
        return tuple(mx.split(self.lin(nn.silu(vec)), 3, axis=-1))


class RMSNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.scale = mx.ones((dim,))


class QKNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.query_norm = RMSNorm(dim)
        self.key_norm = RMSNorm(dim)

    def __call__(self, q, k, freqs):
        return (
            self._apply(q, self.query_norm.scale, freqs),
            self._apply(k, self.key_norm.scale, freqs),
        )

    @staticmethod
    def _apply(x, weight, freqs):
        # Normalize and rotate together in float32, then round once to BF16.
        y = mx.fast.rms_norm(x.astype(mx.float32), weight.astype(mx.float32), 1e-6)
        even, odd = y[..., 0::2], y[..., 1::2]
        cos, sin = (f[:, None] for f in freqs)
        return (
            mx.stack([even * cos - odd * sin, even * sin + odd * cos], axis=-1)
            .reshape(x.shape)
            .astype(x.dtype)
        )


class Block(nn.Module):
    def __init__(self, config: Flux3Config):
        super().__init__()
        hidden = config.hidden
        self.heads = config.heads
        self.q_proj = nn.Linear(hidden, hidden, bias=False)
        self.k_proj = nn.Linear(hidden, hidden, bias=False)
        self.v_proj = nn.Linear(hidden, hidden, bias=False)
        self.mlp_in = nn.Linear(hidden, 2 * int(hidden * config.mlp_ratio), bias=False)
        self.attn_out = nn.Linear(hidden, hidden, bias=False)
        self.mlp_out = nn.Linear(int(hidden * config.mlp_ratio), hidden, bias=False)
        self.norm = QKNorm(hidden // config.heads)

    def __call__(self, x, freqs, modulation):
        shift, scale, gate = modulation
        h = modulate(x, scale, shift)
        b, length, hidden = h.shape

        def split(t):
            return t.reshape(b, length, self.heads, -1).transpose(0, 2, 1, 3)

        q, k = self.norm(split(self.q_proj(h)), split(self.k_proj(h)), freqs)
        v = split(self.v_proj(h))
        attn = mx.fast.scaled_dot_product_attention(q, k, v, scale=q.shape[-1] ** -0.5)
        attn = attn.transpose(0, 2, 1, 3).reshape(b, length, hidden)
        gate_in, up = mx.split(self.mlp_in(h), 2, axis=-1)
        return x + gate * (self.attn_out(attn) + self.mlp_out(nn.silu(gate_in) * up))


class LastLayer(nn.Module):
    def __init__(self, hidden, channels):
        super().__init__()
        self.linear = nn.Linear(hidden, channels, bias=False)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(), nn.Linear(hidden, 2 * hidden, bias=False)
        )

    def __call__(self, x, vec):
        shift, scale = mx.split(self.adaLN_modulation(vec), 2, axis=-1)
        return self.linear(modulate(x, scale, shift))


class Flux3Transformer(nn.Module):
    def __init__(self, config: Flux3Config):
        super().__init__()
        self.config = config
        self.emb_in = {
            m: nn.Linear(config.in_channels[m], config.hidden, bias=False)
            for m in MODES[:2]
        }
        self.txt_in = nn.Linear(config.in_channels["text"], config.hidden, bias=False)
        self.time_in = MLPEmbedder(config.hidden)
        if config.flow_map:
            self.ref_time_in = MLPEmbedder(config.hidden)
        self.early_stream_modulations = {
            m: ModeModulation(config.hidden) for m in ("image", "image_cond", "txt")
        }
        self.single_stream_modulations = {
            m: ModeModulation(config.hidden) for m in ("image", "image_cond", "txt")
        }
        self.content_mode_blocks = {
            m: [Block(config) for _ in range(config.depth)] for m in MODES[:2]
        }
        self.txt_mode_blocks = [Block(config) for _ in range(config.depth)]
        self.single_blocks = [Block(config) for _ in range(config.depth_single)]
        self.final_layer = {
            "image": LastLayer(config.hidden, config.in_channels["image"])
        }
        self.pos_embed = Flux2PosEmbed(theta=config.theta, axes_dim=config.axes_dim)

    def __call__(self, data, ids, timestep, *, target_timestep=None):
        if (target_timestep is not None) != self.config.flow_map:
            raise ValueError(
                "target_timestep is required only for a Flux3 flow-map checkpoint"
            )
        active = [m for m in MODES if m in data and data[m].shape[1] > 0]
        dtype = self.txt_in.weight.dtype
        batch = data["image"].shape[0]
        ref = None
        if target_timestep is not None:
            ref = self.ref_time_in(
                timestep_embedding(mx.full((batch, 1), target_timestep, dtype=dtype))
            )
        x, vec, early, single, freqs = {}, {}, {}, {}, {}
        for mode in active:
            t = timestep if mode == "image" else 0.0
            vec[mode] = self.time_in(
                timestep_embedding(mx.full((batch, 1), t, dtype=dtype))
            )
            if ref is not None:
                vec[mode] = vec[mode] + ref
            key = "txt" if mode == "text" else mode
            early[mode] = self.early_stream_modulations[key](vec[mode])
            single[mode] = self.single_stream_modulations[key](vec[mode])
            proj = self.txt_in if mode == "text" else self.emb_in[mode]
            x[mode] = proj(data[mode].astype(dtype))
            freqs[mode] = self.pos_embed(ids[mode])
            blocks = (
                self.txt_mode_blocks
                if mode == "text"
                else self.content_mode_blocks[mode]
            )
            for block in blocks:
                x[mode] = block(x[mode], freqs[mode], early[mode])
        h = mx.concatenate([x[m] for m in active], axis=1)
        joint_freqs = tuple(
            mx.concatenate([freqs[m][i] for m in active], axis=1) for i in range(2)
        )
        mods = tuple(
            mx.concatenate(
                [mx.broadcast_to(single[m][i], x[m].shape) for m in active], axis=1
            )
            for i in range(3)
        )
        for block in self.single_blocks:
            h = block(h, joint_freqs, mods)
        return self.final_layer["image"](h[:, : data["image"].shape[1]], vec["image"])
