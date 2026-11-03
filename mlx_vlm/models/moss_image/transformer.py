"""MLX port of OpenMOSS's dense MOSS-Image diffusion transformer.

Reference: OpenMOSS-Team/MOSS-Image-SFT/pipeline_moss_image.py.
"""

from __future__ import annotations

import math

import mlx.core as mx
from mlx import nn

from .config import MOSSImageTransformerConfig


def validate_skip_layers(layers, depth, name="skip_layers"):
    layers = () if layers is None else tuple(layers)
    if any(type(index) is not int or not 0 <= index < depth for index in layers):
        raise ValueError(f"{name} must contain integer block indices in [0, {depth})")
    return frozenset(layers)


def situ(hidden, gate):
    return (4 * mx.tanh(gate / 4) * mx.sigmoid(gate)) * (25 * mx.tanh(hidden / 25))


def rotary_positions(height, width, text_length, head_dim, base=10000.0):
    # Four axes (r, f, h, w), with centered image coordinates. Keep phases
    # in fp32 even when the rest of the transformer uses bf16 weights.
    bands = 1 / (base ** (mx.arange(head_dim // 8, dtype=mx.float32) / (head_dim // 8)))
    h, w = mx.meshgrid(mx.arange(height), mx.arange(width), indexing="ij")
    image = (
        mx.stack([mx.zeros_like(h), mx.full(h.shape, text_length), h, w], axis=-1)
        .reshape(-1, 4)
        .astype(mx.float32)
    )
    image = image - mx.array([0, 0, (height - 1) / 2, (width - 1) / 2])
    text = mx.stack(
        [
            mx.zeros(text_length),
            mx.arange(text_length),
            mx.zeros(text_length),
            mx.zeros(text_length),
        ],
        axis=-1,
    )
    phase = mx.concatenate([image, text], axis=0)[..., None] * bands
    return mx.concatenate([mx.sin(phase), mx.cos(phase)], axis=-2).reshape(
        1, -1, head_dim
    )


def _rotate(x, rope):
    sin, cos = mx.split(rope, 2, axis=-1)
    sin = mx.concatenate([sin, sin], axis=-1)[:, :, None]
    cos = mx.concatenate([cos, cos], axis=-1)[:, :, None]
    a, b = mx.split(x, 2, axis=-1)
    return x * cos + mx.concatenate([-b, a], axis=-1) * sin


def _attention(q, k, v, rope, heads):
    shape = (*q.shape[:2], heads, q.shape[-1] // heads)
    q = _rotate(q.reshape(shape), rope).transpose(0, 2, 1, 3)
    k = _rotate(k.reshape(shape), rope).transpose(0, 2, 1, 3)
    v = v.reshape(shape).transpose(0, 2, 1, 3)
    output = mx.fast.scaled_dot_product_attention(
        q.astype(v.dtype), k.astype(v.dtype), v, scale=shape[-1] ** -0.5
    )
    return output.transpose(0, 2, 1, 3).reshape(*shape[:2], -1)


class TextBlock(nn.Module):
    def __init__(self, dim, ffn_dim, heads):
        super().__init__()
        self.heads = heads
        self.norm1, self.norm2 = nn.RMSNorm(dim, eps=1e-6), nn.RMSNorm(dim, eps=1e-6)
        self.q_norm, self.k_norm = nn.RMSNorm(dim // heads, eps=1e-6), nn.RMSNorm(
            dim // heads, eps=1e-6
        )
        self.qkv = nn.Linear(dim, 3 * dim)
        self.proj = nn.Linear(dim, dim)
        self.ffn_gate, self.ffn_up = nn.Linear(dim, ffn_dim), nn.Linear(dim, ffn_dim)
        self.ffn_down = nn.Linear(ffn_dim, dim)

    def __call__(self, x, rope):
        q, k, v = mx.split(self.qkv(self.norm1(x)), 3, axis=-1)
        shape = (*q.shape[:2], self.heads, -1)
        q = self.q_norm(q.reshape(shape)).reshape(q.shape)
        k = self.k_norm(k.reshape(shape)).reshape(k.shape)
        x = x + self.proj(_attention(q, k, v, rope, self.heads))
        y = self.norm2(x)
        return x + self.ffn_down(situ(self.ffn_up(y), self.ffn_gate(y)))


class TextProjection(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.proj = nn.Linear(config.text_dim, config.dim)
        self.blocks = [
            TextBlock(config.dim, config.ffn_dim, config.num_heads) for _ in range(2)
        ]

    def __call__(self, x, rope):
        x = self.proj(x.astype(self.proj.weight.dtype))
        for block in self.blocks:
            x = block(x, rope)
        return x


class Block(nn.Module):
    def __init__(self, dim, ffn_dim, heads):
        super().__init__()
        self.heads = heads
        self.norm1, self.norm2 = nn.RMSNorm(dim, eps=1e-6), nn.RMSNorm(dim, eps=1e-6)
        self.q_norm, self.k_norm = nn.RMSNorm(dim // heads, eps=1e-6), nn.RMSNorm(
            dim // heads, eps=1e-6
        )
        self.proj = {
            name: nn.Linear(dim, dim, bias=False) for name in ("q", "k", "v", "o")
        }
        self.proj["ffn_up_gate"] = nn.Linear(dim, 2 * ffn_dim, bias=False)
        self.proj["ffn_down"] = nn.Linear(ffn_dim, dim, bias=False)

    def __call__(self, x, rope, modulation):
        shift_a, scale_a, gate_a, shift_f, scale_f, gate_f = [
            modulation[:, i, None] for i in range(6)
        ]
        y = (self.norm1(x) * (1 + scale_a) + shift_a).astype(
            self.proj["q"].weight.dtype
        )
        shape = (*y.shape[:2], self.heads, -1)
        q = self.q_norm(self.proj["q"](y).reshape(shape)).reshape(y.shape)
        k = self.k_norm(self.proj["k"](y).reshape(shape)).reshape(y.shape)
        x = (
            x
            + self.proj["o"](_attention(q, k, self.proj["v"](y), rope, self.heads))
            * gate_a
        )
        y = (self.norm2(x) * (1 + scale_f) + shift_f).astype(
            self.proj["q"].weight.dtype
        )
        hidden, gate = mx.split(self.proj["ffn_up_gate"](y), 2, axis=-1)
        return x + self.proj["ffn_down"](situ(hidden, gate)) * gate_f


class MOSSImageTransformer(nn.Module):
    def __init__(self, config: MOSSImageTransformerConfig | None = None):
        super().__init__()
        self.config = c = config or MOSSImageTransformerConfig()
        self.video_embedder = nn.Linear(c.latent_channels, c.dim)
        self.text_embedder = TextProjection(c)
        self.time_embedding = nn.Sequential(
            nn.Linear(c.freq_dim, c.dim), nn.SiLU(), nn.Linear(c.dim, c.dim)
        )
        self.shared_block_modulation = nn.Linear(c.dim, 6 * c.dim)
        self.layers = [
            Block(c.dim, c.ffn_dim, c.num_heads) for _ in range(c.num_layers)
        ]
        self.final_norm_video = nn.RMSNorm(c.dim, eps=1e-6)
        self.final_linear_video = nn.Linear(c.dim, c.latent_channels, bias=False)
        self.video_head_modulation = nn.Linear(c.dim, 2 * c.dim)

    def __call__(
        self,
        hidden_states,
        timestep,
        encoder_hidden_states,
        encoder_attention_mask=None,
        *,
        skip_layers=None,
    ):
        c = self.config
        skip_layers = validate_skip_layers(skip_layers, len(self.layers))
        if hidden_states.ndim != 4 or hidden_states.shape[1] != c.latent_channels:
            raise ValueError("hidden_states must be [B, latent_channels, H, W]")
        batch, _, height, width = hidden_states.shape
        if (
            encoder_hidden_states.ndim != 3
            or encoder_hidden_states.shape[0] != batch
            or encoder_hidden_states.shape[2] != c.text_dim
        ):
            raise ValueError("encoder_hidden_states must be [B, S, text_dim]")
        lengths = None
        if encoder_attention_mask is not None:
            mask = encoder_attention_mask
            if mask.shape != encoder_hidden_states.shape[:2] or not bool(
                mx.all((mask == 0) | (mask == 1)).item()
            ):
                raise ValueError("encoder_attention_mask must be a binary [B, S] mask")
            # Gather valid tokens before text refinement, including non-prefix masks.
            lengths = [
                [i for i, value in enumerate(row) if value] for row in mask.tolist()
            ]
        times = mx.broadcast_to(
            mx.array(timestep, dtype=mx.float32).reshape(-1), (batch,)
        )
        half = c.freq_dim // 2
        phase = times[:, None] * mx.exp(
            -math.log(10000) * mx.arange(half, dtype=mx.float32) / half
        )
        embedding = mx.concatenate([mx.cos(phase), mx.sin(phase)], axis=-1)
        time = self.time_embedding(
            embedding.astype(self.time_embedding.layers[0].weight.dtype)
        ).astype(mx.float32)
        mod = self.shared_block_modulation(
            time.astype(self.shared_block_modulation.weight.dtype)
        ).reshape(batch, 6, -1)
        head = self.video_head_modulation(
            time.astype(self.video_head_modulation.weight.dtype)
        ).reshape(batch, 2, -1)
        outputs = []
        for index in range(batch):
            context = encoder_hidden_states[index : index + 1]
            if lengths is not None:
                context = context[:, mx.array(lengths[index], dtype=mx.int32)]
            if context.shape[1] == 0:
                raise ValueError("Each sample needs at least one valid text token")
            video = (
                hidden_states[index : index + 1]
                .reshape(1, c.latent_channels, -1)
                .transpose(0, 2, 1)
            )
            video = self.video_embedder(video.astype(self.video_embedder.weight.dtype))
            rope = rotary_positions(
                height, width, context.shape[1], c.dim // c.num_heads, c.rope_base
            ).astype(video.dtype)
            text = self.text_embedder(context, rope[:, height * width :])
            x = mx.concatenate([video, text], axis=1)
            for layer_index, layer in enumerate(self.layers):
                if layer_index not in skip_layers:
                    x = layer(x, rope, mod[index : index + 1])
            x = self.final_norm_video(
                x[:, : height * width].astype(self.final_norm_video.weight.dtype)
            )
            x = (
                x * (1 + head[index : index + 1, 1, None])
                + head[index : index + 1, 0, None]
            )
            x = self.final_linear_video(x.astype(self.final_linear_video.weight.dtype))
            outputs.append(
                x.transpose(0, 2, 1).reshape(1, c.latent_channels, height, width)
            )
        return mx.concatenate(outputs, axis=0)

    @staticmethod
    def sanitize(weights):
        # MLX Sequential stores its modules under `layers`. RoPE is derived in
        # fp32 at runtime rather than loaded from the checkpoint's bf16 buffer.
        return {
            key.replace("time_embedding.0.", "time_embedding.layers.0.").replace(
                "time_embedding.2.", "time_embedding.layers.2."
            ): value
            for key, value in weights.items()
            if key != "rope.bands"
        }
