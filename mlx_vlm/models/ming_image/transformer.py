"""Ming-Image NextDiT: a single-stream Lumina-2.0-style diffusion transformer.

The block structure (noise/context refiners, sandwich AdaLN modulation, 3-axis
axial RoPE, joint image+caption self-attention) matches the ``z_image`` port.
Ming adds a second conditioning stream: the caption sequence is
``cap_embedder(cap_feats)`` concatenated with the already-projected direct-VLM
features (``cap_feats_2``). Alignment padding is zero-filled and masked out of
attention (the checkpoint has no learnable pad tokens).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import mlx.core as mx
import mlx.nn as nn

from .config import MingImageDiTConfig

SEQ_MULTIPLE = 32


@dataclass(slots=True)
class MingImageConditioning:
    """Step-invariant DiT inputs, built once and reused across denoising steps."""

    refined_cap: mx.array
    img_cos: mx.array
    img_sin: mx.array
    img_mask: mx.array | None
    cos: mx.array
    sin: mx.array
    mask: mx.array | None
    image_len: int
    padded_image_len: int
    image_shape: tuple[int, int, int]

    def arrays(self) -> list[mx.array]:
        values = [self.refined_cap, self.img_cos, self.img_sin, self.cos, self.sin]
        values.extend(v for v in (self.img_mask, self.mask) if v is not None)
        return values


def _round_up(value: int, multiple: int) -> int:
    return ((value + multiple - 1) // multiple) * multiple


class TimestepEmbedder(nn.Module):
    def __init__(self, out_size: int, frequency_size: int = 256) -> None:
        super().__init__()
        self.frequency_size = frequency_size
        self.linear1 = nn.Linear(frequency_size, 1024)
        self.linear2 = nn.Linear(1024, out_size)

    def __call__(self, t: mx.array) -> mx.array:
        half = self.frequency_size // 2
        freqs = mx.exp(-math.log(10000) * mx.arange(half, dtype=mx.float32) / half)
        args = t.reshape(-1, 1).astype(mx.float32) * freqs[None]
        emb = mx.concatenate([mx.cos(args), mx.sin(args)], axis=-1)
        return self.linear2(nn.silu(self.linear1(emb.astype(t.dtype))))


class FeedForward(nn.Module):
    def __init__(self, dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.w1 = nn.Linear(dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, dim, bias=False)
        self.w3 = nn.Linear(dim, hidden_dim, bias=False)

    def __call__(self, x: mx.array) -> mx.array:
        return self.w2(nn.silu(self.w1(x)) * self.w3(x))


class MRoPE:
    def __init__(self, sections: tuple[int, ...], theta: float) -> None:
        self.sections = sections
        self.theta = theta

    def __call__(self, position_ids: mx.array) -> tuple[mx.array, mx.array]:
        cos_parts, sin_parts = [], []
        for i, s in enumerate(self.sections):
            ids = position_ids[..., i].astype(mx.float32)
            inv_freq = 1.0 / (self.theta ** (mx.arange(0, s, 2, dtype=mx.float32) / s))
            angles = ids[..., None] * inv_freq[None, None, :]
            cos_parts.append(mx.repeat(mx.cos(angles), 2, axis=-1))
            sin_parts.append(mx.repeat(mx.sin(angles), 2, axis=-1))
        return mx.concatenate(cos_parts, axis=-1), mx.concatenate(sin_parts, axis=-1)


def apply_rotary(x: mx.array, cos: mx.array, sin: mx.array) -> mx.array:
    cos = cos[:, :, None, :]
    sin = sin[:, :, None, :]
    *rest, d = x.shape
    x_pairs = x.reshape(*rest, d // 2, 2)
    x_real, x_imag = x_pairs[..., 0], x_pairs[..., 1]
    cos_r = cos.reshape(*cos.shape[:-1], d // 2, 2)[..., 0]
    sin_r = sin.reshape(*sin.shape[:-1], d // 2, 2)[..., 0]
    out_real = x_real * cos_r - x_imag * sin_r
    out_imag = x_real * sin_r + x_imag * cos_r
    return mx.stack([out_real, out_imag], axis=-1).reshape(x.shape)


class Attention(nn.Module):
    def __init__(self, dim: int, num_heads: int, head_dim: int) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.scale = head_dim**-0.5
        self.to_q = nn.Linear(dim, dim, bias=False)
        self.to_k = nn.Linear(dim, dim, bias=False)
        self.to_v = nn.Linear(dim, dim, bias=False)
        self.to_out = [nn.Linear(dim, dim, bias=False)]
        self.norm_q = nn.RMSNorm(head_dim)
        self.norm_k = nn.RMSNorm(head_dim)

    def __call__(
        self,
        x: mx.array,
        cos: mx.array,
        sin: mx.array,
        mask: mx.array | None = None,
    ) -> mx.array:
        b, length, _ = x.shape
        q = self.norm_q(self.to_q(x).reshape(b, length, self.num_heads, self.head_dim))
        k = self.norm_k(self.to_k(x).reshape(b, length, self.num_heads, self.head_dim))
        v = self.to_v(x).reshape(b, length, self.num_heads, self.head_dim)
        q = apply_rotary(q, cos, sin).transpose(0, 2, 1, 3)
        k = apply_rotary(k, cos, sin).transpose(0, 2, 1, 3)
        v = v.transpose(0, 2, 1, 3)
        out = mx.fast.scaled_dot_product_attention(q, k, v, scale=self.scale, mask=mask)
        out = out.transpose(0, 2, 1, 3).reshape(b, length, -1)
        return self.to_out[0](out)


class TransformerBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        head_dim: int,
        mlp_dim: int,
        *,
        modulation: bool,
        adaln_dim: int,
    ) -> None:
        super().__init__()
        self.modulation = modulation
        self.attention = Attention(dim, num_heads, head_dim)
        self.feed_forward = FeedForward(dim, mlp_dim)
        self.attention_norm1 = nn.RMSNorm(dim)
        self.attention_norm2 = nn.RMSNorm(dim)
        self.ffn_norm1 = nn.RMSNorm(dim)
        self.ffn_norm2 = nn.RMSNorm(dim)
        if modulation:
            self.adaLN_modulation = [nn.Linear(adaln_dim, 4 * dim)]

    def __call__(
        self,
        x: mx.array,
        cos: mx.array,
        sin: mx.array,
        adaln_input: mx.array | None = None,
        mask: mx.array | None = None,
    ) -> mx.array:
        if self.modulation:
            scale_msa, gate_msa, scale_mlp, gate_mlp = mx.split(
                self.adaLN_modulation[0](adaln_input), 4, axis=-1
            )
            gate_msa = mx.tanh(gate_msa)[:, None, :]
            gate_mlp = mx.tanh(gate_mlp)[:, None, :]
            scale_msa = (1.0 + scale_msa)[:, None, :]
            scale_mlp = (1.0 + scale_mlp)[:, None, :]
            attn = self.attention(self.attention_norm1(x) * scale_msa, cos, sin, mask)
            x = x + gate_msa * self.attention_norm2(attn)
            ffn = self.feed_forward(self.ffn_norm1(x) * scale_mlp)
            x = x + gate_mlp * self.ffn_norm2(ffn)
        else:
            attn = self.attention(self.attention_norm1(x), cos, sin, mask)
            x = x + self.attention_norm2(attn)
            ffn = self.feed_forward(self.ffn_norm1(x))
            x = x + self.ffn_norm2(ffn)
        return x


class FinalLayer(nn.Module):
    def __init__(self, dim: int, out_dim: int, adaln_dim: int) -> None:
        super().__init__()
        self.norm_final = nn.LayerNorm(dim, affine=False, eps=1e-6)
        self.linear = nn.Linear(dim, out_dim)
        self.adaLN_modulation = [nn.Linear(adaln_dim, dim)]

    def __call__(self, x: mx.array, c: mx.array) -> mx.array:
        scale = 1.0 + self.adaLN_modulation[0](nn.silu(c))[:, None, :]
        return self.linear(self.norm_final(x) * scale)


class MingImageTransformer(nn.Module):
    def __init__(self, config: MingImageDiTConfig) -> None:
        super().__init__()
        self.config = config
        dim = config.dim
        head_dim = dim // config.n_heads
        if config.n_kv_heads != config.n_heads:
            raise ValueError("Ming-Image DiT expects full multi-head attention")
        if sum(config.axes_dims) != head_dim:
            raise ValueError("axes_dims must sum to the attention head dimension")
        adaln_dim = min(dim, config.adaln_embed_dim)
        patch = config.patch_size
        f_patch = config.f_patch_size
        self.patch_size = patch
        self.f_patch_size = f_patch
        self.in_channels = config.in_channels
        self.t_scale = config.t_scale

        self.t_embedder = TimestepEmbedder(adaln_dim)
        self.cap_embedder = [
            nn.RMSNorm(config.cap_feat_dim, eps=config.norm_eps),
            nn.Linear(config.cap_feat_dim, dim),
        ]
        self.x_embedder = nn.Linear(f_patch * patch * patch * config.in_channels, dim)
        self.noise_refiner = [
            TransformerBlock(
                dim,
                config.n_heads,
                head_dim,
                config.intermediate_size,
                modulation=True,
                adaln_dim=adaln_dim,
            )
            for _ in range(config.n_refiner_layers)
        ]
        self.context_refiner = [
            TransformerBlock(
                dim,
                config.n_heads,
                head_dim,
                config.intermediate_size,
                modulation=False,
                adaln_dim=adaln_dim,
            )
            for _ in range(config.n_refiner_layers)
        ]
        self.layers = [
            TransformerBlock(
                dim,
                config.n_heads,
                head_dim,
                config.intermediate_size,
                modulation=True,
                adaln_dim=adaln_dim,
            )
            for _ in range(config.n_layers)
        ]
        self.final_layer = FinalLayer(
            dim, patch * patch * f_patch * config.in_channels, adaln_dim
        )
        self.rope = MRoPE(config.axes_dims, config.rope_theta)

    def _patchify(self, x: mx.array) -> tuple[mx.array, tuple[int, int, int]]:
        b, c, f, h, w = x.shape
        p, pf = self.patch_size, self.f_patch_size
        ft, ht, wt = f // pf, h // p, w // p
        x = x.reshape(b, c, ft, pf, ht, p, wt, p)
        x = x.transpose(0, 2, 4, 6, 3, 5, 7, 1)
        return x.reshape(b, ft * ht * wt, pf * p * p * c), (ft, ht, wt)

    def _unpatchify(self, x: mx.array, shape: tuple[int, int, int]) -> mx.array:
        b = x.shape[0]
        ft, ht, wt = shape
        p, pf = self.patch_size, self.f_patch_size
        c = self.in_channels
        x = x.reshape(b, ft, ht, wt, pf, p, p, c)
        x = x.transpose(0, 7, 1, 4, 2, 5, 3, 6)
        return x.reshape(b, c, ft * pf, ht * p, wt * p)

    @staticmethod
    def _key_mask(valid_len: int, total_len: int) -> mx.array | None:
        if valid_len == total_len:
            return None
        keep = mx.arange(total_len) < valid_len
        return keep.reshape(1, 1, 1, total_len)

    def _positions(
        self, cap_len: int, image_shape: tuple[int, int, int]
    ) -> tuple[mx.array, mx.array]:
        ft, ht, wt = image_shape
        cap_t = mx.arange(1, cap_len + 1, dtype=mx.float32)
        zeros = mx.zeros(cap_len)
        cap_pos = mx.stack([cap_t, zeros, zeros], axis=-1)[None]
        start = cap_len + 1
        f_ids = mx.arange(ft, dtype=mx.float32) + start
        h_ids = mx.arange(ht, dtype=mx.float32)
        w_ids = mx.arange(wt, dtype=mx.float32)
        f_grid = mx.repeat(f_ids, ht * wt)
        h_grid = mx.tile(mx.repeat(h_ids, wt), (ft,))
        w_grid = mx.tile(w_ids, (ft * ht,))
        img_pos = mx.stack([f_grid, h_grid, w_grid], axis=-1)[None]
        return cap_pos, img_pos

    def prepare_conditioning(
        self,
        cap_feats: mx.array,
        cap_feats_2: mx.array | None,
        image_shape: tuple[int, int, int],
    ) -> MingImageConditioning:
        """Precompute the step-invariant caption context, RoPE, and masks.

        The context refiner carries no timestep dependence, so its output and
        every RoPE/mask are identical across the denoising loop. The pipeline
        builds this once and reuses it for every step.
        """
        ft, ht, wt = image_shape
        image_len = ft * ht * wt
        padded_image_len = _round_up(image_len, SEQ_MULTIPLE)

        cap_tokens = self.cap_embedder[1](self.cap_embedder[0](cap_feats))
        if cap_feats_2 is not None:
            cap_tokens = mx.concatenate([cap_tokens, cap_feats_2], axis=1)
        cap_len = cap_tokens.shape[1]
        padded_cap_len = _round_up(cap_len, SEQ_MULTIPLE)
        cap_tokens = self._pad_sequence(cap_tokens, padded_cap_len)

        cap_pos, img_pos = self._positions(padded_cap_len, image_shape)
        img_pos = self._pad_positions(img_pos, padded_image_len)
        img_cos, img_sin = self.rope(img_pos)
        cap_cos, cap_sin = self.rope(cap_pos)
        cap_mask = self._key_mask(cap_len, padded_cap_len)

        for block in self.context_refiner:
            cap_tokens = block(cap_tokens, cap_cos, cap_sin, None, cap_mask)

        return MingImageConditioning(
            refined_cap=cap_tokens,
            img_cos=img_cos,
            img_sin=img_sin,
            img_mask=self._key_mask(image_len, padded_image_len),
            cos=mx.concatenate([img_cos, cap_cos], axis=1),
            sin=mx.concatenate([img_sin, cap_sin], axis=1),
            mask=self._unified_mask(
                image_len, padded_image_len, cap_len, padded_cap_len
            ),
            image_len=image_len,
            padded_image_len=padded_image_len,
            image_shape=image_shape,
        )

    def denoise(
        self, x: mx.array, t: mx.array, conditioning: MingImageConditioning
    ) -> mx.array:
        """One denoising step reusing a precomputed conditioning context."""
        patches, _ = self._patchify(x)
        img_tokens = self._pad_sequence(
            self.x_embedder(patches), conditioning.padded_image_len
        )
        t_emb = self.t_embedder(t * self.t_scale).astype(img_tokens.dtype)
        for block in self.noise_refiner:
            img_tokens = block(
                img_tokens,
                conditioning.img_cos,
                conditioning.img_sin,
                t_emb,
                conditioning.img_mask,
            )
        unified = mx.concatenate([img_tokens, conditioning.refined_cap], axis=1)
        for block in self.layers:
            unified = block(
                unified, conditioning.cos, conditioning.sin, t_emb, conditioning.mask
            )
        img_out = self.final_layer(unified[:, : conditioning.image_len], t_emb)
        return self._unpatchify(img_out, conditioning.image_shape)

    def __call__(
        self,
        x: mx.array,
        t: mx.array,
        cap_feats: mx.array,
        cap_feats_2: mx.array | None = None,
    ) -> mx.array:
        _, _, f, h, w = x.shape
        image_shape = (
            f // self.f_patch_size,
            h // self.patch_size,
            w // self.patch_size,
        )
        conditioning = self.prepare_conditioning(cap_feats, cap_feats_2, image_shape)
        return self.denoise(x, t, conditioning)

    @staticmethod
    def _pad_sequence(tokens: mx.array, target_len: int) -> mx.array:
        pad = target_len - tokens.shape[1]
        if pad == 0:
            return tokens
        b, _, dim = tokens.shape
        return mx.concatenate(
            [tokens, mx.zeros((b, pad, dim), dtype=tokens.dtype)], axis=1
        )

    @staticmethod
    def _pad_positions(positions: mx.array, target_len: int) -> mx.array:
        pad = target_len - positions.shape[1]
        if pad == 0:
            return positions
        return mx.concatenate(
            [positions, mx.zeros((1, pad, 3), dtype=positions.dtype)], axis=1
        )

    @staticmethod
    def _unified_mask(
        image_len: int, padded_image_len: int, cap_len: int, padded_cap_len: int
    ) -> mx.array | None:
        if image_len == padded_image_len and cap_len == padded_cap_len:
            return None
        img_valid = mx.arange(padded_image_len) < image_len
        cap_valid = mx.arange(padded_cap_len) < cap_len
        keep = mx.concatenate([img_valid, cap_valid], axis=0)
        return keep.reshape(1, 1, 1, keep.shape[0])


def sanitize_transformer_weights(weights: dict[str, mx.array]) -> dict[str, mx.array]:
    sanitized: dict[str, mx.array] = {}
    for key, value in weights.items():
        if key.startswith("all_final_layer.2-1."):
            key = "final_layer." + key[len("all_final_layer.2-1.") :]
            key = key.replace("adaLN_modulation.1.", "adaLN_modulation.0.")
        elif key.startswith("all_x_embedder.2-1."):
            key = "x_embedder." + key[len("all_x_embedder.2-1.") :]
        key = key.replace("t_embedder.mlp.0.", "t_embedder.linear1.")
        key = key.replace("t_embedder.mlp.2.", "t_embedder.linear2.")
        sanitized[key] = value
    return sanitized


__all__ = [
    "MingImageConditioning",
    "MingImageTransformer",
    "sanitize_transformer_weights",
]
