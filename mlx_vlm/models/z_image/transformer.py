from __future__ import annotations

import math

import mlx.core as mx
from mlx import nn

from .config import ZImageTransformerConfig

ADALN_EMBED_DIM = 256
SEQ_MULTI_OF = 32


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
    def __init__(
        self,
        sections: tuple[int, ...] = (32, 48, 48),
        theta: float = 256.0,
    ) -> None:
        self.sections = sections
        self.theta = theta

    def compute_freqs(self, position_ids: mx.array) -> tuple[mx.array, mx.array]:
        cos_parts = []
        sin_parts = []
        for i, s in enumerate(self.sections):
            ids = position_ids[..., i].astype(mx.float32)
            inv_freq = 1.0 / (self.theta ** (mx.arange(0, s, 2, dtype=mx.float32) / s))
            theta = ids[..., None] * inv_freq[None, None, :]
            cos_parts.append(mx.repeat(mx.cos(theta), 2, axis=-1))
            sin_parts.append(mx.repeat(mx.sin(theta), 2, axis=-1))
        return mx.concatenate(cos_parts, axis=-1), mx.concatenate(sin_parts, axis=-1)


def apply_rotary(x: mx.array, cos: mx.array, sin: mx.array) -> mx.array:
    cos = cos[:, :, None, :]
    sin = sin[:, :, None, :]
    *rest, D = x.shape
    x_r = x.reshape(*rest, D // 2, 2)
    x_real, x_imag = x_r[..., 0], x_r[..., 1]
    cos_r = cos.reshape(*cos.shape[:-1], D // 2, 2)[..., 0]
    sin_r = sin.reshape(*sin.shape[:-1], D // 2, 2)[..., 0]
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
        # to_out is a list so weight key becomes attention.to_out.0.weight
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
        B, L, _ = x.shape
        q = self.to_q(x).reshape(B, L, self.num_heads, self.head_dim)
        k = self.to_k(x).reshape(B, L, self.num_heads, self.head_dim)
        v = self.to_v(x).reshape(B, L, self.num_heads, self.head_dim)
        q = self.norm_q(q)
        k = self.norm_k(k)
        q = apply_rotary(q, cos, sin)
        k = apply_rotary(k, cos, sin)
        q = q.transpose(0, 2, 1, 3)
        k = k.transpose(0, 2, 1, 3)
        v = v.transpose(0, 2, 1, 3)
        out = mx.fast.scaled_dot_product_attention(q, k, v, scale=self.scale, mask=mask)
        out = out.transpose(0, 2, 1, 3).reshape(B, L, -1)
        return self.to_out[0](out)


class ZImageTransformerBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        head_dim: int,
        mlp_dim: int,
        *,
        modulation: bool = True,
        adaln_dim: int = ADALN_EMBED_DIM,
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
            # List so key becomes adaLN_modulation.0.weight
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
            chunks = self.adaLN_modulation[0](adaln_input)
            scale_msa, gate_msa, scale_mlp, gate_mlp = mx.split(chunks, 4, axis=-1)
            gate_msa = mx.tanh(gate_msa)[:, None, :]
            gate_mlp = mx.tanh(gate_mlp)[:, None, :]
            scale_msa = (1.0 + scale_msa)[:, None, :]
            scale_mlp = (1.0 + scale_mlp)[:, None, :]
            attn_out = self.attention(
                self.attention_norm1(x) * scale_msa, cos, sin, mask
            )
            x = x + gate_msa * self.attention_norm2(attn_out)
            ffn_out = self.feed_forward(self.ffn_norm1(x) * scale_mlp)
            x = x + gate_mlp * self.ffn_norm2(ffn_out)
        else:
            attn_out = self.attention(self.attention_norm1(x), cos, sin, mask)
            x = x + self.attention_norm2(attn_out)
            ffn_out = self.feed_forward(self.ffn_norm1(x))
            x = x + self.ffn_norm2(ffn_out)
        return x


class FinalLayer(nn.Module):
    def __init__(
        self, dim: int, out_dim: int, adaln_dim: int = ADALN_EMBED_DIM
    ) -> None:
        super().__init__()
        self.norm_final = nn.LayerNorm(dim, affine=False, eps=1e-6)
        self.linear = nn.Linear(dim, out_dim)
        # List for .0. key path
        self.adaLN_modulation = [nn.Linear(adaln_dim, dim)]

    def __call__(self, x: mx.array, c: mx.array) -> mx.array:
        scale = 1.0 + self.adaLN_modulation[0](nn.silu(c))[:, None, :]
        return self.linear(self.norm_final(x) * scale)


class _FinalLayerDict(nn.Module):
    def __init__(self, dim: int, out_dim: int) -> None:
        super().__init__()
        # Attribute name uses Python-safe replacement but weight key
        # mapping will handle the '2-1' literal via sanitization
        self.layer = FinalLayer(dim, out_dim)


class _XEmbedderDict(nn.Module):
    def __init__(self, in_features: int, out_features: int) -> None:
        super().__init__()
        self.embed = nn.Linear(in_features, out_features)


class ZImageTransformer(nn.Module):
    def __init__(self, config: ZImageTransformerConfig | None = None) -> None:
        super().__init__()
        if config is None:
            config = ZImageTransformerConfig()
        self.config = config
        dim = config.hidden_size
        num_heads = config.num_attention_heads
        head_dim = dim // num_heads
        if config.num_key_value_heads != num_heads:
            raise ValueError("Z-Image grouped-query attention is not supported")
        if sum(config.rope_sections) != head_dim:
            raise ValueError(
                "Z-Image RoPE sections must sum to the attention head dimension"
            )
        mlp_dim = config.intermediate_size
        patch_size = config.patch_size
        f_patch_size = config.f_patch_size
        in_ch = config.in_channels

        self.patch_size = patch_size
        self.f_patch_size = f_patch_size
        self.in_channels = in_ch

        adaln_dim = min(dim, config.adaln_embed_dim)

        self.t_embedder = TimestepEmbedder(adaln_dim)
        # cap_embedder: list [RMSNorm, Linear] for .0. and .1. keys
        self.cap_embedder = [
            nn.RMSNorm(config.text_embed_dim, eps=config.norm_eps),
            nn.Linear(config.text_embed_dim, dim),
        ]
        self.noise_refiner = [
            ZImageTransformerBlock(
                dim, num_heads, head_dim, mlp_dim, modulation=True, adaln_dim=adaln_dim
            )
            for _ in range(config.n_refiner_layers)
        ]
        self.context_refiner = [
            ZImageTransformerBlock(dim, num_heads, head_dim, mlp_dim, modulation=False)
            for _ in range(config.n_context_refiner_layers)
        ]
        self.layers = [
            ZImageTransformerBlock(
                dim, num_heads, head_dim, mlp_dim, modulation=True, adaln_dim=adaln_dim
            )
            for _ in range(config.num_hidden_layers)
        ]
        # For weight loading, use sanitization to map these special keys
        self.final_layer = FinalLayer(
            dim, patch_size * patch_size * f_patch_size * in_ch, adaln_dim=adaln_dim
        )
        self.x_embedder = nn.Linear(f_patch_size * patch_size * patch_size * in_ch, dim)

        # Learnable padding tokens
        self.x_pad_token = mx.zeros((1, dim))
        self.cap_pad_token = mx.zeros((1, dim))

        self.rope = MRoPE(config.rope_sections, config.rope_theta)
        self.t_scale = config.timestep_scale

    def _patchify(self, x: mx.array) -> tuple[mx.array, tuple[int, int, int]]:
        B, C, F, H, W = x.shape
        pH, pW, pF = self.patch_size, self.patch_size, self.f_patch_size
        Ft, Ht, Wt = F // pF, H // pH, W // pW
        x = x.reshape(B, C, Ft, pF, Ht, pH, Wt, pW)
        x = x.transpose(0, 2, 4, 6, 3, 5, 7, 1)
        x = x.reshape(B, Ft * Ht * Wt, pF * pH * pW * C)
        return x, (Ft, Ht, Wt)

    def _unpatchify(self, x: mx.array, shape: tuple[int, int, int]) -> mx.array:
        B = x.shape[0]
        Ft, Ht, Wt = shape
        pH, pW, pF = self.patch_size, self.patch_size, self.f_patch_size
        C = self.in_channels
        x = x.reshape(B, Ft, Ht, Wt, pF, pH, pW, C)
        x = x.transpose(0, 7, 1, 4, 2, 5, 3, 6)
        return x.reshape(B, C, Ft * pF, Ht * pH, Wt * pW)

    def _build_position_ids(
        self, cap_len: int, Ft: int, Ht: int, Wt: int
    ) -> tuple[mx.array, mx.array]:
        # Caption positions: t=1..cap_len, h=0, w=0
        t_ids = mx.arange(1, cap_len + 1, dtype=mx.float32)
        cap_pos = mx.stack([t_ids, mx.zeros(cap_len), mx.zeros(cap_len)], axis=-1)[None]

        # Image positions
        start_t = cap_len + 1
        f_ids = mx.arange(Ft, dtype=mx.float32) + start_t
        h_ids = mx.arange(Ht, dtype=mx.float32)
        w_ids = mx.arange(Wt, dtype=mx.float32)
        f_grid = mx.repeat(f_ids, Ht * Wt)
        h_grid = mx.tile(mx.repeat(h_ids, Wt), Ft)
        w_grid = mx.tile(w_ids, Ft * Ht)
        img_pos = mx.stack([f_grid, h_grid, w_grid], axis=-1)[None]
        return cap_pos, img_pos

    def __call__(self, x: mx.array, t: mx.array, cap_feats: mx.array) -> mx.array:
        B = x.shape[0]
        patches, (Ft, Ht, Wt) = self._patchify(x)
        image_len = patches.shape[1]
        image_padding = (-image_len) % SEQ_MULTI_OF
        if image_padding:
            patches = mx.concatenate(
                [patches, mx.repeat(patches[:, -1:], image_padding, axis=1)],
                axis=1,
            )
        cap_len = cap_feats.shape[1]
        cap_padding = (-cap_len) % SEQ_MULTI_OF
        if cap_padding:
            cap_feats = mx.concatenate(
                [cap_feats, mx.repeat(cap_feats[:, -1:], cap_padding, axis=1)],
                axis=1,
            )
        padded_cap_len = cap_feats.shape[1]

        # Embed timestep
        t_emb = self.t_embedder(t * self.t_scale)

        # Embed patches and caption
        img_tokens = self.x_embedder(patches)
        if image_padding:
            img_tokens = mx.concatenate(
                [
                    img_tokens[:, :image_len],
                    mx.broadcast_to(
                        self.x_pad_token,
                        (B, image_padding, self.config.hidden_size),
                    ),
                ],
                axis=1,
            )
        cap_tokens = self.cap_embedder[1](self.cap_embedder[0](cap_feats))
        if cap_padding:
            cap_tokens = mx.concatenate(
                [
                    cap_tokens[:, :cap_len],
                    mx.broadcast_to(
                        self.cap_pad_token,
                        (B, cap_padding, self.config.hidden_size),
                    ),
                ],
                axis=1,
            )

        # Position IDs
        cap_pos, img_pos = self._build_position_ids(padded_cap_len, Ft, Ht, Wt)
        if image_padding:
            img_pos = mx.concatenate(
                [img_pos, mx.zeros((1, image_padding, 3), dtype=img_pos.dtype)],
                axis=1,
            )
        img_cos, img_sin = self.rope.compute_freqs(img_pos)
        cap_cos, cap_sin = self.rope.compute_freqs(cap_pos)

        # Noise refiner (image only)
        for block in self.noise_refiner:
            img_tokens = block(img_tokens, img_cos, img_sin, adaln_input=t_emb)

        # Context refiner (caption only)
        for block in self.context_refiner:
            cap_tokens = block(cap_tokens, cap_cos, cap_sin)

        # Unified
        unified = mx.concatenate([img_tokens, cap_tokens], axis=1)
        unified_cos = mx.concatenate([img_cos, cap_cos], axis=1)
        unified_sin = mx.concatenate([img_sin, cap_sin], axis=1)
        for block in self.layers:
            unified = block(
                unified,
                unified_cos,
                unified_sin,
                adaln_input=t_emb,
            )

        # Final layer (image tokens only)
        img_out = self.final_layer(unified[:, :image_len], t_emb)
        return self._unpatchify(img_out, (Ft, Ht, Wt))


def sanitize_transformer_weights(
    weights: dict[str, mx.array],
) -> dict[str, mx.array]:
    sanitized: dict[str, mx.array] = {}
    for key, value in weights.items():
        # Map the source ModuleDict key to the registered MLX module.
        if key.startswith("all_final_layer.2-1."):
            key = "final_layer." + key[len("all_final_layer.2-1.") :]
            key = key.replace("adaLN_modulation.1.", "adaLN_modulation.0.")
        # Map the source ModuleDict key to the registered MLX module.
        elif key.startswith("all_x_embedder.2-1."):
            key = "x_embedder." + key[len("all_x_embedder.2-1.") :]
        key = key.replace("t_embedder.mlp.0.", "t_embedder.linear1.")
        key = key.replace("t_embedder.mlp.2.", "t_embedder.linear2.")
        sanitized[key] = value
    return sanitized


__all__ = ["ZImageTransformer", "sanitize_transformer_weights"]
