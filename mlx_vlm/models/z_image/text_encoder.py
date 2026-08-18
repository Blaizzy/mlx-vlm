"""Qwen3-style text encoder for Z-Image.

Matches weight keys:
- embed_tokens.{weight, scales}
- layers.N.input_layernorm.weight
- layers.N.self_attn.{q_proj,k_proj,v_proj,o_proj}.{weight,scales}
- layers.N.self_attn.{q_norm,k_norm}.weight
- layers.N.post_attention_layernorm.weight
- layers.N.mlp.{gate_proj,up_proj,down_proj}.{weight,scales}
- norm.weight
"""

from __future__ import annotations

import math
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn

from .config import ZImageTextEncoderConfig


class Qwen3Attention(nn.Module):
    def __init__(self, config: ZImageTextEncoderConfig) -> None:
        super().__init__()
        self.num_heads = config.num_attention_heads
        self.num_kv_heads = config.num_key_value_heads
        self.head_dim = config.head_dim
        self.num_kv_groups = self.num_heads // self.num_kv_heads
        self.scale = self.head_dim**-0.5
        dim = config.hidden_size

        self.q_proj = nn.Linear(dim, self.num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(dim, self.num_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(dim, self.num_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, dim, bias=False)
        self.q_norm = nn.RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.k_norm = nn.RMSNorm(self.head_dim, eps=config.rms_norm_eps)

    def __call__(
        self, x: mx.array, cos: mx.array, sin: mx.array, mask: mx.array | None
    ) -> mx.array:
        B, L, _ = x.shape
        q = self.q_proj(x).reshape(B, L, self.num_heads, self.head_dim)
        k = self.k_proj(x).reshape(B, L, self.num_kv_heads, self.head_dim)
        v = self.v_proj(x).reshape(B, L, self.num_kv_heads, self.head_dim)
        q = self.q_norm(q)
        k = self.k_norm(k)
        # Apply RoPE
        q = _apply_rope(q, cos, sin)
        k = _apply_rope(k, cos, sin)
        q = q.transpose(0, 2, 1, 3)
        k = k.transpose(0, 2, 1, 3)
        v = v.transpose(0, 2, 1, 3)
        if self.num_kv_groups > 1:
            k = mx.repeat(k, self.num_kv_groups, axis=1)
            v = mx.repeat(v, self.num_kv_groups, axis=1)
        out = mx.fast.scaled_dot_product_attention(q, k, v, scale=self.scale, mask=mask)
        out = out.transpose(0, 2, 1, 3).reshape(B, L, -1)
        return self.o_proj(out)


def _apply_rope(x: mx.array, cos: mx.array, sin: mx.array) -> mx.array:
    """x: [B, L, H, D], cos/sin: [1, L, 1, D]."""
    x1, x2 = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2 :]
    return mx.concatenate(
        [x1 * cos - x2 * sin, x1 * sin + x2 * cos], axis=-1
    )


class Qwen3MLP(nn.Module):
    def __init__(self, config: ZImageTextEncoderConfig) -> None:
        super().__init__()
        self.gate_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.up_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.down_proj = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)

    def __call__(self, x: mx.array) -> mx.array:
        return self.down_proj(nn.silu(self.gate_proj(x)) * self.up_proj(x))


class Qwen3DecoderLayer(nn.Module):
    def __init__(self, config: ZImageTextEncoderConfig) -> None:
        super().__init__()
        self.self_attn = Qwen3Attention(config)
        self.mlp = Qwen3MLP(config)
        self.input_layernorm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def __call__(
        self, x: mx.array, cos: mx.array, sin: mx.array, mask: mx.array | None
    ) -> mx.array:
        h = x + self.self_attn(self.input_layernorm(x), cos, sin, mask)
        return h + self.mlp(self.post_attention_layernorm(h))


class ZImageTextEncoder(nn.Module):
    """Qwen3-style text encoder matching checkpoint structure."""

    def __init__(self, config: ZImageTextEncoderConfig | None = None) -> None:
        super().__init__()
        if config is None:
            config = ZImageTextEncoderConfig()
        self.config = config
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = [Qwen3DecoderLayer(config) for _ in range(config.num_hidden_layers)]
        self.norm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        # Precompute RoPE frequencies
        self._rope_dim = config.head_dim
        self._rope_theta = config.rope_theta

    def _rope_freqs(self, seq_len: int) -> tuple[mx.array, mx.array]:
        dim = self._rope_dim
        inv_freq = 1.0 / (
            self._rope_theta ** (mx.arange(0, dim, 2, dtype=mx.float32) / dim)
        )
        t = mx.arange(seq_len, dtype=mx.float32)
        freqs = t[:, None] * inv_freq[None, :]
        cos = mx.cos(freqs)[None, :, None, :]  # [1, L, 1, D/2]
        sin = mx.sin(freqs)[None, :, None, :]
        return cos, sin

    def __call__(
        self, input_ids: mx.array, attention_mask: mx.array | None = None
    ) -> mx.array:
        B, L = input_ids.shape
        x = self.embed_tokens(input_ids)
        cos, sin = self._rope_freqs(L)
        # Causal mask
        mask = mx.triu(mx.full((L, L), -math.inf, dtype=mx.float32), k=1)
        mask = mask[None, None, :, :]
        if attention_mask is not None:
            pad_mask = mx.where(attention_mask == 1, 0.0, -math.inf)
            mask = mask + pad_mask[:, None, None, :]
        for layer in self.layers:
            x = layer(x, cos, sin, mask)
        return self.norm(x)


def sanitize_text_encoder_weights(
    weights: dict[str, mx.array],
) -> dict[str, mx.array]:
    """Drop rotary_emb.inv_freq (computed at runtime)."""
    return {k: v for k, v in weights.items() if "rotary_emb" not in k}


__all__ = ["ZImageTextEncoder", "sanitize_text_encoder_weights"]
