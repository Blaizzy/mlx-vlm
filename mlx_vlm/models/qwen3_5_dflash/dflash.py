"""MLX port of the DFlash block-diffusion drafter.

Reference: huggingface.co/z-lab/Qwen3.5-4B-DFlash (dflash.py).

The drafter is a stateless head that runs on top of a target Qwen3.5 model:
    * `noise_embedding` — [B, L, H] embeddings of L mask tokens (from the
       target's embed_tokens) at the positions to be drafted
    * `target_hidden` — [B, T, num_target_layers * H] concatenated hidden
       states collected from the target model at layers `target_layer_ids`,
       spanning the full committed context (T tokens)
    * `position_ids` — [B, T + L] 1D absolute positions for context + noise

The drafter returns [B, L, H] noise-position hidden states. Logits are obtained
externally by projecting through the target model's tied embed_tokens
(``target.embed_tokens.as_linear(...)``).
"""

from typing import Optional

import mlx.core as mx
import mlx.nn as nn

from .config import DFlashConfig


def _rotate_half(x: mx.array) -> mx.array:
    half = x.shape[-1] // 2
    x1 = x[..., :half]
    x2 = x[..., half:]
    return mx.concatenate([-x2, x1], axis=-1)


class _Qwen3RotaryEmbedding:
    """Standard Qwen3 (rotate_half) RoPE with full rotary factor."""

    def __init__(self, head_dim: int, base: float):
        self.head_dim = head_dim
        self.base = base
        self.inv_freq = 1.0 / (
            base ** (mx.arange(0, head_dim, 2, dtype=mx.float32) / head_dim)
        )

    def __call__(self, position_ids: mx.array, dtype):
        # position_ids: [B, S]  ->  cos/sin: [B, S, head_dim]
        pos = position_ids.astype(mx.float32)
        freqs = pos[..., None] * self.inv_freq[None, None, :]  # [B, S, head_dim/2]
        emb = mx.concatenate([freqs, freqs], axis=-1)  # [B, S, head_dim]
        return mx.cos(emb).astype(dtype), mx.sin(emb).astype(dtype)


def _apply_rope(x: mx.array, cos: mx.array, sin: mx.array) -> mx.array:
    # x: [B, num_heads, S, head_dim], cos/sin: [B, 1, S, head_dim]
    return (x * cos) + (_rotate_half(x) * sin)


class DFlashAttention(nn.Module):
    """Non-causal attention where Q comes from noise positions and K/V come
    from the concatenation of target context + noise positions.

    K/V projections are shared between context and noise — the same k_proj /
    v_proj is applied to ``target_hidden`` (already fused via `fc`) and the
    current layer's noise hidden states, then the two are concatenated along
    the sequence dimension. This matches the reference implementation.
    """

    def __init__(self, config: DFlashConfig):
        super().__init__()
        self.num_heads = config.num_attention_heads
        self.num_kv_heads = config.num_key_value_heads
        self.head_dim = config.head_dim
        self.scale = self.head_dim ** -0.5

        self.q_proj = nn.Linear(
            config.hidden_size,
            self.num_heads * self.head_dim,
            bias=config.attention_bias,
        )
        self.k_proj = nn.Linear(
            config.hidden_size,
            self.num_kv_heads * self.head_dim,
            bias=config.attention_bias,
        )
        self.v_proj = nn.Linear(
            config.hidden_size,
            self.num_kv_heads * self.head_dim,
            bias=config.attention_bias,
        )
        self.o_proj = nn.Linear(
            self.num_heads * self.head_dim,
            config.hidden_size,
            bias=config.attention_bias,
        )
        self.q_norm = nn.RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.k_norm = nn.RMSNorm(self.head_dim, eps=config.rms_norm_eps)

    def __call__(
        self,
        noise: mx.array,          # [B, L, H]
        target_hidden: mx.array,  # [B, T, H] (already fused + normed)
        cos: mx.array,            # [B, T+L, head_dim]
        sin: mx.array,            # [B, T+L, head_dim]
        mask: Optional[mx.array] = None,
    ) -> mx.array:
        B, L, _ = noise.shape
        T = target_hidden.shape[1]

        q = self.q_proj(noise).reshape(B, L, self.num_heads, self.head_dim)
        q = self.q_norm(q).transpose(0, 2, 1, 3)  # [B, nH, L, D]

        k_ctx = self.k_proj(target_hidden)
        k_noise = self.k_proj(noise)
        k = mx.concatenate([k_ctx, k_noise], axis=1).reshape(
            B, T + L, self.num_kv_heads, self.head_dim
        )
        k = self.k_norm(k).transpose(0, 2, 1, 3)  # [B, nKV, T+L, D]

        v_ctx = self.v_proj(target_hidden)
        v_noise = self.v_proj(noise)
        v = mx.concatenate([v_ctx, v_noise], axis=1).reshape(
            B, T + L, self.num_kv_heads, self.head_dim
        )
        v = v.transpose(0, 2, 1, 3)  # [B, nKV, T+L, D]

        # RoPE: Q takes the last L positions (noise), K takes all T+L positions.
        cos_kv = cos[:, None, :, :]
        sin_kv = sin[:, None, :, :]
        cos_q = cos[:, None, -L:, :]
        sin_q = sin[:, None, -L:, :]

        q = _apply_rope(q, cos_q, sin_q)
        k = _apply_rope(k, cos_kv, sin_kv)

        out = mx.fast.scaled_dot_product_attention(
            q, k, v, scale=self.scale, mask=mask
        )
        out = out.transpose(0, 2, 1, 3).reshape(B, L, self.num_heads * self.head_dim)
        return self.o_proj(out)


class DFlashMLP(nn.Module):
    def __init__(self, config: DFlashConfig):
        super().__init__()
        self.gate_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.up_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.down_proj = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)

    def __call__(self, x: mx.array) -> mx.array:
        return self.down_proj(nn.silu(self.gate_proj(x)) * self.up_proj(x))


class DFlashDecoderLayer(nn.Module):
    def __init__(self, config: DFlashConfig):
        super().__init__()
        self.self_attn = DFlashAttention(config)
        self.mlp = DFlashMLP(config)
        self.input_layernorm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = nn.RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )

    def __call__(
        self,
        noise: mx.array,
        target_hidden: mx.array,
        cos: mx.array,
        sin: mx.array,
        mask: Optional[mx.array] = None,
    ) -> mx.array:
        h = noise + self.self_attn(
            self.input_layernorm(noise), target_hidden, cos, sin, mask
        )
        h = h + self.mlp(self.post_attention_layernorm(h))
        return h


class DFlashDraftModel(nn.Module):
    """Stateless block-diffusion drafter. See module docstring."""

    def __init__(self, config: DFlashConfig):
        super().__init__()
        self.config = config
        self.layers = [DFlashDecoderLayer(config) for _ in range(config.num_hidden_layers)]
        self.norm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.fc = nn.Linear(
            len(config.target_layer_ids) * config.hidden_size,
            config.hidden_size,
            bias=False,
        )
        self.hidden_norm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = _Qwen3RotaryEmbedding(config.head_dim, config.rope_theta)

    def __call__(
        self,
        noise_embedding: mx.array,        # [B, L, H]
        target_hidden: mx.array,          # [B, T, num_target_layers * H]
        position_ids: Optional[mx.array] = None,
    ) -> mx.array:
        B, L, _ = noise_embedding.shape
        T = target_hidden.shape[1]

        fused = self.hidden_norm(self.fc(target_hidden))  # [B, T, H]

        if position_ids is None:
            position_ids = mx.arange(T + L, dtype=mx.int32)[None, :]
            position_ids = mx.broadcast_to(position_ids, (B, T + L))

        cos, sin = self.rotary_emb(position_ids, noise_embedding.dtype)

        h = noise_embedding
        for layer in self.layers:
            h = layer(h, fused, cos, sin)
        return self.norm(h)

    def sanitize(self, weights: dict) -> dict:
        # Reference repo uses the key prefixes `layers.*`, `fc`, `hidden_norm`,
        # `norm` directly (no outer "model." wrapper) — but some checkpoints
        # do wrap them. Strip an optional leading "model." for robustness.
        out = {}
        for k, v in weights.items():
            if k.startswith("model."):
                k = k[len("model."):]
            out[k] = v
        return out
