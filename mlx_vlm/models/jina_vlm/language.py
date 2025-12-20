"""Language model decoder for Jina VLM in MLX."""

from typing import List, Optional

import mlx.core as mx
import mlx.nn as nn
from mlx_lm.models.base import create_attention_mask, scaled_dot_product_attention
from mlx_lm.models.cache import KVCache

from ..base import LanguageModelOutput
from .config import TextConfig


class RMSNorm(nn.Module):
    """RMS Layer Normalization."""

    def __init__(self, dims: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = mx.ones((dims,))

    def __call__(self, x: mx.array) -> mx.array:
        rms = mx.sqrt(mx.mean(x * x, axis=-1, keepdims=True) + self.eps)
        return self.weight * (x / rms)


class RoPE(nn.Module):
    """Rotary Positional Embeddings."""

    def __init__(self, dims: int, theta: float = 1000000.0):
        super().__init__()
        self.dims = dims
        self.theta = theta
        inv_freq = 1.0 / (theta ** (mx.arange(0, dims, 2).astype(mx.float32) / dims))
        self._inv_freq = inv_freq

    def __call__(self, x: mx.array, offset: int = 0) -> mx.array:
        seq_len = x.shape[2]
        positions = mx.arange(offset, offset + seq_len).astype(mx.float32)
        freqs = positions[:, None] * self._inv_freq[None, :]
        emb = mx.concatenate([freqs, freqs], axis=-1)
        cos = mx.cos(emb)[None, None, :, :]
        sin = mx.sin(emb)[None, None, :, :]
        x1 = x[..., : self.dims // 2]
        x2 = x[..., self.dims // 2 :]
        rotated = mx.concatenate([-x2, x1], axis=-1)
        return (x * cos + rotated * sin).astype(x.dtype)


class Attention(nn.Module):
    """Multi-head attention with GQA and RoPE - matches weight naming: attn.qkv, attn.out"""

    def __init__(self, config: TextConfig):
        super().__init__()
        self.config = config
        self.num_heads = config.num_attention_heads
        self.num_kv_heads = config.num_key_value_heads
        self.head_dim = config.head_dim
        self.num_kv_groups = self.num_heads // self.num_kv_heads
        self.scale = self.head_dim**-0.5

        # Fused QKV projection - named to match weights
        qkv_size = (
            config.num_attention_heads + 2 * config.num_key_value_heads
        ) * config.head_dim
        self.qkv = nn.Linear(config.hidden_size, qkv_size, bias=False)
        self.out = nn.Linear(
            config.num_attention_heads * config.head_dim, config.hidden_size, bias=False
        )

        # QK normalization - named to match weights
        if config.use_qk_norm:
            self.q_norm = RMSNorm(config.head_dim, eps=config.rms_norm_eps)
            self.k_norm = RMSNorm(config.head_dim, eps=config.rms_norm_eps)
        else:
            self.q_norm = None
            self.k_norm = None

        self.rope = RoPE(config.head_dim, theta=config.rope_theta)

    def __call__(
        self,
        x: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional[KVCache] = None,
    ) -> mx.array:
        B, L, _ = x.shape

        # Compute fused QKV
        qkv = self.qkv(x)
        q_size = self.num_heads * self.head_dim
        kv_size = self.num_kv_heads * self.head_dim

        q = qkv[..., :q_size]
        k = qkv[..., q_size : q_size + kv_size]
        v = qkv[..., q_size + kv_size :]

        q = q.reshape(B, L, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)
        k = k.reshape(B, L, self.num_kv_heads, self.head_dim).transpose(0, 2, 1, 3)
        v = v.reshape(B, L, self.num_kv_heads, self.head_dim).transpose(0, 2, 1, 3)

        if self.q_norm is not None:
            q = self.q_norm(q)
            k = self.k_norm(k)

        if cache is not None:
            q = self.rope(q, offset=cache.offset)
            k = self.rope(k, offset=cache.offset)
            k, v = cache.update_and_fetch(k, v)
        else:
            q = self.rope(q)
            k = self.rope(k)

        output = scaled_dot_product_attention(
            q, k, v, cache=cache, scale=self.scale, mask=mask
        )
        output = output.transpose(0, 2, 1, 3).reshape(B, L, -1)
        return self.out(output)


class MLP(nn.Module):
    """MLP with SwiGLU - matches weight naming: ffn.gate_up, ffn.down"""

    def __init__(self, config: TextConfig):
        super().__init__()
        # Fused gate and up projection - named to match weights
        self.gate_up = nn.Linear(
            config.hidden_size, 2 * config.intermediate_size, bias=False
        )
        self.down = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)

    def __call__(self, x: mx.array) -> mx.array:
        gate_up = self.gate_up(x)
        # Jina VLM convention: first half is value, second half is gate (activated)
        up, gate = mx.split(gate_up, 2, axis=-1)
        return self.down(nn.silu(gate) * up)


class TransformerBlock(nn.Module):
    """Transformer block - matches weight naming: attn_norm, ffn_norm"""

    def __init__(self, config: TextConfig, layer_idx: int = 0):
        super().__init__()
        # Named to match weights
        self.attn_norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.attn = Attention(config)
        self.ffn_norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.ffn = MLP(config)

    def __call__(
        self,
        x: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional[KVCache] = None,
    ) -> mx.array:
        h = self.attn(self.attn_norm(x), mask=mask, cache=cache)
        x = x + h
        x = x + self.ffn(self.ffn_norm(x))
        return x


class ExtendedEmbedding(nn.Module):
    """Embedding with additional tokens - matches weight naming: embedding, new_embedding"""

    def __init__(self, vocab_size: int, additional_size: int, dims: int):
        super().__init__()
        self.vocab_size = vocab_size
        self.additional_size = additional_size
        # Named to match weights
        self.embedding = mx.zeros((vocab_size, dims))
        self.new_embedding = mx.zeros((additional_size, dims))

    def __call__(self, x: mx.array) -> mx.array:
        full_embedding = mx.concatenate([self.embedding, self.new_embedding], axis=0)
        return full_embedding[x]


class TextModel(nn.Module):
    """Text decoder model - matches weight naming structure"""

    def __init__(self, config: TextConfig):
        super().__init__()
        self.config = config

        # Named to match weights: language_model.embedding
        if config.additional_vocab_size > 0:
            self.embedding = ExtendedEmbedding(
                config.vocab_size, config.additional_vocab_size, config.hidden_size
            )
        else:
            self.embedding = nn.Embedding(config.vocab_size, config.hidden_size)

        self.layers = [
            TransformerBlock(config, layer_idx=i)
            for i in range(config.num_hidden_layers)
        ]

        # Named to match weights: language_model.ln_f
        self.ln_f = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def __call__(
        self,
        input_ids: mx.array,
        inputs_embeds: Optional[mx.array] = None,
        mask: Optional[mx.array] = None,
        cache: Optional[List[KVCache]] = None,
    ) -> mx.array:
        if inputs_embeds is None:
            x = self.embedding(input_ids)
        else:
            x = inputs_embeds

        for i, layer in enumerate(self.layers):
            layer_cache = cache[i] if cache is not None else None
            x = layer(x, mask=mask, cache=layer_cache)

        return self.ln_f(x)


class LanguageModel(nn.Module):
    """Language model wrapper - the TextModel is accessed as language_model in weights"""

    def __init__(self, config: TextConfig):
        super().__init__()
        self.config = config
        self.model_type = config.model_type
        # This will be loaded under "language_model" prefix
        self.embedding = None  # Handled by sanitize
        self.layers = None  # Handled by sanitize
        self.ln_f = None  # Handled by sanitize

        # Build the actual model components directly here
        # They'll be found via language_model.embedding, language_model.layers, etc.
        if config.additional_vocab_size > 0:
            self.embedding = ExtendedEmbedding(
                config.vocab_size, config.additional_vocab_size, config.hidden_size
            )
        else:
            self.embedding = nn.Embedding(config.vocab_size, config.hidden_size)

        self.layers = [
            TransformerBlock(config, layer_idx=i)
            for i in range(config.num_hidden_layers)
        ]
        self.ln_f = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def __call__(
        self,
        input_ids: mx.array,
        inputs_embeds: Optional[mx.array] = None,
        mask: Optional[mx.array] = None,
        cache: Optional[List[KVCache]] = None,
        **kwargs,
    ) -> LanguageModelOutput:
        if inputs_embeds is None:
            x = self.embedding(input_ids)
        else:
            x = inputs_embeds

        # Initialize cache if needed
        if cache is None:
            cache = [None] * len(self.layers)

        # Create causal attention mask
        mask = create_attention_mask(x, cache)

        for i, layer in enumerate(self.layers):
            layer_cache = cache[i] if cache is not None else None
            x = layer(x, mask=mask, cache=layer_cache)

        hidden_states = self.ln_f(x)
        logits = self.lm_head(hidden_states)
        return LanguageModelOutput(logits=logits)

    def _create_causal_mask(self, seq_len: int, offset: int = 0) -> mx.array:
        total_len = offset + seq_len
        mask = mx.triu(mx.full((seq_len, total_len), float("-inf")), k=offset + 1)
        return mask[None, None, :, :]
