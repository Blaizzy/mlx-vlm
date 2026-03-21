import math
from typing import Optional

import mlx.core as mx
import mlx.nn as nn
from mlx_lm.models.base import create_attention_mask

from ..base import LanguageModelOutput
from .config import ModelConfig, TextConfig


def rotate_half(x):
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return mx.concatenate([-x2, x1], axis=-1)


def apply_rotary_pos_emb(q, k, cos, sin):
    cos = mx.expand_dims(cos, axis=1)
    sin = mx.expand_dims(sin, axis=1)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


class Molmo2RMSNorm(nn.Module):
    def __init__(self, size: int, eps: float = 1e-6):
        super().__init__()
        self.weight = mx.ones((size,))
        self.eps = eps

    def __call__(self, x: mx.array) -> mx.array:
        dtype = x.dtype
        x = x.astype(mx.float32)
        variance = mx.mean(x * x, axis=-1, keepdims=True)
        x = x * mx.rsqrt(variance + self.eps)
        return (self.weight * x).astype(dtype)


class Molmo2RotaryEmbedding(nn.Module):
    def __init__(self, config: TextConfig):
        super().__init__()
        self.dim = config.head_dim
        self.max_position_embeddings = config.max_position_embeddings
        self.rope_theta = config.rope_theta
        inv_freq = 1.0 / (
            self.rope_theta
            ** (mx.arange(0, self.dim, 2, dtype=mx.float32) / self.dim)
        )
        self._inv_freq = inv_freq

    def __call__(self, x, position_ids):
        inv_freq = self._inv_freq
        inv_freq_expanded = mx.broadcast_to(
            inv_freq[None, :, None],
            (position_ids.shape[0], inv_freq.shape[0], 1),
        )
        position_ids_expanded = position_ids[:, None, :].astype(mx.float32)
        freqs = (inv_freq_expanded.astype(mx.float32) @ position_ids_expanded).transpose(
            0, 2, 1
        )
        emb = mx.concatenate([freqs, freqs], axis=-1)
        cos = mx.cos(emb)
        sin = mx.sin(emb)
        return cos.astype(x.dtype), sin.astype(x.dtype)


class Molmo2Attention(nn.Module):
    def __init__(self, config: TextConfig, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.num_heads = config.num_attention_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = config.num_attention_heads // config.num_key_value_heads
        self.head_dim = config.head_dim
        self.scale = self.head_dim**-0.5

        total_qkv = (
            config.num_attention_heads * config.head_dim
            + 2 * config.num_key_value_heads * config.head_dim
        )
        self.att_proj = nn.Linear(config.hidden_size, total_qkv, bias=config.qkv_bias)
        self.attn_out = nn.Linear(
            config.head_dim * config.num_attention_heads,
            config.hidden_size,
            bias=False,
        )

        self.q_norm = None
        self.k_norm = None
        if config.use_qk_norm:
            if config.qk_norm_type == "qwen3":
                self.q_norm = Molmo2RMSNorm(config.head_dim, eps=config.layer_norm_eps)
                self.k_norm = Molmo2RMSNorm(config.head_dim, eps=config.layer_norm_eps)
            else:
                self.q_norm = Molmo2RMSNorm(
                    config.num_attention_heads * config.head_dim, eps=config.layer_norm_eps
                )
                self.k_norm = Molmo2RMSNorm(
                    config.num_key_value_heads * config.head_dim, eps=config.layer_norm_eps
                )
            self.qk_norm_type = config.qk_norm_type
        else:
            self.qk_norm_type = None

    def __call__(
        self,
        hidden_states: mx.array,
        mask: Optional[mx.array] = None,
        cache=None,
        position_embeddings=None,
    ) -> mx.array:
        B, L, _ = hidden_states.shape

        qkv = self.att_proj(hidden_states)

        q_dim = self.num_heads * self.head_dim
        kv_dim = self.num_key_value_heads * self.head_dim
        queries = qkv[..., :q_dim]
        keys = qkv[..., q_dim : q_dim + kv_dim]
        values = qkv[..., q_dim + kv_dim :]

        # Apply QK norm before reshape if not qwen3 style
        if self.q_norm is not None and self.qk_norm_type != "qwen3":
            queries = self.q_norm(queries)
            keys = self.k_norm(keys)

        queries = queries.reshape(B, L, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)
        keys = keys.reshape(B, L, self.num_key_value_heads, self.head_dim).transpose(0, 2, 1, 3)
        values = values.reshape(B, L, self.num_key_value_heads, self.head_dim).transpose(0, 2, 1, 3)

        # Apply QK norm after reshape if qwen3 style (per-head)
        if self.q_norm is not None and self.qk_norm_type == "qwen3":
            queries = self.q_norm(queries)
            keys = self.k_norm(keys)

        # Apply RoPE
        if position_embeddings is not None:
            cos, sin = position_embeddings
            queries, keys = apply_rotary_pos_emb(queries, keys, cos, sin)

        # Update KV cache
        if cache is not None:
            keys, values = cache.update_and_fetch(keys, values)

        # GQA expansion
        if self.num_heads != self.num_key_value_heads:
            keys = mx.repeat(keys, self.num_key_value_groups, axis=1)
            values = mx.repeat(values, self.num_key_value_groups, axis=1)

        # Use mx.fast.scaled_dot_product_attention for efficient attention
        attn_output = mx.fast.scaled_dot_product_attention(
            queries, keys, values, scale=self.scale, mask=mask
        )

        attn_output = attn_output.transpose(0, 2, 1, 3).reshape(B, L, -1)
        return self.attn_out(attn_output)


class LanguageModelMLP(nn.Module):
    def __init__(self, input_dim: int, intermediate_size: int):
        super().__init__()
        self.ff_proj = nn.Linear(input_dim, intermediate_size * 2, bias=False)
        self.ff_out = nn.Linear(intermediate_size, input_dim, bias=False)

    def __call__(self, x: mx.array) -> mx.array:
        x = self.ff_proj(x)
        x, gate = mx.split(x, 2, axis=-1)
        x = nn.silu(gate) * x
        x = self.ff_out(x)
        return x


class Molmo2DecoderLayer(nn.Module):
    def __init__(self, config: TextConfig, layer_idx: int):
        super().__init__()
        self.self_attn = Molmo2Attention(config, layer_idx)
        self.attn_norm = Molmo2RMSNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.mlp = LanguageModelMLP(config.hidden_size, config.intermediate_size)
        self.ff_norm = Molmo2RMSNorm(config.hidden_size, eps=config.layer_norm_eps)

    def __call__(
        self,
        hidden_states: mx.array,
        mask: Optional[mx.array] = None,
        cache=None,
        position_embeddings=None,
    ):
        residual = hidden_states
        hidden_states = self.attn_norm(hidden_states)
        hidden_states = self.self_attn(
            hidden_states, mask=mask, cache=cache, position_embeddings=position_embeddings
        )
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.ff_norm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        return hidden_states


class Molmo2Embedding(nn.Module):
    def __init__(self, num_embeddings: int, num_new_embeddings: int, features: int):
        super().__init__()
        self.embedding = mx.zeros((num_embeddings, features))
        self.new_embedding = mx.zeros((num_new_embeddings, features))

    def __call__(self, x: mx.array) -> mx.array:
        full_embedding = mx.concatenate([self.embedding, self.new_embedding], axis=0)
        return full_embedding[x]


class TextModel(nn.Module):
    def __init__(self, config: TextConfig):
        super().__init__()
        self.config = config
        if config.additional_vocab_size is not None and config.additional_vocab_size > 0:
            self.wte = Molmo2Embedding(
                config.vocab_size,
                config.additional_vocab_size,
                config.hidden_size,
            )
        else:
            self.wte = nn.Embedding(config.vocab_size, config.hidden_size)

        self.blocks = [
            Molmo2DecoderLayer(config, layer_idx)
            for layer_idx in range(config.num_hidden_layers)
        ]
        self.ln_f = Molmo2RMSNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.rotary_emb = Molmo2RotaryEmbedding(config)

    def __call__(
        self,
        input_ids: Optional[mx.array] = None,
        inputs_embeds: Optional[mx.array] = None,
        mask: Optional[mx.array] = None,
        cache=None,
    ):
        if inputs_embeds is None:
            safe_ids = mx.where(input_ids != -1, input_ids, 0)
            inputs_embeds = self.wte(safe_ids)

        h = inputs_embeds

        if cache is None:
            cache = [None] * len(self.blocks)

        if mask is None:
            mask = create_attention_mask(h, cache[0])

        # Position IDs
        if cache[0] is not None:
            offset = cache[0].offset
        else:
            offset = 0
        position_ids = mx.arange(offset, offset + h.shape[1])[None]
        position_embeddings = self.rotary_emb(h, position_ids)

        for layer, c in zip(self.blocks, cache):
            h = layer(h, mask=mask, cache=c, position_embeddings=position_embeddings)

        pre_ln_hidden_state = h
        h = self.ln_f(h)
        return h, pre_ln_hidden_state


class LanguageModel(nn.Module):
    def __init__(self, config: TextConfig, model_config: ModelConfig = None):
        super().__init__()
        self.config = config
        self.model = TextModel(config)

    def __call__(
        self,
        input_ids: Optional[mx.array] = None,
        inputs_embeds: Optional[mx.array] = None,
        mask: Optional[mx.array] = None,
        cache=None,
        **kwargs,
    ):
        out, _ = self.model(input_ids, inputs_embeds, mask, cache)
        lm_head = kwargs.get("lm_head", None)
        if lm_head is not None:
            out = lm_head(out)
            return LanguageModelOutput(logits=out)
        return LanguageModelOutput(logits=out)

    @property
    def layers(self):
        return self.model.blocks

    @staticmethod
    def sanitize(weights):
        sanitized = {}
        for k, v in weights.items():
            # Pass through all weights
            sanitized[k] = v
        return sanitized
