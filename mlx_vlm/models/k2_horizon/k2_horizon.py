from typing import Optional

import mlx.core as mx
import mlx.nn as nn

from .config import K2HorizonConfig


class K2HorizonAttention(nn.Module):
    def __init__(self, config: K2HorizonConfig):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.num_kv_heads = config.num_key_value_heads
        self.head_dim = config.head_dim
        self.num_key_value_groups = self.num_heads // self.num_kv_heads

        self.q_proj = nn.Linear(
            self.hidden_size, self.num_heads * self.head_dim, bias=config.attention_bias
        )
        self.k_proj = nn.Linear(
            self.hidden_size,
            self.num_kv_heads * self.head_dim,
            bias=config.attention_bias,
        )
        self.v_proj = nn.Linear(
            self.hidden_size,
            self.num_kv_heads * self.head_dim,
            bias=config.attention_bias,
        )
        self.o_proj = nn.Linear(
            self.num_heads * self.head_dim, self.hidden_size, bias=config.attention_bias
        )

    def __call__(
        self,
        hidden_states: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional[dict] = None,
    ) -> mx.array:
        batch_size, q_len, _ = hidden_states.shape

        query = self.q_proj(hidden_states)
        key = self.k_proj(hidden_states)
        value = self.v_proj(hidden_states)

        query = query.reshape(
            batch_size, q_len, self.num_heads, self.head_dim
        ).transpose(0, 2, 1, 3)
        key = key.reshape(
            batch_size, q_len, self.num_kv_heads, self.head_dim
        ).transpose(0, 2, 1, 3)
        value = value.reshape(
            batch_size, q_len, self.num_kv_heads, self.head_dim
        ).transpose(0, 2, 1, 3)

        # Apply RoPE (simplified - full RoPE uses frequency calculations)
        # For production use, reference qwen3_5 or internlm3 rope implementation

        # Repeat k/v heads if needed
        if self.num_key_value_groups > 1:
            key = key.repeat_interleave(self.num_key_value_groups, axis=1)
            value = value.repeat_interleave(self.num_key_value_groups, axis=1)

        # Scaled dot-product attention
        scale = 1.0 / mx.sqrt(self.head_dim)
        scores = mx.matmul(query, key.transpose(0, 1, 3, 2)) * scale

        if mask is not None:
            scores = scores + mask

        attn_weights = mx.softmax(scores, axis=-1)
        output = mx.matmul(attn_weights, value)
        output = output.transpose(0, 2, 1, 3).reshape(
            batch_size, q_len, self.hidden_size
        )
        output = self.o_proj(output)
        return output


class K2HorizonMLP(nn.Module):
    def __init__(self, config: K2HorizonConfig):
        super().__init__()
        self.gate_proj = nn.Linear(
            config.hidden_size, config.intermediate_size, bias=False
        )
        self.up_proj = nn.Linear(
            config.hidden_size, config.intermediate_size, bias=False
        )
        self.down_proj = nn.Linear(
            config.intermediate_size, config.hidden_size, bias=False
        )
        self.act_fn = nn.silu

    def __call__(self, x: mx.array) -> mx.array:
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))


class K2HorizonDecoderLayer(nn.Module):
    def __init__(self, config: K2HorizonConfig, layer_idx: int):
        super().__init__()
        self.self_attn = K2HorizonAttention(config)
        self.mlp = K2HorizonMLP(config)
        self.input_layernorm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = nn.RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )
        self.layer_idx = layer_idx

    def __call__(
        self,
        hidden_states: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional[dict] = None,
    ) -> mx.array:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.self_attn(hidden_states, mask=mask, cache=cache)
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
        return hidden_states


class K2HorizonModel(nn.Module):
    def __init__(self, config: K2HorizonConfig):
        super().__init__()
        self.config = config
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = [
            K2HorizonDecoderLayer(config, layer_idx)
            for layer_idx in range(config.num_hidden_layers)
        ]
        self.norm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def __call__(
        self,
        input_ids: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional[dict] = None,
    ) -> mx.array:
        hidden_states = self.embed_tokens(input_ids)
        for layer in self.layers:
            hidden_states = layer(hidden_states, mask=mask, cache=cache)
        hidden_states = self.norm(hidden_states)
        return hidden_states


class Model(nn.Module):
    def __init__(self, config: K2HorizonConfig):
        super().__init__()
        self.config = config
        self.model = K2HorizonModel(config)
        # Note: for full e2e, add lm_head projection if tie_word_embeddings is False
        # The K2-Horizon family uses separate output weights in some variants

    def __call__(
        self,
        input_ids: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional[dict] = None,
    ) -> mx.array:
        hidden_states = self.model(input_ids, mask=mask, cache=cache)
        # Output projection - for full implementation, add separate lm_head
        # if config.tie_word_embeddings is False
        return hidden_states
