import math

import mlx.core as mx
from mlx import nn
from mlx.core.fast import scaled_dot_product_attention

from mlx_vlm.models.bonsai.qwen.rms_norm import Qwen3VLRMSNorm

Qwen3VLKVCache = tuple[mx.array, mx.array, int]


class Qwen3VLAttention(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        num_attention_heads: int,
        num_key_value_heads: int,
        head_dim: int,
        max_position_embeddings: int = 262144,
        rope_theta: float = 1000000.0,
        mrope_section: list[int] | None = None,
        attention_bias: bool = False,
        rms_norm_eps: float = 1e-6,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.head_dim = head_dim
        self.num_key_value_groups = num_attention_heads // num_key_value_heads
        self.scaling = 1.0 / math.sqrt(self.head_dim)
        self.q_proj = nn.Linear(
            hidden_size, num_attention_heads * head_dim, bias=attention_bias
        )
        self.k_proj = nn.Linear(
            hidden_size, num_key_value_heads * head_dim, bias=attention_bias
        )
        self.v_proj = nn.Linear(
            hidden_size, num_key_value_heads * head_dim, bias=attention_bias
        )
        self.o_proj = nn.Linear(
            num_attention_heads * head_dim, hidden_size, bias=attention_bias
        )
        self.q_norm = Qwen3VLRMSNorm(head_dim, eps=rms_norm_eps)
        self.k_norm = Qwen3VLRMSNorm(head_dim, eps=rms_norm_eps)
        self.mrope_section = mrope_section or [24, 20, 20]

    def __call__(
        self,
        hidden_states: mx.array,
        attention_mask: mx.array | None = None,
        position_embeddings: tuple[mx.array, mx.array] | None = None,
        past_key_value: Qwen3VLKVCache | None = None,
        max_cache_length: int | None = None,
    ) -> mx.array | tuple[mx.array, Qwen3VLKVCache]:
        bsz, q_len, _ = hidden_states.shape

        q_proj = self.q_proj(hidden_states)
        k_proj = self.k_proj(hidden_states)
        v_proj = self.v_proj(hidden_states)

        query_states = q_proj.reshape(
            bsz, q_len, self.num_attention_heads, self.head_dim
        )
        key_states = k_proj.reshape(bsz, q_len, self.num_key_value_heads, self.head_dim)
        value_states = v_proj.reshape(
            bsz, q_len, self.num_key_value_heads, self.head_dim
        )

        query_states = self.q_norm(query_states)
        key_states = self.k_norm(key_states)

        query_states = query_states.transpose(0, 2, 1, 3)
        key_states = key_states.transpose(0, 2, 1, 3)
        value_states = value_states.transpose(0, 2, 1, 3)

        if position_embeddings is not None:
            cos, sin = position_embeddings
            query_states, key_states = Qwen3VLAttention._apply_rotary_pos_emb(
                q=query_states,
                k=key_states,
                cos=cos,
                sin=sin,
            )

        if past_key_value is None:
            if max_cache_length is None:
                cache_key_states = key_states
                cache_value_states = value_states
                cache_length = q_len
            else:
                cache_shape = (
                    bsz,
                    self.num_key_value_heads,
                    max_cache_length,
                    self.head_dim,
                )
                cache_key_states = mx.zeros(cache_shape, dtype=key_states.dtype)
                cache_value_states = mx.zeros(cache_shape, dtype=value_states.dtype)
                start_indices = mx.array([0, 0, 0, 0], dtype=mx.int32)
                cache_key_states = mx.slice_update(
                    cache_key_states,
                    key_states,
                    start_indices=start_indices,
                    axes=(0, 1, 2, 3),
                )
                cache_value_states = mx.slice_update(
                    cache_value_states,
                    value_states,
                    start_indices=start_indices,
                    axes=(0, 1, 2, 3),
                )
                cache_length = q_len
        else:
            cache_key_states, cache_value_states, existing_cache_length = past_key_value
            start_indices = mx.array([0, 0, existing_cache_length, 0], dtype=mx.int32)
            cache_key_states = mx.slice_update(
                cache_key_states,
                key_states,
                start_indices=start_indices,
                axes=(0, 1, 2, 3),
            )
            cache_value_states = mx.slice_update(
                cache_value_states,
                value_states,
                start_indices=start_indices,
                axes=(0, 1, 2, 3),
            )
            cache_length = existing_cache_length + q_len

        valid_key_states = cache_key_states[:, :, :cache_length, :]
        valid_value_states = cache_value_states[:, :, :cache_length, :]

        if self.num_key_value_heads != self.num_attention_heads:
            key_states = Qwen3VLAttention._repeat_kv(
                valid_key_states, self.num_key_value_groups
            )
            value_states = Qwen3VLAttention._repeat_kv(
                valid_value_states, self.num_key_value_groups
            )
        else:
            key_states = valid_key_states
            value_states = valid_value_states

        attn_mask = None
        if attention_mask is not None:
            kv_len = key_states.shape[2]
            attn_mask = attention_mask[:, :, :, :kv_len]

        query_states_f32 = query_states.astype(mx.float32)
        key_states_f32 = key_states.astype(mx.float32)
        value_states_f32 = value_states.astype(mx.float32)

        # Use fast MLX attention
        attn_output = scaled_dot_product_attention(
            query_states_f32,
            key_states_f32,
            value_states_f32,
            scale=self.scaling,
            mask=attn_mask,
        )

        attn_output = attn_output.astype(query_states.dtype)
        attn_output = attn_output.transpose(0, 2, 1, 3).reshape(
            bsz, q_len, self.num_attention_heads * self.head_dim
        )
        attn_output = self.o_proj(attn_output)
        return attn_output, (cache_key_states, cache_value_states, cache_length)

    @staticmethod
    def _repeat_kv(hidden_states: mx.array, n_rep: int) -> mx.array:
        shape = hidden_states.shape

        if len(shape) == 5:
            batch, num_key_value_heads, rep, slen, head_dim = shape
            return hidden_states.reshape(
                batch, num_key_value_heads * rep, slen, head_dim
            )

        batch, num_key_value_heads, slen, head_dim = shape
        hidden_states = mx.expand_dims(hidden_states, axis=2)
        hidden_states = mx.broadcast_to(
            hidden_states, (batch, num_key_value_heads, n_rep, slen, head_dim)
        )
        return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)

    @staticmethod
    def _apply_rotary_pos_emb(
        q: mx.array,
        k: mx.array,
        cos: mx.array,
        sin: mx.array,
        unsqueeze_dim: int = 1,
    ) -> tuple[mx.array, mx.array]:
        cos = mx.expand_dims(cos, axis=unsqueeze_dim)
        sin = mx.expand_dims(sin, axis=unsqueeze_dim)
        q_embed = (q * cos) + (Qwen3VLAttention._rotate_half(q) * sin)
        k_embed = (k * cos) + (Qwen3VLAttention._rotate_half(k) * sin)
        return q_embed, k_embed

    @staticmethod
    def _rotate_half(x: mx.array) -> mx.array:
        x1 = x[..., : x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2 :]
        return mx.concatenate([-x2, x1], axis=-1)
