from typing import Optional

import mlx.core as mx
import mlx.nn as nn

from ..base import (
    LanguageModelOutput,
    create_attention_mask,
    scaled_dot_product_attention,
)
from ..cache import KVCache
from .config import ModelConfig


class HrmTextRMSNorm(nn.Module):
    def __init__(self, eps: float = 1e-6):
        super().__init__()
        self.eps = eps

    def __call__(self, x: mx.array) -> mx.array:
        output = x.astype(mx.float32)
        output = output * mx.rsqrt(
            mx.mean(mx.square(output), axis=-1, keepdims=True) + self.eps
        )
        return output.astype(x.dtype)


def _make_attention_mask(
    hidden_states: mx.array,
    cache,
    attention_mask: Optional[mx.array] = None,
    token_type_ids: Optional[mx.array] = None,
    prefix_lm: bool = True,
):
    base_mask = create_attention_mask(hidden_states, cache)
    if attention_mask is None and (token_type_ids is None or not prefix_lm):
        return base_mask

    B, L, _ = hidden_states.shape
    cache_offset = cache.offset if cache is not None else 0
    if cache_offset != 0 or L == 1:
        causal = mx.ones((L, cache_offset + L), dtype=mx.bool_)
    else:
        positions = mx.arange(L)
        causal = positions[:, None] >= positions[None, :]
        if token_type_ids is not None and prefix_lm:
            prefix = token_type_ids.astype(mx.int32) == 1
            causal = causal[None, :, :] | (prefix[:, :, None] & prefix[:, None, :])

    if causal.ndim == 2:
        causal = mx.broadcast_to(causal[None, :, :], (B, *causal.shape))

    if attention_mask is not None and attention_mask.ndim == 2:
        key_mask = attention_mask.astype(mx.bool_)
        if key_mask.shape[-1] != causal.shape[-1]:
            key_mask = key_mask[:, -causal.shape[-1] :]
        causal = causal & key_mask[:, None, :]

    min_dtype = mx.finfo(hidden_states.dtype).min
    return mx.where(causal[:, None, :, :], 0.0, min_dtype).astype(hidden_states.dtype)


class HrmTextMLP(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.gate_proj = nn.Linear(
            config.hidden_size, config.intermediate_size, bias=config.mlp_bias
        )
        self.up_proj = nn.Linear(
            config.hidden_size, config.intermediate_size, bias=config.mlp_bias
        )
        self.down_proj = nn.Linear(
            config.intermediate_size, config.hidden_size, bias=config.mlp_bias
        )

    def __call__(self, x: mx.array) -> mx.array:
        return self.down_proj(nn.silu(self.gate_proj(x)) * self.up_proj(x))


class HrmTextAttention(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.n_heads = config.num_attention_heads
        self.n_kv_heads = config.num_key_value_heads
        self.head_dim = config.head_dim
        self.scale = self.head_dim**-0.5

        dim = config.hidden_size
        self.q_proj = nn.Linear(
            dim, self.n_heads * self.head_dim, bias=config.attention_bias
        )
        self.k_proj = nn.Linear(
            dim, self.n_kv_heads * self.head_dim, bias=config.attention_bias
        )
        self.v_proj = nn.Linear(
            dim, self.n_kv_heads * self.head_dim, bias=config.attention_bias
        )
        self.o_proj = nn.Linear(
            self.n_heads * self.head_dim, dim, bias=config.attention_bias
        )
        self.gate_proj = nn.Linear(
            dim, self.n_heads * self.head_dim, bias=config.attention_bias
        )
        self.rope = nn.RoPE(
            self.head_dim,
            traditional=False,
            base=config.rope_theta,
        )

    def __call__(
        self,
        x: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional[KVCache] = None,
    ) -> mx.array:
        B, L, _ = x.shape

        queries = self.q_proj(x).reshape(B, L, self.n_heads, self.head_dim)
        keys = self.k_proj(x).reshape(B, L, self.n_kv_heads, self.head_dim)
        values = self.v_proj(x).reshape(B, L, self.n_kv_heads, self.head_dim)
        gate = self.gate_proj(x).reshape(B, L, self.n_heads, self.head_dim)

        queries = queries.transpose(0, 2, 1, 3)
        keys = keys.transpose(0, 2, 1, 3)
        values = values.transpose(0, 2, 1, 3)

        offset = cache.offset if cache is not None else 0
        queries = self.rope(queries, offset=offset)
        keys = self.rope(keys, offset=offset)

        if cache is not None:
            keys, values = cache.update_and_fetch(keys, values)

        output = scaled_dot_product_attention(
            queries, keys, values, cache=cache, scale=self.scale, mask=mask
        )
        output = output.transpose(0, 2, 1, 3) * mx.sigmoid(gate)
        output = output.reshape(B, L, -1)
        return self.o_proj(output)


class HrmTextDecoderLayer(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.self_attn = HrmTextAttention(config)
        self.mlp = HrmTextMLP(config)
        self.input_layernorm = HrmTextRMSNorm(config.rms_norm_eps)
        self.post_attention_layernorm = HrmTextRMSNorm(config.rms_norm_eps)

    def __call__(
        self,
        x: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional[KVCache] = None,
    ) -> mx.array:
        h = x + self.self_attn(self.input_layernorm(x), mask=mask, cache=cache)
        return h + self.mlp(self.post_attention_layernorm(h))


class HrmTextStack(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.layers = [
            HrmTextDecoderLayer(config) for _ in range(config.num_layers_per_stack)
        ]
        self.final_norm = HrmTextRMSNorm(config.rms_norm_eps)

    def __call__(
        self,
        x: mx.array,
        mask: Optional[mx.array] = None,
        cache=None,
    ) -> mx.array:
        if cache is None:
            cache = [None] * len(self.layers)
        for layer, layer_cache in zip(self.layers, cache):
            x = layer(x, mask=mask, cache=layer_cache)
        return self.final_norm(x)


class HrmTextModel(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.embedding_scale = config.embedding_scale
        self.L_module = HrmTextStack(config)
        self.H_module = HrmTextStack(config)
        self.z_L_init = mx.zeros((config.hidden_size,))
        raw_bp = list(config.L_bp_cycles)
        self.L_bp_cycles_padded = [1] * max(0, config.H_cycles - len(raw_bp)) + raw_bp

    @property
    def layers(self):
        layers = []
        for _ in range(self.config.H_cycles):
            for _ in range(self.config.L_cycles):
                layers.extend(self.L_module.layers)
            layers.extend(self.H_module.layers)
        return layers

    def __call__(
        self,
        inputs: mx.array,
        inputs_embeds: Optional[mx.array] = None,
        mask: Optional[mx.array] = None,
        attention_mask: Optional[mx.array] = None,
        cache=None,
        token_type_ids: Optional[mx.array] = None,
        **kwargs,
    ) -> mx.array:
        hidden_states_high = (
            self.embed_tokens(inputs) if inputs_embeds is None else inputs_embeds
        )
        hidden_states_high = hidden_states_high * self.embedding_scale
        hidden_states_low = mx.broadcast_to(
            self.z_L_init.astype(hidden_states_high.dtype),
            hidden_states_high.shape,
        )

        if cache is None:
            cache = [None] * self.config.num_hidden_layers

        if mask is None:
            mask = attention_mask
        if mask is not None and getattr(mask, "ndim", None) == 2:
            first_cache = next((c for c in cache if c is not None), None)
            mask = _make_attention_mask(
                hidden_states_high,
                first_cache,
                attention_mask=mask,
                token_type_ids=token_type_ids,
                prefix_lm=self.config.prefix_lm,
            )
        elif mask is None:
            first_cache = next((c for c in cache if c is not None), None)
            mask = _make_attention_mask(
                hidden_states_high,
                first_cache,
                token_type_ids=token_type_ids,
                prefix_lm=self.config.prefix_lm,
            )

        num_layers = self.config.num_layers_per_stack
        for high_cycle_idx in range(self.config.H_cycles):
            for low_cycle_idx in range(self.config.L_cycles):
                cycle_offset = (
                    high_cycle_idx * (self.config.L_cycles + 1) + low_cycle_idx
                ) * num_layers
                hidden_states_low = self.L_module(
                    hidden_states_low + hidden_states_high,
                    mask=mask,
                    cache=cache[cycle_offset : cycle_offset + num_layers],
                )

            cycle_offset = (
                high_cycle_idx * (self.config.L_cycles + 1) + self.config.L_cycles
            ) * num_layers
            hidden_states_high = self.H_module(
                hidden_states_high + hidden_states_low,
                mask=mask,
                cache=cache[cycle_offset : cycle_offset + num_layers],
            )

        return hidden_states_high


class LanguageModel(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.model_type = config.model_type
        self.model = HrmTextModel(config)
        if not config.tie_word_embeddings:
            self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

    @property
    def layers(self):
        return self.model.layers

    def make_cache(self):
        return [KVCache() for _ in range(self.config.num_hidden_layers)]

    def __call__(
        self,
        inputs: mx.array,
        inputs_embeds: Optional[mx.array] = None,
        cache=None,
        mask: Optional[mx.array] = None,
        **kwargs,
    ) -> LanguageModelOutput:
        out = self.model(
            inputs,
            inputs_embeds=inputs_embeds,
            cache=cache,
            mask=mask,
            attention_mask=kwargs.get("attention_mask"),
            token_type_ids=kwargs.get("token_type_ids"),
        )
        if self.config.tie_word_embeddings:
            logits = self.model.embed_tokens.as_linear(out)
        else:
            logits = self.lm_head(out)
        return LanguageModelOutput(logits=logits)
