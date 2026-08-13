# Copyright © 2026 Apple Inc.

from functools import partial
from typing import Any, List, Optional, Tuple

import mlx.core as mx
import mlx.nn as nn
from mlx.nn.layers.distributed import shard_linear

from ..base import (
    LanguageModelOutput,
    create_attention_mask,
    scaled_dot_product_attention,
)
from ..cache import KVCache, RotatingKVCache
from ..rope_utils import initialize_rope
from .config import ModelConfig


@partial(mx.compile, shapeless=True)
def _compute_gate(query: mx.array, weight: mx.array, bias: mx.array) -> mx.array:
    gate_logits = query @ weight[:, None, :].swapaxes(-1, -2)
    gate_logits = gate_logits + bias[..., None, None]
    return mx.sigmoid(gate_logits)


@partial(mx.compile, shapeless=True)
def _silu_mul(gate: mx.array, up: mx.array) -> mx.array:
    return nn.silu(gate) * up


@partial(mx.compile, shapeless=True)
def _mix_attention(
    gate: mx.array, global_attention: mx.array, local_attention: mx.array
) -> mx.array:
    return gate * global_attention + (1 - gate) * local_attention


class LoopGateProjection(nn.Module):
    def __init__(self, num_heads: int, head_dim: int):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.weight = mx.zeros((num_heads, head_dim))
        self.bias = mx.zeros((num_heads,))

    def __call__(self, query: mx.array) -> mx.array:
        return _compute_gate(query, self.weight, self.bias)


class Attention(nn.Module):
    def __init__(self, args: ModelConfig):
        super().__init__()
        dim = args.hidden_size
        self.n_heads = args.num_attention_heads
        self.n_kv_heads = args.num_key_value_heads
        self.head_dim = args.head_dim
        self.scale = args.head_dim**-0.5
        self.q_proj = nn.Linear(
            dim, self.n_heads * self.head_dim, bias=args.attention_bias
        )
        self.k_proj = nn.Linear(
            dim, self.n_kv_heads * self.head_dim, bias=args.attention_bias
        )
        self.v_proj = nn.Linear(
            dim, self.n_kv_heads * self.head_dim, bias=args.attention_bias
        )
        self.o_proj = nn.Linear(
            self.n_heads * self.head_dim, dim, bias=args.attention_bias
        )
        self.rope = initialize_rope(
            self.head_dim,
            args.rope_theta,
            traditional=False,
            scaling_config=args.rope_scaling,
            max_position_embeddings=args.max_position_embeddings,
        )

    def get_qkv(
        self, x: mx.array, offset: int = 0
    ) -> Tuple[mx.array, mx.array, mx.array]:
        batch_size, sequence_length, _ = x.shape
        queries = (
            self.q_proj(x)
            .reshape(batch_size, sequence_length, self.n_heads, -1)
            .transpose(0, 2, 1, 3)
        )
        keys = (
            self.k_proj(x)
            .reshape(batch_size, sequence_length, self.n_kv_heads, -1)
            .transpose(0, 2, 1, 3)
        )
        values = (
            self.v_proj(x)
            .reshape(batch_size, sequence_length, self.n_kv_heads, -1)
            .transpose(0, 2, 1, 3)
        )
        return self.rope(queries, offset=offset), self.rope(keys, offset=offset), values

    def attention(
        self,
        queries: mx.array,
        keys: mx.array,
        values: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional[Any] = None,
    ) -> mx.array:
        return scaled_dot_product_attention(
            queries,
            keys,
            values,
            cache=cache,
            scale=self.scale,
            mask=mask,
        )


class MLP(nn.Module):
    def __init__(self, args: ModelConfig):
        super().__init__()
        self.gate_proj = nn.Linear(
            args.hidden_size, args.intermediate_size, bias=args.mlp_bias
        )
        self.down_proj = nn.Linear(
            args.intermediate_size, args.hidden_size, bias=args.mlp_bias
        )
        self.up_proj = nn.Linear(
            args.hidden_size, args.intermediate_size, bias=args.mlp_bias
        )

    def __call__(self, x: mx.array) -> mx.array:
        return self.down_proj(_silu_mul(self.gate_proj(x), self.up_proj(x)))


class TransformerBlock(nn.Module):
    def __init__(self, args: ModelConfig):
        super().__init__()
        self.self_attn = Attention(args)
        self.mlp = MLP(args)
        self.input_layernorm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)
        self.post_attention_layernorm = nn.RMSNorm(
            args.hidden_size, eps=args.rms_norm_eps
        )


class IQuestLoopCoderModel(nn.Module):
    def __init__(self, args: ModelConfig):
        super().__init__()
        if args.loop_num != 2:
            raise ValueError(f"Only loop_num=2 is supported, got {args.loop_num}")
        self.embed_tokens = nn.Embedding(args.vocab_size, args.hidden_size)
        self.layers = [TransformerBlock(args) for _ in range(args.num_hidden_layers)]
        self.norm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)
        self.gate_projections = [
            LoopGateProjection(args.num_attention_heads, args.head_dim)
            for _ in range(args.num_hidden_layers)
        ]
        self.loop_window_size = args.loop_window_size

    def __call__(
        self,
        inputs: Optional[mx.array],
        cache: Optional[List[Any]] = None,
        inputs_embeds: Optional[mx.array] = None,
    ):
        h = self.embed_tokens(inputs) if inputs_embeds is None else inputs_embeds
        batch_size, sequence_length = h.shape[:2]
        if cache is None:
            cache = [None] * (2 * len(self.layers))

        mask = create_attention_mask(h, cache[0])
        window_mask = create_attention_mask(
            h, cache[len(self.layers)], window_size=self.loop_window_size
        )

        loop1_kv = []
        for layer, layer_cache in zip(self.layers, cache):
            normalized = layer.input_layernorm(h)
            offset = layer_cache.offset if layer_cache is not None else 0
            queries, keys, values = layer.self_attn.get_qkv(normalized, offset)
            if layer_cache is not None:
                keys, values = layer_cache.update_and_fetch(keys, values)
            loop1_kv.append((keys, values))
            attention = layer.self_attn.attention(
                queries, keys, values, mask, cache=layer_cache
            )
            projected = layer.self_attn.o_proj(
                attention.transpose(0, 2, 1, 3).reshape(batch_size, sequence_length, -1)
            )
            h = h + projected
            h = h + layer.mlp(layer.post_attention_layernorm(h))

        for layer, gate_projection, layer_cache, (global_keys, global_values) in zip(
            self.layers,
            self.gate_projections,
            cache[len(self.layers) :],
            loop1_kv,
        ):
            normalized = layer.input_layernorm(h)
            offset = layer_cache.offset if layer_cache is not None else 0
            queries, local_keys, local_values = layer.self_attn.get_qkv(
                normalized, offset
            )
            gate = gate_projection(queries)
            global_attention = layer.self_attn.attention(
                queries,
                global_keys,
                global_values,
                mask,
                cache=layer_cache,
            )
            if layer_cache is not None:
                local_keys, local_values = layer_cache.update_and_fetch(
                    local_keys, local_values
                )
            local_attention = layer.self_attn.attention(
                queries,
                local_keys,
                local_values,
                window_mask,
                cache=layer_cache,
            )
            mixed = _mix_attention(gate, global_attention, local_attention)
            projected = layer.self_attn.o_proj(
                mixed.transpose(0, 2, 1, 3).reshape(batch_size, sequence_length, -1)
            )
            h = h + projected
            h = h + layer.mlp(layer.post_attention_layernorm(h))

        return self.norm(h)


class LanguageModel(nn.Module):
    def __init__(self, args: ModelConfig):
        super().__init__()
        self.args = args
        self.config = args
        self.model_type = args.model_type
        self.model = IQuestLoopCoderModel(args)
        if not args.tie_word_embeddings:
            self.lm_head = nn.Linear(args.hidden_size, args.vocab_size, bias=False)

    def __call__(
        self,
        inputs: Optional[mx.array] = None,
        cache=None,
        input_embeddings: Optional[mx.array] = None,
        inputs_embeds: Optional[mx.array] = None,
        **kwargs,
    ) -> LanguageModelOutput:
        if inputs is None:
            inputs = kwargs.get("input_ids")
        if inputs_embeds is None:
            inputs_embeds = input_embeddings
        hidden_states = self.model(inputs, cache, inputs_embeds)
        if self.args.tie_word_embeddings:
            logits = self.model.embed_tokens.as_linear(hidden_states)
        else:
            logits = self.lm_head(hidden_states)
        return LanguageModelOutput(logits=logits)

    def sanitize(self, weights):
        return weights

    @property
    def layers(self):
        return self.model.layers

    def make_cache(self):
        return [KVCache() for _ in self.layers] + [
            RotatingKVCache(max_size=self.args.loop_window_size) for _ in self.layers
        ]

    def shard(self, group: Optional[mx.distributed.Group] = None):
        group = group or mx.distributed.init()
        size = group.size()
        rank = group.rank()
        for layer_idx, layer in enumerate(self.model.layers):
            layer.self_attn.q_proj = shard_linear(
                layer.self_attn.q_proj, "all-to-sharded", group=group
            )
            layer.self_attn.k_proj = shard_linear(
                layer.self_attn.k_proj, "all-to-sharded", group=group
            )
            layer.self_attn.v_proj = shard_linear(
                layer.self_attn.v_proj, "all-to-sharded", group=group
            )
            layer.self_attn.o_proj = shard_linear(
                layer.self_attn.o_proj, "sharded-to-all", group=group
            )
            layer.self_attn.n_heads //= size
            layer.self_attn.n_kv_heads //= size
            layer.mlp.gate_proj = shard_linear(
                layer.mlp.gate_proj, "all-to-sharded", group=group
            )
            layer.mlp.down_proj = shard_linear(
                layer.mlp.down_proj, "sharded-to-all", group=group
            )
            layer.mlp.up_proj = shard_linear(
                layer.mlp.up_proj, "all-to-sharded", group=group
            )
            gate_projection = self.model.gate_projections[layer_idx]
            heads_per_rank = gate_projection.num_heads // size
            start = rank * heads_per_rank
            end = start + heads_per_rank
            gate_projection.weight = gate_projection.weight[start:end, :]
            gate_projection.bias = gate_projection.bias[start:end]
            gate_projection.num_heads = heads_per_rank
