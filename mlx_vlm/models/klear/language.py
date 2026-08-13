# Copyright © 2025 Apple Inc.

from typing import Any, Optional

import mlx.core as mx
import mlx.nn as nn

from ..activations import swiglu
from ..base import (
    LanguageModelOutput,
    create_attention_mask,
    scaled_dot_product_attention,
)
from ..cache import KVCache
from ..switch_layers import SwitchGLU
from .config import ModelConfig


class KlearAttention(nn.Module):
    def __init__(self, args: ModelConfig):
        super().__init__()
        self.num_attention_heads = args.num_attention_heads
        self.num_key_value_heads = args.num_key_value_heads
        self.head_dim = args.hidden_size // args.num_attention_heads
        self.scale = self.head_dim**-0.5

        self.q_proj = nn.Linear(
            args.hidden_size,
            self.num_attention_heads * self.head_dim,
            bias=args.attention_bias,
        )
        self.k_proj = nn.Linear(
            args.hidden_size,
            self.num_key_value_heads * self.head_dim,
            bias=args.attention_bias,
        )
        self.v_proj = nn.Linear(
            args.hidden_size,
            self.num_key_value_heads * self.head_dim,
            bias=args.attention_bias,
        )
        self.o_proj = nn.Linear(
            self.num_attention_heads * self.head_dim,
            args.hidden_size,
            bias=args.attention_bias,
        )
        self.q_norm = nn.RMSNorm(self.head_dim, eps=args.rms_norm_eps)
        self.k_norm = nn.RMSNorm(self.head_dim, eps=args.rms_norm_eps)
        self.rope = nn.RoPE(self.head_dim, traditional=False, base=args.rope_theta)

    def __call__(
        self,
        x: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional[Any] = None,
    ) -> mx.array:
        batch_size, sequence_length, _ = x.shape
        queries, keys, values = self.q_proj(x), self.k_proj(x), self.v_proj(x)
        queries = self.q_norm(
            queries.reshape(batch_size, sequence_length, self.num_attention_heads, -1)
        ).transpose(0, 2, 1, 3)
        keys = self.k_norm(
            keys.reshape(batch_size, sequence_length, self.num_key_value_heads, -1)
        ).transpose(0, 2, 1, 3)
        values = values.reshape(
            batch_size, sequence_length, self.num_key_value_heads, -1
        ).transpose(0, 2, 1, 3)

        if cache is not None:
            queries = self.rope(queries, offset=cache.offset)
            keys = self.rope(keys, offset=cache.offset)
            keys, values = cache.update_and_fetch(keys, values)
        else:
            queries = self.rope(queries)
            keys = self.rope(keys)

        output = scaled_dot_product_attention(
            queries,
            keys,
            values,
            cache=cache,
            scale=self.scale,
            mask=mask,
        )
        output = output.transpose(0, 2, 1, 3).reshape(batch_size, sequence_length, -1)
        return self.o_proj(output)


class KlearMLP(nn.Module):
    def __init__(self, dim, hidden_dim):
        super().__init__()
        self.gate_proj = nn.Linear(dim, hidden_dim, bias=False)
        self.down_proj = nn.Linear(hidden_dim, dim, bias=False)
        self.up_proj = nn.Linear(dim, hidden_dim, bias=False)

    def __call__(self, x) -> mx.array:
        return self.down_proj(swiglu(self.gate_proj(x), self.up_proj(x)))


class KlearSparseMoeBlock(nn.Module):
    def __init__(self, args: ModelConfig):
        super().__init__()
        self.norm_topk_prob = args.norm_topk_prob
        self.num_experts = args.num_experts
        self.top_k = args.num_experts_per_tok
        self.gate = nn.Linear(args.hidden_size, args.num_experts, bias=False)
        self.experts = SwitchGLU(
            args.hidden_size, args.moe_intermediate_size, args.num_experts
        )
        self.shared_experts = KlearMLP(
            args.hidden_size,
            hidden_dim=args.moe_intermediate_size * args.n_shared_experts,
        )
        self.coefficient = nn.Linear(args.hidden_size, 2)
        self.expert_bias = mx.zeros((self.num_experts,), dtype=mx.float32)

    def __call__(self, x: mx.array) -> mx.array:
        routing_weights = mx.sigmoid(self.gate(x).astype(mx.float32))
        biased_weights = routing_weights + self.expert_bias.reshape((1, 1, -1))
        indices = mx.argpartition(-biased_weights, kth=self.top_k - 1, axis=-1)[
            ..., : self.top_k
        ]
        scores = mx.take_along_axis(routing_weights, indices, axis=-1)
        if self.norm_topk_prob:
            scores = scores / mx.sum(scores, axis=-1, keepdims=True)
        scores = scores.astype(x.dtype)
        expert_output = self.experts(x, indices)
        expert_output = (expert_output * scores[..., None]).sum(axis=-2)
        coefficients = mx.softmax(self.coefficient(x), axis=-1, precise=True)
        shared_output = self.shared_experts(x)
        return (
            expert_output * coefficients[..., :1]
            + shared_output * coefficients[..., 1:]
        )


class KlearDecoderLayer(nn.Module):
    def __init__(self, args: ModelConfig, layer_idx: int):
        super().__init__()
        self.self_attn = KlearAttention(args)
        if (layer_idx not in args.mlp_only_layers) and (
            args.num_experts > 0 and (layer_idx + 1) % args.decoder_sparse_step == 0
        ):
            self.mlp = KlearSparseMoeBlock(args)
        else:
            self.mlp = KlearMLP(args.hidden_size, args.intermediate_size)
        self.input_layernorm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)
        self.post_attention_layernorm = nn.RMSNorm(
            args.hidden_size, eps=args.rms_norm_eps
        )

    def __call__(
        self,
        x: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional[Any] = None,
    ) -> mx.array:
        h = x + self.self_attn(self.input_layernorm(x), mask, cache)
        return h + self.mlp(self.post_attention_layernorm(h))


class KlearModel(nn.Module):
    def __init__(self, args: ModelConfig):
        super().__init__()
        self.embed_tokens = nn.Embedding(args.vocab_size, args.hidden_size)
        self.layers = [
            KlearDecoderLayer(args, layer_idx)
            for layer_idx in range(args.num_hidden_layers)
        ]
        self.norm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)

    def __call__(
        self,
        inputs: Optional[mx.array],
        cache: Optional[Any] = None,
        inputs_embeds: Optional[mx.array] = None,
    ) -> mx.array:
        h = self.embed_tokens(inputs) if inputs_embeds is None else inputs_embeds
        if cache is None:
            cache = [None] * len(self.layers)
        mask = create_attention_mask(h, cache[0])
        for layer, layer_cache in zip(self.layers, cache):
            h = layer(h, mask, layer_cache)
        return self.norm(h)


class LanguageModel(nn.Module):
    def __init__(self, args: ModelConfig):
        super().__init__()
        self.args = args
        self.config = args
        self.model_type = args.model_type
        self.model = KlearModel(args)
        self.lm_head = nn.Linear(args.hidden_size, args.vocab_size, bias=False)

    def __call__(
        self,
        inputs: Optional[mx.array] = None,
        cache: Optional[Any] = None,
        input_embeddings: Optional[mx.array] = None,
        inputs_embeds: Optional[mx.array] = None,
        **kwargs,
    ) -> LanguageModelOutput:
        if inputs is None:
            inputs = kwargs.get("input_ids")
        if inputs_embeds is None:
            inputs_embeds = input_embeddings
        hidden_states = self.model(inputs, cache, inputs_embeds)
        return LanguageModelOutput(logits=self.lm_head(hidden_states))

    def sanitize(self, weights):
        if "model.layers.0.mlp.experts.0.gate_proj.weight" not in weights:
            return weights
        for layer_idx in range(self.args.num_hidden_layers):
            prefix = f"model.layers.{layer_idx}.mlp.experts"
            for name in ("gate_proj", "up_proj", "down_proj"):
                stacked = [
                    weights.pop(f"{prefix}.{expert_idx}.{name}.weight")
                    for expert_idx in range(self.args.num_experts)
                ]
                weights[f"{prefix}.{name}.weight"] = mx.stack(stacked)
        return weights

    @property
    def layers(self):
        return self.model.layers

    def make_cache(self):
        return [KVCache() for _ in self.layers]

    @property
    def quant_predicate(self):
        def predicate(path, _):
            if path.endswith("mlp.gate"):
                return {"group_size": 64, "bits": 8}
            return True

        return predicate

    @property
    def cast_predicate(self):
        def predicate(key):
            return "expert_bias" not in key

        return predicate
