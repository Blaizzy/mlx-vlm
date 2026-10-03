from typing import Any, Optional

import mlx.core as mx
import mlx.nn as nn

from ..base import (
    LanguageModelOutput,
    create_attention_mask,
    scaled_dot_product_attention,
)
from ..cache import KVCache, RotatingKVCache
from ..mlp import SwiGLUMLP
from ..rope_utils import initialize_rope
from ..switch_layers import SwitchGLU
from .config import ModelConfig


class Attention(nn.Module):
    def __init__(self, args: ModelConfig, layer_idx: int):
        super().__init__()
        self.n_heads = args.num_attention_heads
        self.n_kv_heads = args.num_key_value_heads
        self.head_dim = args.head_dim
        self.scale = self.head_dim**-0.5
        self.is_sliding = args.layer_types[layer_idx] == "sliding_attention"

        self.q_proj = nn.Linear(
            args.hidden_size,
            self.n_heads * self.head_dim,
            bias=args.attention_bias,
        )
        self.k_proj = nn.Linear(
            args.hidden_size,
            self.n_kv_heads * self.head_dim,
            bias=args.attention_bias,
        )
        self.v_proj = nn.Linear(
            args.hidden_size,
            self.n_kv_heads * self.head_dim,
            bias=args.attention_bias,
        )
        self.o_proj = nn.Linear(
            self.n_heads * self.head_dim,
            args.hidden_size,
            bias=args.attention_bias,
        )
        self.q_norm = nn.RMSNorm(self.head_dim, eps=args.rms_norm_eps)
        self.k_norm = nn.RMSNorm(self.head_dim, eps=args.rms_norm_eps)

        if self.is_sliding:
            self.rope = initialize_rope(
                dims=self.head_dim,
                base=args.rope_theta,
                traditional=False,
                scaling_config=args.rope_scaling,
                max_position_embeddings=args.max_position_embeddings,
            )

    def __call__(
        self,
        x: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional[Any] = None,
    ) -> mx.array:
        batch_size, sequence_length, _ = x.shape
        queries = self.q_proj(x)
        keys = self.k_proj(x)
        values = self.v_proj(x)

        queries = self.q_norm(
            queries.reshape(batch_size, sequence_length, self.n_heads, self.head_dim)
        ).transpose(0, 2, 1, 3)
        keys = self.k_norm(
            keys.reshape(batch_size, sequence_length, self.n_kv_heads, self.head_dim)
        ).transpose(0, 2, 1, 3)
        values = values.reshape(
            batch_size, sequence_length, self.n_kv_heads, self.head_dim
        ).transpose(0, 2, 1, 3)

        if self.is_sliding:
            offset = cache.offset if cache is not None else 0
            queries = self.rope(queries, offset=offset)
            keys = self.rope(keys, offset=offset)

        if cache is not None:
            keys, values = cache.update_and_fetch(keys, values)

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


class MoEGate(nn.Module):
    def __init__(self, args: ModelConfig):
        super().__init__()
        self.top_k = args.num_experts_per_tok
        self.norm_topk_prob = args.norm_topk_prob
        self.weight = mx.zeros((args.num_experts, args.hidden_size))
        self.e_score_correction_bias = mx.zeros((args.num_experts,))

    def __call__(self, x: mx.array):
        logits = (x @ self.weight.T).astype(mx.float32)
        selection_logits = logits + self.e_score_correction_bias.astype(mx.float32)
        indices = mx.argpartition(-selection_logits, kth=self.top_k - 1, axis=-1)[
            ..., : self.top_k
        ]
        selected_logits = mx.take_along_axis(selection_logits, indices, axis=-1)
        order = mx.stop_gradient(mx.argsort(-selected_logits, axis=-1))
        indices = mx.take_along_axis(indices, order, axis=-1)
        indices = mx.stop_gradient(indices)
        scores = mx.take_along_axis(mx.sigmoid(logits), indices, axis=-1)
        if self.norm_topk_prob:
            scores = scores / (scores.sum(axis=-1, keepdims=True) + 1e-20)
        return indices, scores


class SparseMoeBlock(nn.Module):
    def __init__(self, args: ModelConfig):
        super().__init__()
        self.gate = MoEGate(args)
        self.switch_mlp = SwitchGLU(
            args.hidden_size,
            args.moe_intermediate_size,
            args.num_experts,
        )
        self.shared_experts = SwiGLUMLP(
            args.hidden_size,
            args.shared_expert_intermediate_size,
        )

    def __call__(self, x: mx.array):
        indices, scores = self.gate(x)
        routed = self.switch_mlp(x, indices)
        routed = (routed * scores[..., None]).sum(axis=-2).astype(routed.dtype)
        return routed + self.shared_experts(x)


class DecoderLayer(nn.Module):
    def __init__(self, args: ModelConfig, layer_idx: int):
        super().__init__()
        self.self_attn = Attention(args, layer_idx)
        self.mlp = SparseMoeBlock(args)
        self.input_layernorm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)
        self.post_attn_norm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)
        self.post_attention_layernorm = nn.RMSNorm(
            args.hidden_size, eps=args.rms_norm_eps
        )
        self.post_ffn_norm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)

    def __call__(
        self,
        x: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional[Any] = None,
    ) -> mx.array:
        attention = self.self_attn(self.input_layernorm(x), mask, cache)
        hidden_states = x + self.post_attn_norm(attention)
        routed = self.mlp(self.post_attention_layernorm(hidden_states))
        return hidden_states + self.post_ffn_norm(routed)


class Kolibri1Model(nn.Module):
    def __init__(self, args: ModelConfig):
        super().__init__()
        self.args = args
        self.embed_tokens = nn.Embedding(args.vocab_size, args.hidden_size)
        self.layers = [
            DecoderLayer(args, layer_idx) for layer_idx in range(args.num_hidden_layers)
        ]
        self.norm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)

    def __call__(
        self,
        inputs: mx.array,
        cache=None,
        inputs_embeds: Optional[mx.array] = None,
    ) -> mx.array:
        hidden_states = (
            self.embed_tokens(inputs) if inputs_embeds is None else inputs_embeds
        )
        if cache is None:
            cache = [None] * len(self.layers)

        full_cache = next(
            (
                layer_cache
                for layer, layer_cache in zip(self.layers, cache)
                if not layer.self_attn.is_sliding
            ),
            cache[0],
        )
        sliding_cache = next(
            (
                layer_cache
                for layer, layer_cache in zip(self.layers, cache)
                if layer.self_attn.is_sliding
            ),
            cache[0],
        )
        full_mask = create_attention_mask(hidden_states, full_cache)
        sliding_mask = create_attention_mask(
            hidden_states,
            sliding_cache,
            window_size=self.args.sliding_window,
        )

        for layer, layer_cache in zip(self.layers, cache):
            mask = sliding_mask if layer.self_attn.is_sliding else full_mask
            hidden_states = layer(hidden_states, mask, layer_cache)
        return self.norm(hidden_states)


class LanguageModel(nn.Module):
    def __init__(self, args: ModelConfig):
        super().__init__()
        self.args = args
        self.model_type = args.model_type
        self.model = Kolibri1Model(args)
        if not args.tie_word_embeddings:
            self.lm_head = nn.Linear(args.hidden_size, args.vocab_size, bias=False)

    def __call__(
        self,
        inputs: mx.array,
        cache=None,
        inputs_embeds: Optional[mx.array] = None,
        **kwargs,
    ) -> LanguageModelOutput:
        hidden_states = self.model(inputs, cache, inputs_embeds)
        if self.args.tie_word_embeddings:
            logits = self.model.embed_tokens.as_linear(hidden_states)
        else:
            logits = self.lm_head(hidden_states)
        if self.args.head_dtype == "float32":
            logits = logits.astype(mx.float32)
        return LanguageModelOutput(logits=logits)

    def sanitize(self, weights):
        weights = dict(weights)
        if self.args.tie_word_embeddings:
            weights.pop("lm_head.weight", None)

        for layer_idx in range(self.args.num_hidden_layers):
            prefix = f"model.layers.{layer_idx}"
            source_bias = f"{prefix}.moe.router.expert_bias"
            target_bias = f"{prefix}.mlp.gate.e_score_correction_bias"
            if source_bias in weights:
                weights[target_bias] = weights.pop(source_bias)

            for projection in ("up_proj", "down_proj", "gate_proj"):
                for suffix in ("weight", "scales", "biases"):
                    first_key = f"{prefix}.mlp.experts.0.{projection}.{suffix}"
                    if first_key not in weights:
                        continue
                    values = [
                        weights.pop(
                            f"{prefix}.mlp.experts.{expert_idx}.{projection}.{suffix}"
                        )
                        for expert_idx in range(self.args.num_experts)
                    ]
                    weights[f"{prefix}.mlp.switch_mlp.{projection}.{suffix}"] = (
                        mx.stack(values)
                    )

        return weights

    @property
    def quant_predicate(self):
        def predicate(path, _):
            if path.endswith("model.embed_tokens") or path.endswith("lm_head"):
                return False
            if path.endswith("mlp.gate"):
                return False
            return True

        return predicate

    @property
    def layers(self):
        return self.model.layers

    def make_cache(self):
        return [
            (
                RotatingKVCache(max_size=self.args.sliding_window, keep=0)
                if layer.self_attn.is_sliding
                else KVCache()
            )
            for layer in self.layers
        ]
