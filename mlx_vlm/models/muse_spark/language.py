from typing import Any, Optional

import mlx.core as mx
import mlx.nn as nn

from ..base import (
    LanguageModelOutput,
    create_attention_mask,
    scaled_dot_product_attention,
)
from ..cache import KVCache, RotatingKVCache
from ..muse_glimmer.language import MLP, CenteredRMSNorm, RMSNormNoScale
from ..rope_utils import initialize_rope
from ..switch_layers import SwitchGLU
from .config import TextConfig

# Added to the sum of the top-k router scores.
_ROUTER_SCORE_EPS = 1e-15


def residual_gate_coefficients(
    gate: mx.array, temperature: float, eps: float = 1e-3
) -> tuple[mx.array, mx.array]:
    """The weights of the residual stream and of the branch for a gate."""
    w = gate.astype(mx.float32) / temperature
    beta = mx.sigmoid(w)
    alpha = mx.sqrt(mx.maximum(mx.sigmoid(-w) * (1.0 + beta), eps))
    return alpha, beta


@mx.compile
def _input_scaled_rms_norm(x: mx.array, weight: mx.array, eps: float) -> mx.array:
    dtype = x.dtype
    # Scale first, round to the input dtype, then normalize in FP32.
    x = (x.astype(mx.float32) * (1.0 + weight.astype(mx.float32))).astype(dtype)
    return mx.fast.rms_norm(x.astype(mx.float32), None, eps).astype(dtype)


class InputScaledRMSNorm(nn.Module):
    """Scale by the centered weight (1+w), then normalize without a scale."""

    def __init__(self, dim: int, eps: float):
        super().__init__()
        self.weight = mx.zeros((dim,))
        self.eps = eps

    def __call__(self, x: mx.array) -> mx.array:
        return _input_scaled_rms_norm(x, self.weight, self.eps)


class Attention(nn.Module):
    def __init__(self, args: TextConfig, layer_idx: int):
        super().__init__()
        self.n_heads = args.num_attention_heads
        self.n_kv_heads = args.num_key_value_heads
        self.head_dim = args.head_dim
        self.scale = self.head_dim**-0.5
        self.qk_scale_factor = args.qk_scale_factor
        self.use_rope = bool(args.layer_rope_theta[layer_idx])

        dim = args.hidden_size
        bias = args.attention_bias
        self.q_proj = nn.Linear(dim, self.n_heads * self.head_dim, bias=bias)
        self.k_proj = nn.Linear(dim, self.n_kv_heads * self.head_dim, bias=bias)
        self.v_proj = nn.Linear(dim, self.n_kv_heads * self.head_dim, bias=bias)
        self.gate_proj = nn.Linear(dim, self.n_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.n_heads * self.head_dim, dim, bias=bias)
        self.qk_norm = RMSNormNoScale(args.rms_norm_eps)
        self.o_norm = RMSNormNoScale(args.rms_norm_eps)

        if self.use_rope:
            # RoPE layers use the global theta; layer_rope_theta only marks NoPE.
            theta = float(args.rope_parameters.get("rope_theta", 500000.0))
            scaling_config = dict(args.rope_parameters)
            rope_type = scaling_config.get("type") or scaling_config.get(
                "rope_type", "default"
            )
            self.rope = initialize_rope(
                self.head_dim,
                base=theta,
                traditional=False,
                scaling_config=scaling_config,
                max_position_embeddings=args.max_position_embeddings,
                implementation=(
                    "eager" if rope_type in ("default", "linear") else "fast"
                ),
            )

    def __call__(
        self,
        x: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional[Any] = None,
    ) -> mx.array:
        batch, length, _ = x.shape
        queries = self.q_proj(x).reshape(batch, length, self.n_heads, self.head_dim)
        keys = self.k_proj(x).reshape(batch, length, self.n_kv_heads, self.head_dim)
        values = self.v_proj(x).reshape(batch, length, self.n_kv_heads, self.head_dim)

        queries = self.qk_norm(queries)
        queries = (queries.astype(mx.float32) * self.qk_scale_factor).astype(
            queries.dtype
        )
        queries = queries.transpose(0, 2, 1, 3)
        keys = self.qk_norm(keys).transpose(0, 2, 1, 3)
        values = values.transpose(0, 2, 1, 3)

        if self.use_rope:
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
        # Normalize each head's output before the gate.
        output = self.o_norm(output.transpose(0, 2, 1, 3)).reshape(batch, length, -1)
        output = output * mx.sigmoid(self.gate_proj(x))
        return self.o_proj(output)


class MoEGate(nn.Module):
    """The router and its score bias, both kept in FP32."""

    def __init__(self, dim: int, num_experts: int):
        super().__init__()
        self.weight = mx.zeros((num_experts, dim), dtype=mx.float32)
        self.e_score_correction_bias = mx.zeros((num_experts,), dtype=mx.float32)


class MoE(nn.Module):
    def __init__(self, args: TextConfig):
        super().__init__()
        dim = args.hidden_size
        compressed = args.moe_hidden_size
        self.top_k = args.num_experts_per_tok
        self.router_multiplier = args.output_multiplier
        self.routed_scaling_factor = args.routed_scaling_factor

        self.gate = MoEGate(dim, args.num_local_experts)
        self.pre_expert_proj = nn.Linear(dim, compressed, bias=False)
        self.pre_expert_norm = CenteredRMSNorm(compressed, args.rms_norm_eps)
        self.experts = SwitchGLU(
            compressed, args.moe_intermediate_size, args.num_local_experts
        )
        # The checkpoint stores this norm under the experts.
        self.experts.post_expert_norm = InputScaledRMSNorm(
            compressed, args.post_norm_eps
        )
        self.post_expert_proj = nn.Linear(compressed, dim, bias=False)

    def __call__(self, x: mx.array) -> mx.array:
        logits = x.astype(mx.float32) @ self.gate.weight.astype(mx.float32).T
        scores = mx.sigmoid(logits * self.router_multiplier)
        biased = scores + self.gate.e_score_correction_bias.astype(mx.float32)
        k = self.top_k
        inds = mx.stop_gradient(mx.argpartition(-biased, kth=k - 1, axis=-1)[..., :k])
        weights = mx.take_along_axis(scores, inds, axis=-1)
        weights = weights / (weights.sum(axis=-1, keepdims=True) + _ROUTER_SCORE_EPS)
        weights = weights * self.routed_scaling_factor

        h = self.pre_expert_norm(self.pre_expert_proj(x))
        h = self.experts(h, inds)
        h = self.experts.post_expert_norm(h)
        h = (h * weights[..., None]).astype(x.dtype).sum(axis=-2)
        return self.post_expert_proj(h)


class DecoderLayer(nn.Module):
    def __init__(self, args: TextConfig, layer_idx: int):
        super().__init__()
        dim = args.hidden_size
        self.self_attn = Attention(args, layer_idx)
        if args.mlp_layer_types[layer_idx] == "sparse":
            self.mlp = MoE(args)
        else:
            self.mlp = MLP(args)
        self.input_layernorm = CenteredRMSNorm(dim, args.rms_norm_eps)
        self.post_attention_layernorm = RMSNormNoScale(args.post_norm_eps)
        self.pre_feedforward_layernorm = CenteredRMSNorm(dim, args.rms_norm_eps)
        self.post_feedforward_layernorm = InputScaledRMSNorm(dim, args.post_norm_eps)
        # Sanitize computes these from the residual gates of the checkpoint.
        alpha, beta = residual_gate_coefficients(
            mx.zeros((dim,)), args.residual_gate_temperature
        )
        self.add_attn_alpha, self.add_attn_beta = alpha, beta
        self.add_ffn_alpha, self.add_ffn_beta = alpha, beta
        self.is_sliding = args.layer_types[layer_idx] == "sliding_attention"

    def __call__(
        self,
        x: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional[Any] = None,
    ) -> mx.array:
        h = self.self_attn(self.input_layernorm(x), mask=mask, cache=cache)
        h = self.post_attention_layernorm(h)
        x = (self.add_attn_alpha * x + self.add_attn_beta * h).astype(x.dtype)

        h = self.mlp(self.pre_feedforward_layernorm(x))
        h = self.post_feedforward_layernorm(h)
        return (self.add_ffn_alpha * x + self.add_ffn_beta * h).astype(x.dtype)


class TextModel(nn.Module):
    def __init__(self, args: TextConfig):
        super().__init__()
        self.args = args
        self.embed_tokens = nn.Embedding(args.vocab_size, args.hidden_size)
        self.embed_norm = RMSNormNoScale(args.rms_norm_eps)
        self.layers = [DecoderLayer(args, idx) for idx in range(args.num_hidden_layers)]
        self.norm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)
        self.sliding_window = args.sliding_window

        layer_types = args.layer_types
        self.full_attention_idx = (
            layer_types.index("full_attention")
            if "full_attention" in layer_types
            else None
        )
        self.sliding_attention_idx = (
            layer_types.index("sliding_attention")
            if "sliding_attention" in layer_types
            else None
        )

    def __call__(
        self,
        inputs: Optional[mx.array],
        cache=None,
        inputs_embeds: Optional[mx.array] = None,
    ) -> mx.array:
        hidden_states = inputs_embeds
        if hidden_states is None:
            hidden_states = self.embed_norm(self.embed_tokens(inputs))
        if cache is None:
            cache = [None] * len(self.layers)

        full_mask = sliding_mask = None
        if self.full_attention_idx is not None:
            full_mask = create_attention_mask(
                hidden_states, cache[self.full_attention_idx]
            )
        if self.sliding_attention_idx is not None:
            sliding_mask = create_attention_mask(
                hidden_states,
                cache[self.sliding_attention_idx],
                window_size=self.sliding_window,
            )

        for layer, layer_cache in zip(self.layers, cache):
            mask = sliding_mask if layer.is_sliding else full_mask
            hidden_states = layer(hidden_states, mask=mask, cache=layer_cache)
        return self.norm(hidden_states)


class LanguageModel(nn.Module):
    def __init__(self, args: TextConfig):
        super().__init__()
        self.args = args
        self.config = args
        self.model_type = args.model_type
        self.model = TextModel(args)
        self.lm_head = nn.Linear(args.hidden_size, args.vocab_size, bias=False)
        self.final_logit_softcapping = args.final_logit_softcapping
        self.output_multiplier = args.output_multiplier

    def __call__(
        self,
        inputs: Optional[mx.array] = None,
        cache=None,
        inputs_embeds: Optional[mx.array] = None,
        **kwargs,
    ) -> LanguageModelOutput:
        if inputs is None:
            inputs = kwargs.get("input_ids")
        hidden_states = self.model(inputs, cache=cache, inputs_embeds=inputs_embeds)
        logits = self.lm_head(hidden_states) * self.output_multiplier
        softcap = self.final_logit_softcapping
        logits = mx.tanh(logits / softcap) * softcap
        return LanguageModelOutput(logits=logits)

    @property
    def layers(self):
        return self.model.layers

    @property
    def head_dim(self):
        return self.args.head_dim

    @property
    def n_kv_heads(self):
        return self.args.num_key_value_heads

    def make_cache(self):
        return [
            (
                RotatingKVCache(max_size=self.args.sliding_window)
                if layer.is_sliding
                else KVCache()
            )
            for layer in self.layers
        ]
