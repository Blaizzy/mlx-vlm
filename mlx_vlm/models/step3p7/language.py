from functools import partial
from typing import Any, Dict, Optional

import mlx.core as mx
import mlx.nn as nn

from ..base import (
    LanguageModelOutput,
    create_attention_mask,
    scaled_dot_product_attention,
)
from ..cache import KVCache, RotatingKVCache
from ..rope_utils import initialize_rope
from ..switch_layers import SwiGLU, SwitchGLU
from .config import ModelConfig, TextConfig


@partial(mx.compile, shapeless=True)
def _clamped_swiglu(x, gate, limit):
    gate = mx.clip(nn.silu(gate), a_min=None, a_max=limit)
    x = mx.clip(x, a_min=-limit, a_max=limit)
    return gate * x


class ClampedSwiGLU(nn.Module):
    def __init__(self, limit: float):
        super().__init__()
        self.limit = limit

    def __call__(self, x, gate):
        return _clamped_swiglu(x, gate, self.limit)


class StepRMSNorm(nn.Module):
    def __init__(self, hidden_size: int, eps: float):
        super().__init__()
        self.weight = mx.ones((hidden_size,))
        self.eps = eps

    def __call__(self, x):
        return mx.fast.rms_norm(x, self.weight, self.eps).astype(x.dtype)


class MLP(nn.Module):
    def __init__(self, config: TextConfig, intermediate_size: int, swiglu_limit=0):
        super().__init__()
        self.gate_proj = nn.Linear(config.hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(config.hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, config.hidden_size, bias=False)
        self.limit = swiglu_limit if swiglu_limit and swiglu_limit > 0 else None

    def __call__(self, x):
        if self.limit is not None:
            return self.down_proj(
                _clamped_swiglu(self.up_proj(x), self.gate_proj(x), self.limit)
            )
        return self.down_proj(nn.silu(self.gate_proj(x)) * self.up_proj(x))


@mx.compile
def _moe_gate_select(gates, router_bias, top_k, routed_scaling_factor, norm_topk_prob):
    scores = mx.sigmoid(gates.astype(mx.float32))
    corrected_scores = scores + router_bias
    topk_indices = mx.argpartition(-corrected_scores, kth=top_k - 1, axis=-1)[
        ..., :top_k
    ]
    topk_weights = mx.take_along_axis(scores, topk_indices, axis=-1)
    if norm_topk_prob:
        topk_weights = topk_weights / (
            mx.sum(topk_weights, axis=-1, keepdims=True) + 1e-20
        )
    return topk_indices, topk_weights * routed_scaling_factor


class MoEGate(nn.Module):
    def __init__(self, config: TextConfig):
        super().__init__()
        self.top_k = config.moe_top_k
        self.routed_scaling_factor = config.moe_router_scaling_factor
        self.norm_topk_prob = config.norm_expert_weight
        self.gate = nn.Linear(config.hidden_size, config.moe_num_experts, bias=False)
        self.router_bias = mx.zeros((config.moe_num_experts,), dtype=mx.float32)

    def __call__(self, x):
        return _moe_gate_select(
            self.gate(x),
            self.router_bias,
            self.top_k,
            self.routed_scaling_factor,
            self.norm_topk_prob,
        )


class MoE(nn.Module):
    def __init__(self, config: TextConfig, layer_idx: int):
        super().__init__()
        swiglu_limit = 0
        if config.swiglu_limits and layer_idx < len(config.swiglu_limits):
            swiglu_limit = config.swiglu_limits[layer_idx] or 0
        shared_limit = 0
        if config.swiglu_limits_shared and layer_idx < len(config.swiglu_limits_shared):
            shared_limit = config.swiglu_limits_shared[layer_idx] or 0

        self.gate = MoEGate(config)
        activation = ClampedSwiGLU(swiglu_limit) if swiglu_limit > 0 else SwiGLU()
        self.switch_mlp = SwitchGLU(
            config.hidden_size,
            config.moe_intermediate_size,
            config.moe_num_experts,
            activation=activation,
        )
        self.share_expert = MLP(config, config.share_expert_dim, shared_limit)

    def __call__(self, x):
        topk_indices, topk_weights = self.gate(x)
        routed_output = self.switch_mlp(x, topk_indices)
        routed_output = (
            (routed_output * topk_weights[..., None])
            .sum(axis=-2)
            .astype(routed_output.dtype)
        )
        return routed_output + self.share_expert(x)


class Attention(nn.Module):
    def __init__(self, config: TextConfig, layer_idx: int):
        super().__init__()
        layer_types = config.layer_types or []
        self.is_sliding = (
            layer_types[layer_idx] == "sliding_attention"
            if layer_types
            else layer_idx % 2 == 0
        )
        if self.is_sliding and config.attention_other_setting:
            self.n_heads = config.attention_other_setting["num_attention_heads"]
            self.n_kv_heads = config.attention_other_setting["num_attention_groups"]
        else:
            self.n_heads = config.num_attention_heads
            self.n_kv_heads = config.num_attention_groups
        self.head_dim = config.head_dim
        self.scale = self.head_dim**-0.5
        self.q_proj = nn.Linear(
            config.hidden_size, self.n_heads * self.head_dim, bias=False
        )
        self.k_proj = nn.Linear(
            config.hidden_size, self.n_kv_heads * self.head_dim, bias=False
        )
        self.v_proj = nn.Linear(
            config.hidden_size, self.n_kv_heads * self.head_dim, bias=False
        )
        self.o_proj = nn.Linear(
            self.n_heads * self.head_dim, config.hidden_size, bias=False
        )
        self.q_norm = StepRMSNorm(self.head_dim, config.rms_norm_eps)
        self.k_norm = StepRMSNorm(self.head_dim, config.rms_norm_eps)
        self.use_head_wise_attn_gate = config.use_head_wise_attn_gate
        if self.use_head_wise_attn_gate:
            self.g_proj = nn.Linear(config.hidden_size, self.n_heads, bias=False)

        theta = (
            config.rope_theta[layer_idx]
            if isinstance(config.rope_theta, list)
            else config.rope_theta
        )
        partial = 1.0
        if config.partial_rotary_factors and layer_idx < len(
            config.partial_rotary_factors
        ):
            partial = config.partial_rotary_factors[layer_idx]
        layer_type = layer_types[layer_idx] if layer_types else "full_attention"
        rope_scaling = (
            None
            if config.yarn_only_types and layer_type not in config.yarn_only_types
            else config.rope_scaling
        )
        self.rope = initialize_rope(
            dims=int(self.head_dim * partial),
            base=theta,
            traditional=False,
            scaling_config=rope_scaling,
            max_position_embeddings=config.max_position_embeddings,
        )

    def __call__(self, x, mask=None, cache: Optional[Any] = None):
        b, l, _ = x.shape
        q = self.q_norm(self.q_proj(x).reshape(b, l, self.n_heads, -1)).transpose(
            0, 2, 1, 3
        )
        k = self.k_norm(self.k_proj(x).reshape(b, l, self.n_kv_heads, -1)).transpose(
            0, 2, 1, 3
        )
        v = self.v_proj(x).reshape(b, l, self.n_kv_heads, -1).transpose(0, 2, 1, 3)
        if cache is not None:
            q = self.rope(q, offset=cache.offset)
            k = self.rope(k, offset=cache.offset)
            k, v = cache.update_and_fetch(k, v)
        else:
            q = self.rope(q)
            k = self.rope(k)
        y = scaled_dot_product_attention(
            q, k, v, cache=cache, scale=self.scale, mask=mask
        )
        y = y.transpose(0, 2, 1, 3)
        if self.use_head_wise_attn_gate:
            y = y * mx.sigmoid(self.g_proj(x))[..., None]
        return self.o_proj(y.reshape(b, l, -1))


class DecoderLayer(nn.Module):
    def __init__(self, config: TextConfig, layer_idx: int):
        super().__init__()
        self.self_attn = Attention(config, layer_idx)
        self.is_sliding = self.self_attn.is_sliding
        moe_layers = config.moe_layers_enum
        if isinstance(moe_layers, str):
            moe_layers = {int(i) for i in moe_layers.strip().split(",") if i.strip()}
        else:
            moe_layers = set(moe_layers)
        self.is_moe_layer = layer_idx in moe_layers
        if self.is_moe_layer:
            self.mlp = MoE(config, layer_idx)
        else:
            shared_limit = 0
            if config.swiglu_limits_shared and layer_idx < len(
                config.swiglu_limits_shared
            ):
                shared_limit = config.swiglu_limits_shared[layer_idx] or 0
            self.mlp = MLP(config, config.intermediate_size, shared_limit)
        self.input_layernorm = StepRMSNorm(config.hidden_size, config.rms_norm_eps)
        self.post_attention_layernorm = StepRMSNorm(
            config.hidden_size, config.rms_norm_eps
        )

    def __call__(self, x, mask=None, cache=None):
        h = x + self.self_attn(self.input_layernorm(x), mask=mask, cache=cache)
        return h + self.mlp(self.post_attention_layernorm(h))


class StepTextModel(nn.Module):
    def __init__(self, config: TextConfig):
        super().__init__()
        self.args = config
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = [DecoderLayer(config, i) for i in range(config.num_hidden_layers)]
        self.norm = StepRMSNorm(config.hidden_size, config.rms_norm_eps)
        self._swa_idx = next(
            (i for i, layer in enumerate(self.layers) if layer.is_sliding), None
        )
        self._full_idx = next(
            (i for i, layer in enumerate(self.layers) if not layer.is_sliding), None
        )

    def __call__(self, input_ids, inputs_embeds=None, cache=None):
        h = self.embed_tokens(input_ids) if inputs_embeds is None else inputs_embeds
        if cache is None:
            cache = [None] * len(self.layers)
        full_mask = (
            create_attention_mask(h, cache[self._full_idx])
            if self._full_idx is not None
            else None
        )
        swa_mask = (
            create_attention_mask(
                h, cache[self._swa_idx], window_size=self.args.sliding_window
            )
            if self._swa_idx is not None
            else None
        )
        for layer, c in zip(self.layers, cache):
            mask = swa_mask if layer.is_sliding else full_mask
            h = layer(h, mask=mask, cache=c)
        return self.norm(h)


class LanguageModel(nn.Module):
    def __init__(self, args: TextConfig, config: ModelConfig = None):
        super().__init__()
        self.args = args
        self.model = StepTextModel(args)
        self.lm_head = nn.Linear(args.hidden_size, args.vocab_size, bias=False)

    def __call__(self, input_ids=None, cache=None, inputs_embeds=None, **kwargs):
        if input_ids is None:
            input_ids = kwargs.pop("inputs", None)
        if input_ids is None:
            raise ValueError("input_ids must be provided")
        h = self.model(input_ids, inputs_embeds=inputs_embeds, cache=cache)
        return LanguageModelOutput(logits=self.lm_head(h))

    def make_cache(self):
        return [
            (
                RotatingKVCache(max_size=self.args.sliding_window)
                if layer.is_sliding
                else KVCache()
            )
            for layer in self.layers
        ]

    @property
    def layers(self):
        return self.model.layers

    @property
    def head_dim(self):
        return self.args.head_dim

    @property
    def n_kv_heads(self):
        return self.args.num_attention_groups

    @property
    def quant_predicate(self):
        def predicate(path, _):
            if "mlp.gate.gate" in path:
                return {"group_size": 64, "bits": 8}
            return True

        return predicate

    def sanitize(self, weights: Dict[str, mx.array]):
        remappings = [
            (".moe.gate_proj.", ".mlp.switch_mlp.gate_proj."),
            (".moe.up_proj.", ".mlp.switch_mlp.up_proj."),
            (".moe.down_proj.", ".mlp.switch_mlp.down_proj."),
            (".moe.gate.", ".mlp.gate.gate."),
            (".moe.router_bias", ".mlp.gate.router_bias"),
            (".share_expert.", ".mlp.share_expert."),
        ]
        is_vanilla = any(
            src in k and dst not in k for k in weights for src, dst in remappings
        )
        sanitized = {}
        for key, value in weights.items():
            if ".mtp" in key:
                continue
            if "model.layers." in key:
                parts = key.split(".")
                if "layers" in parts:
                    layer_pos = parts.index("layers")
                    if (
                        layer_pos + 1 < len(parts)
                        and parts[layer_pos + 1].isdigit()
                        and int(parts[layer_pos + 1]) >= self.args.num_hidden_layers
                    ):
                        continue
            for src, dst in remappings:
                if src in key and dst not in key:
                    key = key.replace(src, dst)
                    break
            if is_vanilla and key.endswith(".weight") and "norm" in key:
                value = value + 1
            sanitized[key] = value
        return sanitized
