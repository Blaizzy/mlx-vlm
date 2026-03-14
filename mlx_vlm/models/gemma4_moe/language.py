from functools import partial
from typing import Any, Optional

import mlx.core as mx
import mlx.nn as nn

from ..base import (
    LanguageModelOutput,
    create_attention_mask,
    scaled_dot_product_attention,
)
from ..cache import KVCache, RotatingKVCache
from .config import TextConfig


class RMSNorm(nn.Module):
    """Gemma4 RMSNorm with scale_shift=1.0 (weight + 1.0)."""

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.weight = mx.ones((dim,))
        self.eps = eps

    def __call__(self, x: mx.array) -> mx.array:
        return mx.fast.rms_norm(x, 1.0 + self.weight, self.eps)


class RMSNormNoScale(nn.Module):
    """RMSNorm without learnable scale."""

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps

    def __call__(self, x: mx.array) -> mx.array:
        return mx.fast.rms_norm(x, None, self.eps)


@partial(mx.compile, shapeless=True)
def logit_softcap(softcap, x):
    return mx.tanh(x / softcap) * softcap


class MLP(nn.Module):
    def __init__(self, config: TextConfig):
        super().__init__()
        self.gate_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.down_proj = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)
        self.up_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)

    def __call__(self, x: mx.array) -> mx.array:
        return self.down_proj(nn.gelu_approx(self.gate_proj(x)) * self.up_proj(x))


class Router(nn.Module):
    """Expert router: norm -> scale -> project -> top-k -> renormalize."""

    def __init__(self, config: TextConfig):
        super().__init__()
        self.config = config
        self.norm = RMSNormNoScale(config.hidden_size, eps=config.rms_norm_eps)
        self.proj = nn.Linear(config.hidden_size, config.num_experts, bias=False)
        self.scale = mx.ones((config.hidden_size,))
        self._root_size = config.hidden_size ** -0.5

    def __call__(self, x: mx.array):
        x = self.norm(x)
        x = x * self._root_size
        x = x * self.scale

        expert_scores = self.proj(x)  # [B, S, E]
        router_probs = mx.softmax(expert_scores, axis=-1)

        # Top-k indices and gather their probabilities
        top_k_indices = mx.argpartition(
            -expert_scores, kth=self.config.top_k_experts - 1, axis=-1
        )[..., : self.config.top_k_experts]  # [B, S, K]

        # Gather top-k probabilities and renormalize
        top_k_weights = mx.take_along_axis(
            router_probs, top_k_indices, axis=-1
        )  # [B, S, K]
        top_k_weights = top_k_weights / mx.sum(
            top_k_weights, axis=-1, keepdims=True
        )

        return top_k_indices, top_k_weights


class MoEBlock(nn.Module):
    """Sparse Mixture of Experts — only computes top-k expert MLPs."""

    def __init__(self, config: TextConfig):
        super().__init__()
        E = config.num_experts
        H = config.hidden_size
        I = config.expert_intermediate_size
        self.gate_proj = mx.zeros((E, H, I))
        self.up_proj = mx.zeros((E, H, I))
        self.down_proj = mx.zeros((E, I, H))
        self.per_expert_scale = mx.ones((E,))

    def __call__(
        self, x: mx.array, top_k_indices: mx.array, top_k_weights: mx.array
    ) -> mx.array:
        # x: [B, S, H], top_k_indices: [B, S, K], top_k_weights: [B, S, K]
        B, S, H = x.shape
        K = top_k_indices.shape[-1]

        # Gather only the active expert weights
        flat_idx = top_k_indices.reshape(-1)  # [B*S*K]
        gate_w = self.gate_proj[flat_idx].reshape(B * S, K, -1, self.gate_proj.shape[-1])
        up_w = self.up_proj[flat_idx].reshape(B * S, K, -1, self.up_proj.shape[-1])
        down_w = self.down_proj[flat_idx].reshape(B * S, K, self.down_proj.shape[-2], -1)
        expert_scales = self.per_expert_scale[flat_idx].reshape(B * S, K)

        # x: [B*S, 1, 1, H] for broadcast matmul with [B*S, K, H, I]
        x_flat = x.reshape(B * S, 1, 1, H)

        # SwitchGLU: gelu(x @ gate) * (x @ up), then @ down
        gate = (x_flat @ gate_w).squeeze(-2)  # [B*S, K, I]
        up = (x_flat @ up_w).squeeze(-2)
        activated = nn.gelu_approx(gate) * up
        down = (activated[:, :, None, :] @ down_w).squeeze(-2)  # [B*S, K, H]

        # Weighted sum: [B*S, K, H] * [B*S, K, 1] -> sum K -> [B*S, H]
        weights = (top_k_weights.reshape(B * S, K) * expert_scales)[..., None]
        return (down * weights).sum(axis=1).reshape(B, S, H)


class Attention(nn.Module):
    def __init__(self, config: TextConfig, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.layer_type = config.layer_types[layer_idx]
        self.is_sliding = self.layer_type == "sliding_attention"

        self.head_dim = (
            config.global_head_dim
            if self.layer_type == "full_attention"
            and hasattr(config, "global_head_dim")
            and config.global_head_dim
            else config.head_dim
        )

        dim = config.hidden_size
        self.n_heads = config.num_attention_heads

        # K-eq-V for full attention layers
        self.use_k_eq_v = (
            getattr(config, "attention_k_eq_v", False) and not self.is_sliding
        )
        if self.use_k_eq_v and config.num_global_key_value_heads is not None:
            self.n_kv_heads = config.num_global_key_value_heads
        else:
            self.n_kv_heads = config.num_key_value_heads

        self.scale = 1.0

        self.q_proj = nn.Linear(dim, self.n_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(dim, self.n_kv_heads * self.head_dim, bias=False)
        if not self.use_k_eq_v:
            self.v_proj = nn.Linear(dim, self.n_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.n_heads * self.head_dim, dim, bias=False)

        self.q_norm = RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.k_norm = RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.v_norm = RMSNormNoScale(self.head_dim, eps=config.rms_norm_eps)

        # RoPE
        layer_key = "sliding_attention" if self.is_sliding else "full_attention"
        rope_params = config.rope_parameters.get(layer_key, {})
        rope_theta = rope_params.get("rope_theta", 10000.0)

        self.rope = nn.RoPE(
            self.head_dim,
            traditional=config.rope_traditional,
            base=rope_theta,
        )

    def __call__(
        self,
        x: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional[Any] = None,
    ) -> mx.array:
        B, L, _ = x.shape

        queries = self.q_proj(x).reshape(B, L, self.n_heads, self.head_dim)
        queries = self.q_norm(queries)

        offset = cache.offset if cache is not None else 0

        keys = self.k_proj(x).reshape(B, L, self.n_kv_heads, self.head_dim)

        # k_eq_v: values come from raw k_proj output (before k_norm)
        if self.use_k_eq_v:
            values = keys
        else:
            values = self.v_proj(x).reshape(B, L, self.n_kv_heads, self.head_dim)

        keys = self.k_norm(keys)
        values = self.v_norm(values)
        values = values.transpose(0, 2, 1, 3)

        keys = keys.transpose(0, 2, 1, 3)
        keys = self.rope(keys, offset=offset)

        if cache is not None:
            keys, values = cache.update_and_fetch(keys, values)

        queries = queries.transpose(0, 2, 1, 3)
        queries = self.rope(queries, offset=offset)

        if mask is not None and isinstance(mask, mx.array):
            if mask.shape[-1] != keys.shape[-2]:
                mask = mask[..., -keys.shape[-2] :]

        output = scaled_dot_product_attention(
            queries, keys, values, cache=cache, scale=self.scale, mask=mask
        )
        output = output.transpose(0, 2, 1, 3).reshape(B, L, -1)
        return self.o_proj(output)


class DecoderLayer(nn.Module):
    def __init__(self, config: TextConfig, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.layer_type = config.layer_types[layer_idx]
        self.self_attn = Attention(config, layer_idx)
        self.mlp = MLP(config)
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )
        self.pre_feedforward_layernorm = RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )
        self.post_feedforward_layernorm = RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )

        # MoE
        self.enable_moe = getattr(config, "enable_moe_block", False)
        if self.enable_moe:
            self.router = Router(config)
            self.moe = MoEBlock(config)
            self.post_feedforward_layernorm_1 = RMSNorm(
                config.hidden_size, eps=config.rms_norm_eps
            )
            self.post_feedforward_layernorm_2 = RMSNorm(
                config.hidden_size, eps=config.rms_norm_eps
            )
            self.pre_feedforward_layernorm_2 = RMSNorm(
                config.hidden_size, eps=config.rms_norm_eps
            )

        # Layer scalar for full attention layers
        if self.layer_type == "full_attention":
            self.layer_scalar = mx.ones((1,))
        else:
            self.layer_scalar = None

    def __call__(
        self,
        x: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional[Any] = None,
    ) -> mx.array:
        residual = x
        h = self.input_layernorm(x)
        h = self.self_attn(h, mask, cache)
        h = self.post_attention_layernorm(h)
        h = residual + h

        residual = h

        if self.enable_moe:
            # Parallel dense MLP + MoE
            h1 = self.pre_feedforward_layernorm(h)
            h1 = self.mlp(h1)
            h1 = self.post_feedforward_layernorm_1(h1)

            # Router on raw hidden_states, MoE on normed hidden_states
            top_k_indices, top_k_weights = self.router(h)
            h2 = self.pre_feedforward_layernorm_2(h)
            h2 = self.moe(h2, top_k_indices, top_k_weights)
            h2 = self.post_feedforward_layernorm_2(h2)

            h = h1 + h2
        else:
            h = self.pre_feedforward_layernorm(h)
            h = self.mlp(h)

        # post_feedforward_layernorm and residual apply to both paths
        h = self.post_feedforward_layernorm(h)
        h = residual + h

        if self.layer_scalar is not None:
            h = h * self.layer_scalar

        return h


class ScaledEmbedding(nn.Embedding):
    def __init__(self, num_embeddings: int, dims: int, embed_scale: float = 1.0):
        super().__init__(num_embeddings, dims)
        self.embed_scale = embed_scale

    def __call__(self, x: mx.array) -> mx.array:
        return super().__call__(x) * self.embed_scale

    def as_linear(self, x: mx.array) -> mx.array:
        return x @ self.weight.T


class Gemma4MoETextModel(nn.Module):
    def __init__(self, config: TextConfig):
        super().__init__()
        self.config = config
        self.vocab_size = config.vocab_size
        self.window_size = config.sliding_window
        self.num_hidden_layers = config.num_hidden_layers

        self.embed_tokens = ScaledEmbedding(
            config.vocab_size, config.hidden_size, embed_scale=config.hidden_size**0.5
        )
        self.layers = [
            DecoderLayer(config, layer_idx=i) for i in range(config.num_hidden_layers)
        ]
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def __call__(
        self,
        inputs: mx.array = None,
        inputs_embeds: Optional[mx.array] = None,
        mask: Optional[mx.array] = None,
        cache=None,
    ):
        if inputs_embeds is None:
            h = self.embed_tokens(inputs)
        else:
            h = inputs_embeds

        if cache is None:
            cache = [None] * len(self.layers)

        global_cache_idx = None
        sliding_cache_idx = None
        for i, layer in enumerate(self.layers):
            if layer.layer_type == "full_attention" and global_cache_idx is None:
                global_cache_idx = i
            elif layer.layer_type == "sliding_attention" and sliding_cache_idx is None:
                sliding_cache_idx = i

        if mask is None:
            global_mask = create_attention_mask(
                h, cache[global_cache_idx] if global_cache_idx is not None else None
            )
            sliding_window_mask = create_attention_mask(
                h,
                cache[sliding_cache_idx] if sliding_cache_idx is not None else None,
                window_size=self.window_size,
            )

        for i, (layer, c) in enumerate(zip(self.layers, cache)):
            is_global = layer.layer_type == "full_attention"
            local_mask = mask
            if mask is None and is_global:
                local_mask = global_mask
            elif mask is None:
                local_mask = sliding_window_mask
            h = layer(h, local_mask, c)

        return self.norm(h)


class LanguageModel(nn.Module):
    def __init__(self, config: TextConfig):
        super().__init__()
        self.config = config
        self.model_type = config.model_type
        self.model = Gemma4MoETextModel(config)
        self.final_logit_softcapping = getattr(config, "final_logit_softcapping", None)

    def __call__(
        self,
        inputs: mx.array = None,
        inputs_embeds: Optional[mx.array] = None,
        mask: Optional[mx.array] = None,
        cache=None,
        **kwargs,
    ):
        out = self.model(
            inputs,
            inputs_embeds=inputs_embeds,
            mask=mask,
            cache=cache,
        )
        out = self.model.embed_tokens.as_linear(out)
        if self.final_logit_softcapping is not None:
            out = logit_softcap(self.final_logit_softcapping, out)
        return LanguageModelOutput(logits=out)

    def sanitize(self, weights):
        sanitized = {}
        for k, v in weights.items():
            if "self_attn.rotary_emb" in k:
                continue
            if any(
                s in k for s in ["input_max", "input_min", "output_max", "output_min"]
            ):
                if "vision_tower" not in k:
                    continue
            sanitized[k] = v
        return sanitized

    @property
    def layers(self):
        return self.model.layers

    @property
    def head_dim(self):
        return self.config.head_dim

    @property
    def n_kv_heads(self):
        return self.config.num_key_value_heads

    def make_cache(self):
        caches = []
        for i in range(self.config.num_hidden_layers):
            layer_type = self.config.layer_types[i]
            if layer_type == "full_attention":
                caches.append(KVCache())
            else:
                caches.append(
                    RotatingKVCache(
                        max_size=self.config.sliding_window,
                        keep=0,
                    )
                )
        return caches
