from functools import partial
from typing import Any, Optional

import mlx.core as mx
import mlx.nn as nn
from mlx.nn import RMSNorm

from ..base import (
    LanguageModelOutput,
    create_attention_mask,
    scaled_dot_product_attention,
)
from ..cache import KVCache, RotatingKVCache
from .config import TextConfig
from .rope_utils import initialize_rope


class RMSNormNoScale(nn.Module):
    """RMSNorm without learnable scale (with_scale=False, scale_shift=0.0)."""

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps

    def __call__(self, x: mx.array) -> mx.array:
        return mx.fast.rms_norm(x, None, self.eps)


class RMSNormZeroShift(nn.Module):
    """Gemma4 RMSNorm with scale_shift=0.0 (weight used directly, no +1 offset)."""

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.weight = mx.ones((dim,))
        self.eps = eps

    def __call__(self, x: mx.array) -> mx.array:
        return mx.fast.rms_norm(x, self.weight, self.eps)


@partial(mx.compile, shapeless=True)
def logit_softcap(softcap, x):
    return mx.tanh(x / softcap) * softcap


class MLP(nn.Module):
    def __init__(self, config: TextConfig, layer_idx: int = 0):
        super().__init__()
        first_kv_shared_layer_idx = config.num_hidden_layers - getattr(
            config, "num_kv_shared_layers", 0
        )
        is_kv_shared_layer = layer_idx >= first_kv_shared_layer_idx > 0
        use_double_wide = (
            getattr(config, "use_double_wide_mlp", False) and is_kv_shared_layer
        )
        intermediate_size = config.intermediate_size * (2 if use_double_wide else 1)

        self.gate_proj = nn.Linear(config.hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, config.hidden_size, bias=False)
        self.up_proj = nn.Linear(config.hidden_size, intermediate_size, bias=False)

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
        self.per_expert_scale = mx.ones((config.num_experts,))
        self._root_size = config.hidden_size**-0.5

    def __call__(self, x: mx.array):
        x = self.norm(x)
        x = x * self._root_size
        x = x * self.scale

        expert_scores = self.proj(x)
        router_probs = mx.softmax(expert_scores, axis=-1)

        top_k_indices = mx.argpartition(
            -expert_scores, kth=self.config.top_k_experts - 1, axis=-1
        )[..., : self.config.top_k_experts]

        top_k_weights = mx.take_along_axis(router_probs, top_k_indices, axis=-1)
        top_k_weights = top_k_weights / mx.sum(top_k_weights, axis=-1, keepdims=True)
        top_k_weights = top_k_weights * self.per_expert_scale[top_k_indices]
        return top_k_indices, top_k_weights


class GeGLU(nn.Module):
    """GELU-gated linear unit activation for SwitchGLU."""

    def __call__(self, x, gate):
        return nn.gelu_approx(gate) * x


class Experts(nn.Module):
    """Sparse MoE using mlx_lm SwitchGLU with gather_mm."""

    def __init__(self, config: TextConfig):
        super().__init__()
        from mlx_lm.models.switch_layers import SwitchGLU

        self.switch_glu = SwitchGLU(
            input_dims=config.hidden_size,
            hidden_dims=config.moe_intermediate_size,
            num_experts=config.num_experts,
            activation=GeGLU(),
            bias=False,
        )

    def __call__(
        self, x: mx.array, top_k_indices: mx.array, top_k_weights: mx.array
    ) -> mx.array:
        B, S, H = x.shape
        K = top_k_indices.shape[-1]

        x_flat = x.reshape(B * S, H)
        indices_flat = top_k_indices.reshape(B * S, K)

        expert_out = self.switch_glu(x_flat, indices_flat)

        weights = top_k_weights.reshape(B * S, K)[..., None]
        return (expert_out * weights).sum(axis=-2).reshape(B, S, H)


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

        # K-eq-V for full attention layers (26B/31B models)
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

        # RoPE (with partial rotation support)
        layer_key = "sliding_attention" if self.is_sliding else "full_attention"
        rope_params = config.rope_parameters.get(layer_key, {})
        rope_theta = rope_params.get("rope_theta", 10000.0)

        self.rope = initialize_rope(
            dims=self.head_dim,
            traditional=config.rope_traditional,
            base=rope_theta,
            scaling_config=rope_params,
            max_position_embeddings=config.max_position_embeddings,
        )

        # KV sharing (2B/4B models)
        first_kv_shared_layer_idx = config.num_hidden_layers - getattr(
            config, "num_kv_shared_layers", 0
        )
        self.is_kv_shared_layer = layer_idx >= first_kv_shared_layer_idx > 0

    def __call__(
        self,
        x: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional[Any] = None,
    ) -> mx.array:
        B, L, _ = x.shape

        queries = self.q_proj(x).reshape(B, L, self.n_heads, self.head_dim)
        queries = self.q_norm(queries)

        offset = 0
        if self.is_kv_shared_layer and cache is not None:
            state = cache.state
            keys, values = state[0], state[1]
            # Snapshot via + 0 so cache.update_and_fetch cannot mutate this
            # local alias under batched caches where cache.offset is an
            # mx.array (mx.array.__iadd__ is in place; int.__iadd__ rebinds,
            # so + 0 is safe for both).
            offset = cache.offset + 0
        else:
            if cache is not None:
                offset = cache.offset + 0

            keys = self.k_proj(x).reshape(B, L, self.n_kv_heads, self.head_dim)

            # k_eq_v: values from raw k_proj (before k_norm)
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
        self.mlp = MLP(config, layer_idx)
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

        # MoE (26B model)
        self.enable_moe = getattr(config, "enable_moe_block", False)
        if self.enable_moe:
            self.router = Router(config)
            self.experts = Experts(config)
            self.post_feedforward_layernorm_1 = RMSNorm(
                config.hidden_size, eps=config.rms_norm_eps
            )
            self.post_feedforward_layernorm_2 = RMSNorm(
                config.hidden_size, eps=config.rms_norm_eps
            )
            self.pre_feedforward_layernorm_2 = RMSNorm(
                config.hidden_size, eps=config.rms_norm_eps
            )

        # Per-layer input gating (2B/4B models)
        self.hidden_size_per_layer_input = getattr(
            config, "hidden_size_per_layer_input", 0
        )
        if self.hidden_size_per_layer_input:
            self.per_layer_input_gate = nn.Linear(
                config.hidden_size, self.hidden_size_per_layer_input, bias=False
            )
            self.per_layer_projection = nn.Linear(
                self.hidden_size_per_layer_input, config.hidden_size, bias=False
            )
            self.post_per_layer_input_norm = RMSNorm(
                config.hidden_size, eps=config.rms_norm_eps
            )
        else:
            self.per_layer_input_gate = None
            self.per_layer_projection = None
            self.post_per_layer_input_norm = None

        # Layer scalar (all text layers)
        self.layer_scalar = mx.ones((1,))

    def __call__(
        self,
        x: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional[Any] = None,
        per_layer_input: Optional[mx.array] = None,
    ) -> mx.array:
        residual = x

        h = self.input_layernorm(x)
        h = self.self_attn(h, mask, cache)
        h = self.post_attention_layernorm(h)
        h = residual + h

        residual = h

        if self.enable_moe:
            h1 = self.pre_feedforward_layernorm(h)
            h1 = self.mlp(h1)
            h1 = self.post_feedforward_layernorm_1(h1)

            top_k_indices, top_k_weights = self.router(h)
            h2 = self.pre_feedforward_layernorm_2(h)
            h2 = self.experts(h2, top_k_indices, top_k_weights)
            h2 = self.post_feedforward_layernorm_2(h2)

            h = h1 + h2
        else:
            h = self.pre_feedforward_layernorm(h)
            h = self.mlp(h)

        h = self.post_feedforward_layernorm(h)
        h = residual + h

        # Per-layer input gating
        if (
            self.per_layer_input_gate is not None
            and self.per_layer_projection is not None
            and self.post_per_layer_input_norm is not None
            and per_layer_input is not None
        ):
            residual = h
            gate = self.per_layer_input_gate(h)
            gate = nn.gelu_approx(gate)
            gate = mx.multiply(gate, per_layer_input)
            gate = self.per_layer_projection(gate)
            gate = self.post_per_layer_input_norm(gate)
            h = residual + gate

        if self.layer_scalar is not None:
            h = h * self.layer_scalar

        return h


class Gemma4TextModel(nn.Module):
    def __init__(self, config: TextConfig):
        super().__init__()
        self.config = config
        self.vocab_size = config.vocab_size
        self.window_size = config.sliding_window
        self.sliding_window_pattern = config.sliding_window_pattern
        self.num_hidden_layers = config.num_hidden_layers

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.embed_scale = config.hidden_size**0.5
        self.layers = [
            DecoderLayer(config, layer_idx=i) for i in range(config.num_hidden_layers)
        ]
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        # KV sharing: only non-shared layers own a cache
        self.first_kv_shared_layer_idx = config.num_hidden_layers - getattr(
            config, "num_kv_shared_layers", 0
        )
        concrete_layers = config.layer_types[: self.first_kv_shared_layer_idx]
        self.layer_idx_to_cache_idx = list(range(self.first_kv_shared_layer_idx))
        if self.first_kv_shared_layer_idx < config.num_hidden_layers:
            shared_full_idx = (
                len(concrete_layers) - 1 - concrete_layers[::-1].index("full_attention")
            )
            shared_sliding_idx = (
                len(concrete_layers)
                - 1
                - concrete_layers[::-1].index("sliding_attention")
            )
            for i in range(self.first_kv_shared_layer_idx, config.num_hidden_layers):
                if config.layer_types[i] == "full_attention":
                    self.layer_idx_to_cache_idx.append(shared_full_idx)
                else:
                    self.layer_idx_to_cache_idx.append(shared_sliding_idx)

        # First cache indices by attention type (for mask creation)
        self.first_full_cache_idx = next(
            (i for i, t in enumerate(concrete_layers) if t == "full_attention"), 0
        )
        self.first_sliding_cache_idx = next(
            (i for i, t in enumerate(concrete_layers) if t == "sliding_attention"), 0
        )

        # Per-layer input embeddings (2B/4B models)
        self.hidden_size_per_layer_input = config.hidden_size_per_layer_input
        if self.hidden_size_per_layer_input:
            self.embed_tokens_per_layer = nn.Embedding(
                config.vocab_size_per_layer_input,
                config.num_hidden_layers * config.hidden_size_per_layer_input,
            )
            self.embed_tokens_per_layer_scale = config.hidden_size_per_layer_input**0.5
            self.per_layer_input_scale = 2.0**-0.5
            self.per_layer_projection_scale = config.hidden_size**-0.5
            self.per_layer_model_projection = nn.Linear(
                config.hidden_size,
                config.num_hidden_layers * config.hidden_size_per_layer_input,
                bias=False,
            )
            self.per_layer_projection_norm = RMSNormZeroShift(
                config.hidden_size_per_layer_input, eps=config.rms_norm_eps
            )
        else:
            self.embed_tokens_per_layer = None
            self.per_layer_input_scale = None
            self.per_layer_projection_scale = None
            self.per_layer_model_projection = None
            self.per_layer_projection_norm = None

    def get_per_layer_inputs(self, input_ids: mx.array) -> mx.array:
        result = self.embed_tokens_per_layer(input_ids)
        result = result * self.embed_tokens_per_layer_scale
        return result.reshape(
            *input_ids.shape,
            self.config.num_hidden_layers,
            self.hidden_size_per_layer_input,
        )

    def project_per_layer_inputs(
        self,
        inputs_embeds: mx.array,
        per_layer_inputs: Optional[mx.array] = None,
    ) -> mx.array:
        per_layer_projection = self.per_layer_model_projection(inputs_embeds)
        per_layer_projection = per_layer_projection * self.per_layer_projection_scale
        per_layer_projection = per_layer_projection.reshape(
            *inputs_embeds.shape[:-1],
            self.config.num_hidden_layers,
            self.hidden_size_per_layer_input,
        )
        per_layer_projection = self.per_layer_projection_norm(per_layer_projection)

        if per_layer_inputs is None:
            return per_layer_projection

        return (per_layer_projection + per_layer_inputs) * self.per_layer_input_scale

    def __call__(
        self,
        inputs: mx.array = None,
        inputs_embeds: Optional[mx.array] = None,
        mask: Optional[mx.array] = None,
        cache=None,
        per_layer_inputs: Optional[mx.array] = None,
        **kwargs,
    ):
        if inputs_embeds is None:
            h = self.embed_tokens(inputs)
            h = h * self.embed_scale
        else:
            h = inputs_embeds

        if self.hidden_size_per_layer_input:
            if inputs is not None and per_layer_inputs is None:
                per_layer_inputs = self.get_per_layer_inputs(inputs)
            elif per_layer_inputs is not None:
                # Slice per_layer_inputs to match current chunk (chunked prefill)
                target_len = h.shape[1]
                if per_layer_inputs.shape[1] != target_len:
                    cache_offset = next(
                        (
                            int(c.offset)
                            for c in (cache or [])
                            if c is not None and hasattr(c, "offset")
                        ),
                        0,
                    )
                    max_start = max(per_layer_inputs.shape[1] - target_len, 0)
                    start = min(cache_offset, max_start)
                    per_layer_inputs = per_layer_inputs[:, start : start + target_len]
            if per_layer_inputs is not None or inputs is not None:
                per_layer_inputs = self.project_per_layer_inputs(h, per_layer_inputs)

        if cache is None:
            cache = [None] * self.first_kv_shared_layer_idx

        if mask is None:
            global_mask = create_attention_mask(
                h,
                (
                    cache[self.first_full_cache_idx]
                    if self.first_full_cache_idx < len(cache)
                    else None
                ),
            )
            sliding_window_mask = create_attention_mask(
                h,
                (
                    cache[self.first_sliding_cache_idx]
                    if self.first_sliding_cache_idx < len(cache)
                    else None
                ),
                window_size=self.window_size,
            )

        for i, layer in enumerate(self.layers):
            c = cache[self.layer_idx_to_cache_idx[i]]
            is_global = layer.layer_type == "full_attention"

            local_mask = mask
            if mask is None and is_global:
                local_mask = global_mask
            elif mask is None:
                local_mask = sliding_window_mask

            per_layer_input = None
            if per_layer_inputs is not None:
                per_layer_input = per_layer_inputs[:, :, i, :]

            h = layer(
                h,
                local_mask,
                c,
                per_layer_input=per_layer_input,
            )

        return self.norm(h)


class LanguageModel(nn.Module):
    def __init__(self, config: TextConfig):
        super().__init__()
        self.config = config
        self.model_type = config.model_type
        self.model = Gemma4TextModel(config)
        self.final_logit_softcapping = getattr(config, "final_logit_softcapping", None)

    def __call__(
        self,
        inputs: mx.array = None,
        inputs_embeds: Optional[mx.array] = None,
        mask: Optional[mx.array] = None,
        cache=None,
        per_layer_inputs: Optional[mx.array] = None,
        **kwargs,
    ):
        out = self.model(
            inputs,
            inputs_embeds=inputs_embeds,
            mask=mask,
            cache=cache,
            per_layer_inputs=per_layer_inputs,
            **kwargs,
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
                if "vision_tower" not in k and "audio_tower" not in k:
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

    @property
    def quant_predicate(self):
        def predicate(path, m):
            if not hasattr(m, "to_quantized"):
                return False
            if "router" in path:
                return {"group_size": 64, "bits": 8}
            if path.endswith(("mlp.gate_proj", "mlp.up_proj", "mlp.down_proj")):
                return {"group_size": 64, "bits": 8}
            return True

        return predicate

    def make_cache(self):
        caches = []
        for layer_type in self.config.layer_types[
            : self.model.first_kv_shared_layer_idx
        ]:
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
