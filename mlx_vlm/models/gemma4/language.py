from functools import partial
from typing import Any, List, Optional

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


@partial(mx.compile, shapeless=True)
def geglu(gate, x):
    return nn.gelu_approx(gate) * x


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
        return self.down_proj(geglu(self.gate_proj(x), self.up_proj(x)))


class Router(nn.Module):
    """Expert router: norm -> scale -> project -> top-k -> renormalize."""

    def __init__(self, config: TextConfig):
        super().__init__()
        self.config = config
        self.eps = config.rms_norm_eps
        self.proj = nn.Linear(config.hidden_size, config.num_experts, bias=False)
        self.scale = mx.ones((config.hidden_size,))
        self.per_expert_scale = mx.ones((config.num_experts,))
        self._root_size = config.hidden_size**-0.5

    def __call__(self, x: mx.array):
        x = mx.fast.rms_norm(x, self.scale * self._root_size, self.eps)

        expert_scores = self.proj(x)

        top_k_indices = mx.argpartition(
            expert_scores, kth=-self.config.top_k_experts, axis=-1
        )
        top_k_indices = top_k_indices[..., -self.config.top_k_experts :]

        top_k_weights = mx.take_along_axis(expert_scores, top_k_indices, axis=-1)
        top_k_weights = mx.softmax(top_k_weights, axis=-1)
        top_k_weights = top_k_weights * self.per_expert_scale[top_k_indices]

        return top_k_indices, top_k_weights


class GeGLU(nn.Module):
    """GELU-gated linear unit activation for SwitchGLU."""

    def __call__(self, x, gate):
        return geglu(gate, x)


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
        w = mx.expand_dims(top_k_weights, -1)
        y = self.switch_glu(x, top_k_indices)
        return (w * y).sum(-2)


class Attention(nn.Module):
    def __init__(
        self,
        config: TextConfig,
        layer_idx: int,
        kv_shared_only: bool = False,
    ):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.layer_type = config.layer_types[layer_idx]
        self.is_sliding = self.layer_type == "sliding_attention"
        self.kv_shared_only = kv_shared_only

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
        if not kv_shared_only:
            self.k_proj = nn.Linear(dim, self.n_kv_heads * self.head_dim, bias=False)
            if not self.use_k_eq_v:
                self.v_proj = nn.Linear(
                    dim, self.n_kv_heads * self.head_dim, bias=False
                )
        self.o_proj = nn.Linear(self.n_heads * self.head_dim, dim, bias=False)

        self.q_norm = RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        if not kv_shared_only:
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
        shared_kv: Optional[tuple] = None,
        offset: Optional[Any] = None,
    ) -> mx.array:
        B, L, _ = x.shape

        queries = self.q_proj(x).reshape(B, L, self.n_heads, self.head_dim)
        queries = self.q_norm(queries)

        if shared_kv is not None:
            keys, values = shared_kv
        else:
            keys = self.k_proj(x).reshape(B, L, self.n_kv_heads, self.head_dim)

            # k_eq_v: values from raw k_proj (before k_norm)
            if self.use_k_eq_v:
                values = keys
            else:
                values = self.v_proj(x).reshape(B, L, self.n_kv_heads, self.head_dim)

            offset = mx.array(cache.offset) if cache is not None else 0

            keys = self.k_norm(keys)
            keys = keys.transpose(0, 2, 1, 3)
            keys = self.rope(keys, offset=offset)

            values = self.v_norm(values)
            values = values.transpose(0, 2, 1, 3)

            if cache is not None:
                keys, values = cache.update_and_fetch(keys, values)

        queries = queries.transpose(0, 2, 1, 3)
        queries = self.rope(queries, offset=offset)

        output = scaled_dot_product_attention(
            queries, keys, values, cache=cache, scale=self.scale, mask=mask
        )
        output = output.transpose(0, 2, 1, 3).reshape(B, L, -1)

        return self.o_proj(output), (keys, values), offset


class DecoderLayer(nn.Module):
    def __init__(
        self,
        config: TextConfig,
        layer_idx: int,
        kv_shared_only: bool = False,
    ):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.layer_type = config.layer_types[layer_idx]
        self.self_attn = Attention(config, layer_idx, kv_shared_only=kv_shared_only)
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
        shared_kv: Optional[tuple] = None,
        offset: Optional[Any] = None,
    ) -> mx.array:
        residual = x

        h = self.input_layernorm(x)
        h, shared_kv, offset = self.self_attn(
            h, mask, cache, shared_kv=shared_kv, offset=offset
        )
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

        return h, shared_kv, offset


class Gemma4TextModel(nn.Module):
    def __init__(self, config: TextConfig, kv_shared_only: bool = False):
        super().__init__()
        self.config = config
        self.vocab_size = config.vocab_size
        self.window_size = config.sliding_window
        self.sliding_window_pattern = config.sliding_window_pattern
        self.num_hidden_layers = config.num_hidden_layers

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.embed_scale = config.hidden_size**0.5
        self.layers = [
            DecoderLayer(config, layer_idx=i, kv_shared_only=kv_shared_only)
            for i in range(config.num_hidden_layers)
        ]
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        num_kv_shared = getattr(config, "num_kv_shared_layers", 0)
        self.first_kv_shared_layer_idx = config.num_hidden_layers - num_kv_shared
        self.previous_kvs = list(range(len(self.layers)))
        if num_kv_shared > 0:
            N = len(self.layers)
            M = N - num_kv_shared
            kvs_by_type = {}
            for i in range(M):
                kvs_by_type[self.layers[i].layer_type] = i
            for j in range(M, N):
                self.previous_kvs[j] = kvs_by_type[self.layers[j].layer_type]

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

    def _make_masks(self, h, cache):
        """Create attention masks, deduplicated by layer type."""
        mask = {}
        masks = []
        for l, c in zip(self.layers, cache):
            if l.layer_type not in mask:
                if l.layer_type == "full_attention":
                    mask["full_attention"] = create_attention_mask(h, c)
                elif l.layer_type == "sliding_attention":
                    mask["sliding_attention"] = create_attention_mask(
                        h, c, window_size=self.window_size
                    )
            masks.append(mask[l.layer_type])
        return masks

    def __call__(
        self,
        inputs: mx.array = None,
        inputs_embeds: Optional[mx.array] = None,
        mask: Optional[mx.array] = None,
        cache=None,
        per_layer_inputs: Optional[mx.array] = None,
        capture_layer_ids: Optional[List[int]] = None,
        hidden_sink: Optional[list] = None,
        shared_kv_sink: Optional[dict] = None,
        **kwargs,
    ):
        if inputs_embeds is None:
            h = self.embed_tokens(inputs)
            h = h * self.embed_scale
        else:
            h = inputs_embeds

        # Per-layer inputs (2B/4B models)
        if self.hidden_size_per_layer_input:
            if inputs is not None and per_layer_inputs is None:
                per_layer_inputs = self.get_per_layer_inputs(inputs)
            elif per_layer_inputs is not None:
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

        # Build cache + masks
        if cache is None:
            cache = [None] * len(self.layers)
        else:
            cache = cache + [None] * (len(self.layers) - len(cache))

        if mask is None:
            masks = self._make_masks(h, cache)
        else:
            masks = [mask] * len(self.layers)

        # Forward through layers
        if per_layer_inputs is not None:
            per_layer_inputs = [
                per_layer_inputs[:, :, i, :] for i, _ in enumerate(self.layers)
            ]
        else:
            per_layer_inputs = [None] * len(self.layers)

        capture_set = set(capture_layer_ids) if capture_layer_ids else set()
        intermediates = [(None, None)] * len(self.layers)
        for idx, (layer, c, m, prev_idx, pli) in enumerate(
            zip(
                self.layers,
                cache,
                masks,
                self.previous_kvs,
                per_layer_inputs,
            )
        ):
            kvs, offset = intermediates[prev_idx]
            h, kvs, offset = layer(
                h, m, c, per_layer_input=pli, shared_kv=kvs, offset=offset
            )
            intermediates[idx] = (kvs, offset)
            if hidden_sink is not None and idx in capture_set:
                hidden_sink.append(h)

        if shared_kv_sink is not None:
            for idx, layer in enumerate(self.layers):
                kvs, _ = intermediates[idx]
                if kvs is not None:
                    shared_kv_sink[layer.layer_type] = kvs

        # Match HF's `_can_record_outputs={"hidden_states": Gemma4TextDecoderLayer}`
        # — the recorded value is the LAST decoder layer's output, captured
        # BEFORE the final RMSNorm. The drafter's `pre_projection` was trained
        # against this pre-norm hidden.
        if hidden_sink is not None and not capture_set:
            hidden_sink.append(h)

        h = self.norm(h)

        return h


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
        capture_layer_ids: Optional[List[int]] = None,
        **kwargs,
    ):
        hidden_sink: Optional[list] = (
            []
            if capture_layer_ids is not None or kwargs.pop("return_hidden", False)
            else None
        )
        shared_kv_sink: Optional[dict] = (
            {} if kwargs.pop("return_shared_kv", False) else None
        )
        # Allow callers to pass pre-allocated sinks directly.
        hidden_sink = kwargs.pop("hidden_sink", hidden_sink)
        shared_kv_sink = kwargs.pop("shared_kv_sink", shared_kv_sink)

        out = self.model(
            inputs,
            inputs_embeds=inputs_embeds,
            mask=mask,
            cache=cache,
            per_layer_inputs=per_layer_inputs,
            capture_layer_ids=capture_layer_ids,
            hidden_sink=hidden_sink,
            shared_kv_sink=shared_kv_sink,
            **kwargs,
        )
        out = self.model.embed_tokens.as_linear(out)
        if self.final_logit_softcapping is not None:
            out = logit_softcap(self.final_logit_softcapping, out)
        return LanguageModelOutput(
            logits=out,
            hidden_states=hidden_sink,
            shared_kv_states=shared_kv_sink,
        )

    def rollback_speculative_cache(
        self,
        caches: List[Any],
        gdn_states: Any,
        accepted: Any,
        block_size: int,
    ) -> int:
        """Rewind target KV caches after a speculative-decoding round.

        Gemma 4 has only KV/RotatingKV caches (no SSM/GDN), so this is a
        simple trim + per-row tail-zero. ``gdn_states`` is accepted (and
        ignored) for API parity with qwen3_5's hook.
        """
        del gdn_states  # API-parity placeholder; Gemma 4 has no SSM/GDN state.
        if isinstance(accepted, int):
            accepted = mx.array([accepted])

        max_a = int(accepted.max().item())
        n = max_a + 1
        trim = block_size - n
        is_batch = accepted.size > 1
        valid_ends = accepted + 1

        for c in caches:
            if c is None:
                continue

            if trim > 0 and hasattr(c, "trim"):
                c.trim(trim)
            if is_batch and hasattr(c, "_idx") and c.keys is not None and max_a > 0:
                kv_len = c._idx
                ve = valid_ends.tolist()
                verify_start = kv_len - n
                for bi in range(accepted.shape[0]):
                    start = verify_start + int(ve[bi])
                    if start < kv_len:
                        c.keys[bi, :, start:kv_len, :] = 0
                        c.values[bi, :, start:kv_len, :] = 0
        return max_a

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
