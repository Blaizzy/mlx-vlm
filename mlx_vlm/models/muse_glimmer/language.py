from typing import Any, Optional

import mlx.core as mx
import mlx.nn as nn

from ..activations import swiglu
from ..base import (
    LanguageModelOutput,
    create_attention_mask,
    scaled_dot_product_attention,
)
from ..cache import KVCache, RotatingKVCache
from ..rope_utils import initialize_rope
from .config import TextConfig


@mx.compile
def _centered_rms_norm(x: mx.array, weight: mx.array, eps: float) -> mx.array:
    dtype = x.dtype
    x = x.astype(mx.float32)
    variance = mx.mean(mx.square(x), axis=-1, keepdims=True)
    x = x * mx.rsqrt(variance + eps)
    x = x * (1.0 + weight.astype(mx.float32))
    return x.astype(dtype)


@mx.compile
def _prepare_mlp_input(
    residual: mx.array,
    attention_output: mx.array,
    post_attention_weight: mx.array,
    pre_feedforward_weight: mx.array,
    post_attention_eps: float,
    pre_feedforward_eps: float,
) -> tuple[mx.array, mx.array]:
    hidden_states = residual + _centered_rms_norm(
        attention_output,
        post_attention_weight,
        post_attention_eps,
    )
    mlp_input = _centered_rms_norm(
        hidden_states,
        pre_feedforward_weight,
        pre_feedforward_eps,
    )
    return hidden_states, mlp_input


@mx.compile
def _finish_mlp(
    residual: mx.array,
    mlp_output: mx.array,
    post_feedforward_weight: mx.array,
    post_feedforward_eps: float,
) -> mx.array:
    return residual + _centered_rms_norm(
        mlp_output,
        post_feedforward_weight,
        post_feedforward_eps,
    )


class RMSNormNoScale(nn.Module):
    def __init__(self, eps: float):
        super().__init__()
        self.eps = eps

    def __call__(self, x: mx.array) -> mx.array:
        return mx.fast.rms_norm(x, None, self.eps)


class CenteredRMSNorm(nn.Module):
    """RMSNorm whose checkpoint scale is centered at zero (effective scale 1+w)."""

    def __init__(self, dim: int, eps: float):
        super().__init__()
        self.weight = mx.zeros((dim,))
        self.eps = eps

    def __call__(self, x: mx.array) -> mx.array:
        # Transformers applies the centered scale in FP32 before casting back
        # to the activation dtype. Casting earlier changes BF16 decode choices.
        return _centered_rms_norm(x, self.weight, self.eps)


class MLP(nn.Module):
    def __init__(self, args: TextConfig):
        super().__init__()
        self.gate_proj = nn.Linear(args.hidden_size, args.intermediate_size, bias=False)
        self.up_proj = nn.Linear(args.hidden_size, args.intermediate_size, bias=False)
        self.down_proj = nn.Linear(args.intermediate_size, args.hidden_size, bias=False)

    def __call__(self, x: mx.array) -> mx.array:
        return self.down_proj(swiglu(self.gate_proj(x), self.up_proj(x)))


class Attention(nn.Module):
    def __init__(self, args: TextConfig, layer_idx: int):
        super().__init__()
        self.n_heads = args.num_attention_heads
        self.n_kv_heads = args.num_key_value_heads
        self.head_dim = args.head_dim
        self.scale = self.head_dim**-0.5
        self.qk_scale_factor = args.qk_scale_factor
        self.use_rope = bool(args.layer_rope_theta[layer_idx])
        self.is_sliding = args.layer_types[layer_idx] == "sliding_attention"

        dim = args.hidden_size
        self.q_proj = nn.Linear(
            dim, self.n_heads * self.head_dim, bias=args.attention_bias
        )
        self.k_proj = nn.Linear(
            dim, self.n_kv_heads * self.head_dim, bias=args.attention_bias
        )
        self.v_proj = nn.Linear(
            dim, self.n_kv_heads * self.head_dim, bias=args.attention_bias
        )
        self.gate_proj = nn.Linear(dim, self.n_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(
            self.n_heads * self.head_dim, dim, bias=args.attention_bias
        )
        self.qk_norm = RMSNormNoScale(args.rms_norm_eps)

        theta = (
            float(args.layer_rope_theta[layer_idx])
            if self.use_rope
            else float(args.rope_parameters.get("rope_theta", 500000.0))
        )
        self.rope = initialize_rope(
            self.head_dim,
            base=theta,
            traditional=False,
            scaling_config={"rope_type": "default", "rope_theta": theta},
            max_position_embeddings=args.max_position_embeddings,
            implementation="eager",
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
        output = output.transpose(0, 2, 1, 3).reshape(batch, length, -1)
        output = output * mx.sigmoid(self.gate_proj(x))
        return self.o_proj(output)


class DecoderLayer(nn.Module):
    def __init__(self, args: TextConfig, layer_idx: int):
        super().__init__()
        self.self_attn = Attention(args, layer_idx)
        self.mlp = MLP(args)
        self.input_layernorm = CenteredRMSNorm(args.hidden_size, args.rms_norm_eps)
        self.post_attention_layernorm = CenteredRMSNorm(
            args.hidden_size, args.post_norm_eps
        )
        self.pre_feedforward_layernorm = CenteredRMSNorm(
            args.hidden_size, args.rms_norm_eps
        )
        self.post_feedforward_layernorm = CenteredRMSNorm(
            args.hidden_size, args.post_norm_eps
        )
        self.is_sliding = args.layer_types[layer_idx] == "sliding_attention"

    def __call__(
        self,
        x: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional[Any] = None,
    ) -> mx.array:
        residual = x
        x = self.self_attn(self.input_layernorm(x), mask=mask, cache=cache)
        residual, mlp_input = _prepare_mlp_input(
            residual,
            x,
            self.post_attention_layernorm.weight,
            self.pre_feedforward_layernorm.weight,
            self.post_attention_layernorm.eps,
            self.pre_feedforward_layernorm.eps,
        )
        return _finish_mlp(
            residual,
            self.mlp(mlp_input),
            self.post_feedforward_layernorm.weight,
            self.post_feedforward_layernorm.eps,
        )


class TextModel(nn.Module):
    def __init__(self, args: TextConfig):
        super().__init__()
        self.args = args
        self.embed_tokens = nn.Embedding(args.vocab_size, args.hidden_size)
        self.embed_norm = RMSNormNoScale(args.rms_norm_eps)
        self.layers = [DecoderLayer(args, idx) for idx in range(args.num_hidden_layers)]
        self.norm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)
        self.layer_types = args.layer_types
        self.sliding_window = args.sliding_window

        self.full_attention_idx = self.layer_types.index("full_attention")
        self.sliding_attention_idx = (
            self.layer_types.index("sliding_attention")
            if "sliding_attention" in self.layer_types
            else None
        )

    def __call__(
        self,
        inputs: Optional[mx.array],
        cache=None,
        inputs_embeds: Optional[mx.array] = None,
        capture_layer_ids: Optional[list[int]] = None,
        hidden_sink: Optional[list[mx.array]] = None,
    ) -> mx.array:
        hidden_states = inputs_embeds
        if hidden_states is None:
            hidden_states = self.embed_norm(self.embed_tokens(inputs))
        if cache is None:
            cache = [None] * len(self.layers)

        full_mask = create_attention_mask(hidden_states, cache[self.full_attention_idx])
        sliding_mask = None
        if self.sliding_attention_idx is not None:
            sliding_mask = create_attention_mask(
                hidden_states,
                cache[self.sliding_attention_idx],
                window_size=self.sliding_window,
            )

        capture_set = set(capture_layer_ids) if capture_layer_ids else set()
        for layer_idx, (layer, layer_cache) in enumerate(zip(self.layers, cache)):
            mask = sliding_mask if layer.is_sliding else full_mask
            hidden_states = layer(hidden_states, mask=mask, cache=layer_cache)
            if hidden_sink is not None and layer_idx in capture_set:
                hidden_sink.append(hidden_states)
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
        capture_layer_ids = kwargs.pop("capture_layer_ids", None)
        hidden_sink: Optional[list[mx.array]] = (
            [] if capture_layer_ids is not None else None
        )
        hidden_states = self.model(
            inputs,
            cache=cache,
            inputs_embeds=inputs_embeds,
            capture_layer_ids=capture_layer_ids,
            hidden_sink=hidden_sink,
        )
        logits = self.lm_head(hidden_states) * self.output_multiplier
        softcap = self.final_logit_softcapping
        logits = mx.tanh(logits / softcap) * softcap
        return LanguageModelOutput(logits=logits, hidden_states=hidden_sink)

    def rollback_speculative_cache(
        self,
        caches: list[Any],
        gdn_states: Any,
        accepted: Any,
        block_size: int,
    ) -> int:
        """Rewind target KV caches to the accepted speculative prefix."""

        del gdn_states
        if isinstance(accepted, int):
            accepted = mx.array([accepted])
        if isinstance(accepted, (list, tuple)):
            accepted = mx.array(accepted, dtype=mx.int32)

        max_accepted = int(accepted.max().item())
        retained = max_accepted + 1
        trim = block_size - retained
        is_batch = accepted.size > 1
        valid_ends = accepted + 1

        for cache in caches:
            if cache is None:
                continue
            if trim > 0 and hasattr(cache, "trim"):
                cache.trim(trim)
            if (
                is_batch
                and hasattr(cache, "_idx")
                and cache.keys is not None
                and max_accepted > 0
            ):
                cache_length = cache._idx
                verify_start = cache_length - retained
                for row, valid_end in enumerate(valid_ends.tolist()):
                    start = verify_start + int(valid_end)
                    if start >= cache_length:
                        continue
                    zero_row_tail = getattr(cache, "zero_row_tail", None)
                    if callable(zero_row_tail):
                        zero_row_tail(row, start, cache_length)
                    else:
                        cache.keys[row, :, start:cache_length, :] = 0
                        cache.values[row, :, start:cache_length, :] = 0
        return max_accepted

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
