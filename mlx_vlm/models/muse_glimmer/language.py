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
from ..target_verify import exact_quantized_linear
from .config import TextConfig


@mx.compile
def _centered_rms_norm(x: mx.array, weight: mx.array, eps: float) -> mx.array:
    dtype = x.dtype
    x = x.astype(mx.float32)
    variance = mx.mean(mx.square(x), axis=-1, keepdims=True)
    x = x * mx.rsqrt(variance + eps)
    x = x * (1.0 + weight.astype(mx.float32))
    return x.astype(dtype)


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


def _target_verify_linear(linear, x: mx.array, target_verify: bool) -> mx.array:
    """Use decode-shaped quantized matmuls while verifying draft blocks."""

    if not (
        target_verify
        and x.ndim == 3
        and x.shape[1] > 1
        and isinstance(linear, nn.QuantizedLinear)
    ):
        return linear(x)
    output = exact_quantized_linear(linear, x)
    if output is not None:
        return output
    return mx.concatenate(
        [linear(x[:, index : index + 1]) for index in range(x.shape[1])],
        axis=1,
    )


def _target_verify_linears(linears, x: mx.array, target_verify: bool):
    return tuple(_target_verify_linear(linear, x, target_verify) for linear in linears)


class MLP(nn.Module):
    def __init__(self, args: TextConfig):
        super().__init__()
        self.gate_proj = nn.Linear(args.hidden_size, args.intermediate_size, bias=False)
        self.up_proj = nn.Linear(args.hidden_size, args.intermediate_size, bias=False)
        self.down_proj = nn.Linear(args.intermediate_size, args.hidden_size, bias=False)

    def __call__(self, x: mx.array, target_verify: bool = False) -> mx.array:
        gate, up = _target_verify_linears(
            (self.gate_proj, self.up_proj), x, target_verify
        )
        return _target_verify_linear(
            self.down_proj,
            swiglu(gate, up),
            target_verify,
        )


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
        )

    def __call__(
        self,
        x: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional[Any] = None,
        target_verify: bool = False,
    ) -> mx.array:
        batch, length, _ = x.shape
        queries, keys, values = _target_verify_linears(
            (self.q_proj, self.k_proj, self.v_proj), x, target_verify
        )
        queries = queries.reshape(batch, length, self.n_heads, self.head_dim)
        keys = keys.reshape(batch, length, self.n_kv_heads, self.head_dim)
        values = values.reshape(batch, length, self.n_kv_heads, self.head_dim)

        queries = (self.qk_norm(queries) * self.qk_scale_factor).transpose(0, 2, 1, 3)
        keys = self.qk_norm(keys).transpose(0, 2, 1, 3)
        values = values.transpose(0, 2, 1, 3)

        if self.use_rope:
            offset = cache.offset if cache is not None else 0
            queries = self.rope(queries, offset=offset)
            keys = self.rope(keys, offset=offset)

        if target_verify and length > 1:
            rotate_during_verify = (
                isinstance(cache, RotatingKVCache)
                and cache.offset + length > cache.max_size
            )
            if rotate_during_verify:
                token_outputs = []
                for index in range(length):
                    attention_keys, attention_values = cache.update_and_fetch(
                        keys[:, :, index : index + 1],
                        values[:, :, index : index + 1],
                    )
                    token_outputs.append(
                        scaled_dot_product_attention(
                            queries[:, :, index : index + 1],
                            attention_keys,
                            attention_values,
                            cache=cache,
                            scale=self.scale,
                            mask=None,
                        )
                    )
                output = mx.concatenate(token_outputs, axis=2)
            else:
                if cache is not None:
                    keys, values = cache.update_and_fetch(keys, values)
                prefix_length = keys.shape[-2] - length
                output = mx.concatenate(
                    [
                        scaled_dot_product_attention(
                            queries[:, :, index : index + 1],
                            keys[:, :, : prefix_length + index + 1],
                            values[:, :, : prefix_length + index + 1],
                            cache=cache,
                            scale=self.scale,
                            mask=None,
                        )
                        for index in range(length)
                    ],
                    axis=2,
                )
        else:
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
        gate = _target_verify_linear(self.gate_proj, x, target_verify)
        output = output * mx.sigmoid(gate)
        return _target_verify_linear(self.o_proj, output, target_verify)


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
        target_verify: bool = False,
    ) -> mx.array:
        residual = x
        x = self.self_attn(
            self.input_layernorm(x),
            mask=mask,
            cache=cache,
            target_verify=target_verify,
        )
        x = residual + self.post_attention_layernorm(x)

        residual = x
        x = self.mlp(
            self.pre_feedforward_layernorm(x),
            target_verify=target_verify,
        )
        return residual + self.post_feedforward_layernorm(x)


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
        target_verify: bool = False,
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
            hidden_states = layer(
                hidden_states,
                mask=mask,
                cache=layer_cache,
                target_verify=target_verify,
            )
            if hidden_sink is not None and layer_idx in capture_set:
                hidden_sink.append(hidden_states)
        return self.norm(hidden_states)


class LanguageModel(nn.Module):
    supports_speculative_target_verify = True

    def __init__(self, args: TextConfig):
        super().__init__()
        self.args = args
        self.config = args
        self.model_type = args.model_type
        self.model = TextModel(args)
        self.lm_head = nn.Linear(args.hidden_size, args.vocab_size, bias=False)
        self.final_logit_softcapping = args.final_logit_softcapping
        self.output_multiplier = args.output_multiplier

    def chunked_prefill_policy(
        self,
        *,
        input_ids=None,
        inputs_embeds=None,
        prompt_cache=None,
        draft_model=None,
        draft_kind=None,
        prefill_kwargs=None,
    ) -> bool:
        del input_ids, inputs_embeds, prompt_cache
        if draft_model is None:
            return True
        prefill_kwargs = prefill_kwargs or {}
        if draft_kind in ("dflash", "eagle3"):
            return prefill_kwargs.get("capture_layer_ids") is not None
        return False

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
        target_verify = bool(kwargs.pop("target_verify", False))
        batch_size = (
            inputs_embeds.shape[0] if inputs_embeds is not None else inputs.shape[0]
        )
        target_verify = (
            target_verify
            and batch_size == 1
            and isinstance(self.lm_head, nn.QuantizedLinear)
        )
        hidden_sink: Optional[list[mx.array]] = (
            [] if capture_layer_ids is not None else None
        )
        hidden_states = self.model(
            inputs,
            cache=cache,
            inputs_embeds=inputs_embeds,
            capture_layer_ids=capture_layer_ids,
            hidden_sink=hidden_sink,
            target_verify=target_verify,
        )
        logits = (
            _target_verify_linear(
                self.lm_head,
                hidden_states,
                target_verify,
            )
            * self.output_multiplier
        )
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
