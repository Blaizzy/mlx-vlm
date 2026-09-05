from typing import Any, Optional

import mlx.core as mx
import mlx.nn as nn

from ..activations import swiglu
from ..base import (
    LanguageModelOutput,
    kv_sequence_length,
    scaled_dot_product_attention,
    slice_kv_sequence,
)
from ..exact_speculative_verify import exact_speculative_verify_dense_available
from ..exact_speculative_verify import (
    exact_speculative_verify_weight as _target_verify_weight,
)
from ..quantized_verifier import DEFAULT_QUANTIZED_VERIFIER
from ..quantized_verifier import (
    optimized_affine_argmax as _target_verify_optimized_affine_argmax,
)
from ..quantized_verifier import (
    optimized_affine_linear as _target_verify_optimized_affine_linear,
)
from ..quantized_verifier import (
    optimized_affine_linears as _target_verify_quantized_linears,
)
from ..quantized_verifier import (
    optimized_nvfp4_argmax as _target_verify_optimized_nvfp4_argmax,
)
from ..quantized_verifier import pad_token_mask as _pad_token_mask_to_head
from ..quantized_verifier import (
    singleton_quantized_argmax as _target_verify_singleton_quantized_argmax,
)
from ..quantized_verifier import (
    singleton_quantized_linear as _target_verify_singleton_quantized_linear,
)
from .gated_delta import gated_delta_update_with_states


def _use_target_verify_dense(linear, x: mx.array) -> bool:
    return (
        exact_speculative_verify_dense_available()
        and x.ndim == 3
        and (x.shape[0] > 1 or x.shape[1] > 1)
        and isinstance(linear, (nn.Linear, nn.QuantizedLinear))
    )


def _target_verify_quantized_linear(linear, x: mx.array) -> Optional[mx.array]:
    output = _target_verify_optimized_affine_linear(linear, x)
    if output is not None:
        return output
    return _target_verify_singleton_quantized_linear(linear, x)


def _target_verify_quantized_argmax(
    linear, x: mx.array, token_mask: Optional[mx.array] = None
) -> Optional[mx.array]:
    output = _target_verify_optimized_affine_argmax(linear, x, token_mask=token_mask)
    if output is not None:
        return output
    output = _target_verify_optimized_nvfp4_argmax(linear, x, token_mask=token_mask)
    if output is not None:
        return output
    return _target_verify_singleton_quantized_argmax(linear, x, token_mask=token_mask)


def _can_target_verify_quantized_head(linear) -> bool:
    return DEFAULT_QUANTIZED_VERIFIER.supports(linear)


def _target_verify_timewise(fn, x: mx.array) -> mx.array:
    return mx.concatenate(
        [fn(mx.contiguous(x[:, i : i + 1])) for i in range(x.shape[1])], axis=1
    )


def _target_verify_singletons(fn, x: mx.array) -> mx.array:
    rows = []
    for row in range(x.shape[0]):
        rows.append(
            mx.concatenate(
                [fn(x[row : row + 1, i : i + 1]) for i in range(x.shape[1])],
                axis=1,
            )
        )
    return mx.concatenate(rows, axis=0)


def _target_verify_linear(linear, x: mx.array) -> mx.array:
    if not _use_target_verify_dense(linear, x):
        return linear(x)

    if isinstance(linear, nn.QuantizedLinear):
        out = _target_verify_quantized_linear(linear, x)
        if out is not None:
            return out
        if x.shape[0] > 1:
            return _target_verify_singletons(linear, x)
        return _target_verify_timewise(linear, x)

    if isinstance(linear, nn.Linear) and "bias" not in linear:
        out = _target_verify_weight(linear.weight, x)
        if out is not None:
            return out

    return _target_verify_singletons(linear, x)


def _target_verify_linears(linears, x: mx.array):
    if not (
        x.ndim == 3
        and (x.shape[0] > 1 or x.shape[1] > 1)
        and all(
            isinstance(linear, (nn.Linear, nn.QuantizedLinear)) for linear in linears
        )
    ):
        from .language import _decode_quantized_linears_fused

        out = _decode_quantized_linears_fused(linears, x)
        if out is not None:
            return out
        return tuple(linear(x) for linear in linears)

    out = _target_verify_quantized_linears(linears, x)
    if out is not None:
        return out
    return tuple(_target_verify_linear(linear, x) for linear in linears)


def _target_verify_embedding_as_linear(embedding, x: mx.array):
    if not (x.ndim == 3 and (x.shape[0] > 1 or x.shape[1] > 1)):
        return embedding.as_linear(x)

    out = _target_verify_weight(embedding.weight, x)
    if out is not None:
        return out

    return _target_verify_timewise(embedding.as_linear, x)


class Qwen3_5BatchInvariantForward:
    """Run Qwen3.5 rows with singleton-equivalent reductions."""

    @staticmethod
    def _helpers():
        # Imported lazily because language.py owns the shared cache and ragged
        # attention utilities and imports this verifier at module load time.
        from . import language

        return language

    def _linear(self, linear, x: mx.array) -> mx.array:
        return _target_verify_linear(linear, x)

    def _linears(self, linears, x: mx.array):
        return _target_verify_linears(linears, x)

    def _embedding_as_linear(self, embedding, x: mx.array) -> mx.array:
        return _target_verify_embedding_as_linear(embedding, x)

    def quantized_linear(self, linear, x: mx.array) -> Optional[mx.array]:
        return _target_verify_quantized_linear(linear, x)

    def quantized_argmax(
        self,
        linear,
        x: mx.array,
        token_mask: Optional[mx.array] = None,
    ) -> Optional[mx.array]:
        return _target_verify_quantized_argmax(linear, x, token_mask=token_mask)

    def can_quantized_head(self, linear) -> bool:
        return _can_target_verify_quantized_head(linear)

    def pad_token_mask(self, token_mask: mx.array, n_size: int) -> mx.array:
        return _pad_token_mask_to_head(token_mask, n_size)

    def _attention(
        self,
        attention,
        x,
        mask,
        cache,
        position_ids,
        position_embeddings,
    ):
        batch, length, _ = x.shape
        q_proj_output, keys, values = self._linears(
            (attention.q_proj, attention.k_proj, attention.v_proj), x
        )
        queries, keys, values, gate, mask = attention._prepare_projected_qkv(
            q_proj_output,
            keys,
            values,
            cache,
            position_ids,
            position_embeddings,
            mask,
        )

        left_padded_decode = (
            mask == "left_padded_decode" if isinstance(mask, str) else False
        )
        if left_padded_decode:
            mask = None

        output = None
        if length > 1 or left_padded_decode:
            output = self._helpers()._qwen3_5_left_padded_attention(
                queries,
                keys,
                values,
                cache=cache,
                scale=attention.scale,
                mask=mask,
            )

        if output is None and length == 2:
            if isinstance(mask, str) and mask == "causal":
                key_length = kv_sequence_length(keys)
                prefix_length = key_length - length
                mask = (
                    mx.arange(key_length)[None, None, None, :]
                    < (prefix_length + mx.arange(length) + 1)[None, None, :, None]
                )
            output = scaled_dot_product_attention(
                queries,
                keys,
                values,
                cache=cache,
                scale=attention.scale,
                mask=mask,
            )
        elif output is None and length > 1:
            prefix_length = kv_sequence_length(keys) - length
            output = mx.concatenate(
                [
                    scaled_dot_product_attention(
                        queries[:, :, index : index + 1, :],
                        slice_kv_sequence(keys, prefix_length + index + 1),
                        slice_kv_sequence(values, prefix_length + index + 1),
                        cache=cache,
                        scale=attention.scale,
                        mask=(
                            mask[
                                ...,
                                index : index + 1,
                                : prefix_length + index + 1,
                            ]
                            if isinstance(mask, mx.array) and mask.ndim >= 4
                            else None
                        ),
                    )
                    for index in range(length)
                ],
                axis=2,
            )
        elif output is None:
            output = scaled_dot_product_attention(
                queries,
                keys,
                values,
                cache=cache,
                scale=attention.scale,
                mask=mask,
            )

        output = output.transpose(0, 2, 1, 3).reshape(batch, length, -1)
        return self._linear(attention.o_proj, output * mx.sigmoid(gate))

    def _switch_glu(self, switch_mlp, x, indices):
        if x.ndim != 3 or not all(
            hasattr(switch_mlp, name)
            for name in ("up_proj", "gate_proj", "down_proj", "activation")
        ):
            return switch_mlp(x, indices)

        batch, length, width = x.shape
        top_k = indices.shape[-1]
        flat_x = x.reshape(batch * length, width)
        flat_indices = indices.reshape(batch * length, top_k)
        flat_x = mx.expand_dims(flat_x, (-2, -3))

        up = switch_mlp.up_proj(flat_x, flat_indices, sorted_indices=False)
        gate = switch_mlp.gate_proj(flat_x, flat_indices, sorted_indices=False)
        output = switch_mlp.down_proj(
            switch_mlp.activation(up, gate),
            flat_indices,
            sorted_indices=False,
        )
        return output.squeeze(-2).reshape(batch, length, top_k, -1)

    def _feed_forward(self, feed_forward, x):
        if hasattr(feed_forward, "switch_mlp"):
            gates = mx.softmax(
                self._linear(feed_forward.gate, x),
                axis=-1,
                precise=True,
            )
            top_k = feed_forward.top_k
            indices = mx.argpartition(gates, kth=-top_k, axis=-1)[..., -top_k:]
            scores = mx.take_along_axis(gates, indices, axis=-1)
            scores = scores / scores.sum(axis=-1, keepdims=True)

            shared_output = self._feed_forward(feed_forward.shared_expert, x)
            shared_output = (
                mx.sigmoid(self._linear(feed_forward.shared_expert_gate, x))
                * shared_output
            )
            output = self._switch_glu(feed_forward.switch_mlp, x, indices)
            output = (output * scores[..., None]).sum(axis=-2)
            return output + shared_output

        if all(
            hasattr(feed_forward, name)
            for name in ("gate_proj", "up_proj", "down_proj")
        ):
            gate, up = self._linears((feed_forward.gate_proj, feed_forward.up_proj), x)
            return self._linear(feed_forward.down_proj, swiglu(gate, up))

        return feed_forward(x)

    @staticmethod
    def _normalize_gated_delta_qk(layer, q, k):
        del layer
        inv_scale = k.shape[-1] ** -0.5
        q = (inv_scale**2) * mx.fast.rms_norm(q, None, 1e-6)
        k = inv_scale * mx.fast.rms_norm(k, None, 1e-6)
        return q, k

    def _gated_delta(self, layer, inputs, mask, cache, gdn_sink):
        helpers = self._helpers()
        batch, length, _ = inputs.shape
        mixed_qkv, z, b, a = self._linears(
            (layer.in_proj_qkv, layer.in_proj_z, layer.in_proj_b, layer.in_proj_a),
            inputs,
        )
        z = z.reshape(batch, length, -1, layer.head_v_dim)

        if cache is not None and cache[0] is not None:
            conv_state = cache[0]
            if conv_state.shape[0] != batch:
                conv_state = mx.zeros(
                    (batch, layer.conv_kernel_size - 1, layer.conv_dim),
                    dtype=inputs.dtype,
                )
        else:
            conv_state = mx.zeros(
                (batch, layer.conv_kernel_size - 1, layer.conv_dim),
                dtype=inputs.dtype,
            )

        if mask is not None:
            if mask.shape[0] != batch:
                mask = None
            else:
                mixed_qkv = mx.where(mask[..., None], mixed_qkv, 0)
        conv_input = mx.concatenate([conv_state, mixed_qkv], axis=1)
        if cache is not None:
            n_keep = layer.conv_kernel_size - 1
            if getattr(cache, "lengths", None) is not None:
                ends = mx.clip(cache.lengths, 0, length)
                positions = (ends[:, None] + mx.arange(n_keep))[..., None]
                cache[0] = mx.take_along_axis(conv_input, positions, axis=1)
            else:
                cache[0] = mx.contiguous(conv_input[:, -n_keep:, :])

        conv_output = nn.silu(layer.conv1d(conv_input))
        q, k, v = [
            value.reshape(batch, length, heads, width)
            for value, heads, width in zip(
                mx.split(conv_output, [layer.key_dim, 2 * layer.key_dim], -1),
                [layer.num_k_heads, layer.num_k_heads, layer.num_v_heads],
                [layer.head_k_dim, layer.head_k_dim, layer.head_v_dim],
            )
        ]

        state = cache[1] if cache else None
        if state is not None and state.shape[0] != batch:
            state = None
        q, k = self._normalize_gated_delta_qk(layer, q, k)

        initial_state = state
        output, state, intermediate_states = gated_delta_update_with_states(
            q,
            k,
            v,
            a,
            b,
            layer.A_log,
            layer.dt_bias,
            state,
            mask,
            use_kernel=not layer.training,
            state_steps=length - 1,
        )
        gdn_sink.append(
            (
                q,
                k,
                v,
                a,
                b,
                layer.A_log,
                layer.dt_bias,
                initial_state,
                mask,
                conv_input,
                layer.conv_kernel_size,
                intermediate_states,
            )
        )

        if cache is not None:
            cache[1] = state
            if hasattr(cache, "advance"):
                cache.advance(length)
                helpers._qwen3_5_advance_left_padding_info(cache, length)
                helpers._qwen3_5_advance_lengths_info(cache, length)

        output = layer.norm(output, z)
        return self._linear(layer.out_proj, output.reshape(batch, length, -1))

    def _layer(
        self,
        layer,
        hidden,
        mask,
        cache,
        position_ids,
        position_embeddings,
        gdn_sink,
    ):
        normed = layer.input_layernorm(hidden)
        if layer.is_linear:
            residual = self._gated_delta(
                layer.linear_attn,
                normed,
                mask,
                cache,
                gdn_sink,
            )
        else:
            residual = self._attention(
                layer.self_attn,
                normed,
                mask,
                cache,
                position_ids,
                position_embeddings,
            )
        hidden = hidden + residual
        return hidden + self._feed_forward(
            layer.mlp,
            layer.post_attention_layernorm(hidden),
        )

    def _model(
        self,
        model,
        inputs,
        cache,
        inputs_embeds,
        position_ids,
        capture_layer_ids,
        hidden_sink,
        gdn_sink,
    ):
        helpers = self._helpers()
        hidden = model.embed_tokens(inputs) if inputs_embeds is None else inputs_embeds
        if cache is None:
            cache = [None] * len(model.layers)

        fa_mask = helpers._create_qwen3_5_attention_mask(hidden, cache[model.fa_idx])
        ssm_mask = helpers._create_qwen3_5_ssm_mask(hidden, cache[model.ssm_idx])
        decode_left_padding = (
            getattr(cache[model.fa_idx], "_qwen3_5_decode_left_padding", None)
            if isinstance(fa_mask, str) and fa_mask == "left_padded_decode"
            else None
        )
        helpers._set_qwen3_5_decode_left_padding(
            cache, model.layers, decode_left_padding
        )

        position_embeddings = None
        if position_ids is not None:
            for layer in model.layers:
                if not layer.is_linear:
                    if not layer.self_attn.rotary_emb.fused_apply:
                        position_embeddings = layer.self_attn.rotary_emb(
                            hidden, position_ids
                        )
                    break

        capture_set = set(capture_layer_ids) if capture_layer_ids else set()
        for index, (layer, layer_cache) in enumerate(zip(model.layers, cache)):
            layer_mask = ssm_mask if layer.is_linear else fa_mask
            hidden = self._layer(
                layer,
                hidden,
                layer_mask,
                layer_cache,
                position_ids,
                position_embeddings,
                gdn_sink,
            )
            if hidden_sink is not None and index in capture_set:
                hidden_sink.append(hidden)

        return model.norm(hidden)

    def __call__(
        self,
        language_model,
        inputs,
        *,
        cache: Any = None,
        inputs_embeds: Optional[mx.array] = None,
        position_ids: Optional[mx.array] = None,
        capture_layer_ids: Optional[list[int]] = None,
        return_hidden: bool = False,
        return_shared_kv: bool = False,
        skip_logits: bool = False,
    ) -> LanguageModelOutput:
        hidden_sink: list[mx.array] | None = (
            [] if capture_layer_ids is not None else None
        )
        gdn_sink: list = []
        hidden = self._model(
            language_model.model,
            inputs,
            cache,
            inputs_embeds,
            position_ids,
            capture_layer_ids,
            hidden_sink,
            gdn_sink,
        )
        if return_hidden:
            if hidden_sink is None:
                hidden_sink = []
            hidden_sink.append(hidden)

        if skip_logits:
            logits = None
        elif language_model.args.tie_word_embeddings:
            logits = self._embedding_as_linear(
                language_model.model.embed_tokens, hidden
            )
        else:
            logits = self._linear(language_model.lm_head, hidden)

        return LanguageModelOutput(
            logits=logits,
            hidden_states=hidden_sink,
            gdn_states=gdn_sink,
            shared_kv_states={} if return_shared_kv else None,
        )


Qwen3_5ExactSpeculativeVerifier = Qwen3_5BatchInvariantForward


__all__ = ["Qwen3_5BatchInvariantForward", "Qwen3_5ExactSpeculativeVerifier"]
