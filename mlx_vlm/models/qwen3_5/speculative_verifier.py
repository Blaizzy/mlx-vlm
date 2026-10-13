from functools import lru_cache
from typing import Any, Optional

import mlx.core as mx
import mlx.nn as nn

from ...speculative.cache_state import start_speculative_cache
from ...speculative.ops.linear import (
    _can_target_verify_quantized_head,
    _target_verify_embedding_as_linear,
    _target_verify_linear,
    _target_verify_linears,
    _target_verify_quantized_argmax,
    _target_verify_quantized_linear,
)
from ..activations import swiglu
from ..base import (
    LanguageModelOutput,
    kv_sequence_length,
    scaled_dot_product_attention,
    slice_kv_sequence,
)
from ..quantized_verifier import _masked_argmax
from ..quantized_verifier import pad_token_mask as _pad_token_mask_to_head
from ..simd_mma import simd_mmas, supports_simd_linear, supports_simd_mma
from ..small_row_attention import small_row_attention
from . import gated_delta as gd
from .gated_delta import gated_delta_update


class Qwen3_5BatchInvariantForward:
    """Shared forward and cache handling for singleton-equivalent reductions."""

    pipeline_layers = 0
    uses_simd = False

    def verify(self, language_model, inputs, *, cache=None, **kwargs):
        """Explicit speculative entry; ordinary invariant decode calls __call__."""
        transaction = start_speculative_cache(cache or [], inputs.shape[1])
        try:
            output = self(language_model, inputs, cache=cache, **kwargs)
            output.gdn_states = transaction
            return output
        except BaseException:
            transaction.abort()
            raise

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

    def _attention_kernel(self, attention, queries, keys, values, mask, cache):
        return None

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

        output = self._attention_kernel(attention, queries, keys, values, mask, cache)
        if output is None and (length > 1 or left_padded_decode):
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
            indices = mx.stop_gradient(indices)
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

    @staticmethod
    def _convolve(layer, inputs):
        return nn.silu(layer.conv1d(inputs))

    _gated_delta_update = staticmethod(gated_delta_update)

    def _gated_delta(self, layer, inputs, mask, cache):
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
            cache.update_window(
                0, conv_input, layer.conv_kernel_size - 1, lengths=cache.lengths
            )

        conv_output = self._convolve(layer, conv_input)
        q, k, v = [
            value.reshape(batch, length, heads, width)
            for value, heads, width in zip(
                mx.split(conv_output, [layer.key_dim, 2 * layer.key_dim], -1),
                [layer.num_k_heads, layer.num_k_heads, layer.num_v_heads],
                [layer.head_k_dim, layer.head_k_dim, layer.head_v_dim],
            )
        ]

        q, k = self._normalize_gated_delta_qk(layer, q, k)

        output, _ = self._gated_delta_update(
            q,
            k,
            v,
            a,
            b,
            layer.A_log,
            layer.dt_bias,
            mask=mask,
            use_kernel=not layer.training,
            cache=cache,
        )

        if cache is not None:
            if hasattr(cache, "advance"):
                cache.advance(length)
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
    ):
        normed = layer.input_layernorm(hidden)
        if layer.is_linear:
            residual = self._gated_delta(
                layer.linear_attn,
                normed,
                mask,
                cache,
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
    ):
        helpers = self._helpers()
        pipeline = self.pipeline_layers
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
            )
            if hidden_sink is not None and index in capture_set:
                hidden_sink.append(hidden)
            if (
                pipeline > 0
                and index + 1 < len(model.layers)
                and (index == 0 or (index + 1) % pipeline == 0)
            ):
                mx.async_eval(hidden)

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
        hidden = self._model(
            language_model.model,
            inputs,
            cache,
            inputs_embeds,
            position_ids,
            capture_layer_ids,
            hidden_sink,
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
            shared_kv_states={} if return_shared_kv else None,
        )


@lru_cache(maxsize=1)
def _convolution_kernel():
    return mx.fast.metal_kernel(
        name="verify_depthwise_silu",
        input_names=["x", "w"],
        output_names=["y"],
        source=r"""
            uint i = thread_position_in_grid.x;
            if (i >= B * M * C) return;
            uint c = i % C, t = (i / C) % M, b = i / (M * C);
            float sum = 0.0f;
            for (uint j = 0; j < W; ++j) {
                sum += float(x[(b * (M + W - 1) + t + j) * C + c]) *
                       float(w[c * W + j]);
            }
            T value = T(sum);
            T gate = T(1.0f / (1.0f + exp(-float(value))));
            y[i] = T(float(value) * float(gate));
        """,
    )


@lru_cache(maxsize=2)
def _replay_kernel(masked):
    source = r"""
        uint n = thread_position_in_grid.z, batch = n / HV, hv = n % HV;
        uint hk = hv / (HV / HK);
        uint lane = thread_index_in_simdgroup, dv = thread_position_in_grid.y;
        float state_values[DK / 32];
        for (uint i = 0; i < DK / 32; ++i) {
            state_values[i] = state[(n * DV + dv) * DK + lane * (DK / 32) + i];
        }
        for (uint t = 0; t < uint(lengths[batch]); ++t) {
            if (__MASK__) {
                float dot = 0.0f;
                for (uint i = 0; i < DK / 32; ++i) {
                    uint d = lane * (DK / 32) + i;
                    state_values[i] *= g[(batch * NT + t) * HV + hv];
                    dot += state_values[i] * float(k[((batch * NT + t) * HK + hk) * DK + d]);
                }
                dot = simd_sum(dot);
                float delta = (float(v[((batch * NT + t) * HV + hv) * DV + dv]) - dot) *
                              float(beta[(batch * NT + t) * HV + hv]);
                for (uint i = 0; i < DK / 32; ++i) {
                    uint d = lane * (DK / 32) + i;
                    state_values[i] += float(k[((batch * NT + t) * HK + hk) * DK + d]) * delta;
                }
            }
        }
        for (uint i = 0; i < DK / 32; ++i) {
            out[(n * DV + dv) * DK + lane * (DK / 32) + i] = state_values[i];
        }
    """.replace("__MASK__", "mask[batch * NT + t]" if masked else "true")
    return mx.fast.metal_kernel(
        name=f"verify_replay_state_{int(masked)}",
        input_names=["k", "v", "g", "beta", "state", "lengths"]
        + (["mask"] if masked else []),
        output_names=["out"],
        source=source,
    )


@lru_cache(maxsize=64)
def _acceptance_array(lengths):
    return mx.array(lengths, mx.int32)


def replay_recurrent_state(k, v, g, beta, state, lengths, mask=None):
    """Rebuild accepted prefixes without query dots or intermediate states."""
    batch, tokens, key_heads, key_width = k.shape
    value_heads, value_width = v.shape[-2:]
    inputs = [k, v, g, beta, state, _acceptance_array(tuple(lengths))]
    if mask is not None:
        inputs.append(mask)
    return _replay_kernel(mask is not None)(
        inputs=inputs,
        template=[
            ("NT", tokens),
            ("HK", key_heads),
            ("HV", value_heads),
            ("DK", key_width),
            ("DV", value_width),
        ],
        grid=(32, value_width, batch * value_heads),
        threadgroup=(32, 4, 1),
        output_shapes=[state.shape],
        output_dtypes=[mx.float32],
    )[0]


class Qwen3_5SpeculativeVerifier(Qwen3_5BatchInvariantForward):
    """Use matrix reductions while retaining regular fallbacks and cache ownership."""

    pipeline_layers = 4
    uses_simd = True

    @staticmethod
    def supports_model(language_model):
        from .language import Qwen3_5Model

        model = language_model.model
        if type(model) is not Qwen3_5Model:
            return False
        projection = getattr(model.layers[0].mlp, "gate_proj", None)
        return (
            not language_model.training
            and getattr(model.embed_tokens, "scales", model.embed_tokens.weight).dtype
            == mx.bfloat16
            and supports_simd_linear(projection)
            and supports_simd_mma()
        )

    def _linear(self, linear, x):
        output = simd_mmas((linear,), x, chunked=True)
        return super()._linear(linear, x) if output is None else output[0]

    def _linears(self, linears, x):
        output = simd_mmas(linears, x, chunked=True)
        return super()._linears(linears, x) if output is None else output

    def quantized_linear(self, linear, x):
        output = simd_mmas((linear,), x, chunked=True)
        return super().quantized_linear(linear, x) if output is None else output[0]

    def quantized_argmax(self, linear, x, token_mask=None):
        output = simd_mmas((linear,), x, chunked=True)
        if output is None:
            return super().quantized_argmax(linear, x, token_mask)
        logits = output[0]
        return (
            mx.argmax(logits, axis=-1)
            if token_mask is None
            else _masked_argmax(logits, token_mask)
        )

    def _attention_kernel(self, attention, queries, keys, values, mask, cache):
        padding = getattr(cache, "_qwen3_5_decode_left_padding", None)
        padding_info = self._helpers()._qwen3_5_left_padding_info(cache)
        if (
            hasattr(cache, "bits")
            or padding is not None
            and max(padding, default=0) > 0
            or padding_info is not None
            and padding_info[1] > 0
            or isinstance(mask, str)
            and mask != "causal"
        ):
            return None
        return small_row_attention(
            queries, keys, values, scale=attention.scale, mask=mask, exact=True
        )

    @staticmethod
    def _convolve(layer, inputs):
        conv = layer.conv1d
        if (
            layer.training
            or not mx.metal.is_available()
            or mx.default_device() != mx.gpu
            or inputs.dtype not in (mx.bfloat16, mx.float16)
            or conv.weight.dtype != inputs.dtype
            or "bias" in conv
            or conv.groups != inputs.shape[-1]
            or conv.stride != 1
            or conv.padding != 0
            or conv.dilation != 1
        ):
            return Qwen3_5BatchInvariantForward._convolve(layer, inputs)
        batch, total, channels = inputs.shape
        width = conv.weight.shape[1]
        length = total - width + 1
        return _convolution_kernel()(
            inputs=[inputs, conv.weight],
            template=[
                ("T", inputs.dtype),
                ("B", batch),
                ("M", length),
                ("C", channels),
                ("W", width),
            ],
            grid=(batch * length * channels, 1, 1),
            threadgroup=(256, 1, 1),
            output_shapes=[(batch, length, channels)],
            output_dtypes=[inputs.dtype],
        )[0]

    @staticmethod
    def _gated_delta_update(
        q, k, v, a, b, A_log, dt_bias, *, mask=None, use_kernel=True, cache=None
    ):
        length = q.shape[1]
        initial = cache[1] if cache is not None else None
        if (
            not use_kernel
            or not mx.metal.is_available()
            or mx.default_device() != mx.gpu
            or cache is None
            or cache.history_capacity != length
            or cache._recorded_length(1)
            or k.shape[-1] % 32
            or initial is not None
            and initial.dtype != mx.float32
        ):
            return gd.gated_delta_update(
                q,
                k,
                v,
                a,
                b,
                A_log,
                dt_bias,
                mask=mask,
                use_kernel=use_kernel,
                cache=cache,
            )
        if initial is None or initial.shape[0] != q.shape[0]:
            initial = mx.zeros(
                (q.shape[0], v.shape[-2], v.shape[-1], k.shape[-1]), mx.float32
            )
        g, beta = gd._compute_g_beta(A_log, a, b, dt_bias)
        output, state = gd.gated_delta_kernel(q, k, v, g, beta, initial, mask)

        def replay(lengths):
            return replay_recurrent_state(k, v, g, beta, initial, lengths, mask)

        cache.record_speculative_replay(1, length, state, replay)
        cache[1] = state
        return output, state


# Other model families import this name for the shared forward implementation.
Qwen3_5ExactSpeculativeVerifier = Qwen3_5BatchInvariantForward

__all__ = [
    "Qwen3_5BatchInvariantForward",
    "Qwen3_5SpeculativeVerifier",
    "Qwen3_5ExactSpeculativeVerifier",
]
