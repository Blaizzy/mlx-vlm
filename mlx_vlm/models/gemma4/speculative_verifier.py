from typing import Any, List, Optional

import mlx.core as mx

from ..base import LanguageModelOutput, scaled_dot_product_attention
from ..qwen3_5.speculative_verifier import Qwen3_5ExactSpeculativeVerifier


class Gemma4ExactSpeculativeVerifier(Qwen3_5ExactSpeculativeVerifier):
    """Run Gemma 4 block verification with singleton-equivalent numerics.

    Verifying a block as one ``(1, K, D)`` matmul takes a different kernel path
    than the ``K`` singleton steps base decoding performs, and the last-bit
    difference is enough to move an argmax. The dense and quantized linear
    primitives are inherited; only the Gemma 4 forward is reimplemented here.

    Sliding layers are bounded by their RotatingKVCache, so the resident
    prefix already carries at most ``sliding_window`` keys.
    """

    def _attention(self, attention, x, mask, cache, shared_kv=None, offset=None):
        batch, length, _ = x.shape
        n_heads, n_kv_heads = attention.n_heads, attention.n_kv_heads
        head_dim = attention.head_dim

        queries = self._linear(attention.q_proj, x).reshape(
            batch, length, n_heads, head_dim
        )
        queries = attention.q_norm(queries)

        if shared_kv is not None:
            keys, values = shared_kv
        else:
            # k_eq_v layers publish no v_proj: keys and values share the
            # projection, and values take it before k_norm.
            keys = self._linear(attention.k_proj, x).reshape(
                batch, length, n_kv_heads, head_dim
            )
            if attention.use_k_eq_v:
                values = keys
            else:
                values = self._linear(attention.v_proj, x).reshape(
                    batch, length, n_kv_heads, head_dim
                )

            offset = mx.array(cache.offset) if cache is not None else 0

            keys = attention.k_norm(keys).transpose(0, 2, 1, 3)
            keys = attention.rope(keys, offset=offset)
            values = attention.v_norm(values).transpose(0, 2, 1, 3)

            if cache is not None:
                keys, values = cache.update_and_fetch(keys, values)

        queries = queries.transpose(0, 2, 1, 3)
        queries = attention.rope(queries, offset=offset)

        if length > 1:
            # Sliding layers are already bounded by RotatingKVCache, so the
            # resident prefix needs no extra windowing here.
            prefix_length = keys.shape[-2] - length
            output = mx.concatenate(
                [
                    scaled_dot_product_attention(
                        queries[:, :, index : index + 1, :],
                        keys[..., : prefix_length + index + 1, :],
                        values[..., : prefix_length + index + 1, :],
                        cache=cache,
                        scale=attention.scale,
                        mask=(
                            mask[..., index : index + 1, : prefix_length + index + 1]
                            if isinstance(mask, mx.array) and mask.ndim >= 4
                            else None
                        ),
                    )
                    for index in range(length)
                ],
                axis=2,
            )
        else:
            output = scaled_dot_product_attention(
                queries, keys, values, cache=cache, scale=attention.scale, mask=mask
            )

        output = output.transpose(0, 2, 1, 3).reshape(batch, length, -1)
        return self._linear(attention.o_proj, output), (keys, values), offset

    def _feed_forward(self, mlp, x):
        from .language import geglu

        gate = self._linear(mlp.gate_proj, x)
        up = self._linear(mlp.up_proj, x)
        return self._linear(mlp.down_proj, geglu(gate, up))

    def _layer(self, layer, x, mask, cache, per_layer_input, shared_kv, offset):
        residual = x
        hidden, kvs, offset = self._attention(
            layer.self_attn, layer.input_layernorm(x), mask, cache, shared_kv, offset
        )
        hidden = residual + layer.post_attention_layernorm(hidden)

        residual = hidden
        if getattr(layer, "enable_moe", False):
            inner = layer.mlp(layer.pre_feedforward_layernorm(hidden))
        else:
            inner = self._feed_forward(
                mlp=layer.mlp, x=layer.pre_feedforward_layernorm(hidden)
            )
        hidden = residual + layer.post_feedforward_layernorm(inner)

        if (
            getattr(layer, "per_layer_input_gate", None) is not None
            and getattr(layer, "per_layer_projection", None) is not None
            and getattr(layer, "post_per_layer_input_norm", None) is not None
            and per_layer_input is not None
        ):
            hidden = layer(
                hidden, mask, cache, per_layer_input=per_layer_input, offset=offset
            )[0]
        elif layer.layer_scalar is not None:
            hidden = hidden * layer.layer_scalar

        return hidden, kvs, offset

    def _model(self, model, inputs, cache, input_embeddings, capture_layer_ids, sink):
        hidden = (
            input_embeddings
            if input_embeddings is not None
            else model.embed_tokens(inputs) * model.embed_scale
        )
        if cache is None:
            cache = [None] * len(model.layers)

        masks = model._make_masks(hidden, cache, None)
        capture_set = set(capture_layer_ids) if capture_layer_ids else set()
        intermediates: List[Any] = [(None, None)] * len(model.layers)

        for index, (layer, layer_cache, mask, prev_index) in enumerate(
            zip(model.layers, cache, masks, model.previous_kvs)
        ):
            kvs, offset = intermediates[prev_index]
            hidden, kvs, offset = self._layer(
                layer, hidden, mask, layer_cache, None, kvs, offset
            )
            intermediates[index] = (kvs, offset)
            if sink is not None and index in capture_set:
                sink.append(hidden)

        if sink is not None and not capture_set:
            sink.append(hidden)

        return model.norm(hidden)

    def __call__(
        self,
        language_model,
        inputs,
        *,
        cache: Any = None,
        input_embeddings: Optional[mx.array] = None,
        capture_layer_ids: Optional[List[int]] = None,
    ) -> LanguageModelOutput:
        sink: Optional[List[mx.array]] = [] if capture_layer_ids is not None else None
        hidden = self._model(
            language_model.model,
            inputs,
            cache,
            input_embeddings,
            capture_layer_ids,
            sink,
        )

        head = getattr(language_model, "lm_head", None)
        if head is None:
            logits = self._embedding_as_linear(
                language_model.model.embed_tokens, hidden
            )
        else:
            logits = self._linear(head, hidden)

        softcap = getattr(language_model, "final_logit_softcapping", None)
        if softcap is not None:
            from .language import logit_softcap

            logits = logit_softcap(softcap, logits)

        return LanguageModelOutput(logits=logits, hidden_states=sink)


__all__ = ["Gemma4ExactSpeculativeVerifier"]
