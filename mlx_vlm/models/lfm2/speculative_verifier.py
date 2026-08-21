from typing import Any

import mlx.core as mx

from ..activations import swiglu
from ..base import (
    LanguageModelOutput,
    create_attention_mask,
    create_ssm_mask,
    scaled_dot_product_attention,
)
from ..exact_speculative_verify import (
    exact_speculative_verify_switch_weight,
    exact_speculative_verify_weight,
)


class Lfm2ExactSpeculativeVerifier:
    """Run block verification with singleton-equivalent LFM2 numerics."""

    @staticmethod
    def _linear(linear, x: mx.array) -> mx.array:
        output = exact_speculative_verify_weight(linear.weight, x)
        if output is None:
            return linear(x)
        if hasattr(linear, "bias"):
            output = output + linear.bias
        return output

    def _attention(self, attention, x, mask, cache):
        length = x.shape[1]
        queries = self._linear(attention.q_proj, x)
        keys = self._linear(attention.k_proj, x)
        values = self._linear(attention.v_proj, x)
        queries, keys, values = attention._prepare_projected_qkv(
            queries, keys, values, cache
        )

        if cache is not None and length > 1:
            prefix_length = keys.shape[-2] - length
            output = mx.concatenate(
                [
                    scaled_dot_product_attention(
                        queries[:, :, index : index + 1, :],
                        keys[:, :, : prefix_length + index + 1, :],
                        values[:, :, : prefix_length + index + 1, :],
                        cache=cache,
                        mask=(
                            mask[
                                ...,
                                index : index + 1,
                                : prefix_length + index + 1,
                            ]
                            if isinstance(mask, mx.array) and mask.ndim >= 4
                            else None
                        ),
                        scale=attention.scale,
                    )
                    for index in range(length)
                ],
                axis=2,
            )
        else:
            output = scaled_dot_product_attention(
                queries,
                keys,
                values,
                cache=cache,
                mask=mask,
                scale=attention.scale,
            )

        output = output.transpose(0, 2, 1, 3).reshape(x.shape[0], length, -1)
        return self._linear(attention.out_proj, output)

    def _short_conv(self, conv, x, mask, cache, gdn_sink):
        projected = self._linear(conv.in_proj, x)
        output = conv._convolve_projected(projected, mask, cache, gdn_sink)
        return self._linear(conv.out_proj, output)

    def _feed_forward(self, feed_forward, x):
        from .language import MLP, GatedMLP, Lfm2MoeSparseMoeBlock

        if isinstance(feed_forward, MLP):
            gate = self._linear(feed_forward.w1, x)
            value = self._linear(feed_forward.w3, x)
            return self._linear(feed_forward.w2, swiglu(gate, value))
        if isinstance(feed_forward, GatedMLP):
            gate = self._linear(feed_forward.gate_proj, x)
            value = self._linear(feed_forward.up_proj, x)
            return self._linear(feed_forward.down_proj, swiglu(gate, value))
        if isinstance(feed_forward, Lfm2MoeSparseMoeBlock):
            gates = mx.softmax(
                self._linear(feed_forward.gate, x).astype(mx.float32),
                axis=-1,
            )
            if feed_forward.use_expert_bias:
                gates += feed_forward.expert_bias
            indices = mx.argpartition(gates, kth=-feed_forward.top_k, axis=-1)[
                ..., -feed_forward.top_k :
            ]
            scores = mx.take_along_axis(gates, indices, axis=-1)
            if feed_forward.norm_topk_prob:
                scores /= mx.sum(scores, axis=-1, keepdims=True) + 1e-20
            switch_mlp = feed_forward.switch_mlp
            expert_up = self._switch_linear(switch_mlp.up_proj, x, indices)
            expert_gate = self._switch_linear(switch_mlp.gate_proj, x, indices)
            expert_output = self._switch_linear(
                switch_mlp.down_proj,
                swiglu(expert_gate, expert_up),
                indices,
            )
            return (expert_output * scores.astype(x.dtype)[..., None]).sum(axis=-2)
        return feed_forward(x)

    @staticmethod
    def _switch_linear(linear, x, indices):
        output = exact_speculative_verify_switch_weight(linear.weight, x, indices)
        if output is not None:
            if hasattr(linear, "bias"):
                output = output + linear.bias[indices]
            return output

        # Preserve exactness for unsupported (for example quantized) switch
        # layers by retaining the singleton execution shape.
        rows = []
        input_has_expert_axis = x.shape[:-1] == indices.shape
        for position in range(x.shape[1]):
            row = x[:, position : position + 1]
            row_indices = indices[:, position : position + 1]
            row = mx.expand_dims(row, -2 if input_has_expert_axis else (-2, -3))
            rows.append(linear(row, row_indices).squeeze(-2))
        return mx.concatenate(rows, axis=1)

    def _layer(self, layer, x, mask, cache, gdn_sink):
        normed = layer.operator_norm(x)
        if layer.is_attention_layer:
            residual = self._attention(layer.self_attn, normed, mask, cache)
        else:
            residual = self._short_conv(
                layer.conv,
                normed,
                mask,
                cache,
                gdn_sink,
            )
        hidden = x + residual
        return hidden + self._feed_forward(
            layer.feed_forward,
            layer.ffn_norm(hidden),
        )

    def _model(
        self,
        model,
        inputs,
        cache,
        input_embeddings,
        capture_layer_ids,
        hidden_sink,
        gdn_sink,
    ):
        hidden = (
            input_embeddings
            if input_embeddings is not None
            else model.embed_tokens(inputs)
        )
        if cache is None:
            cache = [None] * len(model.layers)

        attention_mask = create_attention_mask(hidden, cache[model.fa_idx])
        conv_mask = create_ssm_mask(hidden, cache[model.conv_idx])
        capture_set = set(capture_layer_ids) if capture_layer_ids else set()
        for index, (layer, layer_cache) in enumerate(zip(model.layers, cache)):
            mask = attention_mask if layer.is_attention_layer else conv_mask
            hidden = self._layer(
                layer,
                hidden,
                mask,
                layer_cache,
                gdn_sink,
            )
            if hidden_sink is not None and index in capture_set:
                hidden_sink.append(hidden)

        return model.embedding_norm(hidden)

    def __call__(
        self,
        language_model,
        inputs,
        *,
        cache: Any = None,
        input_embeddings: mx.array | None = None,
        capture_layer_ids: list[int] | None = None,
    ) -> LanguageModelOutput:
        hidden_sink: list[mx.array] | None = (
            [] if capture_layer_ids is not None else None
        )
        gdn_sink: list = []
        hidden = self._model(
            language_model.model,
            inputs,
            cache,
            input_embeddings,
            capture_layer_ids,
            hidden_sink,
            gdn_sink,
        )
        if language_model.args.tie_word_embeddings:
            logits = exact_speculative_verify_weight(
                language_model.model.embed_tokens.weight,
                hidden,
            )
            if logits is None:
                logits = language_model.model.embed_tokens.as_linear(hidden)
        else:
            logits = self._linear(language_model.lm_head, hidden)

        return LanguageModelOutput(
            logits=logits,
            hidden_states=hidden_sink,
            gdn_states=gdn_sink,
        )


__all__ = ["Lfm2ExactSpeculativeVerifier"]
