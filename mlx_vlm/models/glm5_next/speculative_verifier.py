from typing import Any, Callable, List, Optional

import mlx.core as mx

from ...speculative.cache_state import (
    rollback_speculative_cache,
    start_speculative_cache,
)
from ..base import LanguageModelOutput, create_ssm_mask, scaled_dot_product_attention
from ..deepseek_v4.hyper_connection import _hc_kernel, hc_expand
from ..exact_speculative_verify import exact_speculative_verify_weight
from ..quantized_verifier import (
    DEFAULT_QUANTIZED_VERIFIER,
    exact_quantized_linear,
    exact_quantized_selected_linear,
    exact_quantized_switch_linear,
)
from ..qwen3_5.speculative_verifier import _target_verify_linear
from ..switch_layers import _gather_sort, _scatter_unsort
from .exact_ops import (
    clamped_swiglu,
    combine_moe_outputs,
    exact_affine_moe_down,
    exact_affine_switch_gate_up,
    exact_dense_block_linear,
    exact_fp32_decode_block_gemv,
    exact_hc_expand,
    exact_hc_norm,
    exact_hc_normalized_norm,
    scaled_rms_norm,
)


class Glm5NextSpeculativeVerifier:
    """Run GLM-5-Next MTP verification and own its rollback state."""

    @staticmethod
    def _helpers():
        # language.py owns the shared sparse gather helper and imports this
        # verifier at module load time, so resolve it lazily.
        from . import language

        return language

    def logits_from_hidden(self, language_model, hidden: mx.array) -> mx.array:
        tied = language_model.args.tie_word_embeddings
        head = (
            language_model.model.embed_tokens.as_linear
            if tied
            else language_model.lm_head
        )
        if not tied:
            logits = exact_dense_block_linear(head, hidden)
            if logits is not None:
                return logits
        if not tied and hidden.ndim == 3 and hidden.shape[1] > 1:
            logits = exact_speculative_verify_weight(
                language_model.lm_head.weight, hidden
            )
            if logits is not None:
                return logits
        return self._block_linear(head, hidden)

    @staticmethod
    def _singleton_linear(linear, x: mx.array) -> mx.array:
        """Use Qwen's fused verifier projection with decode-exact arithmetic."""
        return _target_verify_linear(linear, x)

    def _block_linear(self, linear, x: mx.array) -> mx.array:
        """Fuse verifier time only where the projection remains decode exact."""
        if x.ndim != 3 or x.shape[1] <= 1:
            return linear(x)
        output = exact_dense_block_linear(linear, x)
        if output is not None:
            return output
        # Keep batch as quantized_matmul's M dimension, matching Bx1 decode.
        # Optional B=1 kernels use a different accumulation strategy and are
        # selected by the shared dispatcher only where a caller wants them.
        output = exact_quantized_linear(linear, x)
        if output is not None:
            return output
        return self._singleton_linear(linear, x)

    def _dense_mlp(self, mlp, x):
        gate, up = mx.split(self._block_linear(mlp.gate_up_proj, x), 2, axis=-1)
        hidden = clamped_swiglu(gate, up, mlp.swiglu_limit)
        return self._block_linear(mlp.down_proj, hidden)

    def _moe(self, moe, x, hc_state=None):
        # Routing and shared projections retain the decode Bx1 arithmetic.
        # Selected-expert projections keep verifier time in an outer dimension,
        # so the same exact path works for every batch and quantization format.
        batch, length, width = x.shape
        gate = moe.gate
        fp32_x = x.astype(mx.float32)
        logits = exact_fp32_decode_block_gemv(fp32_x, gate.weight)
        if logits is None:
            gate_parts = []
            for index in range(length):
                part = gate(mx.contiguous(x[:, index : index + 1]))
                mx.async_eval(*part)
                gate_parts.append(part)
            indices = mx.concatenate([part[0] for part in gate_parts], axis=1)
            weights = mx.concatenate([part[1] for part in gate_parts], axis=1)
        else:
            indices, weights = self._helpers()._expert_select(
                logits,
                gate.e_score_correction_bias,
                gate.top_k,
                gate.n_group,
                gate.topk_group,
                gate.routed_scaling_factor,
                gate.norm_topk_prob,
            )
        top_k = indices.shape[-1]
        flat_x = mx.contiguous(x).reshape(batch * length, width)
        flat_indices = indices.reshape(batch * length, top_k)
        switch = moe.switch_mlp
        # Match SwitchGLU's decode-time sorting decision using the target's
        # per-position route count.  Once selected, sort both verifier
        # positions together so all three expert projections reuse the same
        # order and one inverse permutation.
        sort_routes = batch * top_k >= 64
        route_order = None
        if sort_routes:
            projected, route_indices, route_order = _gather_sort(
                mx.expand_dims(x, (-2, -3)), indices
            )
            up = switch.up_proj(projected, route_indices, sorted_indices=True)
            gate = switch.gate_proj(projected, route_indices, sorted_indices=True)
        else:
            gate_up = exact_affine_switch_gate_up(switch, x, indices)
            if gate_up is None:
                up = exact_quantized_switch_linear(switch.up_proj, x, indices)
                gate = exact_quantized_switch_linear(switch.gate_proj, x, indices)
                if up is None or gate is None:
                    projected = mx.expand_dims(flat_x, (-2, -3))
                    up = switch.up_proj(
                        projected,
                        flat_indices,
                        sorted_indices=False,
                    )
                    gate = switch.gate_proj(
                        projected,
                        flat_indices,
                        sorted_indices=False,
                    )
                    up = up.squeeze(-2).reshape(batch, length, top_k, -1)
                    gate = gate.squeeze(-2).reshape(batch, length, top_k, -1)
            else:
                up, gate = gate_up
        activated = switch.activation(up, gate)
        shared_gate, shared_up = mx.split(
            self._block_linear(moe.shared_experts.gate_up_proj, x),
            2,
            axis=-1,
        )
        shared_hidden = clamped_swiglu(
            shared_gate,
            shared_up,
            moe.shared_experts.swiglu_limit,
        )
        shared = self._block_linear(
            moe.shared_experts.down_proj,
            shared_hidden,
        )
        if sort_routes:
            routed = switch.down_proj(
                activated,
                route_indices,
                sorted_indices=True,
            )
            routed = _scatter_unsort(routed, route_order, indices.shape).squeeze(-2)
            if hc_state is not None:
                output = DEFAULT_QUANTIZED_VERIFIER.moe_hc_expand(
                    switch.down_proj,
                    activated,
                    indices,
                    weights,
                    shared,
                    *hc_state,
                    routed=routed,
                )
                if output is not None:
                    return output
            output = combine_moe_outputs(routed, weights, shared)
            if hc_state is not None:
                output = self._hc_expand_timewise(output, *hc_state)
            return output

        if hc_state is not None:
            output = DEFAULT_QUANTIZED_VERIFIER.moe_hc_expand(
                switch.down_proj,
                activated,
                indices,
                weights,
                shared,
                *hc_state,
            )
            if output is not None:
                return output
        fused_moe = exact_affine_moe_down(
            switch.down_proj,
            activated,
            indices,
            weights,
            shared,
        )
        if fused_moe is not None:
            return (
                self._hc_expand_timewise(fused_moe, *hc_state)
                if hc_state
                else fused_moe
            )
        routed = exact_quantized_selected_linear(
            switch.down_proj,
            activated,
            indices,
        )
        if routed is None:
            activated = activated.reshape(batch * length, top_k, 1, -1)
            routed = switch.down_proj(
                activated,
                flat_indices,
                sorted_indices=False,
            )
            routed = routed.squeeze(-2).reshape(batch, length, top_k, -1)
        if hc_state is not None:
            output = DEFAULT_QUANTIZED_VERIFIER.moe_hc_expand(
                switch.down_proj,
                activated,
                indices,
                weights,
                shared,
                *hc_state,
                routed=routed,
            )
            if output is not None:
                return output
        output = combine_moe_outputs(routed, weights, shared)
        if hc_state is not None:
            output = self._hc_expand_timewise(output, *hc_state)
        return output

    def _linear_attention(self, attention, inputs, mask, cache):
        return attention(
            inputs,
            mask,
            cache,
            linear_fn=self._block_linear,
            output_linear_fn=self._block_linear,
            timewise_fn=self._timewise,
            output_gate_fn=self._linear_output_gate_timewise,
            scaled_norm_fn=scaled_rms_norm,
        )

    @staticmethod
    def _linear_output_gate_timewise(norm, output, gate):
        # RMSNorm reduces each head row independently, so verifier time does
        # not alter its accumulation tree. Keeping BxT intact lets the compiled
        # output norm, sigmoid, and gate run as one graph instead of two graphs
        # plus a concatenate for every speculative position.
        result = norm(output) * mx.sigmoid(gate)
        mx.async_eval(result)
        return result

    def _sparse_attention(
        self,
        attention,
        inputs,
        mask,
        cache,
        prev_topk_indices,
    ):
        batch, length, _ = inputs.shape
        q_a, kv_a = mx.split(
            self._block_linear(attention.qkv_a_proj, inputs),
            (attention.q_lora_rank,),
            axis=-1,
        )
        q_resid = self._timewise(attention.q_a_layernorm, q_a)
        q = (
            self._block_linear(attention.q_b_proj, q_resid)
            .reshape(batch, length, attention.num_heads, attention.q_head_dim)
            .transpose(0, 2, 1, 3)
        )
        new_latent = self._timewise(attention.kv_a_layernorm, kv_a)[:, None]
        index_projected = None
        if attention.indexer is not None:
            indexer = attention.indexer
            index_keys = self._timewise(
                indexer.k_norm,
                self._block_linear(indexer.wk, inputs),
            )
            index_gates = exact_fp32_decode_block_gemv(
                inputs.astype(mx.float32),
                indexer.index_kpool_compress_gate,
            )
            if index_gates is None:
                index_gates = mx.concatenate(
                    [
                        inputs[:, index : index + 1].astype(mx.float32)
                        @ indexer.index_kpool_compress_gate.T
                        for index in range(length)
                    ],
                    axis=1,
                )

            # The scorer projections are skipped while all historical pools
            # fit in the exact candidate set. Once the history exceeds top-k,
            # hoist them across verifier time just like the key projection.
            score_pool_limit = indexer.index_topk
            cache_offset = 0 if cache is None else cache[0].offset
            needs_score_projection = isinstance(cache_offset, mx.array) or (
                cache_offset + length > score_pool_limit
            )
            if needs_score_projection:
                index_q = self._block_linear(indexer.wq_b, q_resid).reshape(
                    batch,
                    length,
                    indexer.n_heads,
                    indexer.head_dim,
                )
                index_weights = (
                    self._block_linear(indexer.weights_proj, inputs).astype(mx.float32)
                    * indexer.n_heads**-0.5
                )
            else:
                index_q = index_weights = None
            index_projected = (
                index_keys,
                index_gates,
                index_q,
                index_weights,
            )
        if cache is None:
            kv_cache = index_cache = pool_cache = None
            cache_offset = 0
            latent = new_latent
        else:
            kv_cache = cache[0]
            cache_offset = kv_cache.offset
            if attention.indexer is None:
                index_cache = pool_cache = None
            else:
                index_cache = cache[1]
                pool_cache = cache[2]
            latent, _ = kv_cache.update_and_fetch(
                new_latent,
                mx.zeros((batch, 1, length, 0), dtype=new_latent.dtype),
            )

        if attention.indexer is None:
            if prev_topk_indices is None:
                raise ValueError("Shared indexer layer has no previous top-k indices.")
            topk = prev_topk_indices
        else:
            topk = attention.indexer(
                inputs,
                q_resid,
                mask,
                index_cache,
                pool_cache,
                cache_offset,
                linear_fn=self._block_linear,
                projected=index_projected,
            )

        outputs = []
        for index in range(length):
            output = self._latent_attention(
                attention,
                q[:, :, index : index + 1],
                latent,
                topk[:, index : index + 1],
                kv_cache,
            )
            mx.async_eval(output, topk[:, index : index + 1])
            outputs.append(output)
        return (
            self._block_linear(
                attention.o_proj,
                mx.concatenate(outputs, axis=1),
            ),
            topk,
        )

    def _latent_attention(self, attention, q, latent, topk, cache):
        batch = q.shape[0]
        kv_length = latent.shape[2]
        valid = (topk >= 0) & (topk < kv_length)
        safe = mx.clip(topk, 0, max(kv_length - 1, 0))
        selected = mx.take_along_axis(
            latent,
            safe[:, None, 0, :, None],
            axis=2,
        )
        q = self._head_timewise(attention.embed_q, q)
        output = scaled_dot_product_attention(
            q,
            selected,
            selected,
            cache=cache,
            scale=attention.scale,
            mask=valid[:, None],
        )
        output = self._head_timewise(attention.unembed_out, output)
        return output.transpose(0, 2, 1, 3).reshape(batch, 1, -1)

    @staticmethod
    def _timewise(fn, x: mx.array) -> mx.array:
        return fn(x)

    @staticmethod
    def _head_timewise(fn, x: mx.array) -> mx.array:
        if x.ndim != 4 or x.shape[2] <= 1:
            return fn(x)
        outputs = []
        for index in range(x.shape[2]):
            output = fn(mx.contiguous(x[:, :, index : index + 1]))
            mx.async_eval(output)
            outputs.append(output)
        return mx.concatenate(outputs, axis=2)

    @staticmethod
    def _hc_expand_timewise(x, residual, post, comb):
        output = exact_hc_expand(x, residual, post, comb)
        if output is not None:
            return output
        return hc_expand(x, residual, post, comb)

    @staticmethod
    def _hc_inputs(connection, x):
        y = x.astype(mx.float32)
        if x.shape[1] <= 1 or x.shape[0] > 1:
            normalized = mx.fast.rms_norm(
                y.flatten(-2),
                None,
                connection.norm_eps,
            )
            return y, normalized @ connection.fn.T

        # At B=1, flattening verifier time changes MLX's FP32 matmul reduction
        # order. Keep each mix projection decode-shaped while sharing the
        # Sinkhorn/collapse work across the complete verifier block.
        mixes = []
        normalized_steps = []
        for index in range(x.shape[1]):
            normalized = mx.fast.rms_norm(
                y[:, index : index + 1].flatten(-2),
                None,
                connection.norm_eps,
            )
            normalized_steps.append(normalized)
        for normalized in normalized_steps:
            mixes.append(normalized @ connection.fn.T)
        return y, mx.concatenate(mixes, axis=1)

    def _hc(self, connection, x):
        if _hc_kernel is None:
            return self._timewise(connection, x)
        y, mixes = self._hc_inputs(connection, x)
        return _hc_kernel(
            x,
            y,
            mixes,
            connection.scale,
            connection.base,
            connection.hc_mult,
            connection.sinkhorn_iters,
            connection.hc_eps,
        )

    def _hc_norm(self, connection, norm, x):
        if x.shape[0] == 1:
            output = exact_hc_normalized_norm(connection, norm, x)
            if output is not None:
                return output
        if _hc_kernel is not None:
            _y, mixes = self._hc_inputs(connection, x)
            output = self._hc_norm_from_mixes(connection, norm, x, mixes)
            if output is not None:
                return output
        collapsed, post, comb = self._hc(connection, x)
        return self._timewise(norm, collapsed), post, comb

    @staticmethod
    def _hc_norm_from_mixes(connection, norm, x, mixes):
        return exact_hc_norm(connection, norm, x, mixes)

    def argmax_from_hidden(self, language_model, hidden: mx.array) -> mx.array:
        if not language_model.args.tie_word_embeddings:
            head = language_model.lm_head
            output = DEFAULT_QUANTIZED_VERIFIER.argmax(head, hidden)
            if output is not None:
                return output
        return mx.argmax(self.logits_from_hidden(language_model, hidden), axis=-1)

    def _layer(
        self,
        layer,
        hidden,
        mask,
        cache,
        prev_topk_indices,
    ):
        residual = hidden
        collapsed, post, comb = self._hc_norm(
            layer.attn_hc,
            layer.input_layernorm,
            hidden,
        )
        if layer.block_type == "linear_attention":
            collapsed = self._linear_attention(
                layer.self_attn,
                collapsed,
                mask,
                cache,
            )
            topk = prev_topk_indices
        else:
            collapsed, topk = self._sparse_attention(
                layer.self_attn,
                collapsed,
                mask,
                cache,
                prev_topk_indices,
            )
        hidden = self._hc_expand_timewise(collapsed, residual, post, comb)

        residual = hidden
        collapsed, post, comb = self._hc_norm(
            layer.ffn_hc,
            layer.post_attention_layernorm,
            hidden,
        )
        if hasattr(layer.mlp, "switch_mlp"):
            hidden = self._moe(layer.mlp, collapsed, (residual, post, comb))
        else:
            collapsed = self._dense_mlp(layer.mlp, collapsed)
            hidden = self._hc_expand_timewise(collapsed, residual, post, comb)
        return hidden, topk

    def _model(
        self,
        model,
        inputs,
        inputs_embeds,
        cache,
        attention_mask,
        hidden_sink,
    ):
        hidden = model.embed_tokens(inputs) if inputs_embeds is None else inputs_embeds
        if cache is None:
            cache = [None] * len(model.layers)
        if attention_mask is not None:
            attention_mask = attention_mask.astype(mx.bool_)
            if attention_mask.shape[-1] != hidden.shape[1]:
                attention_mask = attention_mask[..., -hidden.shape[1] :]

        hidden = mx.repeat(hidden[:, :, None], model.config.hc_mult, axis=2)
        topk = None
        for layer, layer_cache in zip(model.layers, cache):
            layer_mask = attention_mask
            if layer.block_type == "linear_attention" and layer_cache is not None:
                layer_mask = create_ssm_mask(hidden[:, :, 0], layer_cache)
            hidden, topk = self._layer(
                layer,
                hidden,
                layer_mask,
                layer_cache,
                topk,
            )

        hidden = self._timewise(model.norm, hidden.mean(axis=2))
        if hidden_sink is not None:
            hidden_sink.append(hidden)
        return hidden

    def __call__(
        self,
        language_model,
        inputs,
        *,
        inputs_embeds=None,
        cache=None,
        attention_mask=None,
        hidden_sink=None,
        return_shared_kv=False,
        skip_logits=False,
    ) -> LanguageModelOutput:
        hidden = self._model(
            language_model.model,
            inputs,
            inputs_embeds,
            cache,
            attention_mask,
            hidden_sink,
        )
        logits = (
            None if skip_logits else self.logits_from_hidden(language_model, hidden)
        )
        return LanguageModelOutput(
            logits=logits,
            hidden_states=hidden_sink,
            shared_kv_states={} if return_shared_kv else None,
        )

    def verify(
        self,
        language_model,
        inputs,
        cache,
        sampler: Optional[Callable[[mx.array], mx.array]] = None,
    ):
        transaction = start_speculative_cache(cache, inputs.shape[1])
        hidden_sink = []
        try:
            output = self(
                language_model,
                inputs,
                cache=cache,
                hidden_sink=hidden_sink,
                return_shared_kv=True,
                skip_logits=True,
            )
        except Exception:
            transaction.abort()
            raise
        hidden = output.hidden_states[-1]
        if sampler is None:
            return hidden, {}, transaction
        return (
            hidden,
            {},
            transaction,
            sampler(self.logits_from_hidden(language_model, hidden)),
        )

    def rollback(
        self,
        language_model,
        caches: List[Any],
        rollback_state,
        accepted,
        block_size: int,
    ) -> int:
        del language_model
        return rollback_speculative_cache(
            caches,
            rollback_state,
            accepted,
            block_size,
        )


__all__ = ["Glm5NextSpeculativeVerifier"]
