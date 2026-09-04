from typing import Any, Callable, List, Optional

import mlx.core as mx

from ...speculative.cache_state import restore_cache_state, snapshot_cache_state
from ..base import LanguageModelOutput, create_ssm_mask, scaled_dot_product_attention
from ..deepseek_v4.hyper_connection import _hc_kernel, hc_expand
from ..exact_speculative_verify import exact_speculative_verify_weight
from ..gated_delta import gated_delta_update
from ..quantized_verifier import (
    DEFAULT_QUANTIZED_VERIFIER,
    exact_quantized_linear,
    exact_quantized_selected_linear,
    exact_quantized_switch_linear,
)
from ..qwen3_5.speculative_verifier import _target_verify_linear
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
        if x.shape[0] == 1:
            return self._singleton_linear(linear, x)
        return self._singleton_linear(linear, x)

    def _dense_mlp(self, mlp, x):
        gate, up = mx.split(self._block_linear(mlp.gate_up_proj, x), 2, axis=-1)
        hidden = clamped_swiglu(gate, up, mlp.swiglu_limit)
        return self._block_linear(mlp.down_proj, hidden)

    def _moe(self, moe, x):
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
        gate_up = exact_affine_switch_gate_up(switch, x, indices)
        if gate_up is None:
            up = exact_quantized_switch_linear(switch.up_proj, x, indices)
            gate = exact_quantized_switch_linear(switch.gate_proj, x, indices)
            if up is None or gate is None:
                projected = mx.expand_dims(flat_x, (-2, -3))
                up = switch.up_proj(projected, flat_indices, sorted_indices=False)
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
        # Affine-4/5 verification can consume the routed and shared expert
        # activations together.  Prefer that path before materializing the
        # shared down projection; it removes one full 4096-wide quantized
        # dispatch per MoE layer.
        fused_moe = (
            exact_affine_moe_down(
                switch.down_proj,
                activated,
                indices,
                weights,
                moe.shared_experts.down_proj,
                shared_hidden,
            )
            if gate_up is not None
            else None
        )
        if fused_moe is not None:
            return fused_moe

        shared = self._block_linear(
            moe.shared_experts.down_proj,
            shared_hidden,
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
        return combine_moe_outputs(routed, weights, shared)

    def _linear_attention(self, attention, inputs, mask, cache):
        updates = []
        output = attention(
            inputs,
            mask,
            cache,
            rollback_sink=updates,
            linear_fn=self._block_linear,
            output_linear_fn=self._block_linear,
            timewise_fn=self._timewise,
            output_gate_fn=self._linear_output_gate_timewise,
            scaled_norm_fn=scaled_rms_norm,
        )
        return output, updates[0][:12]

    @staticmethod
    def _linear_output_gate_timewise(norm, output, gate):
        # RMSNorm reduces each head row independently, so verifier time does
        # not alter its accumulation tree. Keeping BxT intact lets the compiled
        # output norm, sigmoid, and gate run as one graph instead of two graphs
        # plus a concatenate for every speculative position.
        result = norm(output) * mx.sigmoid(gate)
        mx.async_eval(result)
        return result

    @staticmethod
    def _merge_sparse_updates(updates):
        merged = {
            "new_latent": mx.concatenate(
                [update["new_latent"] for update in updates],
                axis=2,
            )
        }
        if "indexer" not in updates[0]:
            return merged
        merged.update(
            indexer=updates[0]["indexer"],
            index_keys=mx.concatenate(
                [update["index_keys"] for update in updates], axis=1
            ),
            index_gates=mx.concatenate(
                [update["index_gates"] for update in updates], axis=1
            ),
            query_valid=mx.concatenate(
                [update["query_valid"] for update in updates], axis=1
            ),
            cache_offset=updates[0]["cache_offset"],
        )
        return merged

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
        outputs = []
        topks = []
        updates = []
        for index in range(inputs.shape[1]):
            step_updates = []
            output, topk = self._attention(
                attention,
                mx.contiguous(inputs[:, index : index + 1]),
                None if mask is None else mask[:, index : index + 1],
                cache,
                (
                    None
                    if prev_topk_indices is None
                    else prev_topk_indices[:, index : index + 1]
                ),
                step_updates,
                projected=(
                    q_resid[:, index : index + 1],
                    q[:, :, index : index + 1],
                    new_latent[:, :, index : index + 1],
                    (
                        None
                        if index_projected is None
                        else tuple(
                            None if value is None else value[:, index : index + 1]
                            for value in index_projected
                        )
                    ),
                ),
                project_output=False,
            )
            mx.async_eval(output, topk)
            outputs.append(output)
            topks.append(topk)
            updates.append(step_updates[0])
        return (
            self._block_linear(
                attention.o_proj,
                mx.concatenate(outputs, axis=1),
            ),
            mx.concatenate(topks, axis=1),
            self._merge_sparse_updates(updates),
        )

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
        # The fixed GLM verifier shape can keep the complete HC ingress in one
        # dispatch: input RMSNorm, mix projection, Sinkhorn/collapse, and the
        # decoder RMSNorm.  Besides avoiding three intermediate arrays, this
        # preserves the decode reduction tree used by the split exact path.
        # The single-row Metal kernel is decode exact at B=1. At B>1 its
        # FP32 post/comb rounding can differ from the target's batched HC path
        # by one ULP, which is enough to flip a later greedy token. Keep the
        # exact split mix/collapse kernel below for batched verification.
        output = (
            exact_hc_normalized_norm(connection, norm, x) if x.shape[0] == 1 else None
        )
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

    def _attention(
        self,
        attention,
        x,
        padding_mask,
        cache,
        prev_topk_indices,
        rollback_sink,
        projected=None,
        project_output=True,
    ):
        batch, length, _ = x.shape
        if projected is None:
            q_a, kv_a = mx.split(
                self._block_linear(attention.qkv_a_proj, x),
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
        else:
            q_resid, q, new_latent, index_projected = projected

        if cache is None:
            kv_cache = index_cache = pool_cache = None
            latent = new_latent
            cache_offset = 0
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

        cache_update = {"new_latent": new_latent}
        if attention.indexer is None:
            if prev_topk_indices is None:
                raise ValueError("Shared indexer layer has no previous top-k indices.")
            topk = prev_topk_indices
        elif length > 1:
            topk_parts = []
            index_updates = []
            for index in range(length):
                index_update = {}
                topk_parts.append(
                    attention.indexer(
                        x[:, index : index + 1],
                        q_resid[:, index : index + 1],
                        (
                            None
                            if padding_mask is None
                            else padding_mask[:, index : index + 1]
                        ),
                        index_cache,
                        pool_cache,
                        cache_offset + index,
                        cache_update_sink=index_update,
                        linear_fn=self._singleton_linear,
                    )
                )
                index_updates.append(index_update)
            topk = mx.concatenate(topk_parts, axis=1)
            cache_update.update(
                indexer=attention.indexer,
                index_keys=mx.concatenate(
                    [update["index_keys"] for update in index_updates], axis=1
                ),
                index_gates=mx.concatenate(
                    [update["index_gates"] for update in index_updates], axis=1
                ),
                query_valid=mx.concatenate(
                    [update["query_valid"] for update in index_updates], axis=1
                ),
                cache_offset=cache_offset,
            )
        else:
            topk = attention.indexer(
                x,
                q_resid,
                padding_mask,
                index_cache,
                pool_cache,
                cache_offset,
                cache_update_sink=cache_update,
                linear_fn=self._singleton_linear,
                projected=index_projected,
            )
        rollback_sink.append(cache_update)

        kv_length = latent.shape[2]
        valid = (topk >= 0) & (topk < kv_length)
        safe = mx.clip(topk, 0, max(kv_length - 1, 0))
        if length == 1:
            selected = mx.take_along_axis(
                latent,
                safe[:, None, 0, :, None],
                axis=2,
            )
            q = self._head_timewise(attention.embed_q, q)
            out = scaled_dot_product_attention(
                q,
                selected,
                selected,
                cache=kv_cache,
                scale=attention.scale,
                mask=valid[:, None],
            )
            out = self._head_timewise(attention.unembed_out, out)
        else:
            # Retain latent-MLA decode arithmetic. Reprojecting the complete
            # latent cache is algebraically equivalent but changes MXFP8
            # rounding and can flip a close greedy target token.
            selected = self._helpers()._sparse_head_gather(latent, safe)
            latent_q = self._head_timewise(attention.embed_q, q)
            folded_q = latent_q.transpose(0, 2, 1, 3).reshape(
                batch * length,
                attention.num_heads,
                1,
                attention.kv_lora_rank,
            )
            folded_mask = valid.reshape(batch * length, 1, 1, topk.shape[-1])
            out = scaled_dot_product_attention(
                folded_q,
                selected,
                selected,
                cache=None,
                scale=attention.scale,
                mask=folded_mask,
            )
            out = out.reshape(
                batch, length, attention.num_heads, attention.kv_lora_rank
            )
            out = out.transpose(0, 2, 1, 3)
            out = self._head_timewise(attention.unembed_out, out)

        out = out.transpose(0, 2, 1, 3).reshape(batch, length, -1)
        if project_output:
            out = self._block_linear(attention.o_proj, out)
        return out, topk

    def _layer(
        self,
        layer,
        hidden,
        mask,
        cache,
        prev_topk_indices,
        rollback_sink,
    ):
        residual = hidden
        collapsed, post, comb = self._hc_norm(
            layer.attn_hc,
            layer.input_layernorm,
            hidden,
        )
        if layer.block_type == "linear_attention":
            collapsed, linear_update = self._linear_attention(
                layer.self_attn,
                collapsed,
                mask,
                cache,
            )
            rollback_sink.append(linear_update)
            topk = prev_topk_indices
        else:
            collapsed, topk, sparse_update = self._sparse_attention(
                layer.self_attn,
                collapsed,
                mask,
                cache,
                prev_topk_indices,
            )
            rollback_sink.append(sparse_update)
        hidden = self._hc_expand_timewise(collapsed, residual, post, comb)

        residual = hidden
        collapsed, post, comb = self._hc_norm(
            layer.ffn_hc,
            layer.post_attention_layernorm,
            hidden,
        )
        if hasattr(layer.mlp, "switch_mlp"):
            collapsed = self._moe(layer.mlp, collapsed)
        else:
            collapsed = self._dense_mlp(layer.mlp, collapsed)
        return self._hc_expand_timewise(collapsed, residual, post, comb), topk

    def _model(
        self,
        model,
        inputs,
        inputs_embeds,
        cache,
        attention_mask,
        hidden_sink,
        rollback_sink,
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
                rollback_sink,
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
        rollback_sink=None,
        return_shared_kv=False,
        skip_logits=False,
    ) -> LanguageModelOutput:
        if rollback_sink is None:
            rollback_sink = []
        hidden = self._model(
            language_model.model,
            inputs,
            inputs_embeds,
            cache,
            attention_mask,
            hidden_sink,
            rollback_sink,
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
        cache_snapshot = snapshot_cache_state(cache, inputs.shape[1])
        hidden_sink = []
        rollback_updates = []
        output = self(
            language_model,
            inputs,
            cache=cache,
            hidden_sink=hidden_sink,
            rollback_sink=rollback_updates,
            return_shared_kv=True,
            skip_logits=True,
        )
        hidden = output.hidden_states[-1]
        rollback_state = (cache_snapshot, rollback_updates)
        if sampler is None:
            return hidden, {}, rollback_state
        return (
            hidden,
            {},
            rollback_state,
            sampler(self.logits_from_hidden(language_model, hidden)),
        )

    @staticmethod
    def _prepare_cache(cache, valid_lengths, right_padding):
        prepare = getattr(cache, "prepare", None)
        if callable(prepare):
            prepare(lengths=valid_lengths, right_padding=right_padding)

    @staticmethod
    def _finalize_cache(cache):
        finalize = getattr(cache, "finalize", None)
        if callable(finalize):
            finalize()

    def _replay_attention_cache(self, cache, update, valid_lengths, keep):
        if cache is None:
            return

        batch = len(valid_lengths)
        ragged = len(set(valid_lengths)) > 1
        right_padding = [keep - length for length in valid_lengths] if ragged else None
        kv_cache = cache[0]
        touched = [kv_cache]

        indexer = update.get("indexer")
        if indexer is not None:
            index_cache = cache[1]
            pool_cache = cache[2]
            touched.extend([index_cache, pool_cache])

        if ragged:
            for entry in touched:
                self._prepare_cache(entry, valid_lengths, right_padding)

        new_latent = update["new_latent"][:, :, :keep]
        kv_cache.update_and_fetch(
            new_latent,
            mx.zeros((batch, 1, keep, 0), dtype=new_latent.dtype),
        )

        if indexer is not None:
            valid = (
                mx.arange(keep)[None] < mx.array(valid_lengths, dtype=mx.int32)[:, None]
            )
            query_valid = update["query_valid"][:, :keep] & valid
            keys = update["index_keys"][:, :keep]
            gates = update["index_gates"][:, :keep]
            for index in range(keep):
                index_cache.update_and_fetch(
                    query_valid[:, None, index : index + 1, None],
                    mx.zeros((batch, 1, 1, 0), dtype=mx.bool_),
                )

                ready_keys, ready_gates, _ = pool_cache.accumulate_windows(
                    keys[:, index : index + 1],
                    gates[:, index : index + 1],
                    update["cache_offset"] + index,
                )
                new_pool_keys = indexer._compress_pools(ready_keys, ready_gates)
                pool_cache.update_and_fetch(new_pool_keys)

        if ragged:
            for entry in touched:
                self._finalize_cache(entry)

    @staticmethod
    def _initial_linear_state(update):
        q, _k, v, *_rest = update
        initial_state = update[7]
        if initial_state is not None:
            return initial_state
        batch = q.shape[0]
        heads, value_dim = v.shape[-2:]
        key_dim = q.shape[-1]
        return mx.zeros(
            (batch, heads, value_dim, key_dim),
            dtype=mx.float32,
        )

    def _replay_linear_caches(self, entries, valid_lengths, keep):
        if not entries:
            return

        if all(len(update) > 13 for _cache, update in entries):
            indices = mx.array(valid_lengths, dtype=mx.int32) - 1
            for cache, update in entries:
                conv_states = update[12]
                ssm_states = update[13]
                conv_indices = indices.reshape(
                    indices.shape[0], 1, *([1] * (conv_states.ndim - 2))
                )
                ssm_indices = indices.reshape(
                    indices.shape[0], 1, *([1] * (ssm_states.ndim - 2))
                )
                cache[0] = mx.take_along_axis(
                    conv_states,
                    conv_indices,
                    axis=1,
                ).squeeze(1)
                cache[1] = cache[2] = None
                cache[3] = mx.take_along_axis(
                    ssm_states,
                    ssm_indices,
                    axis=1,
                ).squeeze(1)
                self._finalize_cache(cache)
            return

        batch = len(valid_lengths)
        valid = mx.arange(keep)[None] < mx.array(valid_lengths, dtype=mx.int32)[:, None]
        q_parts, k_parts, v_parts, a_parts, b_parts = [], [], [], [], []
        A_log_parts, dt_bias_parts, state_parts, mask_parts = [], [], [], []
        lower_bounds = []

        for _cache, update in entries:
            q, k, v, a, b, A_log, dt_bias, *_ = update
            q_parts.append(q[:, :keep])
            k_parts.append(k[:, :keep])
            v_parts.append(v[:, :keep])
            a_parts.append(a[:, :keep])
            b_parts.append(b[:, :keep])
            A_log_parts.append(
                mx.broadcast_to(
                    A_log[None, None],
                    (batch, 1, *A_log.shape),
                )
            )
            dt_bias_parts.append(
                mx.broadcast_to(
                    dt_bias[None, None],
                    (batch, 1, *dt_bias.shape),
                )
            )
            state_parts.append(self._initial_linear_state(update))
            layer_mask = update[8]
            if layer_mask is None:
                layer_mask = valid
            else:
                layer_mask = layer_mask[:, :keep] & valid
            mask_parts.append(layer_mask)
            lower_bounds.append(update[11])

        if len(set(lower_bounds)) != 1:
            raise ValueError("GLM Gated Delta layers must share a lower bound.")

        _output, states = gated_delta_update(
            mx.concatenate(q_parts, axis=0),
            mx.concatenate(k_parts, axis=0),
            mx.concatenate(v_parts, axis=0),
            mx.concatenate(a_parts, axis=0),
            mx.concatenate(b_parts, axis=0),
            mx.concatenate(A_log_parts, axis=0),
            mx.concatenate(dt_bias_parts, axis=0),
            state=mx.concatenate(state_parts, axis=0),
            mask=mx.concatenate(mask_parts, axis=0),
            use_kernel=True,
            lower_bound=lower_bounds[0],
        )

        offset = 0
        for cache, update in entries:
            conv_input = update[9]
            kernel_size = int(update[10])
            cache[0] = mx.concatenate(
                [
                    conv_input[
                        row : row + 1,
                        length : length + kernel_size - 1,
                    ]
                    for row, length in enumerate(valid_lengths)
                ],
                axis=0,
            )
            cache[1] = cache[2] = None
            cache[3] = states[offset : offset + batch]
            offset += batch
            self._finalize_cache(cache)

    def rollback(
        self,
        language_model,
        caches: List[Any],
        rollback_state,
        accepted,
        block_size: int,
    ) -> int:
        del block_size
        if isinstance(accepted, int):
            accepted_values = [accepted]
        elif isinstance(accepted, mx.array):
            accepted_values = [int(value) for value in accepted.tolist()]
        else:
            accepted_values = [int(value) for value in accepted]

        cache_snapshot, rollback_updates = rollback_state
        restore_cache_state(caches, cache_snapshot)
        valid_lengths = [value + 1 for value in accepted_values]
        keep = max(valid_lengths, default=0)
        if not keep:
            return 0

        linear_entries = []
        for layer, cache, update in zip(
            language_model.model.layers,
            caches,
            rollback_updates,
        ):
            if layer.block_type == "linear_attention":
                linear_entries.append((cache, update))
            else:
                self._replay_attention_cache(cache, update, valid_lengths, keep)
        self._replay_linear_caches(linear_entries, valid_lengths, keep)
        return max(accepted_values)


__all__ = ["Glm5NextSpeculativeVerifier"]
