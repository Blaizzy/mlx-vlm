"""GLM speculative operation adapters; model traversal remains in the model."""

from copy import copy

import mlx.core as mx
import mlx.nn as nn

from ...models.base import scaled_dot_product_attention
from ...models.deepseek_v4.hyper_connection import HyperConnection
from ...models.glm5_next.language import (
    Glm5NextAttention,
    Glm5NextIndexer,
    _expert_select,
)
from ...models.quantized_verifier import (
    decode_quantized_argmax,
    exact_quantized_moe_hc_expand,
    exact_quantized_selected_linear,
    exact_quantized_switch_linear,
)
from ...models.switch_layers import _gather_sort, _scatter_unsort
from ..cache_state import rollback_speculative_cache, start_speculative_cache
from ..ops.glm5_next import (
    combine_moe_outputs,
    exact_affine_moe_down,
    exact_affine_switch_gate_up,
    exact_fp32_decode_block_gemv,
)
from ..ops.hyper_connection import HyperConnectionOps
from ..ops.linear import native_batch_linear


class Glm5NextVerifierOps(HyperConnectionOps):
    def logits_from_hidden(self, language_model, hidden: mx.array) -> mx.array:
        tied = language_model.args.tie_word_embeddings
        head = (
            language_model.model.embed_tokens.as_linear
            if tied
            else language_model.lm_head
        )
        return native_batch_linear(head, hidden)

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
            indices, weights = _expert_select(
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
        shared = moe.shared_experts(x)
        if sort_routes:
            routed = switch.down_proj(
                activated,
                route_indices,
                sorted_indices=True,
            )
            routed = _scatter_unsort(routed, route_order, indices.shape).squeeze(-2)
            if hc_state is not None:
                output = exact_quantized_moe_hc_expand(
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
                output = self._hc_expand(output, *hc_state)
            return output

        if hc_state is not None:
            output = exact_quantized_moe_hc_expand(
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
            return self._hc_expand(fused_moe, *hc_state) if hc_state else fused_moe
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
            output = exact_quantized_moe_hc_expand(
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
            output = self._hc_expand(output, *hc_state)
        return output

    def argmax_from_hidden(self, language_model, hidden: mx.array) -> mx.array:
        if not language_model.args.tie_word_embeddings:
            head = language_model.lm_head
            output = decode_quantized_argmax(head, hidden)
            if output is not None:
                return output
        return mx.argmax(self.logits_from_hidden(language_model, hidden), axis=-1)


_OPS = Glm5NextVerifierOps()


class _Projection:
    def __init__(self, module):
        self.module = module

    def __call__(self, x):
        return native_batch_linear(self.module, x)


class _MoE:
    def __init__(self, module):
        self.module = module
        self.shared_experts = _view(module.shared_experts)

    def __getattr__(self, name):
        return getattr(self.module, name)

    def __call__(self, x, hc_state=None):
        return _OPS._moe(self, x, hc_state)


class _HyperConnection(HyperConnection):
    def apply_branch(self, x, norm, branch, *args, **kwargs):
        collapsed, post, comb = _OPS._hc_norm(self, norm, x)
        if isinstance(branch, _MoE):
            return branch(collapsed, (x, post, comb))
        output = branch(collapsed, *args, **kwargs)
        if isinstance(output, tuple):
            return (_OPS._hc_expand(output[0], x, post, comb), *output[1:])
        return _OPS._hc_expand(output, x, post, comb)


class _Indexer(Glm5NextIndexer):
    def _project_queries(self, x, q_resid):
        q = self.wq_b(q_resid).reshape(*x.shape[:2], self.n_heads, self.head_dim)
        weights = self.weights_proj(x).astype(mx.float32) * self.n_heads**-0.5
        return q, weights

    def _project_keys(self, x):
        gates = exact_fp32_decode_block_gemv(
            x.astype(mx.float32), self.index_kpool_compress_gate
        )
        if gates is None:
            gates = mx.concatenate(
                [
                    x[:, index : index + 1].astype(mx.float32)
                    @ self.index_kpool_compress_gate.T
                    for index in range(x.shape[1])
                ],
                axis=1,
            )
        return self.k_norm(self.wk(x)), gates


class _Attention(Glm5NextAttention):
    def _attend(
        self, q, latent, new_latent, topk, kv_cache, projected_cache, last_only=False
    ):
        batch, _, length, _ = q.shape
        outputs = []
        for index in range(length):
            selected_indices = topk[:, index : index + 1]
            valid = (selected_indices >= 0) & (selected_indices < latent.shape[2])
            safe = mx.clip(selected_indices, 0, max(latent.shape[2] - 1, 0))
            selected = mx.take_along_axis(latent, safe[:, None, 0, :, None], axis=2)
            query = self.embed_q(q[:, :, index : index + 1])
            output = scaled_dot_product_attention(
                query,
                selected,
                selected,
                cache=kv_cache,
                scale=self.scale,
                mask=valid[:, None],
            )
            output = self.unembed_out(output)
            mx.async_eval(output)
            outputs.append(output.transpose(0, 2, 1, 3).reshape(batch, 1, -1))
        return mx.concatenate(outputs, axis=1), topk


def _view(module, cls=None):
    """Copy only module containers, retaining the original parameter arrays."""
    result = copy(module)
    if cls is not None:
        result.__class__ = cls
    for name, value in module.items():
        if isinstance(value, (nn.Linear, nn.QuantizedLinear)):
            result[name] = _Projection(value)
    return result


class Glm5NextSpeculativeTarget:
    """Bind after loading/casting weights; keep one view per generation session.

    Containers and projection policies are private to this view. Parameter
    arrays remain shared, and ordinary calls delegate to the serving model.
    """

    def __init__(self, language_model):
        self.language_model = language_model
        self.model = copy(language_model.model)
        layers = []
        for layer in language_model.model.layers:
            view = copy(layer)
            view.attn_hc = copy(layer.attn_hc)
            view.attn_hc.__class__ = _HyperConnection
            view.ffn_hc = copy(layer.ffn_hc)
            view.ffn_hc.__class__ = _HyperConnection
            if layer.is_linear:
                view.self_attn = _view(layer.self_attn)
            else:
                view.self_attn = _view(layer.self_attn, _Attention)
                if layer.self_attn.indexer is not None:
                    view.self_attn.indexer = _view(layer.self_attn.indexer, _Indexer)
            view.mlp = (
                _MoE(layer.mlp)
                if hasattr(layer.mlp, "switch_mlp")
                else _view(layer.mlp)
            )
            layers.append(view)
        self.model.layers = layers

    def __call__(self, *args, **kwargs):
        return self.language_model(*args, **kwargs)

    def __getattr__(self, name):
        return getattr(self.language_model, name)

    def train(self, mode=True):
        self.language_model.train(mode)
        self.model.train(mode)
        return self

    def eval(self):
        return self.train(False)

    def speculative_logits_from_hidden(self, hidden):
        return _OPS.logits_from_hidden(self.language_model, hidden)

    def speculative_argmax_from_hidden(self, hidden):
        return _OPS.argmax_from_hidden(self.language_model, hidden)

    def speculative_verify_hidden(self, inputs, cache):
        transaction = start_speculative_cache(cache, inputs.shape[1])
        try:
            hidden = self.model(inputs, cache=cache)
            return hidden, {}, transaction
        except BaseException:
            transaction.abort()
            raise

    def speculative_verify_logits(self, inputs, cache, sampler):
        hidden, shared, transaction = self.speculative_verify_hidden(inputs, cache)
        try:
            return (
                hidden,
                shared,
                transaction,
                sampler(self.speculative_logits_from_hidden(hidden)),
            )
        except BaseException:
            transaction.abort()
            raise

    @staticmethod
    def rollback_speculative_cache(caches, state, accepted, block_size):
        return rollback_speculative_cache(caches, state, accepted, block_size)
