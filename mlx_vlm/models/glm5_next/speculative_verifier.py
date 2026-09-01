from typing import Any, Callable, List, Optional

import mlx.core as mx

from ..base import LanguageModelOutput, create_ssm_mask, scaled_dot_product_attention
from ..cache import ArraysCache, CacheList, HierarchyCache, KVCache, PoolingCache
from ..deepseek_v4.hyper_connection import hc_expand
from ..exact_speculative_verify import exact_speculative_verify_weight


def _clone_cache_tree(value):
    if isinstance(value, mx.array):
        return mx.array(value)
    if isinstance(value, tuple):
        return tuple(_clone_cache_tree(item) for item in value)
    if isinstance(value, list):
        return [_clone_cache_tree(item) for item in value]
    if isinstance(value, dict):
        return {key: _clone_cache_tree(item) for key, item in value.items()}
    return value


def _snapshot_single_cache(cache, incoming_tokens):
    if isinstance(cache, ArraysCache):
        # GLM's recurrent layer replaces its convolution and GDN arrays instead
        # of updating them in-place. Retaining those references avoids copying
        # roughly one full FP32 recurrent state per linear-attention layer.
        return (
            "arrays",
            list(cache.state),
            cache._left_padding,
            cache._left_padding_advance,
            cache._lengths,
            cache._lengths_advance,
        )
    if isinstance(cache, CacheList):
        return (
            "list",
            [_snapshot_single_cache(child, incoming_tokens) for child in cache.caches],
        )
    if isinstance(cache, PoolingCache):
        remainder = int(cache.remainder)
        total = remainder + int(incoming_tokens)
        overwrite = total % cache.ratio if total >= cache.ratio else total
        preserve = min(remainder, overwrite)
        buf_kv = (
            None
            if preserve == 0 or cache.buf_kv is None
            else _clone_cache_tree(cache.buf_kv[:, :preserve])
        )
        buf_gate = (
            None
            if preserve == 0 or cache.buf_gate is None
            else _clone_cache_tree(cache.buf_gate[:, :preserve])
        )
        return (
            "pooling",
            remainder,
            buf_kv,
            buf_gate,
            None if cache.pooled is None else cache.pooled.shape[1],
        )
    if isinstance(cache, HierarchyCache):
        # HierarchyCache constructs replacement buffers/representatives, so
        # its old array references are immutable during verification.
        return (
            "hierarchy",
            cache.buffer,
            cache.representatives,
            list(cache.remainders),
            list(cache.representative_lengths),
        )
    if isinstance(cache, KVCache):
        return ("append", cache.empty(), int(cache.offset))
    return (
        "full",
        _clone_cache_tree(cache.state),
        _clone_cache_tree(cache.meta_state),
    )


def _snapshot_cache(caches, incoming_tokens):
    return [_snapshot_single_cache(cache, incoming_tokens) for cache in caches]


def _restore_single_cache(cache, snapshot):
    kind = snapshot[0]
    if kind == "arrays":
        (
            _,
            state,
            left_padding,
            left_padding_advance,
            lengths,
            lengths_advance,
        ) = snapshot
        cache.state = list(state)
        cache._left_padding = left_padding
        cache._left_padding_advance = left_padding_advance
        cache._lengths = lengths
        cache._lengths_advance = lengths_advance
        return
    if kind == "list":
        for child, child_snapshot in zip(cache.caches, snapshot[1]):
            _restore_single_cache(child, child_snapshot)
        return
    if kind == "pooling":
        _, remainder, buf_kv, buf_gate, pooled_length = snapshot
        cache.remainder = remainder
        if buf_kv is not None:
            cache.buf_kv[:, : buf_kv.shape[1]] = buf_kv
            cache.buf_gate[:, : buf_gate.shape[1]] = buf_gate
        cache.pooled = (
            None if pooled_length is None else cache.pooled[:, :pooled_length]
        )
        return
    if kind == "hierarchy":
        _, buffer, representatives, remainders, representative_lengths = snapshot
        cache.buffer = buffer
        cache.representatives = representatives
        cache.remainders = remainders
        cache.representative_lengths = representative_lengths
        return
    if kind == "append":
        _, was_empty, offset = snapshot
        if was_empty:
            cache.keys = cache.values = None
        cache.offset = offset
        return
    _, state, meta_state = snapshot
    cache.meta_state = _clone_cache_tree(meta_state)
    cache.state = _clone_cache_tree(state)


def _restore_cache(caches, snapshots):
    for cache, snapshot in zip(caches, snapshots):
        _restore_single_cache(cache, snapshot)


class Glm5NextSpeculativeVerifier:
    """Run GLM-5-Next MTP verification and own its rollback state."""

    @staticmethod
    def _helpers():
        # language.py owns the shared sparse gather helper and imports this
        # verifier at module load time, so resolve it lazily.
        from . import language

        return language

    def logits_from_hidden(self, language_model, hidden: mx.array) -> mx.array:
        if hidden.ndim == 3 and hidden.shape[1] > 1:
            logits = exact_speculative_verify_weight(
                language_model.lm_head.weight, hidden
            )
            if logits is not None:
                return logits
        return language_model.lm_head(hidden)

    def argmax_from_hidden(self, language_model, hidden: mx.array) -> mx.array:
        return mx.argmax(self.logits_from_hidden(language_model, hidden), axis=-1)

    def _attention(
        self,
        attention,
        x,
        padding_mask,
        cache,
        prev_topk_indices,
    ):
        batch, length, _ = x.shape
        q_a, kv_a = mx.split(attention.qkv_a_proj(x), (attention.q_lora_rank,), axis=-1)
        q_resid = attention.q_a_layernorm(q_a)
        q = (
            attention.q_b_proj(q_resid)
            .reshape(batch, length, attention.num_heads, attention.q_head_dim)
            .transpose(0, 2, 1, 3)
        )
        new_latent = attention.kv_a_layernorm(kv_a)[:, None]

        if cache is None:
            kv_cache = index_cache = pool_cache = hierarchy_cache = None
            latent = new_latent
            cache_offset = 0
        else:
            kv_cache = cache[0]
            cache_offset = kv_cache.offset
            if attention.indexer is None:
                index_cache = pool_cache = hierarchy_cache = None
            else:
                index_cache = cache[1]
                pool_cache = cache[2]
                hierarchy_cache = cache[3] if len(cache.caches) >= 5 else None
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
                x,
                q_resid,
                padding_mask,
                index_cache,
                pool_cache,
                hierarchy_cache,
                cache_offset,
            )

        kv_length = latent.shape[2]
        valid = (topk >= 0) & (topk < kv_length)
        safe = mx.clip(topk, 0, max(kv_length - 1, 0))
        if length == 1:
            selected = mx.take_along_axis(
                latent,
                safe[:, None, 0, :, None],
                axis=2,
            )
            q = attention.embed_q(q)
            out = scaled_dot_product_attention(
                q,
                selected,
                selected,
                cache=kv_cache,
                scale=attention.scale,
                mask=valid[:, None],
            )
            out = attention.unembed_out(out)
        else:
            # Retain latent-MLA decode arithmetic. Reprojecting the complete
            # latent cache is algebraically equivalent but changes MXFP8
            # rounding and can flip a close greedy target token.
            selected = self._helpers()._sparse_head_gather(latent, safe)
            latent_q = attention.embed_q(q)
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
            out = attention.unembed_out(out)

        out = out.transpose(0, 2, 1, 3).reshape(batch, length, -1)
        return attention.o_proj(out), topk

    def _layer(self, layer, hidden, mask, cache, prev_topk_indices):
        residual = hidden
        collapsed, post, comb = layer.attn_hc(hidden)
        collapsed = layer.input_layernorm(collapsed)
        if layer.block_type == "linear_attention":
            collapsed = layer.self_attn(collapsed, mask, cache)
            topk = prev_topk_indices
        else:
            collapsed, topk = self._attention(
                layer.self_attn,
                collapsed,
                mask,
                cache,
                prev_topk_indices,
            )
        hidden = hc_expand(collapsed, residual, post, comb)

        residual = hidden
        collapsed, post, comb = layer.ffn_hc(hidden)
        collapsed = layer.mlp(layer.post_attention_layernorm(collapsed))
        return hc_expand(collapsed, residual, post, comb), topk

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

        hidden = model.norm(hidden.mean(axis=2))
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
        cache_snapshot = _snapshot_cache(cache, inputs.shape[1])
        hidden_sink = []
        output = self(
            language_model,
            inputs,
            cache=cache,
            hidden_sink=hidden_sink,
            return_shared_kv=True,
            skip_logits=sampler is None,
        )
        hidden = output.hidden_states[-1]
        rollback_state = (cache_snapshot, inputs)
        if sampler is None:
            return hidden, {}, rollback_state
        return hidden, {}, rollback_state, sampler(output.logits)

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

        cache_snapshot, verify_inputs = rollback_state
        _restore_cache(caches, cache_snapshot)
        valid_lengths = [value + 1 for value in accepted_values]
        keep = max(valid_lengths, default=0)
        if not keep:
            return 0

        ragged = len(set(valid_lengths)) > 1
        if ragged:
            right_padding = [keep - length for length in valid_lengths]
            for cache in caches:
                cache.prepare(lengths=valid_lengths, right_padding=right_padding)
            attention_mask = (
                mx.arange(keep)[None] < mx.array(valid_lengths, dtype=mx.int32)[:, None]
            )
        else:
            right_padding = None
            attention_mask = None

        language_model(
            verify_inputs[:, :keep],
            cache=caches,
            attention_mask=attention_mask,
            skip_logits=True,
        )

        if ragged:
            for cache in caches:
                cache.finalize()
        return max(accepted_values)


__all__ = ["Glm5NextSpeculativeVerifier"]
