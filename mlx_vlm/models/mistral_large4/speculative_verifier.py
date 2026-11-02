"""Verify speculative blocks with decode-equivalent reductions and cache visibility."""

import mlx.core as mx
import mlx.nn as nn

from ...speculative.cache_state import start_speculative_cache
from ...speculative.ops.linear import _target_verify_linear
from ..linear import native_batch_linear, tokenwise
from .block_mla_kernel import block_sparse_mla, project_queries, project_values
from .language import MoE, PackedIndexer, PackedMLP, rotate
from .moe_kernel import masked_qmv, supports_masked_qmv
from .verify_kernel import native_wide_linear
from .verify_ops import activation, combine, expert_factor, local_experts


def _expert_group(language_model):
    return next(
        (
            layer.feed_forward.group
            for layer in language_model.layers
            if isinstance(layer.feed_forward, MoE)
        ),
        None,
    )


def _expert_parameters(language_model, dtype):
    """Reuse tiny derived scales while reading current packed weights each call."""
    if (
        not mx.metal.is_available()
        or mx.default_device() != mx.gpu
        or dtype not in (mx.float16, mx.bfloat16)
    ):
        return {}
    previous = language_model._verification_scales
    cached, parameters, pending = {}, {}, []
    for layer in language_model.layers:
        ffn = layer.feed_forward
        if (
            not isinstance(ffn, MoE)
            or ffn.training
            or ffn.group is None
            or ffn.args.n_group != 1
            or ffn.args.num_experts_per_tok != 4
        ):
            continue
        modules = (ffn.experts.w1, ffn.experts.w3, ffn.experts.w2)
        if not all(supports_masked_qmv(module) for module in modules):
            continue
        factors = []
        for module in modules:
            source = getattr(module, "global_scale", None)
            signature = (module.num_experts, getattr(module, "_scale_name", None))
            entry = previous.get(id(module))
            if entry is None or entry[0] is not source or entry[1] != signature:
                factor = expert_factor(module)
                pending.append(factor)
                entry = source, signature, factor
            cached[id(module)] = entry
            factors.append(entry[2])
        parameters[id(ffn)] = (
            [m.weight for m in modules],
            [m.scales for m in modules],
            factors,
        )
    language_model._verification_scales = cached
    if pending:
        mx.eval(pending)
    return parameters


def linear(module, x, group=None):
    # Batched decode uses B-by-1 reductions, which differ from singleton GEMV.
    if x.shape[0] > 1:
        if isinstance(module, nn.Linear) and "bias" not in module:
            output = native_wide_linear(module.weight, x, group)
            if output is not None:
                return output
        return native_batch_linear(module, x)
    return _target_verify_linear(module, x)


def dense_mlp(mlp, x, group=None):
    if isinstance(mlp, PackedMLP):
        gate, up = mx.split(linear(mlp.gate_up_proj, x, group), 2, axis=-1)
    else:
        gate, up = linear(mlp.w1, x, group), linear(mlp.w3, x, group)
    return linear(mlp.w2, activation(gate, up), group)


def _expert_mlp(experts, x, indices):
    gate = masked_qmv(x, experts.w1, indices)
    up = masked_qmv(x, experts.w3, indices)
    return masked_qmv(activation(gate, up), experts.w2, indices)


def feed_forward(ffn, x, group=None, parameters=None):
    if not isinstance(ffn, MoE):
        return dense_mlp(ffn, x, group)
    if parameters is not None:
        packed, scales, factors = parameters
        outputs, weights = local_experts(
            x,
            ffn.gate.weight,
            packed,
            scales,
            factors,
            ffn.start,
            ffn.args.routed_scaling_factor,
        )
        outputs = mx.distributed.all_sum(outputs, group=ffn.group)
    else:
        return _routed_feed_forward(ffn, x, group)
    result = combine(outputs, weights)
    if ffn.shared_experts is not None:
        result = result + dense_mlp(ffn.shared_experts, x, group)
    return result


def _routed_feed_forward(ffn, x, group):
    indices, weights = tokenwise(ffn.route, x)
    if ffn.group is None:
        if ffn.start or ffn.end != ffn.args.n_routed_experts:
            raise ValueError("An expert partition requires a distributed group")
        outputs = tokenwise(ffn.experts, x, indices)
    else:
        owned = (indices >= ffn.start) & (indices < ffn.end)
        local = mx.where(owned, indices.astype(mx.int32) - ffn.start, -1)
        if (
            not ffn.training
            and mx.metal.is_available()
            and mx.default_device() == mx.gpu
            and all(
                supports_masked_qmv(w)
                for w in (ffn.experts.w1, ffn.experts.w2, ffn.experts.w3)
            )
        ):
            if indices.size >= 32:
                order = mx.argsort(local.reshape(-1))
                inverse = mx.argsort(order)
                routed_x = x.reshape(-1, x.shape[-1])[order // local.shape[-1], None]
                routed_indices = local.reshape(-1)[order, None]
                outputs = _expert_mlp(ffn.experts, routed_x, routed_indices)[inverse]
            else:
                outputs = _expert_mlp(
                    ffn.experts,
                    x.reshape(-1, 1, x.shape[-1]),
                    local.reshape(-1, 1, local.shape[-1]),
                )
            outputs = outputs.reshape(*indices.shape, x.shape[-1])
        else:
            outputs = tokenwise(ffn.experts.masked, x, local)
        outputs = mx.distributed.all_sum(outputs, group=ffn.group)
    result = combine(outputs, weights)
    if ffn.shared_experts is not None:
        result = result + dense_mlp(ffn.shared_experts, x, group)
    return result


def _attention_tokenwise(
    attention, x, positions, cos, sin, cache, shared, topk, group=None
):
    """Project a short block together, advancing latent attention in token order."""
    batch, length, _ = x.shape
    q = (
        linear(attention.wq, x, group)
        .reshape(batch, length, attention.heads, attention.head_dim)
        .transpose(0, 2, 1, 3)
    )
    if attention.rope_dim:
        q = mx.concatenate(
            [
                q[..., : attention.nope_dim],
                rotate(q[..., attention.nope_dim :], cos[:, None], sin[:, None]),
            ],
            axis=-1,
        )
    q = attention.ssmax_scaling(q, positions)
    if attention.source == attention.idx:
        kv = linear(attention.wkv_a_with_mqa, x, group)
        projected_latent = attention.kv_a_norm(kv[..., : attention.latent])[:, None]
        projected_rotary = kv[..., attention.latent :][:, None]
        if attention.rope_dim:
            projected_rotary = rotate(projected_rotary, cos[:, None], sin[:, None])
    outputs = []
    for t in range(length):
        if attention.source == attention.idx:
            latent, rotary = cache.update_latent(
                mx.contiguous(projected_latent[:, :, t : t + 1]),
                mx.contiguous(projected_rotary[:, :, t : t + 1]),
            )
            shared[t][attention.idx] = latent, rotary
        else:
            latent, rotary = shared[t][attention.source]
        mask = mx.broadcast_to(cache.make_mask(1), (batch, 1, 1, latent.shape[2]))
        if attention.indexer is not None:
            topk[t] = attention.indexer(
                mx.contiguous(x[:, t : t + 1]),
                positions[:, t : t + 1],
                cos[:, t : t + 1],
                sin[:, t : t + 1],
                cache,
                mask,
            )
        value = attention._decode_attention(
            q[:, :, t : t + 1], latent, rotary, mask, topk[t]
        )
        outputs.append(value.transpose(0, 2, 1, 3).reshape(batch, 1, -1))
        cache.advance(1)
    return linear(attention.wo, mx.concatenate(outputs, axis=1), group)


def _sparse_attention_block(
    attention, x, positions, cos, sin, cache, shared, topk, group=None
):
    batch, length, _ = x.shape
    q = (
        linear(attention.wq, x, group)
        .reshape(batch, length, attention.heads, attention.head_dim)
        .transpose(0, 2, 1, 3)
    )
    if attention.rope_dim:
        q = mx.concatenate(
            [
                q[..., : attention.nope_dim],
                rotate(q[..., attention.nope_dim :], cos[:, None], sin[:, None]),
            ],
            axis=-1,
        )
    q = attention.ssmax_scaling(q, positions)
    if attention.source == attention.idx:
        kv = linear(attention.wkv_a_with_mqa, x, group)
        projected_latent = attention.kv_a_norm(kv[..., : attention.latent])[:, None]
        projected_rotary = kv[..., attention.latent :][:, None]
        if attention.rope_dim:
            projected_rotary = rotate(projected_rotary, cos[:, None], sin[:, None])
    block_mask = mx.broadcast_to(
        cache.make_mask(length), (batch, 1, length, cache.size() + length)
    )
    selected = []
    counts = []
    for t in range(length):
        if attention.source == attention.idx:
            latent, rotary = cache.update_latent(
                mx.contiguous(projected_latent[:, :, t : t + 1]),
                mx.contiguous(projected_rotary[:, :, t : t + 1]),
            )
            shared[t][attention.idx] = latent, rotary
        else:
            latent, rotary = shared[t][attention.source]
        mask = block_mask[:, :, t : t + 1, : latent.shape[2]]
        if attention.indexer is not None:
            topk[t] = attention.indexer(
                mx.contiguous(x[:, t : t + 1]),
                positions[:, t : t + 1],
                cos[:, t : t + 1],
                sin[:, t : t + 1],
                cache,
                mask,
            )
        indices = mx.sort(topk[t], axis=-1)[:, 0] if topk[t] is not None else None
        selected.append(indices)
        counts.append(indices.shape[-1] if indices is not None else latent.shape[2])
        cache.advance(1)
    width = max(counts)
    if all(idx is None for idx in selected):
        indices = mx.broadcast_to(mx.arange(width)[None, None], (batch, length, width))
    else:
        selected = [
            (
                idx
                if idx is not None
                else mx.broadcast_to(mx.arange(count)[None], (batch, count))
            )
            for idx, count in zip(selected, counts)
        ]
        indices = mx.stack(
            [mx.pad(idx, [(0, 0), (0, width - idx.shape[-1])]) for idx in selected],
            axis=1,
        )
    mask = block_mask
    weight = attention.wkv_b.weight.reshape(
        attention.heads, attention.nope_dim + attention.value_dim, attention.latent
    )
    ql = project_queries(q, weight, attention.nope_dim)
    out = block_sparse_mla(
        ql,
        q[..., attention.nope_dim :],
        latent,
        rotary,
        indices,
        mask,
        attention.head_dim**-0.5,
        counts,
        chunk=64 if batch == 1 else 128,
    )
    out = project_values(out, weight, attention.nope_dim)
    return linear(
        attention.wo, out.transpose(0, 2, 1, 3).reshape(batch, length, -1), group
    )


def attention_block(attention, x, positions, cos, sin, cache, shared, topk, group=None):
    if not mx.metal.is_available() or mx.default_device() != mx.gpu:
        return _attention_tokenwise(
            attention, x, positions, cos, sin, cache, shared, topk, group
        )
    batch, length, _ = x.shape
    if (
        attention.indexer is not None and cache.size() + length > attention.indexer.topk
    ) or any(v is not None for v in topk):
        return _sparse_attention_block(
            attention, x, positions, cos, sin, cache, shared, topk, group
        )
    prefix = cache.size()
    q = (
        linear(attention.wq, x, group)
        .reshape(batch, length, attention.heads, attention.head_dim)
        .transpose(0, 2, 1, 3)
    )
    if attention.rope_dim:
        q = mx.concatenate(
            [
                q[..., : attention.nope_dim],
                rotate(q[..., attention.nope_dim :], cos[:, None], sin[:, None]),
            ],
            axis=-1,
        )
    q = attention.ssmax_scaling(q, positions)
    if attention.source == attention.idx:
        kv = linear(attention.wkv_a_with_mqa, x, group)
        projected_latent = attention.kv_a_norm(kv[..., : attention.latent])[:, None]
        projected_rotary = kv[..., attention.latent :][:, None]
        if attention.rope_dim:
            projected_rotary = rotate(projected_rotary, cos[:, None], sin[:, None])
    block_mask = mx.broadcast_to(
        cache.make_mask(length), (batch, 1, length, cache.size() + length)
    )
    if attention.source == attention.idx:
        latent, rotary = cache.update_latent(projected_latent, projected_rotary)
        for t in range(length):
            shared[t][attention.idx] = (latent, rotary)
    else:
        latent, rotary = shared[-1][attention.source]
    if attention.indexer is not None:
        indexer = attention.indexer
        weight = (
            indexer.wk_weights_proj.weight[: indexer.dim]
            if isinstance(indexer, PackedIndexer)
            else indexer.wk.weight
        )
        k = native_wide_linear(weight, x, group)
        if k is None:
            k = mx.concatenate(
                [x[:, t : t + 1] @ weight.T for t in range(length)], axis=1
            )
        k = indexer.k_norm(k)
        if indexer.use_rope:
            k = mx.concatenate(
                [
                    rotate(k[..., : indexer.rope_dim], cos, sin),
                    k[..., indexer.rope_dim :],
                ],
                axis=-1,
            )
        cache.update_index(k)
        for t in range(length):
            topk[t] = None
    cache.advance(length)
    counts = [prefix + t + 1 for t in range(length)]
    width = max(counts)
    indices = mx.broadcast_to(mx.arange(width)[None, None], (batch, length, width))
    mask = block_mask
    weight = attention.wkv_b.weight.reshape(
        attention.heads, attention.nope_dim + attention.value_dim, attention.latent
    )
    ql = project_queries(q, weight, attention.nope_dim)
    out = block_sparse_mla(
        ql,
        q[..., attention.nope_dim :],
        latent,
        rotary,
        indices,
        mask,
        attention.head_dim**-0.5,
        counts,
        chunk=64 if batch == 1 else 128,
    )
    out = project_values(out, weight, attention.nope_dim)
    return linear(
        attention.wo, out.transpose(0, 2, 1, 3).reshape(batch, length, -1), group
    )


def verify_hidden(language_model, inputs, cache, capture_layer_ids):
    model = language_model.model
    if len(set(capture_layer_ids)) != len(capture_layer_ids) or any(
        i < 0 or i >= len(model.layers) for i in capture_layer_ids
    ):
        raise ValueError("capture_layer_ids must be unique indices inside the model")
    transaction = start_speculative_cache(cache, inputs.shape[1])
    try:
        x = model.embed_tokens(inputs)
        group = _expert_group(language_model)
        parameters = _expert_parameters(language_model, x.dtype)
        b, length, _ = x.shape
        positions = mx.broadcast_to(
            mx.maximum(mx.array(cache[0].offset).reshape(-1, 1) + mx.arange(length), 0),
            (b, length),
        )
        cos, sin = model.rotary_emb(positions, x.dtype)
        shared, topk = [{} for _ in range(length)], [None] * length
        captured = {}
        for i, (layer, entry) in enumerate(zip(model.layers, cache)):
            normalized = layer.attention_norm(x)
            if b > 1 and isinstance(layer.attention.wkv_b, nn.Linear):
                attended = attention_block(
                    layer.attention,
                    normalized,
                    positions,
                    cos,
                    sin,
                    entry,
                    shared,
                    topk,
                    group,
                )
            else:
                attended = []
                for t in range(length):
                    out, topk[t] = layer.attention(
                        mx.contiguous(normalized[:, t : t + 1]),
                        positions[:, t : t + 1],
                        cos[:, t : t + 1],
                        sin[:, t : t + 1],
                        entry,
                        shared[t],
                        topk[t],
                        None,
                    )
                    attended.append(out)
                attended = mx.concatenate(attended, axis=1)
            x = x + attended
            x = x + feed_forward(
                layer.feed_forward,
                layer.ffn_norm(x),
                group,
                parameters.get(id(layer.feed_forward)),
            )
            if mx.default_device() == mx.gpu and (i + 1) % 4 == 0:
                mx.async_eval(x)
            if i in capture_layer_ids:
                captured[i] = x
        return [captured[i] for i in capture_layer_ids], model.norm(x), transaction
    except BaseException:
        transaction.abort()
        raise


def verify_logits(language_model, hidden):
    logits = linear(language_model.lm_head, hidden, _expert_group(language_model))
    cap = language_model.args.final_logit_softcapping
    return mx.tanh(logits / cap) * cap if cap else logits
