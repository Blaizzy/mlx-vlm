"""Verify short blocks using the same attention and reductions as decode.

Attention advances each token's latent/DSA cache in order. Feed-forward work
is batched across the proposed positions, with one expert-parallel reduction
per layer and dense projections preserving the native batch reduction order.
"""

import mlx.core as mx
import mlx.nn as nn

from ...speculative.cache_state import start_speculative_cache
from ...speculative.ops.linear import _target_verify_linear
from ..linear import native_batch_linear, tokenwise
from .language import MoE, PackedMLP, rotate
from .moe_kernel import supports_masked_qmv
from .verify_kernel import native_wide_linear


def linear(module, x):
    # Batched decode uses B-by-1 reductions, which differ from singleton GEMV.
    if x.shape[0] > 1:
        if isinstance(module, nn.Linear) and "bias" not in module:
            output = native_wide_linear(module.weight, x)
            if output is not None:
                return output
        return native_batch_linear(module, x)
    return _target_verify_linear(module, x)


def dense_mlp(mlp, x):
    if isinstance(mlp, PackedMLP):
        gate, up = mx.split(linear(mlp.gate_up_proj, x), 2, axis=-1)
    else:
        gate, up = linear(mlp.w1, x), linear(mlp.w3, x)
    return linear(mlp.w2, nn.silu(gate) * up)


def feed_forward(ffn, x):
    if not isinstance(ffn, MoE):
        return dense_mlp(ffn, x)
    indices, weights = tokenwise(ffn.route, x)
    if ffn.group is None:
        if ffn.start or ffn.end != ffn.args.n_routed_experts:
            raise ValueError("An expert partition requires a distributed group")
        outputs = tokenwise(ffn.experts, x, indices)
    else:
        owned = (indices >= ffn.start) & (indices < ffn.end)
        local = mx.where(owned, indices.astype(mx.int32) - ffn.start, -1)
        if all(
            supports_masked_qmv(w)
            for w in (ffn.experts.w1, ffn.experts.w2, ffn.experts.w3)
        ):
            # The native masked kernel treats every route independently. Flatten
            # positions into rows to retain exactly its decode accumulation.
            if indices.size >= 32:
                order = mx.argsort(local.reshape(-1))
                inverse = mx.argsort(order)
                routed_x = x.reshape(-1, x.shape[-1])[order // local.shape[-1], None]
                routed_indices = local.reshape(-1)[order, None]
                outputs = ffn.experts.masked(routed_x, routed_indices)[inverse]
            else:
                outputs = ffn.experts.masked(
                    x.reshape(-1, 1, x.shape[-1]), local.reshape(-1, 1, local.shape[-1])
                )
            outputs = outputs.reshape(*indices.shape, x.shape[-1])
        else:
            outputs = tokenwise(ffn.experts.masked, x, local)
        outputs = mx.distributed.all_sum(outputs, group=ffn.group)
    weighted = (outputs.astype(mx.float32) * weights[..., None]).astype(x.dtype)
    result = weighted[..., 0, :]
    for i in range(1, ffn.args.num_experts_per_tok):
        result = result + weighted[..., i, :]
    if ffn.shared_experts is not None:
        result = result + dense_mlp(ffn.shared_experts, x)
    return result


def attention_block(attention, x, positions, cos, sin, cache, shared, topk):
    """Project a short block together, advancing latent attention in token order."""
    batch, length, _ = x.shape
    q = (
        linear(attention.wq, x)
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
        kv = linear(attention.wkv_a_with_mqa, x)
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
    return linear(attention.wo, mx.concatenate(outputs, axis=1))


def verify_hidden(language_model, inputs, cache, capture_layer_ids):
    model = language_model.model
    if len(set(capture_layer_ids)) != len(capture_layer_ids) or any(
        i < 0 or i >= len(model.layers) for i in capture_layer_ids
    ):
        raise ValueError("capture_layer_ids must be unique indices inside the model")
    transaction = start_speculative_cache(cache, inputs.shape[1])
    try:
        x = model.embed_tokens(inputs)
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
            x = x + feed_forward(layer.feed_forward, layer.ffn_norm(x))
            if i in capture_layer_ids:
                captured[i] = x
        return [captured[i] for i in capture_layer_ids], model.norm(x), transaction
    except BaseException:
        transaction.abort()
        raise


def verify_logits(language_model, hidden):
    logits = linear(language_model.lm_head, hidden)
    cap = language_model.args.final_logit_softcapping
    return mx.tanh(logits / cap) * cap if cap else logits
