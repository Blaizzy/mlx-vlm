"""Verify short blocks using the same attention and reductions as decode.

Attention advances each token's latent/DSA cache in order. Feed-forward work
is batched across the proposed positions, with one expert-parallel reduction
per layer and singleton-equivalent dense projections.
"""

import mlx.core as mx
import mlx.nn as nn

from ...speculative.cache_state import start_speculative_cache
from ...speculative.ops.linear import _target_verify_linear as linear
from ..linear import tokenwise
from .language import MoE, PackedMLP
from .moe_kernel import supports_masked_qmv


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
            outputs = ffn.experts.masked(
                x.reshape(-1, 1, x.shape[-1]), local.reshape(-1, 1, local.shape[-1])
            ).reshape(*indices.shape, x.shape[-1])
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
            x = x + mx.concatenate(attended, axis=1)
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
