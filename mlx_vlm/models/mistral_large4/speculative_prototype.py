"""Isolated, independently switchable verifier experiments."""

from contextlib import contextmanager

import mlx.core as mx
import mlx.nn as nn

from . import speculative_verifier as verifier
from .block_mla_kernel import block_sparse_mla, project_queries, project_values
from .grouped_moe_kernel import grouped_qmv
from .language import MoE, PackedMLP, rotate
from .moe_kernel import masked_qmv, supports_masked_qmv

_grouped = 0
_compiled = False


@mx.compile
def activation(gate, up):
    return nn.silu(gate) * up


@mx.compile
def combine(outputs, weights):
    weighted = (outputs.astype(mx.float32) * weights[..., None]).astype(outputs.dtype)
    result = weighted[..., 0, :]
    for i in range(1, weights.shape[-1]):
        result = result + weighted[..., i, :]
    return result


def dense_mlp(mlp, x):
    if isinstance(mlp, PackedMLP):
        gate, up = mx.split(verifier.linear(mlp.gate_up_proj, x), 2, axis=-1)
    else:
        gate, up = verifier.linear(mlp.w1, x), verifier.linear(mlp.w3, x)
    return verifier.linear(mlp.w2, activation(gate, up))


def expert_mlp(experts, x, indices):
    grouped = _grouped and indices.size >= 32 and x.size // x.shape[-1] == indices.size

    def projection(value, module):
        return (
            grouped_qmv(value, module, indices, _grouped)
            if grouped
            else masked_qmv(value, module, indices)
        )

    gate, up = projection(x, experts.w1), projection(x, experts.w3)
    hidden = activation(gate, up) if _compiled else nn.silu(gate) * up
    return projection(hidden, experts.w2)


def feed_forward(ffn, x):
    if not isinstance(ffn, MoE):
        return verifier.dense_mlp(ffn, x)
    indices, weights = verifier.tokenwise(ffn.route, x)
    if ffn.group is None:
        if ffn.start or ffn.end != ffn.args.n_routed_experts:
            raise ValueError("An expert partition requires a distributed group")
        outputs = verifier.tokenwise(ffn.experts, x, indices)
    else:
        owned = (indices >= ffn.start) & (indices < ffn.end)
        local = mx.where(owned, indices.astype(mx.int32) - ffn.start, -1)
        if all(
            supports_masked_qmv(w)
            for w in (ffn.experts.w1, ffn.experts.w2, ffn.experts.w3)
        ):
            if indices.size >= 32:
                order = mx.argsort(local.reshape(-1))
                inverse = mx.argsort(order)
                routed_x = x.reshape(-1, x.shape[-1])[order // local.shape[-1], None]
                routed_indices = local.reshape(-1)[order, None]
                outputs = expert_mlp(ffn.experts, routed_x, routed_indices)[inverse]
            else:
                outputs = expert_mlp(
                    ffn.experts,
                    x.reshape(-1, 1, x.shape[-1]),
                    local.reshape(-1, 1, local.shape[-1]),
                )
            outputs = outputs.reshape(*indices.shape, x.shape[-1])
        else:
            outputs = verifier.tokenwise(ffn.experts.masked, x, local)
        outputs = mx.distributed.all_sum(outputs, group=ffn.group)
    if _compiled:
        result = combine(outputs, weights)
    else:
        weighted = (outputs.astype(mx.float32) * weights[..., None]).astype(x.dtype)
        result = weighted[..., 0, :]
        for i in range(1, ffn.args.num_experts_per_tok):
            result = result + weighted[..., i, :]
    if ffn.shared_experts is not None:
        result = result + verifier.dense_mlp(ffn.shared_experts, x)
    return result


def attention_block(attention, x, positions, cos, sin, cache, shared, topk):
    batch, length, _ = x.shape
    q = (
        verifier.linear(attention.wq, x)
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
        kv = verifier.linear(attention.wkv_a_with_mqa, x)
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
    return verifier.linear(
        attention.wo, out.transpose(0, 2, 1, 3).reshape(batch, length, -1)
    )


@contextmanager
def enabled(*, grouped=0, block=False, compiled=False):
    global _grouped, _compiled
    old = (verifier.feed_forward, verifier.dense_mlp, verifier.attention_block)
    _grouped, _compiled = grouped, compiled
    if grouped or compiled:
        verifier.feed_forward = feed_forward
    if compiled:
        verifier.dense_mlp = dense_mlp
    if block:
        verifier.attention_block = attention_block
    try:
        yield
    finally:
        verifier.feed_forward, verifier.dense_mlp, verifier.attention_block = old
        _grouped, _compiled = 0, False
