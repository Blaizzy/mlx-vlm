"""Experiments layered on the 41 tok/s exact verifier."""

from contextlib import contextmanager

import mlx.core as mx

from . import speculative_prototype as proto
from . import speculative_verifier as verifier
from .block_mla_kernel import block_sparse_mla, project_queries, project_values
from .language import PackedIndexer, rotate


def attention_block(attention, x, positions, cos, sin, cache, shared, topk):
    batch, length, _ = x.shape
    if (
        attention.indexer is not None and cache.size() + length > attention.indexer.topk
    ) or any(v is not None for v in topk):
        return proto.attention_block(
            attention, x, positions, cos, sin, cache, shared, topk
        )
    prefix = cache.size()
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
        k = verifier.native_wide_linear(weight, x)
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
    return verifier.linear(
        attention.wo, out.transpose(0, 2, 1, 3).reshape(batch, length, -1)
    )


@contextmanager
def enabled(*, cache_block=False, dense=None, expert=None):
    old_linear = verifier.native_wide_linear
    old_qmv = proto.masked_qmv
    with proto.enabled(block=True, compiled=True):
        if cache_block:
            verifier.attention_block = attention_block
        if dense is not None:
            verifier.native_wide_linear = dense
        if expert is not None:
            proto.masked_qmv = expert
        try:
            yield
        finally:
            verifier.native_wide_linear = old_linear
            proto.masked_qmv = old_qmv
