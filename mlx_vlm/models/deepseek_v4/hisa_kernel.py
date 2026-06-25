"""HISA — Hierarchical Indexing for Sparse Attention (deepseek_v4), batched L>=1.

Replaces the indexer's flat O(Np) top-k scan with a two-stage hierarchy:
  1. coarse: score block-mean representatives, causal-mask future blocks, keep top-Kb;
  2. fine:   gather only the kept blocks' positions and score them -- tiled over L so
             the candidate tensor stays bounded (memory flat in L) -- then top-k.

Implemented entirely in MLX: the tiled `matmul` uses MLX's optimized GEMM, which
benchmarks faster than a hand-written Metal kernel here while keeping memory flat
in L. Matches the exact flat top-k in the keep-all limit; approximates it otherwise.

paper: https://arxiv.org/abs/2603.28458
"""

import mlx.core as mx


def hisa_select(
    q,
    pooled,
    weights,
    scale,
    k,
    index_block,
    index_keep,
    valid_len=None,
    fine_chunk=512,
):
    """Batched (for L>=1) HISA selection. Returns top-k prefix indices (B, L, k).

    q:       (B, n_heads, L, head_dim)   pre-rotated queries
    pooled:  (B, Np, head_dim)           compressed prefix keys
    weights: (B, L, n_heads)             = weights_proj(x) * n_heads**-0.5
    scale:   head_dim ** -0.5
    valid_len: (B, L) int32 = #causally-visible pooled positions per query;
               None => all Np (decode / no future).
    fine_chunk: tile size over L for the fine stage (bounds peak memory).
    """

    B, _, L, D = q.shape
    Np = pooled.shape[1]
    b = index_block
    nb = Np // b
    usable = nb * b
    q = q.astype(mx.float32)
    pooled = pooled.astype(mx.float32)
    if valid_len is None:
        valid_len = mx.full((B, L), Np, dtype=mx.int32)
    valid_len = valid_len.astype(mx.int32)

    wk = weights.astype(mx.float32) * scale  # (B,L,H): per-head mult, scale folded in
    wk_h = wk.transpose(0, 2, 1)[..., None]  # (B,H,L,1)

    rep = pooled[:, :usable].reshape(B, nb, b, D).mean(axis=2)  # (B,nb,D)
    cs = mx.maximum(q @ rep[:, None].swapaxes(-1, -2), 0)  # (B,H,L,nb)
    cscore = (cs * wk_h).sum(axis=1)  # (B,L,nb)
    block_start = mx.arange(nb) * b
    cscore = mx.where(block_start[None, None] < valid_len[..., None], cscore, -1e30)
    Kb = min(index_keep, nb)
    top_blk = mx.argpartition(-cscore, kth=Kb - 1, axis=-1)[..., :Kb].astype(mx.int32)

    C = Kb * b
    chunk = fine_chunk or L
    parts = []
    for s in range(0, L, chunk):
        e = min(s + chunk, L)
        pos_c = (top_blk[:, s:e, :, None] * b + mx.arange(b)).reshape(B, e - s, C)
        idx = mx.broadcast_to(
            pos_c.reshape(B, (e - s) * C)[..., None], (B, (e - s) * C, D)
        )
        cand = mx.take_along_axis(pooled, idx, axis=1).reshape(B, e - s, C, D)
        qbl = q[:, :, s:e].transpose(0, 2, 1, 3)  # (B,chunk,H,D)
        fs = mx.maximum(mx.matmul(qbl, cand.swapaxes(-1, -2)), 0)  # (B,chunk,H,C)
        oc = (fs * wk[:, s:e][..., None]).sum(axis=2)  # (B,chunk,C)
        oc = mx.where(pos_c < valid_len[:, s:e][..., None], oc, -1e30)  # causal
        if chunk < L:
            mx.eval(oc)  # free chunk intermediates
        parts.append(oc)
    fscore = parts[0] if len(parts) == 1 else mx.concatenate(parts, axis=1)

    sel = mx.argpartition(-fscore, kth=k - 1, axis=-1)[..., :k]  # (B,L,k)
    pos = (top_blk[..., None] * b + mx.arange(b)).reshape(B, L, C)
    return mx.take_along_axis(pos, sel, axis=-1)  # (B,L,k)
