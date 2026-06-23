"""MiniMax Sparse Attention (Lightning Indexer + gather block-sparse attention) for M3 — Phase 2.

On sparse layers: the indexer scores queries vs key-blocks and picks the top-k key-blocks per
QUERY-BLOCK (+ the local/current block). The main attention then gathers only those key/value
blocks and attends over them (NSA-style), so attention FLOPs drop ~n_blk/topk at long context.
Vectorized in pure MLX (block-level gather + one batched SDPA-like matmul). When topk covers all
blocks this is bit-for-bit equivalent to full causal attention — the numerical gate.
"""
import mlx.core as mx
import mlx.nn as nn

from .language import GemmaRMSNorm

NEG = -1e30


class M3Indexer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.n_heads = config.index_n_heads
        self.head_dim = config.index_head_dim
        self.block = config.index_block_size
        self.topk = config.index_topk_blocks
        self.local = config.index_local_blocks
        self.q_proj = nn.Linear(config.hidden_size, self.n_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(config.hidden_size, self.head_dim, bias=False)
        self.q_norm = GemmaRMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.k_norm = GemmaRMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.rope = nn.RoPE(config.rotary_dim, traditional=False, base=config.rope_theta)

    def _per_query_block_scores(self, x):
        """[B, Sq, n_kblk] max-pooled key-block scores per query (causal, ref-faithful)."""
        B, S, _ = x.shape
        q = self.q_norm(self.q_proj(x).reshape(B, S, self.n_heads, self.head_dim)).transpose(0, 2, 1, 3)
        k = self.k_norm(self.k_proj(x).reshape(B, S, 1, self.head_dim)).transpose(0, 2, 1, 3)
        q = self.rope(q); k = self.rope(k)
        scores = q.astype(mx.float32) @ k.astype(mx.float32).transpose(0, 1, 3, 2)  # [B,h,Sq,Sk]
        kpos = mx.arange(S); qpos = mx.arange(S)
        scores = mx.where(kpos[None, None, None, :] > qpos[None, None, :, None], NEG, scores)
        nkb = -(-S // self.block); pad = nkb * self.block - S
        if pad:
            scores = mx.concatenate([scores, mx.full((B, self.n_heads, S, pad), NEG)], axis=-1)
        scores = scores.reshape(B, self.n_heads, S, nkb, self.block)
        return scores.max(axis=-1).max(axis=1)  # [B, Sq, nkb]

    def select_per_qblock(self, x):
        """Per query-block top-k key-block indices: [B, n_qblk, topk] (int32, -1 padded)."""
        bs = self._per_query_block_scores(x)            # [B,Sq,nkb]
        B, S, nkb = bs.shape
        nqb = -(-S // self.block); pad = nqb * self.block - S
        if pad:
            bs = mx.concatenate([bs, mx.full((B, pad, nkb), NEG)], axis=1)
        bsq = bs.reshape(B, nqb, self.block, nkb).max(axis=2)  # [B,nqb,nkb] pool over queries in block
        qb = mx.arange(nqb)
        for l in range(self.local):
            loc = mx.maximum(qb - l, 0)
            oh = (mx.arange(nkb)[None, :] == loc[:, None])      # [nqb,nkb]
            bsq = mx.where(oh[None], 1e30, bsq)
        topk = min(self.topk, nkb)
        idx = mx.argsort(-bsq, axis=-1)[..., :topk]             # [B,nqb,topk]
        ts = mx.take_along_axis(bsq, idx, axis=-1)
        return mx.where(ts <= NEG / 2, -1, idx.astype(mx.int32))


def block_sparse_attention(q, k, v, sel, block, scale, valid_len=None):
    """q:[B,Hq,N,d]  k,v:[B,Hkv,N,d]  sel:[B,n_qblk,topk] int32(-1 pad). N % block == 0.
    Returns [B,Hq,N,d]. Equals full causal attention when sel covers all blocks."""
    B, Hq, N, d = q.shape
    Hkv = k.shape[1]
    nblk = N // block
    if valid_len is None:
        valid_len = N
    nqb = sel.shape[1]
    topk = sel.shape[2]
    if Hkv != Hq:
        rep = Hq // Hkv
        k = mx.repeat(k, rep, axis=1); v = mx.repeat(v, rep, axis=1)
    kf = k.reshape(B, Hq, nblk, block * d)
    vf = v.reshape(B, Hq, nblk, block * d)
    safe = mx.where(sel < 0, 0, sel)                            # [B,nqb,topk]
    gi = mx.broadcast_to(safe.reshape(B, 1, nqb * topk), (B, Hq, nqb * topk))
    gi = mx.broadcast_to(gi[..., None], (B, Hq, nqb * topk, block * d)).astype(mx.int32)
    gk = mx.take_along_axis(kf, gi, axis=2).reshape(B, Hq, nqb, topk * block, d)
    gv = mx.take_along_axis(vf, gi, axis=2).reshape(B, Hq, nqb, topk * block, d)
    qb = q.reshape(B, Hq, nqb, block, d)
    scores = (qb @ gk.transpose(0, 1, 2, 4, 3)) * scale         # [B,Hq,nqb,block,topk*block]
    keypos = (safe[..., None] * block + mx.arange(block)[None, None, None, :]).reshape(B, nqb, topk * block)
    qpos = mx.arange(nqb)[:, None] * block + mx.arange(block)[None, :]   # [nqb,block]
    valid = mx.broadcast_to((sel >= 0)[..., None], (B, nqb, topk, block)).reshape(B, nqb, topk * block)
    causal = keypos[:, :, None, :] <= qpos[None, :, :, None]    # [B,nqb,block,topk*block]
    inrange = keypos < valid_len
    mask = causal & valid[:, :, None, :] & inrange[:, :, None, :]
    scores = mx.where(mask[:, None], scores, NEG)
    w = mx.softmax(scores.astype(mx.float32), axis=-1).astype(q.dtype)
    out = w @ gv                                                # [B,Hq,nqb,block,d]
    return out.reshape(B, Hq, N, d)
