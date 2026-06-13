from __future__ import annotations

from types import SimpleNamespace

import mlx.core as mx
import numpy as np

from mlx_vlm.models.minimax_m3.config import TextConfig
from mlx_vlm.models.minimax_m3.language import MiniMaxM3Model
from mlx_vlm.models.minimax_m3.msa import M3Indexer, block_sparse_attention


def _indexer_cfg():
    return SimpleNamespace(
        hidden_size=128, index_n_heads=4, index_head_dim=32, index_block_size=4,
        index_topk_blocks=3, index_local_blocks=1, rotary_dim=16, rope_theta=5e6, rms_norm_eps=1e-6,
    )


def test_indexer_selection_invariants():
    """Per query-block selection must always keep the local (current) block and never a future block."""
    mx.random.seed(0)
    ix = M3Indexer(_indexer_cfg()); ix.eval()
    S, block = 20, 4
    sel = np.array(ix.select_per_qblock(mx.random.normal((1, S, 128))))[0]  # [n_qblk, topk]
    for qb in range(sel.shape[0]):
        kept = {int(i) for i in sel[qb] if i >= 0}
        assert qb in kept, "local/current block must always be selected"
        assert all(b <= qb for b in kept), "must not select a future key-block"


def test_block_sparse_equals_full_causal_when_all_selected():
    """With every key-block selected, the gather block-sparse path must equal full causal attention."""
    mx.random.seed(0)
    B, Hq, Hkv, N, d, block = 1, 8, 2, 24, 16, 4
    nblk = N // block
    q = mx.random.normal((B, Hq, N, d)); k = mx.random.normal((B, Hkv, N, d)); v = mx.random.normal((B, Hkv, N, d))
    scale = d ** -0.5
    kk = mx.repeat(k, Hq // Hkv, axis=1); vv = mx.repeat(v, Hq // Hkv, axis=1)
    s = (q @ kk.transpose(0, 1, 3, 2)) * scale
    cm = mx.arange(N)[None, None, None, :] <= mx.arange(N)[None, None, :, None]
    s = mx.where(cm, s, -1e30)
    ref = mx.softmax(s.astype(mx.float32), axis=-1).astype(q.dtype) @ vv
    sel = mx.broadcast_to(mx.arange(nblk, dtype=mx.int32)[None, None, :], (B, nblk, nblk))
    out = block_sparse_attention(q, k, v, sel, block, scale)
    assert float(mx.max(mx.abs(ref - out))) < 1e-4


def test_model_sparse_matches_full_at_full_coverage():
    """A tiny model whose sparse layers select all blocks (topk>=n_blk) must match full attention."""
    mx.random.seed(0)
    sc = {"use_sparse_attention": True, "sparse_index_dim": 16, "sparse_num_index_heads": 2,
          "sparse_block_size": 4, "sparse_topk_blocks": 16, "sparse_local_block": 1,
          "sparse_attention_freq": [0, 0, 1, 1]}
    tc = TextConfig(hidden_size=64, intermediate_size=64, dense_intermediate_size=128, shared_intermediate_size=64,
                    num_hidden_layers=4, num_attention_heads=4, num_key_value_heads=2, head_dim=16, vocab_size=200,
                    rotary_dim=8, num_local_experts=8, num_experts_per_tok=2, moe_layer_freq=[0, 0, 1, 1],
                    rope_theta=5e6, sparse_attention_config=sc)
    m = MiniMaxM3Model(tc); m.eval()
    ids = mx.array([[i % 200 for i in range(20)]])
    out_sparse = m(ids)
    for layer in m.layers:
        layer.self_attn.indexer = None
    out_full = m(ids)
    assert float(mx.max(mx.abs(out_sparse - out_full))) < 1e-3
