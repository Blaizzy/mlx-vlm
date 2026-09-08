from functools import lru_cache

import mlx.core as mx

_TG = 256

_INDEXER_EPILOGUE_SOURCE = r"""
    uint tid   = thread_position_in_threadgroup.x;
    uint bl    = threadgroup_position_in_grid.y;
    uint stile = threadgroup_position_in_grid.z;

    int Np = meta[0];
    int L  = meta[1];
    int b  = int(bl) / L;
    int l  = int(bl) - b * L;
    uint s = stile * TG + tid;

    threadgroup float wc[NHEADS];
    if (tid < (uint)NHEADS) {
        wc[tid] = weights[(b * L + l) * NHEADS + int(tid)];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (s < (uint)Np) {
        long base = (long(b) * NHEADS * L + l) * Np + long(s);
        long hstride = long(L) * Np;
        float acc = 0.0f;
        for (int h = 0; h < NHEADS; ++h) {
            float g = scores_in[base + long(h) * hstride];
            acc += max(g, 0.0f) * wc[h];
        }
        scores[(b * L + l) * Np + int(s)] = acc * scale[0];
    }
"""


@lru_cache(maxsize=None)
def _indexer_epilogue_kernel(n_heads):
    return mx.fast.metal_kernel(
        name=f"indexer_epilogue_h{n_heads}",
        input_names=["scores_in", "weights", "scale", "meta"],
        output_names=["scores"],
        source=_INDEXER_EPILOGUE_SOURCE,
        ensure_row_contiguous=True,
    )


def indexer_dense_scores_available(dtype, n_heads, head_dim):
    return mx.default_device() == mx.gpu and mx.metal.is_available()


def indexer_dense_scores(q, pooled, weights, scale):
    B, n_heads, L, hd = q.shape
    Np = pooled.shape[1]
    g = q.astype(mx.float32) @ pooled[:, None].swapaxes(-1, -2).astype(mx.float32)
    g = mx.contiguous(g)
    weights = mx.contiguous(weights.astype(mx.float32))
    scale_arr = mx.array([float(scale)], dtype=mx.float32)
    meta = mx.array([Np, L], dtype=mx.int32)

    kernel = _indexer_epilogue_kernel(n_heads)
    s_tiles = (Np + _TG - 1) // _TG
    (scores,) = kernel(
        inputs=[g, weights, scale_arr, meta],
        template=[("NHEADS", n_heads), ("TG", _TG)],
        grid=(_TG, B * L, s_tiles),
        threadgroup=(_TG, 1, 1),
        output_shapes=[(B, L, Np)],
        output_dtypes=[mx.float32],
    )
    return scores
