"""Softmax + greedy top-k expert selection for a few tokens, in one kernel.

Ported from aimd's MLX shim (https://github.com/shuuul/aimd,
``crates/aimd-mlx/shim/shim.cpp``), where these kernels originated.
"""

import mlx.core as mx

from ._common import scalar

# One simdgroup per token. The softmax mirrors mlx's ``softmax_single_row``
# (4 logits per lane, float accumulation, ``fast::exp``, one rounding to T),
# so E <= 128. ``argpartition`` is a stable ascending sort on Metal, so the
# selection keeps positions E-K .. E-1 of that order: a rank count with ties
# broken by position. The optional route scale multiplies the selected T
# scores in T, like ``scores * factor`` on a Python float.
_TOPK_SOURCE = """
    threadgroup float probs[E];
    uint token = threadgroup_position_in_grid.x;
    uint lane = thread_index_in_simdgroup;
    auto row = logits + size_t(token) * E;

    float ld[4];
    for (int i = 0; i < 4; ++i) {
        int c = int(lane) * 4 + i;
        ld[i] = c < E ? float(row[c]) : -INFINITY;
    }
    float maxval = -FLT_MAX;
    for (int i = 0; i < 4; ++i) {
        maxval = (maxval < ld[i]) ? ld[i] : maxval;
    }
    maxval = simd_max(maxval);
    float normalizer = 0.0f;
    for (int i = 0; i < 4; ++i) {
        float e = metal::fast::exp(ld[i] - maxval);
        ld[i] = e;
        normalizer += e;
    }
    normalizer = simd_sum(normalizer);
    normalizer = 1 / normalizer;
    for (int i = 0; i < 4; ++i) {
        int c = int(lane) * 4 + i;
        if (c < E) {
            probs[c] = float(T(ld[i] * normalizer));
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (int t = int(lane); t < E; t += 32) {
        float p = probs[t];
        int pos = 0;
        for (int j = 0; j < E; ++j) {
            float q = probs[j];
            pos += (q < p || (q == p && j < t)) ? 1 : 0;
        }
        if (pos >= E - K) {
            size_t slot = size_t(token) * K + (pos - (E - K));
            inds[slot] = uint32_t(t);
            T s = T(p);
            if (SCALED) {
                s = T(float(s) * float(T(scale)));
            }
            scores[slot] = s;
        }
    }
"""

_topk_kernel = mx.fast.metal_kernel(
    name="moe_softmax_topk",
    input_names=["logits", "scale"],
    output_names=["inds", "scores"],
    source=_TOPK_SOURCE,
)


def moe_softmax_topk(
    logits: mx.array, *, top_k: int, route_scale: float = 1.0
) -> list[mx.array]:
    """``softmax(logits, precise=True)`` followed by the greedy top-k of
    ``MoEGate`` (``argpartition`` + ``take_along_axis``) times
    ``route_scale``, in one kernel. logits is (..., E) with E <= 128; the
    result is ``[inds, scores]`` of shape (..., top_k): uint32 experts in
    ``argpartition`` order and their scores in the logits dtype."""
    E = logits.shape[-1]
    if E > 128 or not 1 <= top_k <= E:
        raise ValueError(f"moe_softmax_topk: unsupported E={E}, top_k={top_k}")
    rows = logits.size // E
    return _topk_kernel(
        inputs=[logits, scalar(float(route_scale))],
        template=[
            ("T", logits.dtype),
            ("E", E),
            ("K", top_k),
            ("SCALED", int(route_scale != 1.0)),
        ],
        grid=(32 * rows, 1, 1),
        threadgroup=(32, 1, 1),
        output_shapes=[logits.shape[:-1] + (top_k,)] * 2,
        output_dtypes=[mx.uint32, logits.dtype],
    )
