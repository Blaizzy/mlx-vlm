# Copyright © 2026 Apple Inc.

"""The q and k head norms and the rotary embedding, in one kernel.

Fuses, for the q and k heads of every token, the head RMSNorm, the weight
multiply and the split-half rotation that Attention does with
mx.fast.rms_norm and RotaryEmbedding. Bitwise equal to that unfused path, and
inference only.
"""

import mlx.core as mx

from ._common import BLOCK_SUM_HEADER, scalar

# The reads per thread and the largest head that mx.fast.rms_norm normalizes in
# one row, from mlx/backend/metal/kernels/defines.h. The sum of squares below
# adds the same values in the same order as that kernel does, which is what
# makes the two norms bitwise equal.
_RMS_N_READS = 4
_RMS_LOOPED_LIMIT = 4096

# One threadgroup per head. Thread t normalizes the N_READS consecutive values
# starting at t * N_READS, as rms_single_row does, and then rotates both the
# value and its partner in the other half of the head. The rotation is
# x * cos + concat([-b, a]) * sin on the two halves, with every product and
# sum rounded to T exactly as the unfused elementwise ops round them.
_LFM3_QK_NORM_ROPE_SOURCE = """
    #pragma METAL fp contract(off)
    constexpr int NR = N_READS;
    constexpr int HALF = D / 2;
    threadgroup float buf[32];
    uint h = threadgroup_position_in_grid.x;
    uint l = threadgroup_position_in_grid.y;
    uint b = threadgroup_position_in_grid.z;
    uint t = thread_position_in_threadgroup.x;
    uint lane = thread_index_in_simdgroup;
    uint sg = simdgroup_index_in_threadgroup;
    bool is_q = h < HQ;
    uint head = is_q ? h : h - HQ;
    uint heads = is_q ? HQ : HK;
    size_t row = size_t(b) * seq_len + l;
    device const T* x = (is_q ? q : k) + (row * heads + head) * D;
    device const T* w = is_q ? q_weight : k_weight;

    float pos;
    if (USE_POS) {
        pos = static_cast<float>(position_ids[row]);
    } else {
        pos = static_cast<float>(
            static_cast<int>(offset[b * OFFSET_STRIDE]) + static_cast<int>(l));
    }

    float v[NR];
    float sum_sq = 0.0f;
    for (int i = 0; i < NR; i++) {
        int d = static_cast<int>(t) * NR + i;
        v[i] = d < D ? static_cast<float>(x[d]) : 0.0f;
        sum_sq += v[i] * v[i];
    }
    float rstd =
        metal::precise::rsqrt(block_sum<TG>(sum_sq, buf, lane, sg) / D + eps);

    device T* o = (is_q ? q_out : k_out) + ((size_t(b) * heads + head) * seq_len + l) * D;
    for (int i = 0; i < NR; i++) {
        int d = static_cast<int>(t) * NR + i;
        if (d < D) {
            int r = d < HALF ? d : d - HALF;
            int p = d < HALF ? d + HALF : d - HALF;
            // The weight scales the rounded norm in T, as the norm module does.
            T a = static_cast<T>(v[i] * rstd);
            a = static_cast<T>(static_cast<float>(a) * static_cast<float>(w[d]));
            T pb = static_cast<T>(static_cast<float>(x[p]) * rstd);
            pb = static_cast<T>(static_cast<float>(pb) * static_cast<float>(w[p]));
            // The caller's inv_freq is zero padded, so NoPE pairs rotate by
            // exactly one.
            T c = static_cast<T>(metal::precise::cos(pos * inv_freq[r]));
            T s = static_cast<T>(metal::precise::sin(pos * inv_freq[r]));
            T first = static_cast<T>(static_cast<float>(a) * static_cast<float>(c));
            T second = static_cast<T>(static_cast<float>(pb) * static_cast<float>(s));
            // A rounded -b * sin is the negation of the rounded b * sin.
            float other = -static_cast<float>(second);
            o[d] = d < HALF
                ? static_cast<T>(static_cast<float>(first) + other)
                : static_cast<T>(static_cast<float>(first) + static_cast<float>(second));
        }
    }
"""

_lfm3_qk_norm_rope_kernel = mx.fast.metal_kernel(
    name="lfm3_qk_norm_rope",
    input_names=[
        "q",
        "k",
        "q_weight",
        "k_weight",
        "inv_freq",
        "eps",
        "offset",
        "position_ids",
        "seq_len",
    ],
    output_names=["q_out", "k_out"],
    source=_LFM3_QK_NORM_ROPE_SOURCE,
    header=BLOCK_SUM_HEADER,
)


def rms_norm_threads(dim: int) -> int:
    """The threads mx.fast.rms_norm gives to a row of dim values."""
    reads = (dim + _RMS_N_READS - 1) // _RMS_N_READS
    return 32 * ((reads + 31) // 32)


def lfm3_qk_norm_rope(
    q: mx.array,
    k: mx.array,
    q_weight: mx.array,
    k_weight: mx.array,
    inv_freq: mx.array,
    *,
    eps: float,
    position_ids: mx.array | None = None,
    offset: int | mx.array = 0,
) -> tuple[mx.array, mx.array]:
    """Norm the head values of q and k and rotate their halves, in one kernel.

    Bitwise equal to the unfused path: mx.fast.rms_norm on q (B, L, n_heads,
    head_dim) and on k (B, L, n_kv_heads, head_dim), then the RotaryEmbedding
    split-half rotation of the transposed (B, heads, L, head_dim) pair. The
    rotation reads the caller's zero-padded fp32 inv_freq, so the caller picks
    the number of rotated angles with partial_rotary_factor, and the padded
    pairs rotate by exactly one. positions are offset + 0, ... offset + L - 1
    unless position_ids (B, L) places the tokens. offset is an int, or an array
    with one offset per batch row. Returns q (B, n_heads, L, head_dim) and k
    (B, n_kv_heads, L, head_dim), which is the layout the attention reads.

    Inference only: needs Metal, float16 or bfloat16 values, and a head_dim
    that mx.fast.rms_norm normalizes in one row (at most 4096). Otherwise
    raises NotImplementedError, so the caller keeps its unfused path.
    """
    batch, seq_len, n_heads, head_dim = q.shape
    n_kv_heads = k.shape[2]
    if q.dtype not in (mx.float16, mx.bfloat16):
        raise NotImplementedError(f"lfm3_qk_norm_rope has no kernel for {q.dtype}")
    if not mx.metal.is_available():
        raise NotImplementedError("lfm3_qk_norm_rope needs Metal")
    if head_dim % 2 or head_dim > _RMS_LOOPED_LIMIT:
        raise NotImplementedError(f"lfm3_qk_norm_rope has no kernel for {head_dim=}")
    for name, weight in [("q_weight", q_weight), ("k_weight", k_weight)]:
        if weight.dtype != q.dtype or weight.shape != (head_dim,):
            raise NotImplementedError(
                f"lfm3_qk_norm_rope expects {name} ({head_dim},) {q.dtype}"
            )
    if inv_freq.dtype != mx.float32 or inv_freq.shape != (head_dim // 2,):
        raise NotImplementedError(
            f"lfm3_qk_norm_rope expects inv_freq ({head_dim // 2},) float32"
        )
    if isinstance(offset, mx.array):
        offset = offset.reshape(-1).astype(mx.int32)
    else:
        offset = mx.array([offset], dtype=mx.int32)
    use_pos = position_ids is not None
    if use_pos:
        if position_ids.size != batch * seq_len:
            raise NotImplementedError("lfm3_qk_norm_rope expects position_ids (B, L)")
        positions = (
            position_ids
            if position_ids.dtype == mx.int32
            else position_ids.astype(mx.int32)
        )
    else:
        positions = offset
    threads = rms_norm_threads(head_dim)
    return _lfm3_qk_norm_rope_kernel(
        inputs=[
            q,
            k,
            q_weight,
            k_weight,
            inv_freq,
            scalar(eps),
            offset,
            positions,
            scalar(seq_len, mx.int32),
        ],
        template=[
            ("T", q.dtype),
            ("D", head_dim),
            ("HQ", n_heads),
            ("HK", n_kv_heads),
            ("TG", threads),
            ("N_READS", _RMS_N_READS),
            ("USE_POS", int(use_pos)),
            ("OFFSET_STRIDE", int(offset.size > 1)),
        ],
        grid=(threads * (n_heads + n_kv_heads), seq_len, batch),
        threadgroup=(threads, 1, 1),
        output_shapes=[
            (batch, n_heads, seq_len, head_dim),
            (batch, n_kv_heads, seq_len, head_dim),
        ],
        output_dtypes=[q.dtype, q.dtype],
    )
