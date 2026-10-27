# Copyright © 2026 Apple Inc.

"""The cached ShortConv operator over several tokens (B * x, the causal depthwise conv over the
state and the new tokens, the C gate, and the state shift) in one kernel.

It matches the standard ops bitwise: B * x is rounded as the materialized product is, the conv
accumulates the taps in order with fma in float32 and rounds once, as mx.conv1d's depthwise kernel
does, and the gate rounds once more. The single-token step is the ShortConv step kernel.
"""

import mlx.core as mx

# L tokens. Row r of the padded sequence is state row r for r < K - 1, else the B * x of
# token r - (K - 1); output t reads rows t .. t + K - 1, and the new state is the last K - 1
# rows. bcx is (B, L, 3D) and state (B, K - 1, D), both row contiguous.
_SEQ_SOURCE = """
    const uint c = thread_position_in_grid.x;
    const uint t = thread_position_in_grid.y;
    const uint b = thread_position_in_grid.z;
    const uint L = bcx_shape[1];
    if (c >= D) return;

    const device T* rows = bcx + size_t(b) * L * 3 * D;
    // ROW(r): the padded row r, with B * x rounded to T as the materialized product is.
#define ROW(r)                                                                  \\
    ((r) < K - 1                                                                \\
         ? (HAS_STATE ? state[(size_t(b) * (K - 1) + (r)) * D + c] : T(0))      \\
         : static_cast<T>(                                                      \\
               static_cast<float>(rows[size_t((r) - (K - 1)) * 3 * D + c]) *    \\
               static_cast<float>(rows[size_t((r) - (K - 1)) * 3 * D + 2 * D + c])))

    if (t < L) {
        const device T* wc = weight + c * K;
        float acc = 0.0f;
        for (uint i = 0; i < K; ++i) {
            acc = metal::fma(static_cast<float>(ROW(t + i)), static_cast<float>(wc[i]), acc);
        }
        T conv = static_cast<T>(acc);
        if (HAS_BIAS) {
            conv = static_cast<T>(static_cast<float>(conv) + static_cast<float>(bias[c]));
        }
        y[(size_t(b) * L + t) * D + c] = static_cast<T>(
            static_cast<float>(conv) * static_cast<float>(rows[size_t(t) * 3 * D + D + c]));
    }

    if (t < K - 1) {
        new_state[(size_t(b) * (K - 1) + t) * D + c] = ROW(L + t);
    }
#undef ROW
"""

_kernel = None


def short_conv_sequence(bcx, state, weight, bias=None):
    """y = C * conv1d(concat(state, B * x)) and the new state, for bcx (B, L, 3D) holding
    [B | C | x], state (B, K - 1, D) or None (zeros) and a depthwise weight (D, K, 1).
    Returns y (B, L, D) and the new state (B, K - 1, D)."""
    global _kernel
    if _kernel is None:
        _kernel = mx.fast.metal_kernel(
            name="lfm2_short_conv_seq",
            input_names=["bcx", "state", "weight", "bias"],
            output_names=["y", "new_state"],
            source=_SEQ_SOURCE,
        )
    batch, length, three_d = bcx.shape
    dim = three_d // 3
    k = weight.shape[1]
    dtype = bcx.dtype
    return _kernel(
        inputs=[
            bcx,
            state if state is not None else weight,
            weight,
            bias if bias is not None else weight,
        ],
        template=[
            ("T", dtype),
            ("D", dim),
            ("K", k),
            ("HAS_BIAS", bias is not None),
            ("HAS_STATE", state is not None),
        ],
        grid=(dim, max(length, k - 1), batch),
        threadgroup=(min(256, dim), 1, 1),
        output_shapes=[(batch, length, dim), (batch, k - 1, dim)],
        output_dtypes=[dtype, dtype],
    )
