"""One token through its routed affine 4-bit SwiGLU experts, in two kernels.

Ported from aimd's MLX shim (https://github.com/shuuul/aimd,
``crates/aimd-mlx/shim/shim.cpp``), where these kernels originated.
"""

from typing import Optional, Sequence

import mlx.core as mx

# mlx ``qmv_impl`` layout: a row of N inputs is read in 256-value blocks; in
# block i lane l takes u32 word l + 32 i (8 values), loads the matching 8
# activations (float, pre-scaled by 16^-j so nibbles are masked in place, mlx
# ``load_vector`` / ``qdot``), and applies them to all M x R rows it covers.
# Grouping (fours), order and rounding follow ``gather_qmm`` exactly; the
# per-row sum does not depend on R or M.
_QDOT_HEADER = """
template <int N, int GS, int R, int M, typename T, typename S>
inline void qdot_rows(const device T* x, const device uint32_t* const (&w)[M],
                      const device S* const (&s)[M],
                      const device S* const (&b)[M], thread float (&out)[M * R],
                      uint lane) {
    constexpr int W = N / 8;
    constexpr int G = N / GS;
    for (int r = 0; r < M * R; ++r) out[r] = 0.0f;
    for (int c = int(lane); c < W; c += 32) {
        float xr[8];
        float xsum = 0.0f;
        for (int j = 0; j < 8; j += 4) {
            float x0 = float(x[c * 8 + j]), x1 = float(x[c * 8 + j + 1]);
            float x2 = float(x[c * 8 + j + 2]), x3 = float(x[c * 8 + j + 3]);
            xsum += x0 + x1 + x2 + x3;
            xr[j] = x0;
            xr[j + 1] = x1 / 16.0f;
            xr[j + 2] = x2 / 256.0f;
            xr[j + 3] = x3 / 4096.0f;
        }
        int g = (c * 8) / GS;
        uint32_t q[M * R];
        for (int m = 0; m < M; ++m) {
            for (int r = 0; r < R; ++r) {
                q[m * R + r] = w[m][r * W + c];
            }
        }
        for (int m = 0; m < M; ++m) {
            for (int r = 0; r < R; ++r) {
                uint32_t qq = q[m * R + r];
                float qx = 0.0f;
                for (int j = 0; j < 8; j += 4) {
                    uint32_t w16 = (qq >> (4 * j)) & 0xFFFFu;
                    qx += xr[j] * float(w16 & 0x000Fu) +
                          xr[j + 1] * float(w16 & 0x00F0u) +
                          xr[j + 2] * float(w16 & 0x0F00u) +
                          xr[j + 3] * float(w16 & 0xF000u);
                }
                out[m * R + r] +=
                    float(s[m][r * G + g]) * qx + float(b[m][r * G + g]) * xsum;
            }
        }
    }
    for (int r = 0; r < M * R; ++r) out[r] = simd_sum(out[r]);
}

// mlx ``Sigmoid`` in T arithmetic (each op rounds to T), then the compiled
// ``swiglu``'s two multiplies.
template <typename T>
inline T swiglu_t(float g, float u) {
    T gt = T(g);
    T e = T(metal::precise::exp(metal::abs(float(gt))));
    T y = T(1.0f / float(T(1.0f + float(e))));
    T sig = float(gt) < 0.0f ? y : T(1.0f - float(y));
    T act = T(float(gt) * float(sig));
    return T(float(act) * float(T(u)));
}
"""

# Threadgroup: 2 simdgroups x 4 rows of one expert, gate and up together.
# Grid z < K is routed expert inds[z]; z >= K covers the shared MLP's rows
# (I2 of them, I per slot) after the routed intermediates in h.
_GATE_UP_SOURCE = """
    constexpr int RS = 4;
    uint lane = thread_index_in_simdgroup;
    uint z = threadgroup_position_in_grid.z;
    uint row0 = (threadgroup_position_in_grid.y * 2 +
                 simdgroup_index_in_threadgroup) * RS;
    constexpr int W = D / 8;
    constexpr int G = D / GS;
    float acc[2 * RS];
    if (z < K) {
        size_t base = size_t(inds[z]) * I + row0;
        const device uint32_t* const wp[2] = {gw + base * W, uw + base * W};
        const device S* const sp[2] = {gs + base * G, us + base * G};
        const device S* const bp[2] = {gb + base * G, ub + base * G};
        qdot_rows<D, GS, RS, 2>(x, wp, sp, bp, acc, lane);
        row0 += z * I;
    } else {
        size_t srow0 = size_t(z - K) * I + row0;
        if (srow0 + RS > I2) {
            return;
        }
        const device uint32_t* const wp[2] = {sgw + srow0 * W, suw + srow0 * W};
        const device S* const sp[2] = {sgs + srow0 * G, sus + srow0 * G};
        const device S* const bp[2] = {sgb + srow0 * G, sub + srow0 * G};
        qdot_rows<D, GS, RS, 2>(x, wp, sp, bp, acc, lane);
        row0 = K * I + srow0;
    }
    if (lane < RS) {
        float g = acc[0], u = acc[RS];
        for (int j = 1; j < RS; ++j) {
            if (int(lane) == j) { g = acc[j]; u = acc[RS + j]; }
        }
        h[row0 + lane] = swiglu_t<T>(g, u);
    }
"""

# Threadgroup: K simdgroups (one per selected expert) x 4 output rows, plus
# two for the shared MLP's down projection (2 rows each over its I2-wide
# intermediate). The results meet in threadgroup memory for the
# score-weighted sum, which rounds to T per step like mlx's T-typed ``sum``,
# then the shared output and up to two addends follow, each rounding like
# mlx ``add``.
_DOWN_SOURCE = """
    constexpr int RS = 4;
    constexpr int SH = I2 > 0 ? 1 : 0;
    threadgroup float part[K + SH][RS];
    uint k = simdgroup_index_in_threadgroup;
    uint lane = thread_index_in_simdgroup;
    uint row0 = threadgroup_position_in_grid.y * RS;
    if (k < K) {
        constexpr int W = I / 8;
        constexpr int G = I / GS;
        size_t base = size_t(inds[k]) * D + row0;
        float acc[RS];
        const device uint32_t* const wp[1] = {dw + base * W};
        const device S* const sp[1] = {ds + base * G};
        const device S* const bp[1] = {db + base * G};
        qdot_rows<I, GS, RS, 1>(h + k * I, wp, sp, bp, acc, lane);
        if (lane < RS) {
            float d = acc[0];
            for (int j = 1; j < RS; ++j) {
                if (int(lane) == j) d = acc[j];
            }
            part[k][lane] = float(T(float(T(d)) * float(scores[k])));
        }
    } else if (SH) {
        constexpr int W2 = I2 / 8;
        constexpr int G2 = I2 / GS;
        uint hidx = k - K;
        size_t base = size_t(row0 + 2 * hidx);
        float acc[2];
        const device uint32_t* const wp[1] = {sdw + base * W2};
        const device S* const sp[1] = {sds + base * G2};
        const device S* const bp[1] = {sdb + base * G2};
        qdot_rows<I2, GS, 2, 1>(h + K * I, wp, sp, bp, acc, lane);
        if (lane < 2) {
            part[K][2 * hidx + lane] = float(T(lane == 0 ? acc[0] : acc[1]));
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (k == 0 && lane < RS) {
        T acc2 = T(0.0f);
        for (int j = 0; j < K; ++j) {
            acc2 = T(float(acc2) + part[j][lane]);
        }
        if (SH) acc2 = T(float(acc2) + part[K][lane]);
        if (NADD > 0) acc2 = T(float(acc2) + float(a0[row0 + lane]));
        if (NADD > 1) acc2 = T(float(acc2) + float(a1[row0 + lane]));
        y[row0 + lane] = acc2;
    }
"""

_gate_up_kernel = mx.fast.metal_kernel(
    name="moe_gate_up_4bit",
    input_names=[
        "x",
        "inds",
        "gw",
        "gs",
        "gb",
        "uw",
        "us",
        "ub",
        "sgw",
        "sgs",
        "sgb",
        "suw",
        "sus",
        "sub",
    ],
    output_names=["h"],
    header=_QDOT_HEADER,
    source=_GATE_UP_SOURCE,
)

_down_kernel = mx.fast.metal_kernel(
    name="moe_down_4bit",
    input_names=[
        "h",
        "inds",
        "scores",
        "dw",
        "ds",
        "db",
        "sdw",
        "sds",
        "sdb",
        "a0",
        "a1",
    ],
    output_names=["y"],
    header=_QDOT_HEADER,
    source=_DOWN_SOURCE,
)


def moe_swiglu_4bit(
    x: mx.array,
    inds: mx.array,
    scores: mx.array,
    gate: Sequence[mx.array],
    up: Sequence[mx.array],
    down: Sequence[mx.array],
    *,
    group_size: int,
    shared: Optional[Sequence[Sequence[mx.array]]] = None,
    addends: Optional[Sequence[mx.array]] = None,
) -> mx.array:
    """``sum_k scores[k] * down_e(silu(gate_e(x)) * up_e(x))`` with
    ``e = inds[k]`` over affine 4-bit expert stacks, then the shared SwiGLU
    MLP of x and each addend, added in that order. x has D values (any
    shape; the result has the same), inds K uint32 values (in range; not
    bounds-checked), scores K values in x's dtype. gate/up/down are each
    ``(weight, scales, biases)`` shaped (E, out, in/8) uint32 and (E, out,
    in/group_size) (2-d for one expert), with gate/up out=I, in=D and down
    out=D, in=I. shared,
    when given, is the dense MLP's ``(gate, up, down)`` triples shaped as
    one expert with intermediate I2. Two kernels replace the gather_qmm,
    swiglu, weighting, sum, shared MLP and add dispatches, and the values
    are bit-identical to them. Addends (at most two, D values in x's dtype)
    round like mlx ``add``."""
    gw, gs, gb = gate
    uw, us, ub = up
    dw, ds, db = down
    adds = list(addends or [])
    if inds.dtype != mx.uint32 or scores.dtype != x.dtype:
        raise ValueError("moe_swiglu_4bit: inds must be uint32, scores x's dtype")
    if gw.ndim not in (2, 3) or uw.ndim != gw.ndim or dw.ndim != gw.ndim:
        raise ValueError("moe_swiglu_4bit: expert stacks must be 2-d or 3-d")
    D = x.size
    I = gw.shape[-2]
    K = inds.size
    if (
        gw.shape[-1] * 8 != D
        or uw.shape != gw.shape
        or dw.shape[-2] != D
        or dw.shape[-1] * 8 != I
        or dw.size != gw.size // (I * (D // 8)) * D * (I // 8)
        or scores.size != K
        or group_size <= 0
        or group_size % 8
        or D % group_size
        or I % group_size
        or I % 8
        or D % 4
        or not 1 <= K <= 30
    ):
        raise ValueError("moe_swiglu_4bit: bad 4-bit expert shapes")
    if shared is None:
        I2 = 0
        sgate, sup, sdown = gate, up, down
    else:
        sgate, sup, sdown = shared
        I2 = sgate[0].shape[-2]
        if (
            sgate[0].shape[-2:] != (I2, D // 8)
            or sup[0].shape[-2:] != (I2, D // 8)
            or sdown[0].shape[-2:] != (D, I2 // 8)
            or I2 % group_size
            or I2 % 8
            or any(a.size != a.shape[-2] * a.shape[-1] for t in shared for a in t)
        ):
            raise ValueError("moe_swiglu_4bit: bad shared MLP shapes")
    if len(adds) > 2 or any(a.size != D or a.dtype != x.dtype for a in adds):
        raise ValueError("moe_swiglu_4bit: at most two D-sized addends of x's dtype")
    n_add = len(adds)
    # Unused addend slots take any D-sized array of x's dtype.
    while len(adds) < 2:
        adds.append(x)
    template = [
        ("D", D),
        ("I", I),
        ("I2", I2),
        ("K", K),
        ("GS", int(group_size)),
        ("NADD", n_add),
        ("T", x.dtype),
        ("S", gs.dtype),
    ]
    shared_slots = -(-I2 // I)
    h = _gate_up_kernel(
        inputs=[x, inds, gw, gs, gb, uw, us, ub, *sgate, *sup],
        template=template,
        grid=(64, I // 8, K + shared_slots),
        threadgroup=(64, 1, 1),
        output_shapes=[(K * I + I2,)],
        output_dtypes=[x.dtype],
    )[0]
    simdgroups = K + (2 if I2 else 0)
    return _down_kernel(
        inputs=[h, inds, scores, dw, ds, db, *sdown, adds[0], adds[1]],
        template=template,
        grid=(32 * simdgroups, D // 4, 1),
        threadgroup=(32 * simdgroups, 1, 1),
        output_shapes=[x.shape],
        output_dtypes=[x.dtype],
    )[0]
