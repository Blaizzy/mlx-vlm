"""Metal kernels for Unlimited-OCR-2 DASH attention.

``dash_decode`` fuses one decode step of a layer: block routing
(``_decode_select``) and the gathered attention over the selected blocks plus
the local window (``DashAttention._decode``), reading keys and values straight
from the cache buffer. One threadgroup per head; each repeats the (cheap)
routing so the step is a single launch.

Ported from aimd's MLX shim (https://github.com/shuuul/aimd,
``crates/aimd-mlx/shim/shim.cpp``), where these kernels originated.
"""

import math

import mlx.core as mx

_HAS_METAL = mx.metal.is_available()

# Threadgroup memory budget: the routing arrays hold CAP floats + CAP bools.
MAX_CAPACITY = 4096

_DECODE_SOURCE = """
    constexpr int NSG = N / 32;           // one simdgroup per 32 gathered keys
    constexpr int TG = N;
    constexpr int R = D / 32;             // dims per lane
    constexpr int RI = DI / 32;
    constexpr int NREM = K * BS;
    constexpr float SCORE_NEG = -1.0e9f;
    constexpr float ENTMAX_NEG = -1.0e4f;
    constexpr float CLAMP = 1.0e4f;

    threadgroup float xs[CAP];
    threadgroup bool sel[CAP];
    threadgroup int top[TOPM];
    threadgroup float bias_s[K];
    threadgroup bool valid_s[K];
    threadgroup int flags[NSG];
    threadgroup float sg_m[NSG];
    threadgroup float sg_l[NSG];
    threadgroup float sg_o[NSG * D];

    uint h = threadgroup_position_in_grid.x;
    uint tid = thread_position_in_threadgroup.x;
    uint lane = thread_index_in_simdgroup;
    uint sg = simdgroup_index_in_threadgroup;
    int p = pos[0];
    int last_remote = p / BS - LB;

    // 1. Block scores and the candidate set.
    float qi[RI];
    for (int i = 0; i < RI; i++) {
        qi[i] = idx_q[lane + 32 * i];
    }
    bool any_local = false;
    for (int b = sg; b < CAP; b += NSG) {
        float acc = 0.0f;
        for (int i = 0; i < RI; i++) {
            acc += qi[i] * summaries[size_t(b) * DI + lane + 32 * i];
        }
        acc = simd_sum(acc);
        if (lane == 0) {
            float z = metal::clamp(acc * consts[3], -CLAMP, CLAMP);
            bool remote = b <= last_remote;
            bool s = remote && cand[b];
            xs[b] = z;
            sel[b] = s;
            any_local = any_local || s;
        }
    }
    any_local = simd_any(any_local);
    if (lane == 0) {
        flags[sg] = any_local ? 1 : 0;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    bool any_sel = false;
    for (int g = 0; g < NSG; g++) {
        any_sel = any_sel || flags[g] != 0;
    }
    float scale = consts[0];
    for (int b = tid; b < CAP; b += TG) {
        bool s = any_sel ? sel[b] : (b <= last_remote);
        sel[b] = s;
        xs[b] = s ? xs[b] * scale : ENTMAX_NEG;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // 2. Stable top-TOPM (descending, ties -> lower index, like argsort).
    for (int b = tid; b < CAP; b += TG) {
        float xb = xs[b];
        int rank = 0;
        for (int j = 0; j < CAP && rank < TOPM; j++) {
            float xj = xs[j];
            rank += (xj > xb || (xj == xb && j < b)) ? 1 : 0;
        }
        if (rank < TOPM) {
            top[rank] = b;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // 3. 1.5-entmax prior over the sorted top scores and the block bias.
    if (tid == 0) {
        float xv[TOPM];
        float tau[TOPM];
        float x0 = xs[top[0]];
        float cs = 0.0f, cs2 = 0.0f;
        for (int i = 0; i < TOPM; i++) {
            float x = (xs[top[i]] - x0) / 2.0f;
            xv[i] = x;
            cs += x;
            cs2 += x * x;
            float rho = float(i + 1);
            float mean = cs / rho;
            float mean_sq = cs2 / rho;
            float ss = rho * (mean_sq - mean * mean);
            float delta = (1.0f - ss) / rho;
            tau[i] = mean - metal::sqrt(metal::max(delta, 0.0f));
        }
        int support = 0;
        for (int i = 0; i < TOPM; i++) {
            support += tau[i] <= xv[i] ? 1 : 0;
        }
        float ts = tau[metal::max(support, 1) - 1];
        float fl = consts[1];
        float logp[K];
        float sum = 0.0f;
        int cnt = 0;
        for (int i = 0; i < K; i++) {
            float pr = metal::max(xv[i] - ts, 0.0f);
            logp[i] = metal::log(metal::max(pr * pr, fl));
            bool v = sel[top[i]];
            valid_s[i] = v;
            sum += v ? logp[i] : 0.0f;
            cnt += v ? 1 : 0;
        }
        float mu = sum / float(metal::max(cnt, 1));
        for (int i = 0; i < K; i++) {
            // The reference adds the bias through a mask in the query dtype.
            float bv = valid_s[i] ? (logp[i] - mu) * consts[4] : 0.0f;
            bias_s[i] = static_cast<float>(static_cast<T>(bv));
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // 4. Gathered attention: simdgroup sg streams keys [32 sg, 32 sg + 32).
    int S = keys_shape[2];
    auto kh = keys + size_t(h / (HQ / HK)) * S * D;
    auto vh = values + size_t(h / (HQ / HK)) * S * D;
    float qv[R];
    for (int i = 0; i < R; i++) {
        qv[i] = static_cast<float>(q[size_t(h) * D + lane + 32 * i]) * consts[2];
    }
    float m = -INFINITY, l = 0.0f;
    float o[R];
    for (int i = 0; i < R; i++) {
        o[i] = 0.0f;
    }
    for (int j = 0; j < 32; j++) {
        int t = sg * 32 + j;
        int key;
        float add;
        bool ok;
        if (t < NREM) {
            int slot = t / BS;
            key = top[slot] * BS + t % BS;
            ok = valid_s[slot] && key <= p - LW;
            add = bias_s[slot];
        } else {
            key = p - LW + 1 + (t - NREM);
            ok = key >= 0;
            add = 0.0f;
        }
        if (!ok) {
            continue;
        }
        float acc = 0.0f;
        for (int i = 0; i < R; i++) {
            acc += qv[i] * static_cast<float>(kh[size_t(key) * D + lane + 32 * i]);
        }
        float s = simd_sum(acc) + add;
        float mn = metal::max(m, s);
        float a = metal::fast::exp(m - mn);
        float e = metal::fast::exp(s - mn);
        l = l * a + e;
        for (int i = 0; i < R; i++) {
            o[i] = o[i] * a + e * static_cast<float>(vh[size_t(key) * D + lane + 32 * i]);
        }
        m = mn;
    }
    if (lane == 0) {
        sg_m[sg] = m;
        sg_l[sg] = l;
    }
    for (int i = 0; i < R; i++) {
        sg_o[sg * D + lane + 32 * i] = o[i];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid < D) {
        float mm = -INFINITY;
        for (int g = 0; g < NSG; g++) {
            mm = metal::max(mm, sg_m[g]);
        }
        float ll = 0.0f, acc = 0.0f;
        for (int g = 0; g < NSG; g++) {
            float w = sg_m[g] == -INFINITY ? 0.0f : metal::fast::exp(sg_m[g] - mm);
            ll += sg_l[g] * w;
            acc += sg_o[g * D + tid] * w;
        }
        out[size_t(h) * D + tid] = static_cast<T>(acc / ll);
    }
"""

_decode_kernel = (
    mx.fast.metal_kernel(
        name="dash_decode",
        input_names=[
            "idx_q",
            "summaries",
            "cand",
            "pos",
            "consts",
            "q",
            "keys",
            "values",
        ],
        output_names=["out"],
        source=_DECODE_SOURCE,
    )
    if _HAS_METAL
    else None
)


def decode_supported(p, n_heads: int, head_dim: int, index_dim: int, cap: int) -> bool:
    n = p.topk * p.block_size + p.local_window
    return (
        _decode_kernel is not None
        and mx.default_device() == mx.gpu
        and n % 32 == 0
        and n <= 1024
        and head_dim % 32 == 0
        and index_dim % 32 == 0
        and min(p.topm, cap) >= p.topk
        and cap <= MAX_CAPACITY
    )


def decode_consts(idx_scale: float, floor: float, scale: float, p, index_dim: int):
    """Per-layer f32 constants of ``dash_decode``."""
    return mx.array(
        [idx_scale, floor, scale, 1.0 / math.sqrt(index_dim), 1.0 / p.sigma],
        dtype=mx.float32,
    )


def dash_decode(q, keys, values, idx_q, summaries, cand, pos, consts, p):
    """One decode step of DASH attention.

    ``q`` [1, HQ, 1, D] (rotated), ``keys``/``values`` the full cache buffers
    [1, HK, S_alloc, D], ``idx_q`` [1, DI] f32, ``summaries`` [cap, DI] f32,
    ``cand`` [1, cap] bool, ``pos`` int32 scalar, ``consts`` from
    ``decode_consts``. Returns [1, HQ, 1, D].
    """
    _, hq, _, d = q.shape
    cap, di = summaries.shape
    n = p.topk * p.block_size + p.local_window
    return _decode_kernel(
        inputs=[
            idx_q,
            summaries,
            cand,
            pos.reshape(1),
            consts,
            q,
            keys,
            values,
        ],
        template=[
            ("T", q.dtype),
            ("N", n),
            ("D", d),
            ("DI", di),
            ("HQ", hq),
            ("HK", keys.shape[1]),
            ("CAP", cap),
            ("BS", p.block_size),
            ("LB", p.local_blocks),
            ("LW", p.local_window),
            ("K", p.topk),
            ("TOPM", min(p.topm, cap)),
        ],
        grid=(hq * n, 1, 1),
        threadgroup=(n, 1, 1),
        output_shapes=[q.shape],
        output_dtypes=[q.dtype],
    )[0]
