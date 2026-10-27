"""Metal kernels for Unlimited-OCR-2 DASH attention.

``dash_prefill`` is block-sparse prefill attention: each simdgroup takes 8
queries, walks the union of their selected blocks and local window in key
order and runs QK^T / PV on 8x8 simdgroup matrices with an online softmax, so
the cost follows the attended keys instead of the [L, S] grid.

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

_PREFILL_SOURCE = """
    constexpr int NF = D / 8;             // 8-wide dim fragments
    constexpr int NBK = BK / 8;           // 8-key fragments per staged chunk
    constexpr int TQ = 8 * NSG;           // queries per threadgroup tile
    constexpr int TGS = 32 * NSG;
    threadgroup atomic_uint bits[(1 + NSG) * MAXW];  // tile union + per simdgroup
    threadgroup int tsel[TQ * K];
    threadgroup float tbias[TQ * K];
    constexpr int LD = D + 4;             // padded rows: no bank conflicts
    threadgroup float ks[BK * LD];
    threadgroup float vs[BK * LD];

    uint t = threadgroup_position_in_grid.x;
    uint h = threadgroup_position_in_grid.y;
    uint tid = thread_position_in_threadgroup.x;
    uint sg = simdgroup_index_in_threadgroup;
    uint lane = thread_index_in_simdgroup;
    int offset = params[0];
    int L = params[1];
    float scale = params_f[0];
    int t0 = (offset / 32) * 32 + TQ * int(t);
    int a0 = t0 + 8 * int(sg);
    int amin = metal::max(a0, offset);
    int amax = metal::min(a0 + 7, offset + L - 1);
    int tmin = metal::max(t0, offset);
    int tmax = metal::min(t0 + TQ - 1, offset + L - 1);
    threadgroup atomic_uint* ubits = bits;
    threadgroup atomic_uint* sbits = bits + (1 + sg) * MAXW;

    // Union of the blocks the tile (and each simdgroup) attends, as bitmaps.
    for (int w = tid; w < (1 + NSG) * MAXW; w += TGS) {
        atomic_store_explicit(&bits[w], 0u, memory_order_relaxed);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (lane < 8) {
        int rel = a0 + int(lane) - offset;
        bool ok = rel >= 0 && rel < L;
        for (int i = 0; i < K; i++) {
            int b = ok ? topi[rel * K + i] : -1;
            tsel[(8 * sg + lane) * K + i] = b;
            tbias[(8 * sg + lane) * K + i] = ok ? bias[rel * K + i] : 0.0f;
            if (b >= 0) {
                atomic_fetch_or_explicit(&sbits[b / 32], 1u << (b % 32), memory_order_relaxed);
                atomic_fetch_or_explicit(&ubits[b / 32], 1u << (b % 32), memory_order_relaxed);
            }
        }
    }
    if (amin <= amax) {
        for (int b = metal::max(amin - LW + 1, 0) / BS + lane; b <= amax / BS; b += 32) {
            atomic_fetch_or_explicit(&sbits[b / 32], 1u << (b % 32), memory_order_relaxed);
            atomic_fetch_or_explicit(&ubits[b / 32], 1u << (b % 32), memory_order_relaxed);
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // 8x8 fragment coordinates of this lane: (fm, fn) and (fm, fn + 1).
    short qid = lane / 4;
    short fm = (qid & 4) + ((lane / 2) % 4);
    short fn = (qid & 2) * 2 + (lane % 2) * 2;
    int a = a0 + fm;
    int rel = a - offset;
    bool row_ok = rel >= 0 && rel < L;
    bool sg_live = amin <= amax;
    int S_alloc = keys_shape[2];
    int kvh = h / (HQ / HK);
    auto kh = keys + size_t(kvh) * S_alloc * D;
    auto vh = values + size_t(kvh) * S_alloc * D;
    auto qr = q + (size_t(h) * L + (row_ok ? rel : 0)) * D;

    constexpr int LQ = D + 4;
    threadgroup float qs[TQ * LQ];
    simdgroup_matrix<float, 8, 8> Of[NF];
    simdgroup_matrix<float, 8, 8> Sf[NBK];
    for (int c = 0; c < NF; c++) {
        qs[(8 * sg + fm) * LQ + 8 * c + fn] = row_ok ? static_cast<float>(qr[8 * c + fn]) * scale : 0.0f;
        qs[(8 * sg + fm) * LQ + 8 * c + fn + 1] = row_ok ? static_cast<float>(qr[8 * c + fn + 1]) * scale : 0.0f;
        Of[c] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
    }
    float m = -INFINITY;
    float l = 0.0f;

    // The tile's blocks in key order.
    threadgroup int ulist[TQ * K + 8];
    threadgroup int nu_s;
    if (tid == 0) {
        int n = 0;
        for (int w = 0; w <= tmax / BS / 32; w++) {
            uint word = atomic_load_explicit(&ubits[w], memory_order_relaxed);
            while (word != 0u) {
                ulist[n++] = w * 32 + ctz(word);
                word &= word - 1u;
            }
        }
        nu_s = n;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    constexpr int CPB = BS / BK;
    constexpr int EPT = BK * D / TGS;     // staged elements per thread
    int nchunks = nu_s * CPB;
    T kr[EPT], vr[EPT];
    {
        int k0 = ulist[0] * BS;
        for (int e = 0; e < EPT; e++) {
            kr[e] = kh[size_t(k0) * D + tid * EPT + e];
            vr[e] = vh[size_t(k0) * D + tid * EPT + e];
        }
    }
    for (int ci = 0; ci < nchunks; ci++) {
        int b = ulist[ci / CPB];
        int k0 = b * BS + (ci % CPB) * BK;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (int e = 0; e < EPT; e++) {
            int i = tid * EPT + e;
            ks[(i / D) * LD + i % D] = static_cast<float>(kr[e]);
            vs[(i / D) * LD + i % D] = static_cast<float>(vr[e]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (ci + 1 < nchunks) {
            // Prefetch the next chunk while this one computes.
            int n0 = ulist[(ci + 1) / CPB] * BS + ((ci + 1) % CPB) * BK;
            for (int e = 0; e < EPT; e++) {
                kr[e] = kh[size_t(n0) * D + tid * EPT + e];
                vr[e] = vh[size_t(n0) * D + tid * EPT + e];
            }
        }
        uint mine = atomic_load_explicit(&sbits[b / 32], memory_order_relaxed);
        if (!sg_live || !((mine >> (b % 32)) & 1u)) {
            continue;
        }
        float badd = -INFINITY;
        for (int i = 0; i < K; i++) {
            int r = (8 * sg + fm) * K + i;
            badd = tsel[r] == b ? tbias[r] : badd;
        }
        {
            {
                // S = (scale Q) K^T over the chunk.
                for (int j = 0; j < NBK; j++) {
                    Sf[j] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
                    for (int c = 0; c < NF; c++) {
                        simdgroup_matrix<float, 8, 8> Qf, Kf;
                        simdgroup_load(Qf, qs + 8 * sg * LQ + 8 * c, LQ);
                        simdgroup_load(Kf, ks + 8 * j * LD + 8 * c, LD, ulong2(0, 0), true);
                        simdgroup_multiply_accumulate(Sf[j], Qf, Kf, Sf[j]);
                    }
                }

                // Local window unbiased, remote keys get the block bias, causal.
                float rm = -INFINITY;
                for (int j = 0; j < NBK; j++) {
                    for (int e = 0; e < 2; e++) {
                        int key = k0 + 8 * j + fn + e;
                        float add = key > a ? -INFINITY : (key > a - LW ? 0.0f : badd);
                        float sv = row_ok ? Sf[j].thread_elements()[e] + add : -INFINITY;
                        Sf[j].thread_elements()[e] = sv;
                        rm = metal::max(rm, sv);
                    }
                }
                rm = metal::max(rm, simd_shuffle_xor(rm, ushort(1)));
                rm = metal::max(rm, simd_shuffle_xor(rm, ushort(8)));
                float mn = metal::max(m, rm);
                float factor = m == -INFINITY ? 0.0f : metal::fast::exp(m - mn);
                float rs = 0.0f;
                for (int j = 0; j < NBK; j++) {
                    for (int e = 0; e < 2; e++) {
                        float sv = Sf[j].thread_elements()[e];
                        float pe = sv == -INFINITY ? 0.0f : metal::fast::exp(sv - mn);
                        Sf[j].thread_elements()[e] = pe;
                        rs += pe;
                    }
                }
                rs += simd_shuffle_xor(rs, ushort(1));
                rs += simd_shuffle_xor(rs, ushort(8));
                l = l * factor + rs;
                m = mn;

                // O = factor O + P V.
                for (int c = 0; c < NF; c++) {
                    Of[c].thread_elements()[0] *= factor;
                    Of[c].thread_elements()[1] *= factor;
                    for (int j = 0; j < NBK; j++) {
                        simdgroup_matrix<float, 8, 8> Vf;
                        simdgroup_load(Vf, vs + 8 * j * LD + 8 * c, LD);
                        simdgroup_multiply_accumulate(Of[c], Sf[j], Vf, Of[c]);
                    }
                }
            }
        }
    }
    if (row_ok) {
        float inv = 1.0f / l;
        auto orow = out + (size_t(h) * L + rel) * D;
        for (int c = 0; c < NF; c++) {
            orow[8 * c + fn] = static_cast<T>(Of[c].thread_elements()[0] * inv);
            orow[8 * c + fn + 1] = static_cast<T>(Of[c].thread_elements()[1] * inv);
        }
    }
"""

# Bitmap words per simdgroup: prompts of up to 32 * MAXW blocks.
_PREFILL_MAXW = 128
_PREFILL_NSG = 4
_PREFILL_BK = 8

_prefill_kernel = (
    mx.fast.metal_kernel(
        name="dash_prefill",
        input_names=["q", "keys", "values", "topi", "bias", "params", "params_f"],
        output_names=["out"],
        source=_PREFILL_SOURCE,
        header="#include <metal_simdgroup_matrix>\n",
    )
    if _HAS_METAL
    else None
)


def prefill_supported(p, head_dim: int, total: int) -> bool:
    return (
        _prefill_kernel is not None
        and mx.default_device() == mx.gpu
        and head_dim % 8 == 0
        and p.block_size % 8 == 0
        and 8192 // (head_dim * 4) >= 8
        and -(-total // p.block_size) <= 32 * _PREFILL_MAXW
    )


def dash_prefill(q, keys, values, topi, bias, valid, offset: int, p, scale: float):
    """Block-sparse prefill attention.

    ``q`` [1, HQ, L, D] (rotated), ``keys``/``values`` the full cache buffers
    [1, HK, S_alloc, D] holding positions ``< offset + L``; ``topi``/``bias``/
    ``valid`` [L, K] from ``route_select``. Returns [1, HQ, L, D].
    """
    _, hq, L, d = q.shape
    sel = mx.where(valid, topi.astype(mx.int32), -1)
    # The reference adds the bias through a mask in the query dtype.
    b = bias.astype(q.dtype).astype(mx.float32)
    nsg = _PREFILL_NSG
    tiles = -(-(offset + L - offset // 32 * 32) // (8 * nsg))
    return _prefill_kernel(
        inputs=[
            q,
            keys,
            values,
            sel,
            b,
            mx.array([offset, L], dtype=mx.int32),
            mx.array([scale], dtype=mx.float32),
        ],
        template=[
            ("T", q.dtype),
            ("D", d),
            ("HQ", hq),
            ("HK", keys.shape[1]),
            ("BS", p.block_size),
            ("LW", p.local_window),
            ("K", topi.shape[-1]),
            ("MAXW", _PREFILL_MAXW),
            # Staged keys per chunk: f32 K and V chunks fit in 16 KB.
            ("BK", _PREFILL_BK),
            ("NSG", nsg),
        ],
        grid=(tiles * 32 * nsg, hq, 1),
        threadgroup=(32 * nsg, 1, 1),
        output_shapes=[q.shape],
        output_dtypes=[q.dtype],
    )[0]


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
