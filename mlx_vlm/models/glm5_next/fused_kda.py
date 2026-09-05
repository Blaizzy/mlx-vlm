"""Fused single-token decode step for the GLM-5-Next KDA (Kimi Delta Attention) core.

At B=1, S=1 the post-projection half of ``Glm5NextLinearAttention.__call__`` is a
long tail of tiny elementwise / small-reduction kernels: the causal conv1d window
update, silu, two L2 norms, the forget-gate softplus-free "safe gate", the sigmoid
beta, the gated delta-rule state update and finally the gated RMSNorm.  Each of
those is a separate GPU dispatch, and at 34 KDA layers per decode step the launch
overhead dominates the (tiny) arithmetic.

This module folds *all* of them into ONE ``mx.fast.metal_kernel`` launch per layer.
One threadgroup handles one head, so the two cross-``head_dim`` reductions that the
chain needs (the L2 norms over the key axis and the RMSNorm over the value axis)
both stay inside threadgroup memory.  The recurrent state ``[head_dim, head_dim]``
lives in device memory and is streamed through registers exactly once.

The arithmetic is a line-by-line transcription of the eager path, including where
it rounds to the input dtype: ``mx.conv1d`` writes bfloat16, ``nn.silu`` rounds
twice (sigmoid then product), ``gated_delta_kernel`` writes its output ``y`` in the
input dtype before the gated RMSNorm reads it back as float32, and ``beta`` is a
bfloat16 sigmoid.  State accumulation stays float32, matching the eager kernel.

Not a drop-in for prefill: this is decode-only (B=1, S=1, no SSM mask).
``Glm5NextLinearAttention`` falls back to the eager path whenever any of those
preconditions does not hold.
"""

import logging
from typing import Optional, Tuple

import mlx.core as mx

logger = logging.getLogger(__name__)

# MLX's own elementwise ops, transcribed so the fused kernel rounds identically:
#   mlx/backend/metal/kernels/unary_ops.h -> Sigmoid / Exp / Rsqrt
_HEADER = """
// MLX's Sigmoid (mlx/backend/metal/kernels/unary_ops.h), with two subtleties
// that each cost exactness if ignored:
//
//  * Instantiate on the SAME type the eager op used.  Metal's native `bfloat`
//    rounds after every arithmetic step, so evaluating in float32 and rounding
//    once at the end disagrees with mx.sigmoid(bfloat16) on ~15% of elements.
//
//  * Pick the right exp.  MLX's Sigmoid is written with `metal::exp`, which
//    resolves to the precise implementation inside MLX's *precompiled* Metal
//    library but to the fast approximation inside a JIT'd kernel (mx.compile
//    and mx.fast.metal_kernel share that JIT).  In the KDA chain both appear:
//    `nn.silu` and `compute_g_safe` are mx.compile'd (-> _fast), while the beta
//    sigmoid and the output-gate sigmoid are plain ops (-> _precise).  MLX's
//    Exp op spells `metal::precise::exp` in source, so mx.exp is precise in
//    both.  Mixing them up is a 1-ulp disagreement on ~0.1-50% of elements.
template <typename U>
inline U mlx_sigmoid_precise(U x) {
  U e = static_cast<U>(metal::precise::exp(metal::abs(x)));
  U y = static_cast<U>(1) / (static_cast<U>(1) + e);
  return (x < 0) ? y : (static_cast<U>(1) - y);
}

template <typename U>
inline U mlx_sigmoid_fast(U x) {
  U e = static_cast<U>(metal::exp(metal::abs(x)));
  U y = static_cast<U>(1) / (static_cast<U>(1) + e);
  return (x < 0) ? y : (static_cast<U>(1) - y);
}

// `(x * x).sum(-1)` in MLX materialises the squares before reducing, so the
// multiply rounds before the add.  Writing `acc += v * v` here would let the
// compiler contract it into an fma and silently change the last bit of every
// L2 / RMS norm, so contraction is disabled for this one helper.
#pragma clang fp contract(off)
inline float sq_acc(float acc, float v) {
  return v * v + acc;
}
#pragma clang fp contract(on)

// `a + dt_bias` must promote exactly the way MLX's binary op does, and the two
// operands need not share a dtype: glm5_next's cast_predicate leaves A_log /
// dt_bias in whatever the conversion produced (float32 or the model dtype).
// Metal's promotion matches MLX's here -- bfloat + float -> float, bfloat +
// bfloat -> bfloat -- so plain `+` on the stored types is the faithful form,
// and evaluating it in float32 unconditionally would NOT be.
template <typename U, typename V>
inline float promote_add(U a, V b) {
  return float(a + b);
}
"""

_SOURCE = """
  const uint h    = threadgroup_position_in_grid.z;
  const uint lane = thread_position_in_threadgroup.x;
  const uint ty   = thread_position_in_threadgroup.y;
  const uint tid  = thread_index_in_threadgroup;

  constexpr int NT   = 32 * TY;      // threads per threadgroup
  constexpr int RBLK = D / 128;      // MLX row-reduce blocks (32 lanes x 4 reads)
  constexpr int REXTRA = D - RBLK * 128;
  constexpr int NDK  = D / 32;       // key-dim elements held per thread
  constexpr int NDV  = D / TY;       // value-dim rows walked per thread
  constexpr uint QKVD = (uint)(H * D);
  constexpr uint CDIM = 3u * QKVD;   // conv1d channel count

  threadgroup float sq[D];
  threadgroup float sk[D];
  threadgroup float sv[D];
  threadgroup float sg[D];
  threadgroup float sgate[D];
  threadgroup float sy[D];
  threadgroup float shr[3];          // 0: rsqrt(q), 1: rsqrt(k), 2: beta

  // Issue the recurrent-state read first: it is 4 MB per layer and by far the
  // longest-latency operation here, so it overlaps the conv / gate / norm work
  // below instead of starting after three threadgroup barriers.
  device const ST* si = state_in  + (size_t)h * D * D;
  device ST*       so = state_out + (size_t)h * D * D;
  float st[NDV][NDK];
  for (int j = 0; j < NDV; ++j) {
    uint dv = ty + (uint)TY * (uint)j;
    for (int i = 0; i < NDK; ++i) {
      st[j][i] = float(si[(size_t)dv * D + NDK * lane + i]);
    }
  }

  // ---------------------------------------------------------------- phase 0a
  // Depthwise causal conv1d over the K-tap window [conv_state ; x_t], then silu,
  // and shift the cached window.  This head owns 3*D of the 3*H*D channels
  // (one slice each of the q / k / v thirds of the fused in-projection).
  for (uint idx = tid; idx < 3u * (uint)D; idx += NT) {
    uint part = idx / (uint)D;
    uint d    = idx - part * (uint)D;
    uint c    = part * QKVD + h * (uint)D + d;
    device const T* wc = conv_w + (size_t)c * K;
    float acc = 0.0f;
    for (uint j = 0; j + 1 < (uint)K; ++j) {
      acc += float(conv_state[(size_t)j * CDIM + c]) * float(wc[j]);
    }
    T xnew = (part == 0u) ? mq[h * (uint)D + d]
           : ((part == 1u) ? mk[h * (uint)D + d] : mv[h * (uint)D + d]);
    acc += float(xnew) * float(wc[K - 1]);

    T xb  = static_cast<T>(acc);           // mx.conv1d writes its output in T
    T sig = mlx_sigmoid_fast(xb);          // nn.silu = x * mx.sigmoid(x), compiled
    T sl  = xb * sig;
    if (part == 0u)      sq[d] = float(sl);
    else if (part == 1u) sk[d] = float(sl);
    else                 sv[d] = float(sl);

    // new window = [old[1 .. K-2], x_t]
    for (uint j = 0; j + 2 < (uint)K; ++j) {
      conv_state_out[(size_t)j * CDIM + c] = conv_state[(size_t)(j + 1) * CDIM + c];
    }
    conv_state_out[(size_t)(K - 2) * CDIM + c] = xnew;
  }

  // ---------------------------------------------------------------- phase 0b
  // Safe forget gate  g = exp(lb * sigmoid(exp(A_log) * (a + dt_bias)))  in fp32,
  // beta = sigmoid(b) rounded to T, and the output gate pulled into shared mem.
  {
    // mx.exp(A_log.astype(float32)) -> the cast is explicit in the eager path,
    // so this is float regardless of how A_log was stored.
    float a_exp = metal::precise::exp(float(A_log[h]));
    for (uint d = tid; d < (uint)D; d += NT) {
      float av = promote_add(a[h * (uint)D + d], dt_bias[h * (uint)D + d]);
      sg[d]    = metal::precise::exp(lower_bound * mlx_sigmoid_fast<float>(a_exp * av));
      sgate[d] = float(gate[h * (uint)D + d]);
    }
    if (tid == 0u) {
      shr[2] = float(mlx_sigmoid_precise(bvec[h]));  // beta = mx.sigmoid(b), in T
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  // ---------------------------------------------------------------- phase 0c
  // q = l2norm(q) * D^-0.5 ; k = l2norm(k), both rounded back to T.
  if (simdgroup_index_in_threadgroup == 0u) {
    // Same partition and same accumulation order as MLX's row_reduce_simple
    // (N_READS = 4 contiguous elements per lane, then simd_sum), so the fp32
    // sum is bit-identical to `(x * x).sum(-1)`.
    float pq = 0.0f, pk = 0.0f;
    for (int blk = 0; blk < RBLK; ++blk) {
      uint base = (uint)(blk * 128) + 4u * lane;
      for (int i = 0; i < 4; ++i) {
        pq = sq_acc(pq, sq[base + i]);
        pk = sq_acc(pk, sk[base + i]);
      }
    }
    uint base = (uint)(RBLK * 128) + 4u * lane;
    if (4u * lane + 4u <= (uint)REXTRA) {
      for (int i = 0; i < 4; ++i) {
        pq = sq_acc(pq, sq[base + i]);
        pk = sq_acc(pk, sk[base + i]);
      }
    } else {
      for (int i = 0; 4u * lane + (uint)i < (uint)REXTRA; ++i) {
        pq = sq_acc(pq, sq[base + i]);
        pk = sq_acc(pk, sk[base + i]);
      }
    }
    pq = simd_sum(pq);
    pk = simd_sum(pk);
    if (lane == 0u) {
      shr[0] = metal::precise::rsqrt(pq + 1.0e-6f);
      shr[1] = metal::precise::rsqrt(pk + 1.0e-6f);
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  {
    float rq = shr[0], rk = shr[1];
    for (uint d = tid; d < (uint)D; d += NT) {
      sq[d] = float(static_cast<T>((sq[d] * rq) * qscale));
      sk[d] = float(static_cast<T>(sk[d] * rk));
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  // ----------------------------------------------------------------- phase 1
  // Gated delta rule, one time step.  Identical arithmetic and identical simd
  // reduction partition to models/gated_delta.py's kernel: lane `lane` owns key
  // elements [NDK*lane, NDK*lane + NDK).
  {
    float beta = shr[2];
    for (int j = 0; j < NDV; ++j) {
      uint dv = ty + (uint)TY * (uint)j;
      float kv = 0.0f;
      for (int i = 0; i < NDK; ++i) {
        uint s = NDK * lane + i;
        st[j][i] = st[j][i] * sg[s];
        kv += st[j][i] * sk[s];
      }
      kv = simd_sum(kv);
      float delta = (sv[dv] - kv) * beta;
      float o = 0.0f;
      for (int i = 0; i < NDK; ++i) {
        uint s = NDK * lane + i;
        st[j][i] = st[j][i] + sk[s] * delta;
        o += st[j][i] * sq[s];
      }
      o = simd_sum(o);
      if (thread_index_in_simdgroup == 0u) {
        sy[dv] = float(static_cast<T>(o));   // gated_delta writes y in T
      }
      for (int i = 0; i < NDK; ++i) {
        so[(size_t)dv * D + NDK * lane + i] = static_cast<ST>(st[j][i]);
      }
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  // ----------------------------------------------------------------- phase 2
  // Gated RMSNorm over the value axis, in fp32, then back to T.
  if (simdgroup_index_in_threadgroup == 0u) {
    float po = 0.0f;
    for (int blk = 0; blk < RBLK; ++blk) {
      uint base = (uint)(blk * 128) + 4u * lane;
      for (int i = 0; i < 4; ++i) po = sq_acc(po, sy[base + i]);
    }
    uint base = (uint)(RBLK * 128) + 4u * lane;
    if (4u * lane + 4u <= (uint)REXTRA) {
      for (int i = 0; i < 4; ++i) po = sq_acc(po, sy[base + i]);
    } else {
      for (int i = 0; 4u * lane + (uint)i < (uint)REXTRA; ++i) {
        po = sq_acc(po, sy[base + i]);
      }
    }
    po = simd_sum(po);
    if (lane == 0u) {
      shr[0] = metal::precise::rsqrt(po / (float)D + norm_eps);
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  {
    float rn = shr[0];
    for (uint d = tid; d < (uint)D; d += NT) {
      float x = sy[d] * rn;
      x = float(o_w[d]) * x;
      x = x * mlx_sigmoid_precise<float>(sgate[d]);
      y[h * (uint)D + d] = static_cast<T>(x);
    }
  }
"""

# ---------------------------------------------------------------------------
# Optional extra fold: f_b_proj / g_b_proj done in-kernel.
#
# Both are Linear(head_dim, num_heads*head_dim) affine-quantized GEMVs whose only
# consumer is this kernel, so folding them in removes two dispatches per layer at
# the cost of streaming ~2 MB more weight.
#
# It is written as a transcription of MLX's affine ``qmv_quad`` path
# (mlx/backend/metal/kernels/quantized.h: load_vector + qdot + quad_sum, bits==8)
# rather than as a generic dot product: one quad per output row, quad lane `l`
# owning x[VPT*l : VPT*l+VPT] with VPT = head_dim/4, one scale/bias pair per
# thread, and the row total closed by quad_sum.  Same partition and same
# accumulation order as MLX, so the folded projection is bit-identical to
# mx.quantized_matmul -- verified for head_dim in {64, 128}.  (A plain
# per-element ``x * (scale*q + bias)`` dot instead disagrees on ~0.01% of
# elements, which is enough to flip greedy tokens.)
#
# Element e of row r is byte e of the row for bits==8:
#   ((w[r, e/4] >> (8 * (e % 4))) & 0xff) * scales[r, e/GS] + biases[r, e/GS]
_QPROJ_COMPUTE = """
  // ------------------------------------------------------------- phase 0a-pre
  {
    constexpr int VPT   = D / 4;        // values_per_thread
    constexpr int SSPT  = GS / VPT;     // scale_step_per_thread
    constexpr int NQUAD = NT / 4;
    constexpr int NG    = D / GS;
    const uint qg   = tid / 4u;         // quad index within the threadgroup
    const uint qlid = tid % 4u;         // lane within the quad

    // Rows [0, D) are f_b_proj, [D, 2D) are g_b_proj; a quad owns one row.
    for (uint t = qg; t < 2u * (uint)D; t += (uint)NQUAD) {
      uint proj = t / (uint)D;
      uint d    = t - proj * (uint)D;
      uint row  = h * (uint)D + d;
      device const T* xp = (proj == 0u ? fa : ga) + qlid * (uint)VPT;
      float xs = 0.0f;                                  // load_vector's `sum`
      for (int i = 0; i < VPT; ++i) xs += float(xp[i]);
      size_t wb = (size_t)row * D + (size_t)qlid * VPT; // byte offset (bits == 8)
      uint gi = row * (uint)NG + qlid / (uint)SSPT;
      float sc = float(proj == 0u ? fbs[gi] : gbs[gi]);
      float bi = float(proj == 0u ? fbb[gi] : gbb[gi]);
      float accum = 0.0f;                               // qdot
      for (int i = 0; i < VPT; ++i) {
        size_t bo = wb + (size_t)i;
        uint word = (proj == 0u) ? fbw[bo / 4u] : gbw[bo / 4u];
        accum += float(xp[i]) * float((word >> (8u * (uint)(bo % 4u))) & 0xffu);
      }
      float r = quad_sum(sc * accum + xs * bi);
      if (qlid == 0u) {
        if (proj == 0u) sa[d] = float(static_cast<T>(r));      // f_b_proj -> T
        else            sgate[d] = float(static_cast<T>(r));   // g_b_proj -> T
      }
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
"""


def _qproj_source(source: str) -> str:
    """Derive the fold-the-GEMVs source from the validated base source.

    Every substitution is asserted: a silently non-matching replace would leave
    `a` / `gate` referenced by a kernel whose signature no longer declares them,
    which only shows up as a Metal compile error at first use.
    """

    def sub(text: str, old: str, new: str) -> str:
        if old not in text:
            raise RuntimeError(f"qproj source rewrite did not match: {old!r}")
        return text.replace(old, new, 1)

    out = sub(
        source,
        "  threadgroup float shr[3];",
        "  threadgroup float sa[D];          // in-kernel f_b_proj output\n"
        "  threadgroup float shr[3];",
    )
    out = sub(
        out,
        "  // ---------------------------------------------------------------- phase 0a\n",
        _QPROJ_COMPUTE
        + "\n  // ---------------------------------------------------------------- phase 0a\n",
    )
    # sa[d] holds the bf16-rounded f_b_proj output, so cast back to T before the
    # add: `a + dt_bias` must promote from the stored dtypes, not from float.
    out = sub(
        out,
        "      float av = promote_add(a[h * (uint)D + d], dt_bias[h * (uint)D + d]);",
        "      float av = promote_add("
        "static_cast<T>(sa[d]), dt_bias[h * (uint)D + d]);",
    )
    out = sub(out, "      sgate[d] = float(gate[h * (uint)D + d]);\n", "")
    return out


_INPUT_NAMES = [
    "mq",
    "mk",
    "mv",
    "conv_state",
    "conv_w",
    "a",
    "bvec",
    "A_log",
    "dt_bias",
    "state_in",
    "gate",
    "o_w",
    "lower_bound",
    "qscale",
    "norm_eps",
]
_OUTPUT_NAMES = ["y", "state_out", "conv_state_out"]

# "a" / "gate" (the f_b_proj / g_b_proj outputs) are replaced by their inputs and
# quantized weights when the projections are folded in.
_QPROJ_INPUT_NAMES = [n for n in _INPUT_NAMES if n not in ("a", "gate")] + [
    "fa",
    "fbw",
    "fbs",
    "fbb",
    "ga",
    "gbw",
    "gbs",
    "gbb",
]

_KERNELS = {}
_KERNEL_TRIED = False

# Threadgroup y-extent: 32 * TY threads per threadgroup, one threadgroup per head.
#
# TY only controls how many value-dim rows each thread walks (NDV = head_dim / TY)
# and how many output rows each quad walks in the projection fold.  Every
# reduction -- the simd_sum over the 32 key lanes, the 32-lane row reduction in
# the two norms, and quad_sum in the folded GEMV -- is over the same lanes with
# the same operand order at every TY, so lowering it is partition-preserving and
# stays bit-identical.  That matters because `maxTotalThreadsPerThreadgroup` is a
# *per-pipeline* limit driven by register pressure: some GPUs (notably the
# virtualized ones on CI runners) admit 1024 threads for the base kernel but cap
# the higher-pressure qproj pipeline lower.  We probe downwards.
_TY_CANDIDATES = (32, 16, 8, 4)


def _kernel(kind: str = "base"):
    """``kind`` in {"base", "qproj"}; ``None`` if Metal is unavailable.

    Two objects rather than one: mx.fast.metal_kernel derives the function
    signature from input_names/output_names, so each variant needs its own.
    """
    global _KERNEL_TRIED
    if not _KERNEL_TRIED:
        _KERNEL_TRIED = True
        if mx.metal.is_available():
            _KERNELS["base"] = mx.fast.metal_kernel(
                name="glm5_kda_decode_step",
                input_names=_INPUT_NAMES,
                output_names=_OUTPUT_NAMES,
                header=_HEADER,
                source=_SOURCE,
            )
            _KERNELS["qproj"] = mx.fast.metal_kernel(
                name="glm5_kda_decode_step_qproj",
                input_names=_QPROJ_INPUT_NAMES,
                output_names=_OUTPUT_NAMES,
                header=_HEADER,
                source=_qproj_source(_SOURCE),
            )
    return _KERNELS.get(kind)


# (kind, dtype, state dtype, H, D, K, bits, group_size) -> usable TY, or None if
# the device cannot run this variant at any admissible threadgroup size.
_TY_PROBE_CACHE = {}


def _probe_launch(kind, ty, dt, st, num_heads, head_dim, conv_kernel_size, bits, gs):
    """Run one throwaway launch at ``ty`` and force it, so the driver's
    per-pipeline threadgroup limit is exercised here rather than mid-forward."""
    h, d, k = num_heads, head_dim, conv_kernel_size
    zeros = lambda shape, dtype: mx.zeros(shape, dtype=dtype)  # noqa: E731
    args = dict(
        q_in=zeros((1, 1, h * d), dt),
        k_in=zeros((1, 1, h * d), dt),
        v_in=zeros((1, 1, h * d), dt),
        conv_state=zeros((1, k - 1, 3 * h * d), dt),
        conv_w=zeros((3 * h * d, k, 1), dt),
        b=zeros((1, 1, h), dt),
        A_log=zeros((h,), mx.float32),
        dt_bias=zeros((h * d,), mx.float32),
        state=zeros((1, h, d, d), st),
        o_weight=zeros((d,), dt),
    )
    if kind == "qproj":
        pack = 32 // bits
        w = zeros((h * d, d // pack), mx.uint32)
        sc = zeros((h * d, d // gs), dt)
        proj = _ProbeLinear(w, sc, bits, gs)
        args["a"] = None
        args["gate"] = None
        qproj = (zeros((1, 1, d), dt), proj, zeros((1, 1, d), dt), proj)
    else:
        args["a"] = zeros((1, 1, h * d), dt)
        args["gate"] = zeros((1, 1, h * d), dt)
        qproj = None
    outs = fused_kda_decode_step(
        args["q_in"],
        args["k_in"],
        args["v_in"],
        args["conv_state"],
        args["conv_w"],
        args["a"],
        args["b"],
        args["A_log"],
        args["dt_bias"],
        args["state"],
        args["gate"],
        args["o_weight"],
        num_heads=h,
        head_dim=d,
        conv_kernel_size=k,
        lower_bound=-5.0,
        norm_eps=1e-5,
        ty=ty,
        qproj=qproj,
    )
    mx.eval(outs)


class _ProbeLinear:
    """Minimal stand-in carrying just what the folded GEMV reads."""

    def __init__(self, weight, scales, bits, group_size):
        self.weight = weight
        self.scales = scales
        self.biases = scales
        self.bits = bits
        self.group_size = group_size


def fused_kda_probe(
    *,
    kind: str,
    num_heads: int,
    head_dim: int,
    conv_kernel_size: int,
    dtype,
    state_dtype,
    bits: Optional[int] = None,
    group_size: Optional[int] = None,
) -> Optional[int]:
    """Largest threadgroup extent this device will run ``kind`` at, else ``None``.

    ``maxTotalThreadsPerThreadgroup`` is a *per-pipeline* limit set by register
    pressure, not a device constant: a GPU can admit 1024 threads for the base
    kernel and cap the heavier qproj pipeline lower (CI's virtualized runners cap
    it at 640).  MLX reports that as a ValueError at eval time, far from the call
    site, so probe it up front and remember the answer.  Lowering TY is
    partition-preserving -- every reduction keeps the same lanes and the same
    operand order -- so a degraded launch is still bit-identical.
    """
    key = (
        kind,
        dtype,
        state_dtype,
        num_heads,
        head_dim,
        conv_kernel_size,
        bits,
        group_size,
    )
    if key in _TY_PROBE_CACHE:
        return _TY_PROBE_CACHE[key]
    result = None
    for ty in _TY_CANDIDATES:
        if head_dim % ty:
            continue
        try:
            _probe_launch(
                kind,
                ty,
                dtype,
                state_dtype,
                num_heads,
                head_dim,
                conv_kernel_size,
                bits,
                group_size,
            )
        except ValueError as exc:
            if "threads per threadgroup" not in str(exc):
                raise
            continue
        except RuntimeError as exc:  # kernel would not build on this device
            logger.info("glm5_next fused KDA (%s) unavailable: %s", kind, exc)
            break
        result = ty
        break
    _TY_PROBE_CACHE[key] = result
    if result is None:
        logger.info(
            "glm5_next fused KDA (%s) declined: this device's threadgroup limit "
            "is below the kernel's requirement at every supported size",
            kind,
        )
    elif result != _TY_CANDIDATES[0]:
        logger.info(
            "glm5_next fused KDA (%s) running at a reduced threadgroup "
            "(%d threads): the device caps this pipeline below %d.  Results are "
            "unchanged; this only lowers occupancy.",
            kind,
            32 * result,
            32 * _TY_CANDIDATES[0],
        )
    return result


def fused_kda_supported(
    *,
    num_heads: int,
    head_dim: int,
    conv_kernel_size: int,
    lower_bound: Optional[float],
) -> bool:
    """Shape/feature preconditions for the fused kernel (config-level, not per-step).

    ``A_log`` / ``dt_bias`` / the gated-RMSNorm weight may be stored either in
    float32 or in the model dtype (``cast_predicate`` does not pin them), and the
    kernel reads whichever it finds -- ``a + dt_bias`` promotes in Metal exactly
    as it does in MLX.

    Bit-identity with the eager path is verified for ``head_dim <= 128`` (where
    MLX's row reduction uses a 32-wide thread row, which the in-kernel L2 / RMS
    reductions mirror).  Larger head dims stay numerically correct but may differ
    from the eager path in the last bit of those two reductions.
    """
    if not mx.metal.is_available() or mx.default_device() != mx.gpu:
        return False
    if lower_bound is None:
        return False  # only the "safe gate" branch is transcribed
    if conv_kernel_size < 2:
        return False
    if head_dim % 32 != 0:
        return False
    if not any(head_dim % ty == 0 for ty in _TY_CANDIDATES):
        return False
    if num_heads <= 0:
        return False
    return _kernel("base") is not None


def fused_kda_qproj_supported(f_b_proj, g_b_proj, *, head_dim: int) -> bool:
    """Can f_b_proj / g_b_proj be folded into the kernel?

    Requires both to be affine-quantized with the same bits/group_size, a packed
    layout the kernel can address, and a lane-aligned input dim.
    """
    mods = (f_b_proj, g_b_proj)
    if not all(hasattr(m, "scales") and hasattr(m, "biases") for m in mods):
        return False
    if len({getattr(m, "mode", "affine") for m in mods}) != 1:
        return False
    if getattr(mods[0], "mode", "affine") != "affine":
        return False
    if len({m.bits for m in mods}) != 1 or len({m.group_size for m in mods}) != 1:
        return False
    bits, group_size = mods[0].bits, mods[0].group_size
    # The in-kernel GEMV transcribes MLX's affine qmv_quad, which MLX only
    # dispatches for these shapes -- outside them the fold would still be
    # correct but no longer bit-identical, so decline instead.
    if bits != 8:
        return False
    if head_dim not in (64, 128):
        return False
    values_per_thread = head_dim // 4
    if group_size % values_per_thread or head_dim % group_size:
        return False
    pack = 32 // bits
    for m in mods:
        if m.weight.dtype != mx.uint32:
            return False
        if m.weight.shape[-1] != head_dim // pack:
            return False
        if m.scales.shape[-1] != head_dim // group_size:
            return False
        if m.scales.shape != m.biases.shape:
            return False
    return _kernel("qproj") is not None


def fused_kda_decode_step(
    q_in: mx.array,
    k_in: mx.array,
    v_in: mx.array,
    conv_state: mx.array,
    conv_w: mx.array,
    a: Optional[mx.array],
    b: mx.array,
    A_log: mx.array,
    dt_bias: mx.array,
    state: mx.array,
    gate: Optional[mx.array],
    o_weight: mx.array,
    *,
    num_heads: int,
    head_dim: int,
    conv_kernel_size: int,
    lower_bound: float,
    norm_eps: float,
    ty: int = 32,
    qproj: Optional[Tuple] = None,
) -> Tuple[mx.array, ...]:
    """One fused KDA decode step.

    Args (all B=1, S=1):
      q_in, k_in, v_in: ``[1, 1, H*D]`` pre-conv projections, dtype ``T``.
      conv_state:       ``[1, K-1, 3*H*D]`` cached conv window, dtype ``T``.
      conv_w:           ``[3*H*D, K, 1]`` depthwise conv weight, dtype ``T``.
      a:                ``[1, 1, H*D]`` forget-gate ``f_b_proj`` output, dtype ``T``.
      b:                ``[1, 1, H]`` beta logits, dtype ``T``.
      A_log:            ``[H]`` float32.   dt_bias: ``[H*D]`` float32.
      state:            ``[1, H, D, D]`` recurrent state (float32 in practice).
      gate:             ``[1, 1, H*D]`` ``g_b_proj`` output, dtype ``T``.
      o_weight:         ``[D]`` gated-RMSNorm weight, dtype ``T``.

    Returns ``(y, state_out, conv_state_out)`` where ``y`` is ``[1, 1, H*D]`` and is
    exactly what the eager path feeds to ``o_proj``.

    """
    H, D = num_heads, head_dim
    dt = q_in.dtype
    kernel = _kernel("qproj" if qproj is not None else "base")
    out_shapes = [(1, 1, H * D), state.shape, conv_state.shape]
    out_dtypes = [dt, state.dtype, dt]
    head = [q_in, k_in, v_in, conv_state, conv_w]
    tail = [b, A_log, dt_bias, state, o_weight]
    scalars = [float(lower_bound), float(head_dim**-0.5), float(norm_eps)]
    template = [
        ("T", dt),
        ("ST", state.dtype),
        ("H", num_heads),
        ("D", head_dim),
        ("K", conv_kernel_size),
        ("TY", ty),
    ]
    if qproj is not None:
        fa, f_b_proj, ga, g_b_proj = qproj
        inputs = (
            head
            + tail
            + scalars
            + [
                fa,
                f_b_proj.weight,
                f_b_proj.scales,
                f_b_proj.biases,
                ga,
                g_b_proj.weight,
                g_b_proj.scales,
                g_b_proj.biases,
            ]
        )
        template += [("BITS", f_b_proj.bits), ("GS", f_b_proj.group_size)]
    else:
        inputs = head + [a] + tail[:4] + [gate, o_weight] + scalars
    return kernel(
        inputs=inputs,
        template=template,
        grid=(32, ty, num_heads),
        threadgroup=(32, ty, 1),
        output_shapes=out_shapes,
        output_dtypes=out_dtypes,
    )
