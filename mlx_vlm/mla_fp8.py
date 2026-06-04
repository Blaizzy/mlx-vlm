"""FP8 KV cache for MLA-attention models with a fused absorbed-decode Metal kernel.

MLA (Multi-head Latent Attention, DeepSeek-V2/V3, GLM-4-MoE, Kimi, ...) compresses the
KV of every token to a low-rank latent ``c_t`` of size ``kv_lora_rank`` plus a small shared
RoPE key ``k_pe_t`` of size ``qk_rope_head_dim``. mlx-vlm currently *expands* that latent
back to full per-head keys/values before caching, which throws away MLA's entire memory
advantage (tens of KB per token for DeepSeek-V2-class configs).

This module caches the latent directly and stores it in FP8 (e4m3, per-group scales) — the
same layout vLLM / SGLang use for DeepSeek — while keeping the RoPE key in the model dtype:

    512 fp8 latent + (512/128)*4 fp32 scales + 64 * 2 bf16 rope = 512 + 16 + 128 = 656 B/token

Decoding runs in the *absorbed* MLA form: the ``kv_b_proj`` nope/value matrices are folded
into the query (``embed_q``) and output (``unembed_out``) projections so the cached latent
serves as both K and V (an MQA over the latent). The fused Metal kernel streams the FP8
latent once per (batch, head), dequantises in registers via a 256-entry lookup table, and
runs an online-softmax flash decode — so the FP8 bandwidth saving turns directly into speed.

Everything here is opt-in; nothing imports it unless a model explicitly enables the FP8 path.
"""

from __future__ import annotations

import os
from typing import Optional, Tuple

import mlx.core as mx

# ``mx.to_fp8`` rounds to nearest and saturates at this value (verified against the runtime;
# note ``mx.from_fp8`` can *decode* up to 480, but ``to_fp8`` never emits codes above 448).
E4M3_MAX = 448.0
DEFAULT_GROUP_SIZE = 128
_SIMD = 32  # lanes per SIMD-group / threadgroup for the decode kernel

#: Environment flag that turns the FP8 MLA cache on for models that support it.
MLA_FP8_ENV = "MLX_VLM_MLA_FP8"
MLA_FP8_GROUP_ENV = "MLX_VLM_MLA_FP8_GROUP"


def mla_fp8_enabled() -> bool:
    """Whether the opt-in FP8 MLA KV cache is enabled (off by default)."""
    return os.environ.get(MLA_FP8_ENV, "0").strip().lower() in (
        "1",
        "true",
        "yes",
        "on",
    )


def mla_fp8_group_size() -> int:
    """Group size for the FP8 latent quantisation (``MLX_VLM_MLA_FP8_GROUP``)."""
    try:
        return int(os.environ.get(MLA_FP8_GROUP_ENV, str(DEFAULT_GROUP_SIZE)))
    except ValueError:
        return DEFAULT_GROUP_SIZE


def _metal_available() -> bool:
    try:
        return mx.metal.is_available()
    except Exception:
        return False


_FP8_LUT: Optional[mx.array] = None


def fp8_lut() -> mx.array:
    """256-entry fp32 table mapping every e4m3 byte to its value (== ``mx.from_fp8``)."""
    global _FP8_LUT
    if _FP8_LUT is None:
        _FP8_LUT = mx.from_fp8(mx.arange(256, dtype=mx.uint8), dtype=mx.float32)
    return _FP8_LUT


def quantize_latent_fp8(
    latent: mx.array, group_size: int = DEFAULT_GROUP_SIZE
) -> Tuple[mx.array, mx.array]:
    """Quantise the last axis of ``latent`` to FP8 with per-group absmax scales.

    Returns ``(codes, scales)`` where ``codes`` is uint8 with the same shape as ``latent``
    and ``scales`` is fp32 with the group axis (last dim // ``group_size``) instead.
    """
    *lead, d = latent.shape
    if d % group_size != 0:
        raise ValueError(f"latent dim {d} not divisible by group_size {group_size}")
    ng = d // group_size
    g = latent.reshape(*lead, ng, group_size).astype(mx.float32)
    amax = mx.max(mx.abs(g), axis=-1, keepdims=True)
    scale = mx.maximum(amax / E4M3_MAX, mx.array(1e-12, mx.float32))
    codes = mx.to_fp8((g / scale).reshape(*lead, d))
    return codes, scale.reshape(*lead, ng).astype(mx.float32)


def dequantize_latent_fp8(
    codes: mx.array,
    scales: mx.array,
    group_size: int = DEFAULT_GROUP_SIZE,
    dtype: mx.Dtype = mx.float32,
) -> mx.array:
    """Inverse of :func:`quantize_latent_fp8`."""
    *lead, d = codes.shape
    if d % group_size != 0:
        raise ValueError(f"latent dim {d} not divisible by group_size {group_size}")
    ng = d // group_size
    f = mx.from_fp8(codes, dtype=mx.float32).reshape(*lead, ng, group_size)
    f = f * scales[..., None]
    return f.reshape(*lead, d).astype(dtype)


class Fp8MLAKVCache:
    """Stepped KV cache holding the MLA latent in FP8 and the RoPE key in model dtype.

    Layout (all leading dim is batch ``B``, sequence axis is 1):
      * ``latent_codes``  : ``(B, T, D)``  uint8   — e4m3 codes for the ``kv_lora_rank`` latent
      * ``latent_scales`` : ``(B, T, D//group)`` fp32 — per-group absmax scales
      * ``k_pe``          : ``(B, T, R)``  model dtype — the shared RoPE key
    """

    step = 256

    def __init__(self, group_size: int = DEFAULT_GROUP_SIZE):
        self.group_size = group_size
        self.offset = 0
        self.latent_codes: Optional[mx.array] = None
        self.latent_scales: Optional[mx.array] = None
        self.k_pe: Optional[mx.array] = None

    def _grow(self, B: int, D: int, ng: int, R: int, L: int, dtype: mx.Dtype):
        # Allocate only enough *new* slots for the incoming L tokens (a few steps), then
        # concatenate onto the existing buffer — mirrors mlx-lm's KVCache so growth stays
        # O(step) per call instead of re-allocating for prev+L every time.
        prev = self.offset
        n_steps = (self.step + L - 1) // self.step
        cap = n_steps * self.step
        new_codes = mx.zeros((B, cap, D), mx.uint8)
        new_scales = mx.zeros((B, cap, ng), mx.float32)
        new_kpe = mx.zeros((B, cap, R), dtype)
        if self.latent_codes is not None:
            if prev % self.step != 0:
                self.latent_codes = self.latent_codes[:, :prev]
                self.latent_scales = self.latent_scales[:, :prev]
                self.k_pe = self.k_pe[:, :prev]
            self.latent_codes = mx.concatenate([self.latent_codes, new_codes], axis=1)
            self.latent_scales = mx.concatenate(
                [self.latent_scales, new_scales], axis=1
            )
            self.k_pe = mx.concatenate([self.k_pe, new_kpe], axis=1)
        else:
            self.latent_codes, self.latent_scales, self.k_pe = (
                new_codes,
                new_scales,
                new_kpe,
            )

    def update_and_fetch(
        self, latent: mx.array, k_pe: mx.array
    ) -> Tuple[mx.array, mx.array, mx.array]:
        """Append ``latent`` (B,L,D) and ``k_pe`` (B,L,R); return views up to the new offset."""
        B, L, D = latent.shape
        R = k_pe.shape[-1]
        ng = D // self.group_size
        codes, scales = quantize_latent_fp8(latent, self.group_size)
        prev = self.offset
        if self.latent_codes is None or prev + L > self.latent_codes.shape[1]:
            self._grow(B, D, ng, R, L, k_pe.dtype)
        self.offset += L
        self.latent_codes[:, prev : self.offset] = codes
        self.latent_scales[:, prev : self.offset] = scales
        self.k_pe[:, prev : self.offset] = k_pe
        return (
            self.latent_codes[:, : self.offset],
            self.latent_scales[:, : self.offset],
            self.k_pe[:, : self.offset],
        )

    def size(self) -> int:
        return self.offset

    def is_trimmable(self) -> bool:
        return True

    def trim(self, n: int) -> int:
        n = min(self.offset, n)
        self.offset -= n
        return n

    @property
    def state(self):
        if self.latent_codes is None:
            return None
        return (
            self.latent_codes[:, : self.offset],
            self.latent_scales[:, : self.offset],
            self.k_pe[:, : self.offset],
        )

    @state.setter
    def state(self, v):
        self.latent_codes, self.latent_scales, self.k_pe = v
        self.offset = self.latent_codes.shape[1]

    def bytes_per_token(self, D: int, R: int, rope_bytes: int = 2) -> int:
        ng = D // self.group_size
        return D + ng * 4 + R * rope_bytes


# ---------------------------------------------------------------------------
# Absorbed-MLA decode: fused Metal kernel + pure-MLX reference
# ---------------------------------------------------------------------------

#: Context tokens per split-K tile (compile-time; sizes the threadgroup score buffer).
#: Tuned so long contexts spawn many threadgroups (occupancy) with small partial buffers.
DECODE_TILE = 512

_PARTIAL_KERNEL = None
_COMBINE_KERNEL = None


def _partial_kernel():
    """Two-pass split-K flash decode: one threadgroup (32 lanes) per (batch*head, tile).

    A serial online-softmax loop with a per-token ``simd_sum`` has no ILP and loses badly to
    MLX's GEMM-based SDPA. Instead we split each tile into two passes that have no serial
    dependency: pass 1 computes all scores (lanes stride over tokens, each does a full-D dot
    against the query staged in threadgroup memory — no cross-lane reduction), then a single
    ``simd_max`` gives the tile max; pass 2 accumulates the weighted latent sum (lanes own
    disjoint dims). The latent is read twice but both passes pipeline.
    """
    global _PARTIAL_KERNEL
    if _PARTIAL_KERNEL is not None:
        return _PARTIAL_KERNEL
    source = """
        uint lane = thread_position_in_grid.x;
        uint bh   = thread_position_in_grid.y;
        uint tile = thread_position_in_grid.z;
        int H        = meta[1];
        int ctx      = meta[2];
        int ng       = meta[3];
        int n_tiles  = meta[5];
        int b   = (int)bh / H;
        float sc = scale[0];

        threadgroup float q_sh[Dim];
        threadgroup float s_sh[TileT];
        #pragma clang loop unroll(full)
        for (int i = 0; i < DimsPerLane; ++i) {
            int d = lane + i * 32;
            q_sh[d] = (float)q[bh * Dim + d];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        int t0 = (int)tile * TileT;
        int t1 = min(t0 + TileT, ctx);

        // Pass 1: scores. Each lane handles a strided set of tokens; full-D dot, no reduction.
        float lmax = -1e30f;
        for (int t = t0 + (int)lane; t < t1; t += 32) {
            int lbase = (b * ctx + t) * Dim;
            int sbase = (b * ctx + t) * ng;
            float dot = 0.0f;
            for (int d = 0; d < Dim; ++d) {
                float c = lut[(int)latent_codes[lbase + d]] * latent_scales[sbase + (d / GroupSize)];
                dot += q_sh[d] * c;
            }
            float s = sc * dot + (float)pe[bh * ctx + t];
            s_sh[t - t0] = s;
            lmax = fmax(lmax, s);
        }
        float m_tile = simd_max(lmax);
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Pass 2: weighted latent sum. Each lane owns disjoint dims; sequential over tokens.
        float acc[DimsPerLane];
        #pragma clang loop unroll(full)
        for (int i = 0; i < DimsPerLane; ++i) acc[i] = 0.0f;
        float l = 0.0f;
        int n = t1 - t0;
        for (int u = 0; u < n; ++u) {
            int t = t0 + u;
            float p = fast::exp(s_sh[u] - m_tile);
            l += p;
            int lbase = (b * ctx + t) * Dim;
            int sbase = (b * ctx + t) * ng;
            #pragma clang loop unroll(full)
            for (int i = 0; i < DimsPerLane; ++i) {
                int d = lane + i * 32;
                float c = lut[(int)latent_codes[lbase + d]] * latent_scales[sbase + (d / GroupSize)];
                acc[i] += p * c;
            }
        }

        int pidx = (int)bh * n_tiles + (int)tile;
        if (lane == 0) { partial_m[pidx] = m_tile; partial_l[pidx] = l; }
        #pragma clang loop unroll(full)
        for (int i = 0; i < DimsPerLane; ++i)
            partial_acc[pidx * Dim + lane + i * 32] = acc[i];
    """
    _PARTIAL_KERNEL = mx.fast.metal_kernel(
        name="mla_fp8_decode_partial",
        input_names=[
            "q",
            "pe",
            "latent_codes",
            "latent_scales",
            "lut",
            "meta",
            "scale",
        ],
        output_names=["partial_m", "partial_l", "partial_acc"],
        source=source,
    )
    return _PARTIAL_KERNEL


def _combine_kernel():
    """Merge the per-tile flash partials for each (batch*head) into the final output."""
    global _COMBINE_KERNEL
    if _COMBINE_KERNEL is not None:
        return _COMBINE_KERNEL
    source = """
        uint lane = thread_position_in_grid.x;
        uint bh   = thread_position_in_grid.y;
        int n_tiles = meta[0];

        float gm = -1e30f;
        for (int tile = 0; tile < n_tiles; ++tile)
            gm = fmax(gm, partial_m[(int)bh * n_tiles + tile]);

        float l = 0.0f;
        float acc[DimsPerLane];
        #pragma clang loop unroll(full)
        for (int i = 0; i < DimsPerLane; ++i) acc[i] = 0.0f;

        for (int tile = 0; tile < n_tiles; ++tile) {
            int pidx = (int)bh * n_tiles + tile;
            float w = fast::exp(partial_m[pidx] - gm);
            l += partial_l[pidx] * w;
            #pragma clang loop unroll(full)
            for (int i = 0; i < DimsPerLane; ++i)
                acc[i] += partial_acc[pidx * Dim + lane + i * 32] * w;
        }
        float inv = (l > 0.0f) ? (1.0f / l) : 0.0f;
        #pragma clang loop unroll(full)
        for (int i = 0; i < DimsPerLane; ++i)
            out[bh * Dim + lane + i * 32] = (T_out)(acc[i] * inv);
    """
    _COMBINE_KERNEL = mx.fast.metal_kernel(
        name="mla_fp8_decode_combine",
        input_names=["partial_m", "partial_l", "partial_acc", "meta"],
        output_names=["out"],
        source=source,
    )
    return _COMBINE_KERNEL


def _mla_fp8_decode_metal(
    q: mx.array,
    pe: mx.array,
    latent_codes: mx.array,
    latent_scales: mx.array,
    scale: float,
    group_size: int,
    tile: int = DECODE_TILE,
) -> mx.array:
    B, H, D = q.shape
    if D % _SIMD != 0:
        raise ValueError(
            f"kv_lora_rank ({D}) must be divisible by {_SIMD} for the 'kernel' decode path; "
            f"use method='mlx' or method='sdpa'."
        )
    T = latent_codes.shape[1]
    ng = D // group_size
    BH = B * H
    n_tiles = max(1, (T + tile - 1) // tile)
    lut = fp8_lut()
    pe = pe.astype(mx.float32)
    scale_arr = mx.array([scale], dtype=mx.float32)

    partial_m, partial_l, partial_acc = _partial_kernel()(
        inputs=[
            q,
            pe,
            latent_codes,
            latent_scales,
            lut,
            mx.array([B, H, T, ng, tile, n_tiles], dtype=mx.int32),
            scale_arr,
        ],
        template=[
            ("T_out", q.dtype),
            ("Dim", D),
            ("DimsPerLane", D // _SIMD),
            ("GroupSize", group_size),
            ("TileT", tile),
        ],
        grid=(_SIMD, BH, n_tiles),
        threadgroup=(_SIMD, 1, 1),
        output_shapes=[(BH, n_tiles), (BH, n_tiles), (BH, n_tiles, D)],
        output_dtypes=[mx.float32, mx.float32, mx.float32],
    )

    out = _combine_kernel()(
        inputs=[partial_m, partial_l, partial_acc, mx.array([n_tiles], dtype=mx.int32)],
        template=[("T_out", q.dtype), ("Dim", D), ("DimsPerLane", D // _SIMD)],
        grid=(_SIMD, BH, 1),
        threadgroup=(_SIMD, 1, 1),
        output_shapes=[(B, H, D)],
        output_dtypes=[q.dtype],
    )[0]
    return out


_DEQUANT_KERNEL = None


def _dequant_kernel():
    """Single-pass fused FP8 latent dequant (codes -> LUT -> * group scale -> bf16/fp16)."""
    global _DEQUANT_KERNEL
    if _DEQUANT_KERNEL is not None:
        return _DEQUANT_KERNEL
    source = """
        uint idx = thread_position_in_grid.x;
        int total = meta[0];
        if (idx >= (uint)total) return;
        int Dd = meta[1], ng = meta[2], group = meta[3];
        int row = (int)idx / Dd;
        int d = (int)idx % Dd;
        out[idx] = (T_out)(lut[(int)codes[idx]] * scales[row * ng + d / group]);
    """
    _DEQUANT_KERNEL = mx.fast.metal_kernel(
        name="mla_fp8_dequant",
        input_names=["codes", "scales", "lut", "meta"],
        output_names=["out"],
        source=source,
    )
    return _DEQUANT_KERNEL


def dequantize_latent_fp8_metal(
    codes: mx.array,
    scales: mx.array,
    group_size: int = DEFAULT_GROUP_SIZE,
    dtype: mx.Dtype = mx.bfloat16,
) -> mx.array:
    """Fused-kernel dequant — much faster than the multi-pass ``mx.from_fp8`` path."""
    D = codes.shape[-1]
    ng = D // group_size
    total = 1
    for s in codes.shape:
        total *= s
    grid = ((total + 255) // 256) * 256
    out = _dequant_kernel()(
        inputs=[
            codes,
            scales,
            fp8_lut(),
            mx.array([total, D, ng, group_size], mx.int32),
        ],
        template=[("T_out", dtype)],
        grid=(grid, 1, 1),
        threadgroup=(256, 1, 1),
        output_shapes=[codes.shape],
        output_dtypes=[dtype],
    )[0]
    return out


def _mla_fp8_decode_sdpa(
    q: mx.array,
    pe: mx.array,
    latent_codes: mx.array,
    latent_scales: mx.array,
    scale: float,
    group_size: int,
) -> mx.array:
    """Fast path: fused-dequant the latent then run MLX's GEMM/MMA SDPA (absorbed: K=V=latent)."""
    latent = dequantize_latent_fp8_metal(
        latent_codes, latent_scales, group_size, q.dtype
    )
    k = latent[:, None]  # (B, 1, T, D) shared across query heads
    mask = pe[:, :, None, :].astype(q.dtype)
    o = mx.fast.scaled_dot_product_attention(
        q[:, :, None, :], k, k, scale=scale, mask=mask
    )
    return o[:, :, 0, :]


def _mla_fp8_decode_mlx(
    q: mx.array,
    pe: mx.array,
    latent_codes: mx.array,
    latent_scales: mx.array,
    scale: float,
    group_size: int,
) -> mx.array:
    """Absorbed decode via MLX's fused einsum (fp32).

    This is both the fp32 ground truth *and*, empirically, the fastest decode on MLX
    0.31.2: MLX fuses the lazy FP8 dequant into the matmul, so no bf16 latent is ever
    materialised and the GEMMs use the matrix units. See ``tests/bench_mla_fp8.py``.
    """
    c = dequantize_latent_fp8(latent_codes, latent_scales, group_size, mx.float32)
    dot = mx.einsum("bhd,btd->bht", q.astype(mx.float32), c)
    s = scale * dot + pe.astype(mx.float32)
    p = mx.softmax(s, axis=-1, precise=True)
    o = mx.einsum("bht,btd->bhd", p, c)
    return o.astype(q.dtype)


# Backwards-compatible alias: the MLX path doubles as the numerical reference.
_mla_fp8_decode_reference = _mla_fp8_decode_mlx

#: Decode strategies:
#:  * ``mlx``    — fused einsum (default; fastest on MLX 0.31.2; runs on any backend).
#:  * ``kernel`` — all-in-one scalar Metal kernel; no bf16 materialisation, lowest extra
#:                 memory, but compute-bound and slower than ``mlx`` (kept for that niche
#:                 and as the explicit "absorbed MLA Metal kernel"; an MMA version is TODO).
#:  * ``sdpa``   — fused dequant Metal kernel + MLX SDPA; materialises a bf16 latent.
DECODE_METHODS = ("mlx", "kernel", "sdpa")
MLA_FP8_DECODE_ENV = "MLX_VLM_MLA_FP8_DECODE"


def mla_fp8_decode_method() -> str:
    """Decode strategy from ``MLX_VLM_MLA_FP8_DECODE`` (defaults to ``mlx``)."""
    m = os.environ.get(MLA_FP8_DECODE_ENV, "mlx").strip().lower()
    return m if m in DECODE_METHODS else "mlx"


def mla_fp8_decode(
    q: mx.array,
    pe: mx.array,
    latent_codes: mx.array,
    latent_scales: mx.array,
    scale: float,
    group_size: int = DEFAULT_GROUP_SIZE,
    method: Optional[str] = None,
) -> mx.array:
    """Single-token absorbed-MLA attention over the FP8 latent.

    Args:
      q:             ``(B, H, D)`` absorbed query (``embed_q(q_nope)``), D == kv_lora_rank.
      pe:            ``(B, H, T)`` additive RoPE scores ``scale * (q_pe . k_pe)``.
      latent_codes:  ``(B, T, D)`` uint8 e4m3 latent.
      latent_scales: ``(B, T, D//group_size)`` fp32 group scales.
      scale:         attention softmax scale.
      method:        one of :data:`DECODE_METHODS`; ``None`` -> ``mlx`` (or the env override
                     only via the model path). Metal-only methods fall back to ``mlx``.
    Returns ``(B, H, D)`` latent-space output (caller applies ``unembed_out``).
    """
    if method is None:
        method = "mlx"
    if method in ("kernel", "sdpa") and not _metal_available():
        method = "mlx"
    if method == "kernel":
        return _mla_fp8_decode_metal(
            q, pe, latent_codes, latent_scales, scale, group_size
        )
    if method == "sdpa":
        return _mla_fp8_decode_sdpa(
            q, pe, latent_codes, latent_scales, scale, group_size
        )
    return _mla_fp8_decode_mlx(q, pe, latent_codes, latent_scales, scale, group_size)


def mla_fp8_attention(
    qa: mx.array,
    q_pe: mx.array,
    latent: mx.array,
    k_pe: mx.array,
    cache: Fp8MLAKVCache,
    scale: float,
    method: Optional[str] = None,
    mask: Optional[mx.array] = None,
) -> mx.array:
    """Absorbed-MLA attention (prefill + decode) returning latent-space output.

    Args:
      qa:     ``(B, H, L, D)`` absorbed query (``embed_q(q_nope)``).
      q_pe:   ``(B, H, L, R)`` RoPE query (already rotated).
      latent: ``(B, L, D)`` new latent to cache (post ``kv_a_layernorm``).
      k_pe:   ``(B, L, R)`` new RoPE key to cache (already rotated).
      method: decode strategy, see :func:`mla_fp8_decode` (``None`` picks the best).
      mask:   optional additive attention bias ``(.., L, T)`` (e.g. padding) applied on top
              of the built-in causal mask during prefill. ``None`` / non-array is ignored.
    Returns ``(B, H, L, D)`` latent-space output (caller applies ``unembed_out``).
    """
    B, H, L, D = qa.shape
    latent_codes, latent_scales, k_pe_all = cache.update_and_fetch(latent, k_pe)
    T = latent_codes.shape[1]
    # pe scores: (B, H, L, T) = scale * q_pe @ k_pe_all^T (rope key shared across heads)
    pe = scale * (q_pe @ k_pe_all[:, None].swapaxes(-1, -2))

    if L == 1:
        o = mla_fp8_decode(
            qa[:, :, 0, :],
            pe[:, :, 0, :],
            latent_codes,
            latent_scales,
            scale,
            cache.group_size,
            method=method,
        )
        return o[:, :, None, :]

    # Prefill: dequantise the cache and run full attention with a causal mask so the
    # stored (quantised) KV is what gets attended to — consistent with later decodes.
    c = dequantize_latent_fp8(latent_codes, latent_scales, cache.group_size, mx.float32)
    dot = mx.einsum("bhld,btd->bhlt", qa.astype(mx.float32), c)
    s = scale * dot + pe.astype(mx.float32)
    prev = cache.offset - L
    qpos = prev + mx.arange(L)[:, None]
    kpos = mx.arange(T)[None, :]
    s = mx.where(kpos <= qpos, s, mx.array(-mx.inf, mx.float32))
    if isinstance(mask, mx.array):  # e.g. a padding mask for batched prefill
        s = s + mask.astype(mx.float32)
    p = mx.softmax(s, axis=-1, precise=True)
    o = mx.einsum("bhlt,btd->bhld", p, c)
    return o.astype(qa.dtype)
