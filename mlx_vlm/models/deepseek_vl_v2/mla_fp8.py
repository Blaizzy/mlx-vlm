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

#: Head-block size for the MMA kernel: 8 query heads share one simdgroup and one streamed
#: C tile (MQA -> the latent is read once per tile and reused across the head-block). 8 is
#: the simdgroup-matrix tile edge, so QK^T / PV map straight onto 8x8 matrix ops.
MMA_HEAD_BLOCK = 8

#: Context tokens per split-K tile for the MMA kernel (one tile per resident simdgroup).
MMA_TILE = 64

#: Simdgroups co-resident per threadgroup. They share the 16 KB staged query (read once)
#: and each drives a different split-K tile, so one simdgroup's FP8 dequant overlaps
#: another's matrix ops -- the structural win over a single-simdgroup decode. Bounded by the
#: 32 KB threadgroup budget: 16 KB query + MMA_WARPS x (score + streamed-C + temps).
MMA_WARPS = 3

#: D-chunk width for streaming the FP8 latent in the MMA kernel. Each chunk dequantises an
#: 8-token x MMA_CHUNK_W slice into threadgroup memory once and feeds (MMA_CHUNK_W/8) matrix
#: ops, amortising the dequant barrier; MMA_CHUNK_W/8 also bounds the PV register pressure.
MMA_CHUNK_W = 64

_PARTIAL_KERNEL = None
_PARTIAL_KERNEL_MMA = None
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


def _partial_kernel_mma():
    """MMA-tiled split-K flash decode on the 8x8 simdgroup matrix units.

    Eight query heads form a head-block and share one simdgroup. Because the MLA latent is
    MQA (shared across heads), those eight heads read the same streamed C tile, so each FP8
    sub-tile is dequantised once and reused across the head-block. Both matmuls map onto the
    matrix units:

      * QK^T:  S(8h x 8t) += Q(8h x 8d) @ C(8t x 8d)^T   -- C loaded transposed
      * PV:    O(8h x 8d) += P(8h x 8t) @ C(8t x 8d)

    ``MMA_WARPS`` simdgroups are co-resident per threadgroup. They share the staged float
    query (loaded once) and each drives a different split-K tile, so one simdgroup's FP8
    dequant overlaps another's matrix ops -- this is what keeps the matrix units fed in this
    latency-bound single-token decode.

    The FP8 dequant (inline e4m3 decode * group scale) is fused into the matmul feed: the
    latent is streamed straight from device memory as 1 byte/elem and dequantised into a
    small ``8 x ChunkW`` threadgroup scratch just before each ``simdgroup_load`` -- no T x D
    bf16/fp32 latent is ever materialised, so device traffic for the dominant latent is
    ~half the bf16 cost.

    Output is the same split-K partial layout as :func:`_partial_kernel` (running max + the
    tile's unnormalised exp-sum + the weighted latent sum), so :func:`_combine_kernel` merges
    both kernels' partials identically.
    """
    global _PARTIAL_KERNEL_MMA
    if _PARTIAL_KERNEL_MMA is not None:
        return _PARTIAL_KERNEL_MMA
    # Inline OCP-e4m3 -> float (S.EEEE.MMM, bias 7, no inf/NaN in this runtime). Bit-exact
    # with the LUT but with no dependent memory gather, so the dequant pipelines on-SIMD.
    header = """
        inline float e4m3f(uchar bcode) {
            uint e = (bcode >> 3) & 0xFu;
            uint m = bcode & 0x7u;
            float sign = (bcode & 0x80u) ? -1.0f : 1.0f;
            float val = (e == 0u)
                ? ((float)m * 0.001953125f)                       // 2^-6 / 8 (subnormal)
                : (1.0f + (float)m * 0.125f) * exp2((float)((int)e - 7));
            return sign * val;
        }
    """
    source = """
        uint lane = thread_position_in_threadgroup.x;   // 0..31 within the simdgroup
        uint warp = thread_position_in_threadgroup.z;   // 0..NW-1 (one tile each)
        uint hblk = threadgroup_position_in_grid.y;     // (batch * n_head_blocks) + head_block
        uint tgz  = threadgroup_position_in_grid.z;     // tile-group index
        int H        = meta[1];
        int ctx      = meta[2];
        int ng       = meta[3];
        int n_tiles  = meta[5];
        int nhb      = meta[6];                  // head-blocks per batch = ceil(H / 8)
        int b   = (int)hblk / nhb;
        int hb  = (int)hblk % nhb;
        int h0  = hb * 8;                         // first head of this block
        float sc = scale[0];
        int tile = (int)tgz * NW + (int)warp;     // this simdgroup's split-K tile

        const int CWB = ChunkW / 8;               // d-blocks per streamed C chunk
        const int NCH = Dim / ChunkW;             // number of D chunks

        // Q (8 heads x Dim) staged once as float and shared by all NW simdgroups -> matrix
        // loads work for bf16/fp16 queries and Q is read once per threadgroup. Per-simdgroup
        // scratch streams C in 8t x ChunkW chunks, dequantised inline from FP8 just-in-time.
        threadgroup float q_sh[8 * Dim];
        threadgroup float cchunk[NW][8 * ChunkW];   // 8 tokens x ChunkW dequantised latent
        threadgroup float c8[NW][64];               // 8x8 store temp (scores / output tiles)
        threadgroup float p_sh[NW][64];             // 8h x 8t probability tile (PV phase)
        threadgroup float s_sh[NW][8 * TileTok];    // scores, then probabilities (8h x TileT)

        for (int i = (int)(warp * 32 + lane); i < 8 * Dim; i += NW * 32) {
            int hh = i / Dim;
            int dd = i % Dim;
            int hidx = h0 + hh;
            q_sh[i] = (hidx < H) ? (float)q[((b * H) + hidx) * Dim + dd] : 0.0f;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (tile >= n_tiles) return;              // tail simdgroup with no tile

        threadgroup float* cc = cchunk[warp];
        threadgroup float* my8 = c8[warp];
        threadgroup float* myp = p_sh[warp];
        threadgroup float* ss = s_sh[warp];

        int t0 = tile * TileTok;
        int t1 = min(t0 + TileTok, ctx);
        int nvalid = t1 - t0;
        int ntb = (nvalid + 7) / 8;               // token-blocks in this tile

        // ---- Phase 1: scores S(8h x TileT) = sc * (Q @ C^T) + pe ----
        for (int tb = 0; tb < ntb; ++tb) {
            int tt0 = t0 + tb * 8;
            simdgroup_float8x8 Sacc = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
            for (int ch = 0; ch < NCH; ++ch) {
                int dch = ch * ChunkW;
                // Dequant an 8 x ChunkW chunk (inline e4m3 * group scale); one simdgroup
                // barrier amortised over the CWB matmuls below.
                for (int i = (int)lane; i < 8 * ChunkW; i += 32) {
                    int tr = i / ChunkW;
                    int d  = dch + (i % ChunkW);
                    int t  = tt0 + tr;
                    float val = 0.0f;
                    if (t < ctx) {
                        int lbase = (b * ctx + t) * Dim + d;
                        int sbase = (b * ctx + t) * ng + (d / GroupSize);
                        val = e4m3f(latent_codes[lbase]) * latent_scales[sbase];
                    }
                    cc[i] = val;
                }
                simdgroup_barrier(mem_flags::mem_threadgroup);
                #pragma clang loop unroll(full)
                for (int j = 0; j < CWB; ++j) {
                    simdgroup_float8x8 Qm, Cm;
                    simdgroup_load(Qm, q_sh + (dch + j * 8), Dim);            // 8h x 8d
                    simdgroup_load(Cm, cc + j * 8, ChunkW, ulong2(0, 0), true); // ->8d x 8t
                    simdgroup_multiply_accumulate(Sacc, Qm, Cm, Sacc);
                }
                simdgroup_barrier(mem_flags::mem_threadgroup);
            }
            simdgroup_store(Sacc, my8, 8);                                   // 8h x 8t
            simdgroup_barrier(mem_flags::mem_threadgroup);
            for (int e = (int)lane; e < 64; e += 32) {
                int hh = e / 8;
                int tr = e % 8;
                int tcol = tb * 8 + tr;
                if (tcol < nvalid) {
                    int hidx = h0 + hh;
                    float s = sc * my8[e];
                    if (hidx < H) s += (float)pe[((b * H) + hidx) * ctx + (t0 + tcol)];
                    ss[hh * TileTok + tcol] = s;
                }
            }
            simdgroup_barrier(mem_flags::mem_threadgroup);
        }

        // ---- Online softmax over this tile's tokens, per head (8 lanes own 8 heads) ----
        if ((int)lane < 8) {
            int hh = (int)lane;
            float m = -1e30f;
            for (int u = 0; u < nvalid; ++u) m = fmax(m, ss[hh * TileTok + u]);
            float l = 0.0f;
            for (int u = 0; u < nvalid; ++u) {
                float p = fast::exp(ss[hh * TileTok + u] - m);
                ss[hh * TileTok + u] = p;     // overwrite scores with probabilities
                l += p;
            }
            int hidx = h0 + hh;
            if (hidx < H) {
                int pidx = ((b * H) + hidx) * n_tiles + tile;
                partial_m[pidx] = m;
                partial_l[pidx] = l;
            }
        }
        simdgroup_barrier(mem_flags::mem_threadgroup);

        // ---- Phase 3: O(8h x D) = P(8h x T) @ C(T x D), one D chunk at a time. Each chunk
        // keeps CWB output accumulators in registers (bounded), streams C once per token-
        // block, and stores its 8h x ChunkW slice of the partial output. ----
        for (int ch = 0; ch < NCH; ++ch) {
            int dch = ch * ChunkW;
            simdgroup_float8x8 Oacc[CWB];
            #pragma clang loop unroll(full)
            for (int j = 0; j < CWB; ++j)
                Oacc[j] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);

            for (int tb = 0; tb < ntb; ++tb) {
                int tt0 = t0 + tb * 8;
                for (int i = (int)lane; i < 8 * ChunkW; i += 32) {
                    int tr = i / ChunkW;
                    int d  = dch + (i % ChunkW);
                    int t  = tt0 + tr;
                    float val = 0.0f;
                    if (t < ctx) {
                        int lbase = (b * ctx + t) * Dim + d;
                        int sbase = (b * ctx + t) * ng + (d / GroupSize);
                        val = e4m3f(latent_codes[lbase]) * latent_scales[sbase];
                    }
                    cc[i] = val;
                }
                for (int e = (int)lane; e < 64; e += 32) {
                    int hh = e / 8;
                    int tr = e % 8;
                    int tcol = tb * 8 + tr;
                    myp[e] = (tcol < nvalid) ? ss[hh * TileTok + tcol] : 0.0f;
                }
                simdgroup_barrier(mem_flags::mem_threadgroup);
                simdgroup_float8x8 Pm;
                simdgroup_load(Pm, myp, 8);                                 // 8h x 8t
                #pragma clang loop unroll(full)
                for (int j = 0; j < CWB; ++j) {
                    simdgroup_float8x8 Cm;
                    simdgroup_load(Cm, cc + j * 8, ChunkW);                 // 8t x 8d
                    simdgroup_multiply_accumulate(Oacc[j], Pm, Cm, Oacc[j]);
                }
                simdgroup_barrier(mem_flags::mem_threadgroup);
            }

            #pragma clang loop unroll(full)
            for (int j = 0; j < CWB; ++j) {
                int d0 = dch + j * 8;
                simdgroup_store(Oacc[j], my8, 8);                          // 8h x 8d
                simdgroup_barrier(mem_flags::mem_threadgroup);
                for (int e = (int)lane; e < 64; e += 32) {
                    int hh = e / 8;
                    int dc = e % 8;
                    int hidx = h0 + hh;
                    if (hidx < H) {
                        int pidx = ((b * H) + hidx) * n_tiles + tile;
                        partial_acc[pidx * Dim + d0 + dc] = my8[e];
                    }
                }
                simdgroup_barrier(mem_flags::mem_threadgroup);
            }
        }
    """
    _PARTIAL_KERNEL_MMA = mx.fast.metal_kernel(
        name="mla_fp8_decode_partial_mma",
        input_names=[
            "q",
            "pe",
            "latent_codes",
            "latent_scales",
            "meta",
            "scale",
        ],
        output_names=["partial_m", "partial_l", "partial_acc"],
        source=source,
        header=header,
    )
    return _PARTIAL_KERNEL_MMA


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


def _mla_fp8_decode_metal_mma(
    q: mx.array,
    pe: mx.array,
    latent_codes: mx.array,
    latent_scales: mx.array,
    scale: float,
    group_size: int,
    tile: int = MMA_TILE,
) -> mx.array:
    """MMA-tiled FP8-fused split-K flash decode (the ``kernel`` path).

    Maps the absorbed-MLA decode onto the 8x8 simdgroup matrix units, streaming the FP8
    latent in ``MMA_CHUNK_W``-wide chunks and dequantising them inline just before each
    matmul (see :func:`_partial_kernel_mma`). Requires ``D % 32 == 0`` -- the shared
    :func:`_combine_kernel` lane-stripes the partial output in groups of 32, so a tail of
    ``D % 32`` dims would be dropped. Every MLA ``kv_lora_rank`` (512, ...) satisfies this;
    :func:`mla_fp8_decode` falls back to ``mlx`` otherwise.
    """
    B, H, D = q.shape
    if D % _SIMD != 0:
        raise ValueError(
            f"kv_lora_rank ({D}) must be divisible by {_SIMD} for the MMA 'kernel' decode "
            f"path; use method='mlx' or method='sdpa'."
        )
    T = latent_codes.shape[1]
    ng = D // group_size
    # Largest <= MMA_CHUNK_W d-chunk width that is a multiple of 8 and evenly divides D
    # (D % 32 == 0 guarantees at least 32 works). Keeps streamed-C tiles aligned to D.
    chunk_w = max(w for w in range(8, min(MMA_CHUNK_W, D) + 1, 8) if D % w == 0)
    nhb = (H + MMA_HEAD_BLOCK - 1) // MMA_HEAD_BLOCK
    n_tiles = max(1, (T + tile - 1) // tile)
    nw = MMA_WARPS
    n_tilegroups = (n_tiles + nw - 1) // nw  # threadgroups along the split-K axis
    pe = pe.astype(mx.float32)
    scale_arr = mx.array([scale], dtype=mx.float32)

    partial_m, partial_l, partial_acc = _partial_kernel_mma()(
        inputs=[
            q,
            pe,
            latent_codes,
            latent_scales,
            mx.array([B, H, T, ng, tile, n_tiles, nhb], dtype=mx.int32),
            scale_arr,
        ],
        template=[
            ("Dim", D),
            ("GroupSize", group_size),
            ("TileTok", tile),
            ("ChunkW", chunk_w),
            ("NW", nw),
        ],
        grid=(_SIMD, B * nhb, n_tilegroups * nw),
        threadgroup=(_SIMD, 1, nw),
        output_shapes=[(B * H, n_tiles), (B * H, n_tiles), (B * H, n_tiles, D)],
        output_dtypes=[mx.float32, mx.float32, mx.float32],
    )

    out = _combine_kernel()(
        inputs=[partial_m, partial_l, partial_acc, mx.array([n_tiles], dtype=mx.int32)],
        template=[("T_out", q.dtype), ("Dim", D), ("DimsPerLane", D // _SIMD)],
        grid=(_SIMD, B * H, 1),
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

#: Decode strategies (``mlx`` is the default and the fastest correct path on MLX 0.31.2):
#:  * ``mlx``        — fused einsum (default). MLX fuses the lazy FP8 dequant into the matmul,
#:                     so no bf16 latent is materialised and the GEMMs use the matrix units.
#:  * ``sdpa``       — fused dequant Metal kernel + MLX SDPA; materialises a bf16 latent.
#:  * ``kernel``     — scalar all-in-one flash Metal kernel; no bf16 materialisation, lowest
#:                     extra memory, but compute-bound (per-lane scalar dot). Needs ``D % 32 == 0``.
#:  * ``kernel_mma`` — MMA-tiled FP8-fused flash kernel (8x8 simdgroup matrix units). Numerically
#:                     correct (cosine ~1.0) but a *measured perf regression* vs ``mlx``/``kernel``
#:                     for B=1 single-token decode: that shape is overhead/latency-bound (not FLOP-
#:                     or bandwidth-bound), so the matrix units stay starved and a JIT
#:                     ``metal_kernel`` cannot out-pipeline MLX's compiled SDPA. Kept as a reference
#:                     implementation and for batched-decode evaluation — not for default use.
#:
#: NOTE: FP8-MLA is a *memory* win (~43% smaller latent; reaches contexts the expanded cache OOMs
#: on), not a single-token decode *speed* win — at B=1 bandwidth is not the limiter, so FP8 only
#: adds dequant cost. See ``tests/bench_mla_fp8.py``.
DECODE_METHODS = ("mlx", "kernel", "kernel_mma", "sdpa")
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
    if method in ("kernel", "kernel_mma", "sdpa") and not _metal_available():
        method = "mlx"
    if method == "kernel":
        # Scalar Metal kernel. The split-K combine needs D % 32 == 0 (every MLA kv_lora_rank
        # satisfies this); fall back rather than raise on an exotic dim.
        if q.shape[-1] % _SIMD == 0:
            return _mla_fp8_decode_metal(
                q, pe, latent_codes, latent_scales, scale, group_size
            )
        method = "mlx"
    if method == "kernel_mma":
        # MMA reference path (non-default; slower than ``mlx`` at B=1 — see DECODE_METHODS).
        # Same D % 32 == 0 requirement; fall back rather than miscompute.
        if q.shape[-1] % _SIMD == 0:
            return _mla_fp8_decode_metal_mma(
                q, pe, latent_codes, latent_scales, scale, group_size
            )
        method = "mlx"
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
