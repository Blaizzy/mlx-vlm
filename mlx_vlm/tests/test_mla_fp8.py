"""Tests for the FP8 MLA KV cache and the absorbed-decode Metal kernel.

Covers: e4m3 LUT exactness, grouped quant round-trip error, the 656 B/token layout,
cache append/grow/trim/state, the fused Metal decode kernel vs the fp32 reference, and the
full prefill+decode attention path vs a from-scratch recompute.
"""

import math

import mlx.core as mx
import pytest

from mlx_vlm.mla_fp8 import (
    Fp8MLAKVCache,
    _mla_fp8_decode_metal,
    _mla_fp8_decode_metal_mma,
    _mla_fp8_decode_reference,
    dequantize_latent_fp8,
    fp8_lut,
    mla_fp8_attention,
    mla_fp8_decode,
    mla_fp8_enabled,
    mla_fp8_group_size,
    quantize_latent_fp8,
)

requires_metal = pytest.mark.skipif(
    not mx.metal.is_available(), reason="Metal kernels are unavailable on this host"
)

# DeepSeek-V2/V3 attention dims used throughout.
KV_LORA = 512
ROPE = 64
Q_HEAD_DIM = 192  # qk_nope_head_dim(128) + qk_rope_head_dim(64)
SCALE = 1.0 / math.sqrt(Q_HEAD_DIM)


def _max_rel_err(a, b):
    a = a.astype(mx.float32)
    b = b.astype(mx.float32)
    denom = float(mx.max(mx.abs(b)))
    return float(mx.max(mx.abs(a - b))) / (denom if denom else 1.0)


def _cosine(a, b):
    a = a.astype(mx.float32).reshape(-1)
    b = b.astype(mx.float32).reshape(-1)
    return float(mx.sum(a * b) / (mx.sqrt(mx.sum(a * a)) * mx.sqrt(mx.sum(b * b))))


# --------------------------------------------------------------------------- #
# FP8 quant primitives
# --------------------------------------------------------------------------- #


def test_fp8_lut_matches_from_fp8():
    lut = fp8_lut()
    ref = mx.from_fp8(mx.arange(256, dtype=mx.uint8), dtype=mx.float32)
    assert lut.shape == (256,)
    assert lut.dtype == mx.float32
    # Bit-exact (no NaN codes in this runtime's e4m3).
    assert float(mx.sum(mx.abs(lut - ref))) == 0.0


@pytest.mark.parametrize("group_size", [64, 128, 256, 512])
def test_quant_roundtrip_error(group_size):
    mx.random.seed(0)
    latent = mx.random.normal((3, 17, KV_LORA)).astype(mx.float32)
    codes, scales = quantize_latent_fp8(latent, group_size)
    assert codes.dtype == mx.uint8
    assert codes.shape == latent.shape
    assert scales.shape == (3, 17, KV_LORA // group_size)
    deq = dequantize_latent_fp8(codes, scales, group_size)
    rel = float(mx.sqrt(mx.mean((deq - latent) ** 2)) / mx.sqrt(mx.mean(latent**2)))
    # e4m3 with absmax group scales: ~2.5% RMS on N(0,1); allow headroom.
    assert rel < 0.04, rel


def test_quant_zero_and_saturation():
    latent = mx.zeros((1, 1, KV_LORA))
    latent = mx.concatenate([latent, mx.full((1, 1, KV_LORA), 1e4)], axis=1)
    codes, scales = quantize_latent_fp8(latent, 128)
    deq = dequantize_latent_fp8(codes, scales, 128)
    assert float(mx.max(mx.abs(deq[:, 0]))) == 0.0  # all-zero group stays zero
    # Large-magnitude group survives (scaled into e4m3 range), finite, right ballpark.
    assert bool(mx.all(mx.isfinite(deq)))
    assert _max_rel_err(deq[:, 1], latent[:, 1]) < 0.05


def test_quant_requires_divisible_group():
    with pytest.raises(ValueError):
        quantize_latent_fp8(mx.zeros((1, 1, 500)), 128)
    with pytest.raises(ValueError):
        dequantize_latent_fp8(mx.zeros((1, 1, 500), mx.uint8), mx.zeros((1, 1, 4)), 128)


@requires_metal
def test_kernel_rejects_non_simd_lora():
    # kv_lora_rank not divisible by 32 is unsupported by both Metal kernels (the split-K
    # combine lane-stripes the output in groups of 32). The low-level scalar kernel raises;
    # the public 'kernel'/'kernel_mma' dispatch falls back to the mlx path instead.
    D, G = 48, 48  # 48 % 32 != 0
    codes, scales = quantize_latent_fp8(mx.random.normal((1, 4, D)), G)
    q = mx.random.normal((1, 2, D)).astype(mx.float32)
    pe = mx.random.normal((1, 2, 4)).astype(mx.float32)
    with pytest.raises(ValueError):
        _mla_fp8_decode_metal(q, pe, codes, scales, SCALE, G)
    # public dispatch: D % 32 != 0 transparently falls back to mlx, matching its output.
    ref = _mla_fp8_decode_reference(q, pe, codes, scales, SCALE, G)
    for method in ("kernel", "kernel_mma", "mlx"):
        out = mla_fp8_decode(q, pe, codes, scales, SCALE, G, method=method)
        assert out.shape == (1, 2, D)
        assert _max_rel_err(out, ref) == 0.0


# --------------------------------------------------------------------------- #
# Cache
# --------------------------------------------------------------------------- #


def test_bytes_per_token_is_656():
    cache = Fp8MLAKVCache(group_size=128)
    assert cache.bytes_per_token(KV_LORA, ROPE) == 656
    bf16_latent = KV_LORA * 2 + ROPE * 2  # 1152
    assert cache.bytes_per_token(KV_LORA, ROPE) / bf16_latent < 0.58  # ~43% smaller


def test_cache_append_grow_and_offset():
    mx.random.seed(1)
    cache = Fp8MLAKVCache(group_size=128)
    B = 2
    # Cross the step boundary (step=256) in one prefill, then decode tokens.
    P = 300
    lat = mx.random.normal((B, P, KV_LORA)).astype(mx.bfloat16)
    kpe = mx.random.normal((B, P, ROPE)).astype(mx.bfloat16)
    c, s, k = cache.update_and_fetch(lat, kpe)
    assert cache.offset == P
    assert (
        c.shape == (B, P, KV_LORA) and s.shape == (B, P, 4) and k.shape == (B, P, ROPE)
    )
    for _ in range(5):
        c, s, k = cache.update_and_fetch(
            mx.random.normal((B, 1, KV_LORA)).astype(mx.bfloat16),
            mx.random.normal((B, 1, ROPE)).astype(mx.bfloat16),
        )
    assert cache.offset == P + 5
    assert c.shape[1] == P + 5


def test_cache_trim_and_state():
    mx.random.seed(2)
    cache = Fp8MLAKVCache(group_size=128)
    lat = mx.random.normal((1, 10, KV_LORA)).astype(mx.bfloat16)
    kpe = mx.random.normal((1, 10, ROPE)).astype(mx.bfloat16)
    cache.update_and_fetch(lat, kpe)
    state = cache.state
    assert state[0].shape[1] == 10
    assert cache.trim(3) == 3
    assert cache.offset == 7
    # state setter restores offset
    cache.state = state
    assert cache.offset == 10


def test_cache_trim_then_grow_preserves_data():
    # Exercises the prev % step != 0 regrow branch and checks earlier tokens survive.
    mx.random.seed(21)
    G = 128
    cache = Fp8MLAKVCache(group_size=G)
    lat1 = mx.random.normal((1, 260, KV_LORA)).astype(mx.bfloat16)  # crosses step=256
    kpe1 = mx.random.normal((1, 260, ROPE)).astype(mx.bfloat16)
    cache.update_and_fetch(lat1, kpe1)
    ref = dequantize_latent_fp8(*quantize_latent_fp8(lat1[:, :100], G), G)

    cache.trim(10)  # offset 250 -> not a multiple of step
    assert cache.offset == 250
    lat2 = mx.random.normal((1, 300, KV_LORA)).astype(mx.bfloat16)
    kpe2 = mx.random.normal((1, 300, ROPE)).astype(mx.bfloat16)
    codes, scales, _ = cache.update_and_fetch(
        lat2, kpe2
    )  # forces grow with prev%step!=0
    assert cache.offset == 550
    got = dequantize_latent_fp8(codes[:, :100], scales[:, :100], G)
    assert _max_rel_err(got, ref) == 0.0  # first 100 tokens intact through trim+grow


# --------------------------------------------------------------------------- #
# Reference correctness (no Metal needed)
# --------------------------------------------------------------------------- #


def test_reference_matches_dense_softmax():
    mx.random.seed(3)
    B, H, T = 2, 4, 23
    lat = mx.random.normal((B, T, KV_LORA)).astype(mx.float32)
    codes, scales = quantize_latent_fp8(lat, 128)
    q = mx.random.normal((B, H, KV_LORA)).astype(mx.float32)
    pe = mx.random.normal((B, H, T)).astype(mx.float32)
    out = _mla_fp8_decode_reference(q, pe, codes, scales, SCALE, 128)
    c = dequantize_latent_fp8(codes, scales, 128)
    s = SCALE * mx.einsum("bhd,btd->bht", q, c) + pe
    p = mx.softmax(s, axis=-1, precise=True)
    expect = mx.einsum("bht,btd->bhd", p, c)
    assert _max_rel_err(out, expect) < 1e-5


def test_decode_dispatch_fallback_equals_reference():
    mx.random.seed(4)
    B, H, T = 1, 3, 11
    lat = mx.random.normal((B, T, KV_LORA)).astype(mx.float32)
    codes, scales = quantize_latent_fp8(lat, 128)
    q = mx.random.normal((B, H, KV_LORA)).astype(mx.float32)
    pe = mx.random.normal((B, H, T)).astype(mx.float32)
    fallback = mla_fp8_decode(q, pe, codes, scales, SCALE, 128, method="mlx")
    ref = _mla_fp8_decode_reference(q, pe, codes, scales, SCALE, 128)
    assert _max_rel_err(fallback, ref) == 0.0


# --------------------------------------------------------------------------- #
# Metal kernel vs reference
# --------------------------------------------------------------------------- #


@requires_metal
@pytest.mark.parametrize("group_size", [64, 128])
@pytest.mark.parametrize("B,H,T", [(1, 1, 1), (2, 8, 37), (1, 16, 256), (3, 4, 129)])
def test_metal_decode_matches_reference_fp32(group_size, B, H, T):
    mx.random.seed(B * 100 + H * 10 + T)
    lat = mx.random.normal((B, T, KV_LORA)).astype(mx.float32)
    codes, scales = quantize_latent_fp8(lat, group_size)
    q = mx.random.normal((B, H, KV_LORA)).astype(mx.float32)
    pe = mx.random.normal((B, H, T)).astype(mx.float32)
    ker = _mla_fp8_decode_metal(q, pe, codes, scales, SCALE, group_size)
    ref = _mla_fp8_decode_reference(q, pe, codes, scales, SCALE, group_size)
    assert ker.shape == (B, H, KV_LORA)
    assert _max_rel_err(ker, ref) < 2e-3
    assert _cosine(ker, ref) > 0.9999


@requires_metal
@pytest.mark.parametrize("group_size", [64, 128])
@pytest.mark.parametrize(
    "B,H,T",
    [
        (1, 1, 1),  # single head -> single (padded) head-block
        (2, 8, 37),  # exactly one head-block, multi-batch
        (1, 16, 256),  # two full head-blocks, one split-K tile boundary
        (3, 4, 129),  # tail head-block (H=4 < 8) across batches
        (1, 12, 300),  # H=12 -> a full block + a 4-head tail block, multi-tile split-K
        (1, 20, 700),  # H not divisible by 8 with several split-K tiles
        (1, 128, 600),  # production head count, multi-tile
    ],
)
def test_mma_kernel_matches_reference_fp32(group_size, B, H, T):
    # The MMA 'kernel' path: validates QK^T/PV on the matrix units and the H%8 tail block.
    mx.random.seed(B * 1000 + H * 10 + T + group_size)
    lat = mx.random.normal((B, T, KV_LORA)).astype(mx.float32)
    codes, scales = quantize_latent_fp8(lat, group_size)
    q = mx.random.normal((B, H, KV_LORA)).astype(mx.float32)
    pe = mx.random.normal((B, H, T)).astype(mx.float32)
    ker = _mla_fp8_decode_metal_mma(q, pe, codes, scales, SCALE, group_size)
    ref = _mla_fp8_decode_reference(q, pe, codes, scales, SCALE, group_size)
    assert ker.shape == (B, H, KV_LORA)
    assert _max_rel_err(ker, ref) < 2e-3
    assert _cosine(ker, ref) > 0.9999


@requires_metal
def test_mma_kernel_bf16_dtype():
    # MMA kernel preserves the query dtype on output (bf16 in, bf16 out).
    mx.random.seed(91)
    B, H, T = 2, 24, 200
    lat = mx.random.normal((B, T, KV_LORA)).astype(mx.bfloat16)
    codes, scales = quantize_latent_fp8(lat, 128)
    q = mx.random.normal((B, H, KV_LORA)).astype(mx.bfloat16)
    pe = mx.random.normal((B, H, T)).astype(mx.float32)
    ker = _mla_fp8_decode_metal_mma(q, pe, codes, scales, SCALE, 128)
    ref = _mla_fp8_decode_reference(q, pe, codes, scales, SCALE, 128)
    assert ker.dtype == mx.bfloat16
    assert _cosine(ker, ref) > 0.999


@requires_metal
def test_metal_decode_bf16():
    mx.random.seed(9)
    B, H, T = 2, 8, 64
    lat = mx.random.normal((B, T, KV_LORA)).astype(mx.bfloat16)
    codes, scales = quantize_latent_fp8(lat, 128)
    q = mx.random.normal((B, H, KV_LORA)).astype(mx.bfloat16)
    pe = mx.random.normal((B, H, T)).astype(mx.float32)
    ker = _mla_fp8_decode_metal(q, pe, codes, scales, SCALE, 128)
    ref = _mla_fp8_decode_reference(q, pe, codes, scales, SCALE, 128)
    assert ker.dtype == mx.bfloat16
    assert _cosine(ker, ref) > 0.999


@requires_metal
@pytest.mark.parametrize("group_size", [64, 128])
def test_fused_dequant_kernel_matches_mlx(group_size):
    from mlx_vlm.mla_fp8 import dequantize_latent_fp8_metal

    mx.random.seed(5)
    lat = mx.random.normal((2, 70, KV_LORA)).astype(mx.float32)
    codes, scales = quantize_latent_fp8(lat, group_size)
    ker = dequantize_latent_fp8_metal(codes, scales, group_size, mx.float32)
    ref = dequantize_latent_fp8(codes, scales, group_size, mx.float32)
    assert ker.shape == lat.shape
    assert _max_rel_err(ker, ref) == 0.0  # same LUT + same scales -> bit-identical


@requires_metal
@pytest.mark.parametrize("B,H,T", [(1, 1, 1), (2, 8, 37), (1, 16, 300)])
def test_sdpa_decode_matches_reference(B, H, T):
    mx.random.seed(B * 7 + H + T)
    lat = mx.random.normal((B, T, KV_LORA)).astype(mx.bfloat16)
    codes, scales = quantize_latent_fp8(lat, 128)
    q = mx.random.normal((B, H, KV_LORA)).astype(mx.bfloat16)
    pe = mx.random.normal((B, H, T)).astype(mx.float32)
    sdpa = mla_fp8_decode(q, pe, codes, scales, SCALE, 128, method="sdpa")
    ref = _mla_fp8_decode_reference(q, pe, codes, scales, SCALE, 128)
    assert _cosine(sdpa, ref) > 0.999


# --------------------------------------------------------------------------- #
# Full attention path (prefill + decode) vs from-scratch recompute
# --------------------------------------------------------------------------- #


def _full_recompute(cache, group_size, qa1, qpe1, scale):
    """Single-query attention over the whole (quantised) cache, in fp32."""
    upto = cache.offset
    c = dequantize_latent_fp8(
        cache.latent_codes[:, :upto], cache.latent_scales[:, :upto], group_size
    )
    kpe = cache.k_pe[:, :upto].astype(mx.float32)
    dot = mx.einsum("bhd,btd->bht", qa1.astype(mx.float32), c)
    pe = scale * mx.einsum("bhr,btr->bht", qpe1.astype(mx.float32), kpe)
    p = mx.softmax(scale * dot + pe, axis=-1, precise=True)
    return mx.einsum("bht,btd->bhd", p, c)


@pytest.mark.parametrize("method", ["mlx", "kernel", "kernel_mma", "sdpa"])
def test_attention_decode_matches_full_recompute(method):
    if method != "mlx" and not mx.metal.is_available():
        pytest.skip("Metal unavailable")
    mx.random.seed(11)
    B, H, G = 2, 8, 128
    P, steps = 20, 6
    Ttot = P + steps
    lat = mx.random.normal((B, Ttot, KV_LORA)).astype(mx.bfloat16)
    kpe = mx.random.normal((B, Ttot, ROPE)).astype(mx.bfloat16)
    qa = mx.random.normal((B, H, Ttot, KV_LORA)).astype(mx.bfloat16)
    qpe = mx.random.normal((B, H, Ttot, ROPE)).astype(mx.bfloat16)

    cache = Fp8MLAKVCache(group_size=G)
    mla_fp8_attention(
        qa[:, :, :P], qpe[:, :, :P], lat[:, :P], kpe[:, :P], cache, SCALE, method=method
    )
    worst = 0.0
    for i in range(steps):
        pos = P + i
        o = mla_fp8_attention(
            qa[:, :, pos : pos + 1],
            qpe[:, :, pos : pos + 1],
            lat[:, pos : pos + 1],
            kpe[:, pos : pos + 1],
            cache,
            SCALE,
            method=method,
        )
        ref = _full_recompute(cache, G, qa[:, :, pos], qpe[:, :, pos], SCALE)
        worst = max(worst, _max_rel_err(o[:, :, 0], ref))
    assert cache.offset == Ttot
    assert worst < 2e-2, worst


def test_attention_prefill_is_causal():
    """A token's prefill output must not depend on future tokens."""
    mx.random.seed(13)
    B, H, P, G = 1, 2, 12, 128
    lat = mx.random.normal((B, P, KV_LORA)).astype(mx.bfloat16)
    kpe = mx.random.normal((B, P, ROPE)).astype(mx.bfloat16)
    qa = mx.random.normal((B, H, P, KV_LORA)).astype(mx.bfloat16)
    qpe = mx.random.normal((B, H, P, ROPE)).astype(mx.bfloat16)

    c1 = Fp8MLAKVCache(group_size=G)
    full = mla_fp8_attention(qa, qpe, lat, kpe, c1, SCALE, method="mlx")

    # Perturb the last token's latent; outputs for earlier tokens must be unchanged.
    lat2 = mx.array(lat)
    lat2[:, -1] = lat2[:, -1] + 5.0
    c2 = Fp8MLAKVCache(group_size=G)
    full2 = mla_fp8_attention(qa, qpe, lat2, kpe, c2, SCALE, method="mlx")
    assert _max_rel_err(full[:, :, :-1], full2[:, :, :-1]) < 1e-6
    assert (
        _max_rel_err(full[:, :, -1:], full2[:, :, -1:]) > 1e-3
    )  # last token did change


# --------------------------------------------------------------------------- #
# Env flag + end-to-end model layer parity (absorbed FP8 vs expanded bf16)
# --------------------------------------------------------------------------- #


def test_env_flag_parsing(monkeypatch):
    monkeypatch.delenv("MLX_VLM_MLA_FP8", raising=False)
    assert mla_fp8_enabled() is False
    for v in ("1", "true", "YES", "On"):
        monkeypatch.setenv("MLX_VLM_MLA_FP8", v)
        assert mla_fp8_enabled() is True
    monkeypatch.setenv("MLX_VLM_MLA_FP8", "0")
    assert mla_fp8_enabled() is False
    monkeypatch.delenv("MLX_VLM_MLA_FP8_GROUP", raising=False)
    assert mla_fp8_group_size() == 128
    monkeypatch.setenv("MLX_VLM_MLA_FP8_GROUP", "64")
    assert mla_fp8_group_size() == 64
    monkeypatch.setenv("MLX_VLM_MLA_FP8_GROUP", "garbage")
    assert mla_fp8_group_size() == 128  # falls back


def _build_attention(q_lora_rank):
    from mlx_vlm.models.deepseek_vl_v2.config import TextConfig
    from mlx_vlm.models.deepseek_vl_v2.language import DeepseekV2Attention

    cfg = TextConfig(
        hidden_size=256,
        num_attention_heads=4,
        num_key_value_heads=4,
        kv_lora_rank=256,
        q_lora_rank=q_lora_rank,
        qk_rope_head_dim=64,
        qk_nope_head_dim=128,
        v_head_dim=128,
        rope_scaling=None,
    )
    attn = DeepseekV2Attention(cfg)
    mx.eval(attn.parameters())
    return attn, cfg


@pytest.mark.parametrize("q_lora_rank", [None, 192])
def test_absorbed_fp8_matches_expanded_layer(q_lora_rank):
    """The opt-in absorbed FP8 path must equal the default expanded path up to FP8 error.

    This pins the kv_b_proj -> (embed_q, unembed_out) derivation: a wrong transpose would
    collapse the cosine similarity, whereas FP8 latent quant only perturbs it by ~2%.
    """
    from mlx_vlm.models.base import create_attention_mask
    from mlx_vlm.models.cache import KVCache

    mx.random.seed(0)
    attn, cfg = _build_attention(q_lora_rank)
    B, P, steps = 1, 12, 4
    x_all = 0.5 * mx.random.normal((B, P + steps, cfg.hidden_size)).astype(mx.bfloat16)

    expanded = KVCache()
    fp8 = Fp8MLAKVCache(group_size=128)
    xp = x_all[:, :P]
    oe = attn(xp, create_attention_mask(xp, [expanded]), expanded)
    of = attn(xp, None, fp8)
    assert _cosine(of, oe) > 0.995
    for i in range(steps):
        xd = x_all[:, P + i : P + i + 1]
        oe = attn(xd, None, expanded)
        of = attn(xd, None, fp8)
        assert _cosine(of, oe) > 0.995
    assert expanded.offset == fp8.offset == P + steps
