"""Benchmark the FP8 MLA KV cache: memory, decode latency, accuracy.

Run:  python -m mlx_vlm.tests.bench_mla_fp8
Prints markdown tables suitable for the PR description. Needs no model weights.
"""

import time

import mlx.core as mx

from mlx_vlm.mla_fp8 import (
    DEFAULT_GROUP_SIZE,
    _mla_fp8_decode_metal,
    _mla_fp8_decode_metal_mma,
    _mla_fp8_decode_reference,
    _mla_fp8_decode_sdpa,
    dequantize_latent_fp8,
    quantize_latent_fp8,
)

# (name, layers, heads, kv_lora_rank, qk_nope, qk_rope, v_head)
CONFIGS = [
    ("DeepSeek-V2", 60, 128, 512, 128, 64, 128),
    ("DeepSeek-V3 / R1", 61, 128, 512, 128, 64, 128),
    ("DeepSeek-VL2-tiny", 12, 16, 512, 128, 64, 128),
]


def _bytes_per_token(layers, H, lora, nope, rope, v, group):
    q_head = nope + rope
    expanded = layers * H * (q_head + v) * 2  # bf16 per-head K (q_head) + V (v)
    bf16_latent = layers * (lora + rope) * 2
    fp8_latent = layers * (lora * 1 + (lora // group) * 4 + rope * 2)
    return expanded, bf16_latent, fp8_latent


def memory_table(group=DEFAULT_GROUP_SIZE):
    print("\n### Memory — KV cache bytes/token (batch 1)\n")
    print(
        "| model | expanded bf16 (current mlx-vlm) | absorbed bf16 latent | "
        "absorbed FP8 (this PR) | FP8 vs bf16-latent | FP8 vs expanded |"
    )
    print("|---|--:|--:|--:|--:|--:|")
    for name, layers, H, lora, nope, rope, v in CONFIGS:
        exp, bf16, fp8 = _bytes_per_token(layers, H, lora, nope, rope, v, group)
        print(
            f"| {name} | {exp/1024:,.1f} KB | {bf16/1024:,.1f} KB | {fp8/1024:,.1f} KB | "
            f"{(1-fp8/bf16)*100:.0f}% smaller | {fp8/exp*100:.1f}% ({exp/fp8:.0f}× smaller) |"
        )

    print("\n### Total KV cache @ 128K context (batch 1)\n")
    print("| model | absorbed bf16 latent | absorbed FP8 (this PR) | saved |")
    print("|---|--:|--:|--:|")
    C = 128 * 1024 / 1e9  # tokens at 128K context, scaled to GB
    for name, layers, H, lora, nope, rope, v in CONFIGS:
        _, bf16, fp8 = _bytes_per_token(layers, H, lora, nope, rope, v, group)
        print(
            f"| {name} | {bf16 * C:.2f} GB | {fp8 * C:.2f} GB | {(bf16 - fp8) * C:.2f} GB |"
        )


def _time(fn, iters=50, warmup=5):
    for _ in range(warmup):
        mx.eval(fn())
    mx.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        mx.eval(fn())
    mx.synchronize()
    return (time.perf_counter() - t0) / iters * 1e3  # ms/iter


def _bf16_absorbed_decode(q, pe, latent_bf16, scale):
    """Realistic bf16 absorbed decode baseline (what an unquantised latent cache would do)."""
    B, H, D = q.shape
    k = latent_bf16[:, None]  # (B,1,T,D) shared across heads
    mask = pe[:, :, None, :].astype(
        latent_bf16.dtype
    )  # SDPA mask must match output dtype
    out = mx.fast.scaled_dot_product_attention(
        q[:, :, None, :], k, k, scale=scale, mask=mask
    )
    return out[:, :, 0, :]


def latency_table(B=1, H=128, D=512, group=DEFAULT_GROUP_SIZE):
    print(f"\n### Decode latency — B={B}, H={H}, kv_lora_rank={D} (single token)\n")
    print(
        "bf16 absorbed SDPA is the *hypothetical* unquantised-latent baseline (2× the "
        "memory). The current mlx-vlm expands MLA to a per-head MHA cache that OOMs well "
        "before these context lengths, so it is not a runnable baseline here.\n"
    )
    print(
        "| context | bf16 absorbed SDPA (2× mem) | FP8 — mlx (default) | "
        "FP8 — kernel (scalar) | FP8 — kernel_mma (MMA) | FP8 — sdpa | best FP8 vs bf16 |"
    )
    print("|--:|--:|--:|--:|--:|--:|--:|")
    scale = 1.0 / (192**0.5)
    mx.random.seed(0)
    for T in [1024, 4096, 16384, 65536]:
        latent = mx.random.normal((B, T, D)).astype(mx.bfloat16)
        codes, scales = quantize_latent_fp8(latent, group)
        q = mx.random.normal((B, H, D)).astype(mx.bfloat16)
        pe = mx.random.normal((B, H, T)).astype(mx.float32)
        t_bf16 = _time(lambda: _bf16_absorbed_decode(q, pe, latent, scale))
        t_mlx = _time(
            lambda: _mla_fp8_decode_reference(q, pe, codes, scales, scale, group)
        )
        t_scalar = _time(
            lambda: _mla_fp8_decode_metal(q, pe, codes, scales, scale, group)
        )
        t_mma = _time(
            lambda: _mla_fp8_decode_metal_mma(q, pe, codes, scales, scale, group)
        )
        t_sdpa = _time(lambda: _mla_fp8_decode_sdpa(q, pe, codes, scales, scale, group))
        best = min(t_mlx, t_scalar, t_mma, t_sdpa)
        print(
            f"| {T//1024}K | {t_bf16:.3f} ms | {t_mlx:.3f} ms | {t_scalar:.3f} ms | "
            f"{t_mma:.3f} ms | {t_sdpa:.3f} ms | {t_bf16/best:.2f}× |"
        )


def accuracy_table(B=1, H=32, D=512, group=DEFAULT_GROUP_SIZE):
    print(f"\n### Accuracy — FP8 MMA kernel vs fp32 reference (B={B}, H={H}, D={D})\n")
    print("| latent dist | context | cosine | RMS rel-err | max rel-err |")
    print("|---|--:|--:|--:|--:|")
    scale = 1.0 / (192**0.5)
    mx.random.seed(1)
    dists = {
        "N(0,1)": lambda s: mx.random.normal(s),
        "N(0,1)+outliers": lambda s: mx.random.normal(s)
        * (1 + 6 * (mx.random.uniform(shape=s) > 0.97)),
        "heavy-tail (t)": lambda s: mx.random.normal(s)
        / mx.maximum(mx.abs(mx.random.normal(s)), 0.3),
    }
    for dname, gen in dists.items():
        for T in [2048, 16384]:
            latent = gen((B, T, D)).astype(mx.bfloat16)
            codes, scales = quantize_latent_fp8(latent, group)
            q = mx.random.normal((B, H, D)).astype(mx.float32)
            pe = mx.random.normal((B, H, T)).astype(mx.float32)
            ker = _mla_fp8_decode_metal_mma(q, pe, codes, scales, scale, group).astype(
                mx.float32
            )
            ref = _mla_fp8_decode_reference(q, pe, codes, scales, scale, group).astype(
                mx.float32
            )
            a, b = ker.reshape(-1), ref.reshape(-1)
            cos = float(
                mx.sum(a * b) / (mx.sqrt(mx.sum(a * a)) * mx.sqrt(mx.sum(b * b)))
            )
            rms = float(mx.sqrt(mx.mean((a - b) ** 2)) / mx.sqrt(mx.mean(b * b)))
            mx_rel = float(mx.max(mx.abs(a - b)) / mx.max(mx.abs(b)))
            print(f"| {dname} | {T//1024}K | {cos:.6f} | {rms:.2e} | {mx_rel:.2e} |")

    print("\n### Quantisation round-trip RMS rel-error (latent only)\n")
    print("| latent dist | group=128 | group=64 |")
    print("|---|--:|--:|")
    for dname, gen in dists.items():
        row = [dname]
        for g in (128, 64):
            latent = gen((4, 256, D)).astype(mx.float32)
            codes, scales = quantize_latent_fp8(latent, g)
            deq = dequantize_latent_fp8(codes, scales, g)
            rms = float(
                mx.sqrt(mx.mean((deq - latent) ** 2)) / mx.sqrt(mx.mean(latent**2))
            )
            row.append(f"{rms*100:.2f}%")
        print("| " + " | ".join(row) + " |")


if __name__ == "__main__":
    print("# FP8 MLA KV cache benchmark")
    print(f"\nMLX {mx.__version__}, Metal available: {mx.metal.is_available()}")
    memory_table()
    if mx.metal.is_available():
        # Sweep batch size: B=1 single-token decode is overhead-bound (FP8 = memory win, not
        # speed). Batched serving (B>1) feeds the matrix units — the regime where FP8 + MMA
        # could turn its bandwidth saving into a decode speedup. This sweep is the deciding data.
        for B in (1, 4, 8):
            latency_table(B=B)
    accuracy_table()
