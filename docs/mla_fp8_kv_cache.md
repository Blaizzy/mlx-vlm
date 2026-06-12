# FP8 KV cache for MLA models

`mlx_vlm.models.deepseek_vl_v2.mla_fp8` adds a vLLM/SGLang-style **FP8 KV cache for MLA-attention models**
(DeepSeek-V2/V3, GLM-4-MoE, Kimi, …), served through an **absorbed-MLA** decode path. It is
**opt-in and off by default** — the standard path is untouched unless you set an env flag.

## Why

MLA compresses every token's KV to a low-rank latent `c_t` of size `kv_lora_rank` (e.g. 512)
plus a small shared RoPE key `k_pe_t` of size `qk_rope_head_dim` (e.g. 64). Today mlx-vlm
**expands** that latent back into a full per-head MHA cache before storing it, which throws
away MLA's entire memory advantage:

| DeepSeek-V2 (60 layers, 128 heads) | KV cache bytes/token |
|---|--:|
| current mlx-vlm — expanded bf16 | **~4.8 MB** |
| absorbed bf16 latent | 67.5 KB |
| **absorbed FP8 latent (this feature)** | **38.4 KB** |

The FP8 latent is **~43 % smaller than the bf16 latent** and **~125× smaller than the current
expanded cache**. At 128 K context (batch 1) the latent cache drops from **9.06 GB → 5.16 GB**;
versus the expanded cache, long context simply was not reachable before.

Layout matches vLLM's DeepSeek-V3 KV: `512` fp8 latent + `4×4` fp32 group scales + `64×2` bf16
rope = **656 B/token**.

## How it works

* The `kv_lora_rank` latent is quantised to **FP8 (e4m3)** with **per-128-element group
  scales** (absmax / 448); the RoPE key stays in the model dtype.
* Decode runs in the **absorbed** form: `kv_b_proj` is folded into the query (`embed_q`) and
  output (`unembed_out`) projections so the cached latent serves as both K and V (an MQA over
  the latent). These projections are derived lazily from `kv_b_proj` (handling quantised
  weights), exactly as mlx-lm's `deepseek_v3`.
* Three decode strategies are available (`MLX_VLM_MLA_FP8_DECODE`):
  * `mlx` *(default)* — fused einsum; fastest on MLX 0.31.2 (MLX fuses the FP8 dequant into
    the matmul, so no bf16 latent is materialised and the matrix units are used).
  * `kernel` — an all-in-one scalar **Metal** absorbed-decode kernel (split-K flash, online
    softmax). No bf16 materialisation → lowest extra memory, but compute-bound and slower than
    `mlx`. An MMA (simdgroup-matrix) version that could beat `mlx`/SDPA is future work.
  * `sdpa` — a fused FP8-dequant Metal kernel followed by MLX SDPA.

## Usage

```bash
export MLX_VLM_MLA_FP8=1            # enable the FP8 MLA KV cache (default: off)
export MLX_VLM_MLA_FP8_GROUP=128    # latent quant group size (default: 128)
export MLX_VLM_MLA_FP8_DECODE=mlx   # mlx (default) | kernel | sdpa
```

Wired into `deepseek_vl_v2` (DeepSeek-VL2 small/base). With the flag off, behaviour is
byte-identical to before. Other MLA models can adopt it by routing their MLA attention through
`mlx_vlm.models.deepseek_vl_v2.mla_fp8.mla_fp8_attention` and returning `Fp8MLAKVCache` from `make_cache`.

## Performance (M-series, MLX 0.31.2, B=1, H=128, kv_lora_rank=512)

Decode latency per token. `bf16 absorbed SDPA` is the *hypothetical* unquantised-latent
baseline (2× the memory); the current expanded cache OOMs before these contexts.

| context | bf16 absorbed SDPA (2× mem) | FP8 `mlx` (default) | FP8 `kernel` | FP8 `sdpa` |
|--:|--:|--:|--:|--:|
| 1K  | 0.53 ms | **0.33 ms** | 0.83 ms | 0.33 ms |
| 4K  | 0.37 ms | **0.43 ms** | 1.81 ms | 0.53 ms |
| 16K | 0.76 ms | **1.11 ms** | 5.65 ms | 1.34 ms |
| 64K | 2.18 ms | **3.86 ms** | 20.7 ms | 4.54 ms |

The FP8 default decode is at parity-to-faster at short context and ~1.8× the bf16-latent
baseline at 64 K (the cost of dequant), while using **~43 %** less KV memory — and it runs at
contexts the current expanded cache cannot reach at all.

## Accuracy

* Grouped FP8 round-trip RMS error on the latent: **~2.5 %** (`N(0,1)`), stable under outlier
  and heavy-tailed distributions.
* Fused kernel / SDPA decode vs the fp32 reference: **cosine 1.000000**.
* Absorbed FP8 path vs the default expanded bf16 path on a real `DeepseekV2Attention` layer:
  **cosine ≈ 0.9997** (the expected FP8 quant error; a wrong absorbed-weight transpose would
  collapse this).
* Full 6-layer `deepseek_vl_v2` language model, **4-bit quantized** (so `kv_b_proj` is a real
  `QuantizedLinear`), prefill + 24 greedy decode steps, FP8 cache vs bf16 expanded cache:
  next-token **logit cosine 0.9993**. (Greedy token agreement is only meaningful with trained
  weights — random init gives near-flat logits where argmax flips on tiny perturbations.)

See `mlx_vlm/tests/test_mla_fp8.py` (correctness) and `mlx_vlm/tests/bench_mla_fp8.py`
(reproduce the tables above).
