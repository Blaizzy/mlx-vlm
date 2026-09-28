# EmbeddingGemma 2 performance and quantization

Measured on 2026-09-28 on an Apple M5 Max with 128 GiB unified memory, MLX
0.32.2, and the EAP Transformers 5.18.0.dev0 processor. Checkpoint revision:
`fc77679a26fcb86250765859d04ce2fcc6cb0b2c`. Extras revision:
`f6c512df20896fd06f85d39db10c45a0a9849ef8`.

## Method

Each checkpoint runs in a fresh process, sequentially, with three warmups and ten
measured forwards per workload. Two passes use opposite format orders
(BF16 → 8 → 6 → 4, then 4 → 6 → 8 → BF16); table medians combine all 20 samples.
Timings include the native encoder, modality
towers, mean pooling, and normalization, with explicit MLX synchronization.
Preprocessing, file decoding, model loading, and first-use compilation are
excluded. `MLX_ENABLE_TF32=0` is used consistently. Results are median batch
latencies; throughput is batch size divided by median latency.

The text workloads contain exactly 128 or 512 tokens per row. Images use the
extras `cat.jpg` at the default 280-token budget (270 sequence tokens including
wrappers). Audio uses the 5.855-second, 16 kHz mono `speech.wav` (151 tokens).
Video uses the five-second `sample_video.mp4`, sampled at 1 FPS (five frames,
662 sequence tokens). The mixed batch contains one image, one audio clip, one
video, and one text input, padded to 662 tokens per row.

Quantization is the standard affine MLX-VLM conversion with group size 64 and
BF16 remaining weights/activations. It quantizes the text encoder and audio
projection; the vision/audio towers and vision projection remain BF16 under the
converter's default policy. Consequently, whole-model savings are smaller than
those for the text encoder alone. File and weight sizes below use decimal GB.

## Latency and throughput

| Workload | BF16 | 8-bit | 6-bit | 4-bit |
| --- | ---: | ---: | ---: | ---: |
| Text: 1 × 128 tokens | 5.88 ms | 7.16 ms | 7.23 ms | 7.13 ms |
| Text: 32 × 128 tokens | 38.94 ms | 41.17 ms | 42.01 ms | 40.67 ms |
| Text: 1 × 512 tokens | 8.70 ms | 12.81 ms | 13.06 ms | 12.72 ms |
| Text: 32 × 512 tokens | 182.72 ms | 189.81 ms | 193.54 ms | 190.98 ms |
| Image: 1 × 270 tokens | 44.50 ms | 46.36 ms | 46.56 ms | 47.10 ms |
| Audio: 5.855 s, 151 tokens | 24.02 ms | 25.04 ms | 25.12 ms | 25.00 ms |
| Video: 5 s / 5 frames, 662 tokens | 100.06 ms | 99.32 ms | 100.52 ms | 100.57 ms |
| Mixed: 4 rows × 662 padded tokens | 178.16 ms | 179.34 ms | 180.41 ms | 179.40 ms |

For the 32 × 128-token text batch, throughput is **821.8 embeddings/s** (BF16), **777.2 embeddings/s** (8-bit), **761.6 embeddings/s** (6-bit), **786.8 embeddings/s** (4-bit).

## Size, memory, and numerical accuracy

| Format | All weights | Text encoder weights | Peak MLX allocation¹ | Min cosine vs FP32 |
| --- | ---: | ---: | ---: | ---: |
| BF16 | 1.489 GB | 0.542 GB | 3.108 GB | 0.999791 |
| 8-bit | 1.234 GB | 0.288 GB | 2.853 GB | 0.999662 |
| 6-bit | 1.166 GB | 0.220 GB | 2.785 GB | 0.997708 |
| 4-bit | 1.098 GB | 0.153 GB | 2.717 GB | 0.967160 |

¹ Maximum across the eight measured workloads.

Peak memory is **MLX active allocation**, including weights, prepared inputs, and
forward intermediates. It excludes Python/PyTorch preprocessing allocations and
the allocator cache; it is not total process RSS.

Each numerical comparison uses the supplied PyTorch reference in float32. All
20 documentation examples execute completely, with 32 paired forwards per
format. Every forward also compares normalized 128/256/512-dimensional prefixes.

- BF16 baseline: minimum cosine **0.99979059**, maximum component error **0.00271**.
- 8-bit: minimum cosine **0.99966160**, maximum component error **0.00389**.
- 6-bit: minimum cosine **0.99770791**, maximum component error **0.00748**.
- 4-bit: minimum cosine **0.96716032**, maximum component error **0.03374**.

The quantization acceptance limits were cosine >= 0.99 (full and truncated
vectors), component error <= 0.03, and valid-token relative L2 error <= 1.0.
8-bit and 6-bit pass all 20 examples. **4-bit fails these limits in 18 examples**;
these are numerical fidelity failures, with finite unit-normalized outputs.
The runner records failures while continuing all forwards within each example.
These sample-level numerical checks are not a retrieval benchmark such as MTEB.

**BF16 is fastest for text, image, and audio workloads on this M5 Max.** Video
and mixed-batch differences are within the observed run-to-run variation.
Use 8-bit when weight memory matters: it saves 17.1% of total weight storage (46.9% for the text
encoder) while closely preserving embeddings. 6-bit saves 21.7% overall and has
more numerical drift. Standard 4-bit saves 26.2% overall but has substantial
embedding drift, so it is a poor default for accuracy-sensitive retrieval.
The tested affine quantization formats mainly save memory; there is no
consistent latency improvement. The reverse-order pass confirms the text
slowdown. Video medians shifted by up to 5.2% between passes, exceeding the
sub-1% combined difference between BF16 and 8-bit. All four formats still rank
the documented Mars passage above Venus in the single
retrieval smoke test.

For batch-size sweeps across all four modalities, see
[batch scaling measurements](BATCHED_BENCHMARKS.md).

## Reproduce

Install the EAP wheel and audio/video dependencies described in the
[model README](README.md). Download the sample media without the extras' other
large packages:

```sh
hf download gg-hf-em/embeddinggemma-2-eap-extras --repo-type dataset \
  --revision f6c512df20896fd06f85d39db10c45a0a9849ef8 \
  --include 'assets/*' --local-dir embeddinggemma2-extras

mlx_vlm.convert --hf-path gg-hf-em/embeddinggemma-2 \
  --revision fc77679a26fcb86250765859d04ce2fcc6cb0b2c \
  --mlx-path embeddinggemma2-bf16 --dtype bfloat16

# Repeat with --q-bits 4, 6, and 8, using separate output directories.
mlx_vlm.convert --hf-path gg-hf-em/embeddinggemma-2 \
  --revision fc77679a26fcb86250765859d04ce2fcc6cb0b2c \
  --mlx-path embeddinggemma2-8bit --dtype bfloat16 \
  -q --q-bits 8 --q-group-size 64

MLX_ENABLE_TF32=0 python examples/benchmark_embedding_gemma2.py \
  --model-path embeddinggemma2-8bit --assets embeddinggemma2-extras/assets \
  --warmups 3 --repeats 10 --output benchmark-8bit.json

python examples/validate_embedding_gemma2.py --dtype bfloat16 \
  --mlx-model-path embeddinggemma2-8bit --min-cosine 0.99 \
  --max-error 0.03 --max-token-error 1.0 --output-dir accuracy-8bit
```

The benchmark JSON includes individual timing samples, p90 latency, token/frame
counts, weight sizes, peak active allocation, and the documented Mars/Venus
retrieval smoke test. Run formats sequentially without other GPU workloads.
