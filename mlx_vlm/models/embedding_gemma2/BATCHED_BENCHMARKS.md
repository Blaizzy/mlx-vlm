# EmbeddingGemma 2 batch scaling

Measured on Apple M5 Max (128 GiB), MLX 0.32.2, on 2026-09-28 using the same
checkpoint and EAP fixtures as [the initial benchmarks](BENCHMARKS.md).

## Method

Batches of **1, 2, 4, 8, 16, and 32** were measured for 128-token text, 512-token
text, images, audio, and video, with the full model resident. Each batch repeats
the same fixture with independent rows. This measures homogeneous batch scaling
at fixed sequence lengths; it does not represent a variable-length production
queue. Applicable media towers run on every forward, with no feature caching.

Each format runs in a fresh process. Two passes use opposite format orders:
BF16 → 8-bit → 6-bit → 4-bit, then the reverse. Each point has three warmups and
five measured forwards per pass, for **10 timing samples per point**. Results
combine both passes. Preprocessing and decoding are excluded; the entire native
forward, media towers, pooling, and normalization are timed and synchronized.
`MLX_ENABLE_TF32=0` is set for every run.

All variants use the standard conversion policy: affine group-64 quantization of
the text encoder/audio projection, with vision/audio towers and vision projection
remaining BF16. The media inputs are a 270-token image, a 5.855-second audio clip
(151 tokens), and a five-second video sampled at 1 FPS (five frames, 662 tokens).

Each batch is also checked against encoding the same fixture alone using the
same checkpoint. This consistency check is separate from quantization fidelity
against the float32 reference. In particular, the 4-bit checkpoint's earlier
minimum cosine of 0.96716 versus FP32 still applies.

## Results

BF16 128-token text rises from **167.3 to 817.4 embeddings/s** at batch 32 (**4.9×**). For 512-token text, the best measured BF16 throughput is **189.9 embeddings/s at batch 8**. Images and video gain little from larger batches at these visual budgets. BF16 video throughput changes from 9.75 to 8.80 clips/s at batch 32 while peak active memory grows from 3.10 to 33.66 GB. Larger batches are therefore workload-dependent, rather than a uniform speedup.

BF16 throughput (embeddings/s):

| Batch | Text 128 | Text 512 | Image | Audio | Video |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 167.3 | 115.0 | 22.8 | 38.5 | 9.7 |
| 2 | 296.7 | 153.2 | 23.9 | 58.4 | 9.6 |
| 4 | 475.4 | 174.9 | 23.6 | 72.8 | 9.6 |
| 8 | 675.7 | 189.9 | 22.7 | 84.0 | 9.6 |
| 16 | 775.2 | 183.1 | 22.1 | 88.8 | 8.9 |
| 32 | 817.4 | 177.2 | 21.2 | 85.4 | 8.8 |

BF16 latency and memory at batch 32:

| Workload | Full-batch latency | Latency per input | Peak active memory |
| --- | ---: | ---: | ---: |
| 128-token text | 39.1 ms | 1.22 ms | 2.51 GB |
| 512-token text | 180.6 ms | 5.64 ms | 2.76 GB |
| Image | 1508.7 ms | 47.15 ms | 14.44 GB |
| Audio | 374.8 ms | 11.71 ms | 3.53 GB |
| Video | 3635.0 ms | 113.59 ms | 33.66 GB |

Batch-32 throughput by format (embeddings/s):

| Format | Text 128 | Text 512 | Image | Audio | Video |
| --- | ---: | ---: | ---: | ---: | ---: |
| BF16 | 817.4 | 177.2 | 21.2 | 85.4 | 8.8 |
| 8-bit | 774.7 | 167.7 | 20.7 | 85.2 | 9.0 |
| 6-bit | 761.3 | 159.0 | 20.9 | 85.5 | 9.0 |
| 4-bit | 790.2 | 170.1 | 20.9 | 86.2 | 8.9 |

Peak memory means active MLX allocations, including weights, prepared inputs and
forward intermediates. Values are decimal GB and exclude the allocator cache,
Python/PyTorch allocations, and CPU preprocessing. Latency is for the entire batch;
throughput is batch size divided by median batch latency.

All **240 batch-versus-single checks** (30 workload/batch combinations × four formats × two passes) passed. Minimum cosine similarity was **0.99988627** against the same checkpoint's single-input output.

## Reproduce

Use the converted checkpoints and downloaded extras from
[the setup instructions](BENCHMARKS.md#reproduce). Run formats sequentially:

```sh
MLX_ENABLE_TF32=0 python examples/benchmark_embedding_gemma2.py \
  --model-path embeddinggemma2-bf16 \
  --assets embeddinggemma2-extras/assets \
  --batch-sizes 1 2 4 8 16 32 --warmups 3 --repeats 5 \
  --output batched-bf16-pass1.json
```

Repeat for 8/6/4-bit checkpoints, then repeat in reverse format order. The JSON
records individual timing samples, median/p90 latency, embeddings per second,
latency per embedding, peak active memory, and batch-versus-single cosine.
