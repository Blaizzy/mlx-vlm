---
name: serving-and-performance
description: Use this skill when the user asks about MLX-VLM server throughput, latency, memory, batching, prompt caching, KV cache quantization, TurboQuant, speculative decoding, streaming, metrics, slow generation, OOMs, or production-style local serving.
---

# Serving And Performance

Use this workflow for performance, memory, and serving behavior in `mlx_vlm.server`, `mlx_vlm.generate`, `mlx_vlm.speculative`, and related cache/quantization utilities.

## Baseline Before Tuning

1. Record the exact model, prompt/media inputs, batch/concurrency, max tokens, and hardware.
2. Reproduce with a minimal command or request.
3. Separate prefill, media encoding, and decode symptoms. Image/video-heavy workloads often bottleneck differently from text-only decode.
4. Capture server `/metrics` when working through the API.

## Main Knobs

Server startup:

```bash
python -m mlx_vlm server --model <model> --port 8080
```

Speculative decoding:

```bash
python -m mlx_vlm server \
  --model <target-model> \
  --draft-model <draft-model> \
  --draft-kind dflash
```

KV cache quantization:

```bash
python -m mlx_vlm server \
  --model <model> \
  --kv-bits 3.5 \
  --kv-quant-scheme turboquant
```

Useful flags include `--draft-kind`, `--draft-block-size`, `--kv-bits`, `--kv-quant-scheme`, `--kv-group-size`, `--quantized-kv-start`, `--max-kv-size`, `--vision-cache-size`, `--enable-thinking`, and `--thinking-budget`.

## Decision Guide

- Use continuous batching for concurrent server workloads; verify behavior through `/metrics`.
- Use `VisionFeatureCache` when repeated image content appears across turns or requests.
- Use speculative decoding only with a compatible drafter family and validate output quality. Structured outputs may have additional constraints.
- Use KV cache quantization for memory pressure or long contexts; verify quality and speed because quantization can trade accuracy, memory, and kernel cost.
- Tune thinking defaults separately from decode performance. Thinking tokens can dominate latency even when kernels are fast.

## Validation

- Compare baseline and tuned runs with the same prompt/media, max tokens, and sampling settings.
- Track prompt tokens/s, generation tokens/s, wall time, peak memory, cache hit behavior, and output correctness.
- For server changes, test streaming and non-streaming if both are affected.
- For batching changes, include mixed text-only and image requests when the code path claims to support both.
- Run focused tests or smoke scripts first; only run broad suites after shared server or generation utilities change.
