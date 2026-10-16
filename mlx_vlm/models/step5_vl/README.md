# Step 5 Preview

Native MLX support for StepFun's `step5_vl` architecture, including the perception
encoder, FP32 residual stream, windowed attention, region-selected attention,
and mixed BF16/NVFP4 routed experts. The tokenizer and image preprocessing use
the same format as Step 3.7.

The NVFP4 checkpoint is larger than the RAM in most Macs. Repack it once into a
disk-backed expert store; the original checkpoint remains intact:

```bash
python -m mlx_vlm.moe_offload \
  --build /path/to/Step-5-Preview-NVFP4 \
  --out /path/to/Step-5-Preview-NVFP4-offload

python -m mlx_vlm.generate \
  --model /path/to/Step-5-Preview-NVFP4-offload \
  --prefill-step-size 64 \
  --prompt "What is 2 + 2?" --max-tokens 128 --temperature 0
```

Repacking preserves packed NVFP4 weights, each expert's global scale, and the
three BF16 expert layers. The always-resident weights occupy approximately
32 GB; the expert store occupies approximately 358 GB. Allow additional memory
for the expert cache, vision activations, and the KV cache. Disk speed and expert
cache hit rate substantially affect generation speed. The default expert-cache
budget uses the GPU's recommended working set minus resident weights and a
reserve. On the tested 128 GB Mac, `--expert-cache-gb 60` keeps short text/image
runs around 94 GB of peak MLX memory; a larger KV cache needs additional headroom.

A local run on a 128 GB Apple M5 Max with MLX 0.32.3 produced the same complete
outputs and stopped normally before and after the offloader optimization:

| Input | Final answer | Original decode tokens/s (24 GB cache) | Optimized decode tokens/s (60 GB cache) | Response seconds, before → after |
| --- | --- | --- | --- | --- |
| `What is 2 + 2? Give only the answer.` | `4` | 0.70 | 1.60 | 58.6 → 33.2 |
| Image of a red square, asking its color | `Red` | 0.57 | 1.47 | 199.9 → 68.6 |

At the same 24 GB cache size, the first decode optimization reached 1.22
tokens/s on the text case. The final path batches selected experts into gathered
matrix multiplications, applies scalar NVFP4 global factors in the native
kernel, reloads a layer's metadata once per routed set, and retains a bounded
amount of reusable allocator memory. Prefill batches eight experts at a time
to bound temporary weight copies. Image prefill improved from 1.54 to 4.83
tokens/s. Both cases still generated their normal reasoning and final answer
(33 and 42 tokens); the timings do not suppress reasoning.

These are short correctness and performance checks, not quality or long-context
benchmarks. SSD and OS file-cache state affect timings; the 358 GB expert corpus
still exceeds physical memory.

Sparse-region selection uses the published weighted-ReLU indexer, FP8-rounded
region summaries, causal region selection, and the current partial region.
This initial implementation uses MLX attention masks and retains indexer K/Z
history; million-token inference has not been validated. The three auxiliary
MTP layers are omitted from ordinary generation; speculative MTP decoding is
not implemented for this model.

References: [checkpoint](https://huggingface.co/stepfun-ai/Step-5-Preview-NVFP4),
[StepFun reference inference](https://huggingface.co/stepfun-ai/Step-5-Preview/tree/main/inference).
