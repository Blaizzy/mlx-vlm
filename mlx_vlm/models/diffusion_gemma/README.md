# DiffusionGemma4

DiffusionGemma4 is a block-diffusion language model port for
`google/diffusiongemma-26B-A4B-it`. It generates a canvas of noisy token ids and
repeatedly denoises that canvas before appending the accepted tokens to the
prefix.

## Model

| | |
|---|---|
| **Model ID** | `google/diffusiongemma-26B-A4B-it` |
| **Model type** | `diffusion_gemma` |
| **Architecture** | `DiffusionGemma4ModelForBlockDiffusion` |
| **Generation** | Block diffusion with a 256-token canvas |
| **Language layers** | 30 |
| **Hidden size** | 2816 |
| **Attention** | 5 sliding-window layers, then 1 full-attention layer, repeated |
| **Sliding window** | 1024 |
| **MoE** | 128 experts, top-8 routing, 704 expert hidden size |
| **Dense MLP** | 2112 intermediate size |
| **Vocabulary** | 262,144 |

This implementation supports text, image, and video inputs. Image prompts and
sampled video frames run through the Gemma 4 vision tower, and the encoder
applies bidirectional attention within each vision-token block (matching the
checkpoint's `use_bidirectional_attention: "vision"` setting). Multiple images
or videos per prompt are supported. Audio inputs are rejected for now.

## CLI Usage

Basic generation:

```bash
uv run mlx_vlm.generate \
  --model google/diffusiongemma-26B-A4B-it \
  --prompt "Explain why the sky is blue." \
  --max-tokens 120 \
  --temperature 0.0 \
  --trust-remote-code \
  --skip-special-tokens
```

Describe an image:

```bash
uv run mlx_vlm.generate \
  --model google/diffusiongemma-26B-A4B-it \
  --prompt "Describe this image in one short paragraph." \
  --image /path/to/image.png \
  --max-tokens 128 \
  --temperature 0.0 \
  --trust-remote-code \
  --skip-special-tokens
```

When run in a terminal, generation shows a live canvas view: the full
sequence generated so far — finalized text plus the in-flight canvas with
`[Mask]` placeholders — wrapped to the terminal width and redrawn in place on
every denoising step. When the canvas grows taller than the terminal, the
view switches to the alternate screen buffer and restores it when generation
finishes. The view adds no measurable throughput cost; it is skipped
automatically for piped output, and can be disabled programmatically by
passing `diffusion_show_unmasking=False` to `generate`/`stream_generate`.

Use the confidence-threshold sampler. This can reduce denoising work on short
and medium generations, but lower thresholds can hurt quality:

```bash
uv run mlx_vlm.generate \
  --model google/diffusiongemma-26B-A4B-it \
  --prompt "Write a practical explanation of diffusion language model performance." \
  --max-tokens 96 \
  --diffusion-sampler confidence-threshold \
  --threshold 0.8 \
  --temperature 0.0 \
  --trust-remote-code \
  --skip-special-tokens
```

Useful diffusion options:

- `--max-denoising-steps`: maximum denoising iterations per canvas. Defaults
  to the checkpoint's generation config (48 for the RC checkpoints). The
  stable-and-confident stopping criteria usually converges canvases in far
  fewer steps, so this cap is cheap; set it lower to hard-cap throughput.
- `--seed`: seed the PRNG before generation. Diffusion canvases start from
  random noise, so temperature-0 runs are only reproducible with a fixed seed.
- `--diffusion-min-canvas-length`: minimum active canvas for partial blocks.
  Smaller values reduce work for short completions.
- `--diffusion-max-canvas-length`: maximum active canvas length. Defaults to
  the checkpoint canvas length; set it lower to trade quality for throughput
  on long generations.
- `--diffusion-full-canvas`: always denoise the checkpoint canvas length, even
  for a partial final block.
- `--diffusion-sampler entropy-bound`: checkpoint-style entropy-bound canvas
  update.
- `--diffusion-sampler confidence-threshold`: commit high-confidence canvas
  positions early.
- `--threshold`: probability threshold for the confidence sampler.

## Output Stats

Verbose CLI output reports the standard prompt and generation throughput,
like other diffusion models in mlx-vlm:

```text
Prompt: 29 tokens, 142.641 tokens-per-sec
Generation: 120 tokens, 24.488 tokens-per-sec
Peak memory: 50.883 GB
```

Note that generation throughput counts the final emitted tokens; a short
reply still denoises a full canvas, so short generations report lower
tokens-per-second than long ones. The diffusion counters (canvas tokens,
denoising steps, token-steps) remain available programmatically on the
streamed `GenerationResult` objects.

## Architecture Notes

The decoder is a hybrid dense + MoE transformer:

- Attention is dense, with sliding-window layers and periodic full-attention
  layers.
- Each layer has a dense GeGLU MLP branch.
- Each layer also routes tokens through a sparse MoE branch using top-8 of 128
  experts.
- The dense and expert branches are combined before the residual update.

The repeated denoising loop is the main cost. Profiling shows that most canvas
time is spent in the decoder layers, especially the MoE expert path. A normal
KV cache helps reuse the encoded prefix, but it cannot cache the changing
canvas states across denoising steps because the canvas is bidirectional and is
updated every step.

## Numerical Correctness

The MLX port was checked against the checkpoint's bundled Transformers wheel:

- plain forward max absolute difference: `1.34110451e-07`
- masked self-conditioned forward max absolute difference: `1.58324838e-07`
- cache-only decoder max absolute difference: `1.34110451e-07`
- static cache vs dynamic cache max absolute difference: `0`

Argmax outputs matched in these checks.

## Current Limitations

- Text, image, and video generation are supported.
- Batch generation is not supported for this diffusion path.
- The confidence-threshold sampler is an optimization heuristic. It is useful
  for some short and medium outputs, but conservative settings or the default
  checkpoint sampler are better for long-form quality.
