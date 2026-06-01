# Nemotron Labs Diffusion

Nemotron Labs Diffusion is a text-only diffusion language model from NVIDIA. The same checkpoint supports autoregressive decoding, block diffusion decoding, and linear self-speculative decoding.

Capabilities:
- **Text generation** - normal autoregressive generation through the standard `mlx_vlm.generate` path
- **Diffusion generation** - masked block denoising with live visualization when `--verbose` is enabled
- **Linear self-speculation** - diffusion drafting with autoregressive verification using `--gen-kwargs`
- **Thinking mode** - chat-template support through `--enable-thinking`

## Model

| Model | Type | Params | Context | Modalities |
|---|---|---:|---:|---|
| `nvidia/Nemotron-Labs-Diffusion-8B` | Dense diffusion LM | 8B | 262k | Text |

## Install

```sh
pip install -U mlx-vlm
```

## CLI

### Autoregressive generation

By default, Nemotron uses the normal autoregressive generation path.

```sh
mlx_vlm.generate \
  --model nvidia/Nemotron-Labs-Diffusion-8B \
  --prompt "Write a short story about a clockmaker." \
  --max-tokens 256 \
  --temperature 0.0
```

### Diffusion generation

Pass `generation_mode="diffusion"` through `--gen-kwargs` to use the masked diffusion path.
`generation_mode` is a model-specific generation kwarg interpreted by the Nemotron backend.
Nemotron defaults to the upstream/Transformers transfer policy with a 32-step denoising cap and a 0.9 transfer threshold.
This native mode also uses a Transformers-parity runtime for the denoise encoder.
The upstream mode alias `generation_mode="dlm"` is also accepted.
Sampler variants from the NVIDIA evaluation harness can be selected with `sampler`.
Supported values are `native` (default), `confidence_threshold_bound`, `fixed`, `confidence_threshold_ref`, and `cumulative_error`.
For faster MLX experiments, opt into the bounded sampler with `sampler="confidence_threshold_bound"`; it uses `min_threshold=0.45` by default and keeps the optimized MLX kernels.
For profiling, `head_scoring="chunked"` scores masked rows without concatenating full vocabulary logits; the default remains `head_scoring="full"` because it is usually faster on MLX's optimized matmul path.
For mixed AR+dLM experiments, pass `ar_weight` between `0.0` and `1.0`; this adds an AR causal block forward during denoising and is disabled by default.

```sh
mlx_vlm.generate \
  --model nvidia/Nemotron-Labs-Diffusion-8B \
  --prompt "Write a short story about a clockmaker." \
  --max-tokens 256 \
  --max-denoising-steps 16 \
  --temperature 0.0 \
  --gen-kwargs '{"generation_mode": "diffusion"}' \
  --verbose
```

### Linear self-speculative generation

Use `--gen-kwargs` for model-specific generation options. `generation_mode="linear_speculative"` is passed through to the Nemotron backend, where it enables the linear self-speculative path.
The bundled `linear_spec_lora` adapter is loaded automatically when available.
The upstream mode alias `generation_mode="linear_spec"` is also accepted.

```sh
mlx_vlm.generate \
  --model nvidia/Nemotron-Labs-Diffusion-8B \
  --prompt "Write a short story about a clockmaker." \
  --max-tokens 256 \
  --temperature 0.0 \
  --gen-kwargs '{"generation_mode": "linear_speculative"}'
```

### Thinking mode

```sh
mlx_vlm.generate \
  --model nvidia/Nemotron-Labs-Diffusion-8B \
  --prompt "Solve this step by step: if a train travels 180 km in 2.5 hours, what is its average speed?" \
  --enable-thinking \
  --max-tokens 512 \
  --temperature 0.0
```

## Python

### Basic text generation

```python
from mlx_vlm import generate, load
from mlx_vlm.prompt_utils import apply_chat_template

model, processor = load("nvidia/Nemotron-Labs-Diffusion-8B")

prompt = apply_chat_template(
    processor,
    model.config,
    "Write a short story about a clockmaker.",
)

result = generate(
    model=model,
    processor=processor,
    prompt=prompt,
    max_tokens=256,
    temperature=0.0,
)
print(result.text)
```

### Diffusion generation

```python
from mlx_vlm import generate, load
from mlx_vlm.prompt_utils import apply_chat_template

model, processor = load("nvidia/Nemotron-Labs-Diffusion-8B")

prompt = apply_chat_template(
    processor,
    model.config,
    "Write a short story about a clockmaker.",
)

result = generate(
    model=model,
    processor=processor,
    prompt=prompt,
    max_tokens=256,
    max_denoising_steps=16,
    temperature=0.0,
    generation_mode="diffusion",
)
print(result.text)
```

### Linear self-speculative generation

```python
from mlx_vlm import generate, load
from mlx_vlm.prompt_utils import apply_chat_template

model, processor = load("nvidia/Nemotron-Labs-Diffusion-8B")

prompt = apply_chat_template(
    processor,
    model.config,
    "Write a short story about a clockmaker.",
)

result = generate(
    model=model,
    processor=processor,
    prompt=prompt,
    max_tokens=256,
    temperature=0.0,
    generation_mode="linear_speculative",
)
print(result.text)
```

## Architecture

- **Backbone** - dense decoder-only Ministral-style transformer
- **Layers** - 34 transformer layers
- **Hidden size** - 4096
- **Attention** - 32 query heads, 8 KV heads, 128 head dimension
- **MLP** - SwiGLU with 14336 intermediate size
- **RoPE** - long-context YaRN/Llama 4-style scaling parameters from the checkpoint
- **Diffusion head** - untied output projection over the 131072-token vocabulary
- **Mask token** - `mask_token_id=100`

## Notes

- The model is text-only. Image, audio, and video inputs are not supported.
- AR generation should use the normal CLI without diffusion-specific arguments.
- Diffusion generation uses masked block denoising. `--verbose` shows the block visualization as masks are filled.
- The default diffusion schedule uses 32 denoising steps and a 0.9 confidence threshold. Lower `--max-denoising-steps` for speed experiments, but quality can degrade quickly.
- Diffusion generation records model-level stats such as `diffusion_denoise_nfe`, `diffusion_post_block_nfe`, and `diffusion_tokens_per_denoise_forward`. Use `head_scoring="chunked"` to profile the non-materializing confidence scorer.
- Diffusion and linear self-speculative generation are exposed through the model-specific `generation_mode` kwarg, for example `--gen-kwargs '{"generation_mode": "diffusion"}'`.
- Upstream mode names are accepted as aliases: `dlm` for diffusion and `linear_spec` for linear self-speculation.
- The optional `linear_spec_lora` adapter included in the Hugging Face repo is used only during the diffusion draft phase of linear self-speculation.
