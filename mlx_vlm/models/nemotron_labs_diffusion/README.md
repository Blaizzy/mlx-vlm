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

Pass `generation_mode="diffusion"` to use the masked diffusion path.
Nemotron defaults to 8 denoising steps in this mode so the diffusion path transfers multiple tokens per forward pass.

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

Use `--gen-kwargs` for model-specific generation options. The bundled `linear_spec_lora` adapter is loaded automatically when available.

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
- The default diffusion schedule uses 8 denoising steps and no confidence threshold for speed. Increase `--max-denoising-steps` or pass `--threshold` if you want more conservative denoising.
- Diffusion and linear self-speculative generation are exposed through `generation_mode`, for example `--gen-kwargs '{"generation_mode": "diffusion"}'`.
- The optional `linear_spec_lora` adapter included in the Hugging Face repo is used only during the diffusion draft phase of linear self-speculation.
