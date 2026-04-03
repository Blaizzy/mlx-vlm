# Gemma 4

Gemma 4 is a family of multimodal models from Google supporting text, image, audio, and thinking (chain-of-thought reasoning). This unified module handles all Gemma 4 variants — from lightweight 2B to dense 31B and MoE 26B.

Capabilities:
- **Text generation** — single and multi-turn conversations
- **Thinking mode** — chain-of-thought reasoning with `--enable-thinking`
- **Image understanding** — single and multi-image analysis
- **Audio understanding** — speech transcription and audio analysis (2B/4B models)
- **Mixture of Experts** — sparse MoE with SwitchGLU and gather_mm (26B-A4B)

## Models

| Model | Type | Params | Memory | Vision | Audio | K-eq-V |
|-------|------|--------|--------|--------|-------|--------|
| `google/gemma-4-e2b-it` | Dense | 2B | ~5 GB | Yes | Yes | No |
| `google/gemma-4-e4b-it` | Dense | 4B | ~16 GB | Yes | Yes | No |
| `google/gemma-4-26b-a4b-it` | MoE | 26B (4B active) | ~52 GB | Yes | No | Yes |
| `google/gemma-4-31b-it` | Dense | 31B | ~63 GB | Yes | No | Yes |

> **K-eq-V**: Full attention layers reuse key projections as values (no separate `v_proj`), reducing parameters and memory while using `num_global_key_value_heads` for the KV dimension.

## Install

```sh
pip install -U mlx-vlm
```

## CLI

### Text generation

```sh
python -m mlx_vlm.generate \
  --model google/gemma-4-e4b-it \
  --prompt "What is the capital of France?" \
  --max-tokens 500 \
  --temperature 1.0 --top-p 0.95 --top-k 64
```

### Image understanding

```sh
python -m mlx_vlm.generate \
  --model google/gemma-4-e4b-it \
  --image path/to/image.jpg \
  --prompt "Describe this image." \
  --max-tokens 500 \
  --temperature 1.0 --top-p 0.95 --top-k 64
```

### Audio understanding (2B/4B only)

```sh
python -m mlx_vlm.generate \
  --model google/gemma-4-e4b-it \
  --audio path/to/audio.wav \
  --prompt "Transcribe this audio" \
  --max-tokens 500 \
  --temperature 1.0 --top-p 0.95 --top-k 64
```

### Thinking mode (chain-of-thought)

```sh
python -m mlx_vlm.generate \
  --model google/gemma-4-e4b-it \
  --prompt "I want to do a car wash that is 50 meters away, should I walk or drive?" \
  --enable-thinking \
  --max-tokens 2000 \
  --temperature 1.0 --top-p 0.95 --top-k 64
```

### Image + thinking

```sh
python -m mlx_vlm.generate \
  --model google/gemma-4-31b-it \
  --image path/to/image.jpg \
  --prompt "Describe this image." \
  --enable-thinking \
  --max-tokens 2000 \
  --temperature 1.0 --top-p 0.95 --top-k 64
```

### Thinking with budget

```sh
python -m mlx_vlm.generate \
  --model google/gemma-4-e4b-it \
  --prompt "Explain quantum entanglement." \
  --enable-thinking \
  --thinking-budget 512 \
  --max-tokens 2000 \
  --temperature 1.0 --top-p 0.95 --top-k 64
```

## Python

### Basic text generation

```python
from mlx_vlm import load, generate
from mlx_vlm.prompt_utils import apply_chat_template

model, processor = load("google/gemma-4-e4b-it")

prompt = apply_chat_template(
    processor, model.config, "Write a poem about the ocean."
)

result = generate(
    model=model,
    processor=processor,
    prompt=prompt,
    max_tokens=500,
    temperature=1.0,
    top_p=0.95,
    top_k=64,
)
print(result)
```

### Image understanding

```python
from mlx_vlm import load, generate
from mlx_vlm.prompt_utils import apply_chat_template

model, processor = load("google/gemma-4-31b-it")

image = ["path/to/image.jpg"]
prompt = apply_chat_template(
    processor, model.config, "Describe this image.",
    num_images=1,
)

result = generate(
    model=model,
    processor=processor,
    prompt=prompt,
    image=image,
    max_tokens=500,
    temperature=1.0,
    top_p=0.95,
    top_k=64,
)
print(result)
```

### Audio understanding

```python
from mlx_vlm import load, generate
from mlx_vlm.prompt_utils import apply_chat_template

model, processor = load("google/gemma-4-e4b-it")

prompt = apply_chat_template(
    processor, model.config, "Transcribe this audio",
    num_audios=1,
)

result = generate(
    model=model,
    processor=processor,
    prompt=prompt,
    audio=["path/to/audio.wav"],
    max_tokens=500,
    temperature=1.0,
    top_p=0.95,
    top_k=64,
)
print(result)
```

### Thinking mode

```python
from mlx_vlm import load, generate
from mlx_vlm.prompt_utils import apply_chat_template

model, processor = load("google/gemma-4-e4b-it")

prompt = apply_chat_template(
    processor, model.config,
    "Explain quantum entanglement in simple terms.",
    chat_template_kwargs={"enable_thinking": True},
)

result = generate(
    model=model,
    processor=processor,
    prompt=prompt,
    max_tokens=2000,
    temperature=1.0,
    top_p=0.95,
    top_k=64,
)
print(result)
```

## Architecture

All Gemma 4 variants share a common architecture with conditional features:

- **Sliding + full attention** — pattern of 5 sliding window layers + 1 full attention layer
- **KV sharing** — later layers reuse K/V from earlier layers (2B/4B models)
- **K-eq-V attention** — full attention layers use key states as values (26B/31B models)
- **Per-layer inputs** — additional per-layer token embeddings (2B/4B models)
- **MoE** — 128 experts with top-8 routing using `gather_mm` (26B model only)
- **SigLIP2 vision encoder** — shared across all variants
- **Conformer audio encoder** — 12 conformer blocks (2B/4B models only)

## Notes

- Thinking mode uses `<|channel>...<channel|>` delimiters to separate reasoning from the final answer.
- Audio uses a Conformer-based encoder with 128-bin mel spectrogram input (16kHz). Audio tokens are dynamically expanded based on duration (~40ms per token, up to 750 tokens).
- Supported audio formats: WAV, MP3, FLAC, and other formats handled by `soundfile`.
- The 26B MoE model uses sparse expert routing via `mlx_lm.SwitchGLU` with `mx.gather_mm` — only the top-8 experts are computed per token.
- The 31B dense model is bandwidth-bound at ~5 tok/s on 300 GB/s systems (62.5 GB / 300 GB/s).
