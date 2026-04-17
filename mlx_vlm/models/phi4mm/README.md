# Phi-4 Multimodal (phi4mm)

Phi-4 Multimodal is a tri-modal model supporting text, image, and audio understanding.

## Architecture

| Component | Details |
|---|---|
| **Language model** | Phi-4 (32 layers, 3072 hidden, 24 heads, 8 KV heads) |
| **Vision encoder** | SigLIP-2 (27 layers, 1152 hidden, 16 heads) |
| **Audio encoder** | Cascades Conformer (24 blocks, 1024 dim, 16 heads) |
| **Vision projector** | 2-layer MLP (1152 &rarr; 3072 &rarr; 3072, GELU) |
| **Audio projector** | 2-layer MLP with speech/vision modes |

### LoRA switching

The original checkpoint ships with **two LoRA adapters** applied to the LLM backbone:

- **Vision LoRA** (r=256, alpha=512) &mdash; merged at load time by default.
- **Speech LoRA** (r=320, alpha=640) &mdash; stored for runtime switching.

`set_modality()` automatically selects the correct LoRA (or both) based on the input types.

## Model

- Hugging Face ID: `microsoft/Phi-4-multimodal-instruct`
- Remote processor code is ported in-tree, so `--trust-remote-code` is optional.

## CLI

### Text understanding

```sh
mlx_vlm.generate \
  --model microsoft/Phi-4-multimodal-instruct \
  --prompt "Explain the theory of relativity in simple terms." \
  --max-tokens 256
```

### Image understanding

```sh
mlx_vlm.generate \
  --model microsoft/Phi-4-multimodal-instruct \
  --image /path/to/image.jpg \
  --prompt "Describe this image." \
  --max-tokens 256
```

### Audio understanding

```sh
mlx_vlm.generate \
  --model microsoft/Phi-4-multimodal-instruct \
  --audio /path/to/audio.wav \
  --prompt "" \
  --max-tokens 256
```

### Multi-modal (image + audio)

```sh
mlx_vlm.generate \
  --model microsoft/Phi-4-multimodal-instruct \
  --image /path/to/image.jpg \
  --audio /path/to/audio.wav \
  --prompt "" \
  --max-tokens 256
```

## Python

```python
from mlx_vlm import load, generate
from mlx_vlm.prompt_utils import apply_chat_template

model, processor = load("microsoft/Phi-4-multimodal-instruct")

image = ["/path/to/image.jpg"]
audio = ["/path/to/audio.wav"]
prompt = "What animals are in the image?"

formatted_prompt = apply_chat_template(
    processor,
    model.config,
    prompt,
    num_images=len(image),
    num_audios=len(audio),
)

result = generate(
    model=model,
    processor=processor,
    prompt=formatted_prompt,
    image=image,
    audio=audio,
    max_tokens=256,
    temperature=0.0,
)
print(result.text)
```

## Quantization

```sh
mlx_vlm.convert \
  --model microsoft/Phi-4-multimodal-instruct \
  -q \
  --mlx-path Phi-4-multimodal-instruct-4bit
```

During quantization the model pre-merges both LoRA adapters into the LLM weights
and quantizes only the language model. Vision encoder, audio encoder, and
projectors are kept in bfloat16.

After quantization, LoRA switching is disabled (not needed since both adapters
are baked in).

## Notes

- Audio input is a 16 kHz mono waveform; the processor handles resampling automatically.
- The `<|image_1|>` / `<|audio_1|>` placeholders are inserted by `apply_chat_template`.
