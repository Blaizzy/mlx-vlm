# MiniCPM-o (MiniCPM-o-4_5)

MiniCPM-o is an omni model that supports text, image, and audio understanding.

This MLX-VLM integration includes:
- Custom MiniCPM-o processor registration
- Image + audio input preprocessing
- Prompt placeholder handling for `<image>` and `<audio>`

## Model

- Hugging Face ID: `openbmb/MiniCPM-o-4_5`
- Remote tokenizer/processor code is ported in-tree, so `--trust-remote-code` is not required.

## Install

```sh
pip install -U mlx-vlm
```

## CLI

### Image understanding

```sh
uv run mlx_vlm.generate \
  --model openbmb/MiniCPM-o-4_5 \
  --image /path/to/image.jpg \
  --prompt "Describe this image briefly." \
  --max-tokens 128 \
  --temperature 0
```

### Audio understanding

```sh
uv run mlx_vlm.generate \
  --model openbmb/MiniCPM-o-4_5 \
  --audio /path/to/audio.wav \
  --prompt "Describe this audio briefly." \
  --max-tokens 256 \
  --temperature 0
```

### Multi-modal (image + audio)

```sh
uv run mlx_vlm.generate \
  --model openbmb/MiniCPM-o-4_5 \
  --image /path/to/image.jpg \
  --audio /path/to/audio.wav \
  --prompt "Describe what you see and hear." \
  --max-tokens 256 \
  --temperature 0
```

### Disable thinking in chat template

```sh
uv run mlx_vlm.generate \
  --model openbmb/MiniCPM-o-4_5 \
  --audio /path/to/audio.wav \
  --prompt "Describe this audio briefly." \
  --chat-template-kwargs '{"enable_thinking": false}' \
  --max-tokens 256 \
  --temperature 0
```

## Python

```python
from mlx_vlm import load, generate
from mlx_vlm.prompt_utils import apply_chat_template

model, processor = load(
    "openbmb/MiniCPM-o-4_5",
)

image = ["/path/to/image.jpg"]
audio = ["/path/to/audio.wav"]
prompt = "Summarize the visual and audio content."

formatted_prompt = apply_chat_template(
    processor,
    model.config,
    prompt,
    num_images=len(image),
    num_audios=len(audio),
    chat_template_kwargs={"enable_thinking": False},
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

## Notes

- You usually should not manually add `<image>` or `<audio>` markers when using `apply_chat_template`.
- For CLI, `--audio` currently behaves as a single-audio path in template counting.
