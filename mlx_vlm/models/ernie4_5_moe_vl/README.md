# ERNIE 4.5 VL

ERNIE 4.5 VL is a multimodal MoE model from Baidu for image understanding and visual reasoning.

This MLX-VLM integration includes:
- an in-tree ERNIE tokenizer and processor
- ERNIE-specific chat formatting handled by `apply_chat_template`
- support for the ERNIE 4.5 VL architecture added to the standard `load()` and `generate()` flow

## Model

- Hugging Face ID: `baidu/ERNIE-4.5-VL-28B-A3B-Thinking`
- Architecture: multimodal MoE vision-language model
- Best for: general image understanding, document analysis, charts, screenshots, and visually grounded question answering

## Install

```sh
pip install -U mlx-vlm
```

## CLI

### Image understanding

```sh
uv run mlx_vlm.generate \
  --model baidu/ERNIE-4.5-VL-28B-A3B-Thinking \
  --image /path/to/image.jpg \
  --prompt "Describe this image." \
  --max-tokens 512 \
  --temperature 0
```

### Visual question answering

```sh
uv run mlx_vlm.generate \
  --model baidu/ERNIE-4.5-VL-28B-A3B-Thinking \
  --image /path/to/screenshot.png \
  --prompt "What is the main action shown in this screenshot?" \
  --max-tokens 512 \
  --temperature 0
```

## Python

```python
from mlx_vlm import load, generate
from mlx_vlm.prompt_utils import apply_chat_template

model, processor = load("baidu/ERNIE-4.5-VL-28B-A3B-Thinking")

image = ["/path/to/image.jpg"]
prompt = "Describe the important details in this image."

formatted_prompt = apply_chat_template(
    processor,
    model.config,
    prompt,
    num_images=len(image),
)

result = generate(
    model=model,
    processor=processor,
    prompt=formatted_prompt,
    image=image,
    max_tokens=512,
    temperature=0.0,
)
print(result.text)
```

## Notes

- You usually should not manually add image placeholders when using `apply_chat_template`.
- When you pass chat messages through MLX-VLM, the ERNIE-specific image message format is handled for you automatically.
- Local image paths and image URLs can both be used as inputs.
