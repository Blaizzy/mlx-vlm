# Ternary Bonsai 2

Ternary Bonsai 2 is PrismML's 27-billion-parameter vision-language model, derived
from Qwen3.8-27B. It supports text generation, visual question answering, and
reasoning over images and video, with a context window of up to 262,144 tokens.

The model uses ternary language weights to reduce its memory and storage
requirements for local inference on Apple silicon. The complete MLX checkpoint,
including the vision encoder, occupies approximately 8.6 GB on disk.

## Supported Models

- [prism-ml/Ternary-Bonsai-2-27B-mlx-2bit](https://huggingface.co/prism-ml/Ternary-Bonsai-2-27B-mlx-2bit)

## CLI

```bash
python -m mlx_vlm.generate \
    --model prism-ml/Ternary-Bonsai-2-27B-mlx-2bit \
    --image path/to/image.jpg \
    --prompt "Describe this image." \
    --max-tokens 256
```

Omit `--image` for text-only generation. The checkpoint's chat template enables
thinking by default and accepts `enable_thinking=False` for direct answers.

## Python

```python
from mlx_vlm import generate, load
from mlx_vlm.prompt_utils import apply_chat_template

model, processor = load("prism-ml/Ternary-Bonsai-2-27B-mlx-2bit")
prompt = apply_chat_template(
    processor,
    model.config,
    "Describe this image.",
    num_images=1,
    enable_thinking=False,
)
result = generate(
    model, processor, prompt, image="path/to/image.jpg", max_tokens=256
)
print(result.text)
```
