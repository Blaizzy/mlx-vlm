# MLX-VLM

MLX-VLM is a package for inference and fine-tuning of Vision Language Models (VLMs) on Apple silicon using [MLX](https://github.com/ml-explore/mlx).

It provides:

- A command line interface for quick generation.
- A Gradio powered chat UI.
- A Python API for scripting and server integration.
- Support for multi-image chat and video understanding with select models.

## Installation

Install the package from PyPI:

```bash
pip install mlx-vlm
```

Check out the rest of the documentation for examples and usage details.

## Quick Examples

### Language-only

```python
from mlx_vlm import load, generate
from mlx_vlm.prompt_utils import apply_chat_template
from mlx_vlm.utils import load_config

model_path = "mlx-community/Qwen2-VL-2B-Instruct-4bit"
model, processor = load(model_path)
config = load_config(model_path)

prompt = "Explain the importance of the moon."
formatted = apply_chat_template(processor, config, prompt)
output = generate(model, processor, formatted, verbose=False)
print(output)
```

### Single Image

```python
image = ["path/to/photo.jpg"]
prompt = "Describe this image."
formatted = apply_chat_template(processor, config, prompt, num_images=1)
output = generate(model, processor, formatted, image, verbose=False)
print(output)
```

### Multi-image

```python
images = ["image1.jpg", "image2.jpg"]
prompt = "Compare these images."
formatted = apply_chat_template(processor, config, prompt, num_images=len(images))
output = generate(model, processor, formatted, images, verbose=False)
print(output)
```

### Video

```bash
python -m mlx_vlm.video_generate --model mlx-community/Qwen2-VL-2B-Instruct-4bit \
    --video path/to/video.mp4 --max-tokens 100
```
