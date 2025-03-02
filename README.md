[![Upload Python Package](https://github.com/Blaizzy/mlx-vlm/actions/workflows/python-publish.yml/badge.svg)](https://github.com/Blaizzy/mlx-vlm/actions/workflows/python-publish.yml)
# MLX-VLM

MLX-VLM is a package for inference and fine-tuning of Vision Language Models (VLMs) on your Mac using MLX.

## Table of Contents
- [Installation](#installation)
- [Usage](#usage)
  - [Command Line Interface (CLI)](#command-line-interface-cli)
  - [Chat UI with Gradio](#chat-ui-with-gradio)
  - [Python Script](#python-script)
- [Multi-Image Chat Support](#multi-image-chat-support)
  - [Supported Models](#supported-models)
  - [Usage Examples](#usage-examples)
- [Fine-tuning](#fine-tuning)

## Installation

The easiest way to get started is to install the `mlx-vlm` package using pip:

```sh
pip install mlx-vlm
```

## Usage

### Command Line Interface (CLI)

Generate output from a model using the CLI:

```sh
python -m mlx_vlm.generate --model mlx-community/Qwen2-VL-2B-Instruct-4bit --max-tokens 100 --temp 0.0 --image http://images.cocodataset.org/val2017/000000039769.jpg
```

### Chat UI with Gradio

Launch a chat interface using Gradio:

```sh
python -m mlx_vlm.chat_ui --model mlx-community/Qwen2-VL-2B-Instruct-4bit
```

### Python Script

Here's an example of how to use MLX-VLM in a Python script:

```python
import mlx.core as mx
from mlx_vlm import load, generate
from mlx_vlm.prompt_utils import apply_chat_template
from mlx_vlm.utils import load_config

# Load the model
model_path = "mlx-community/Qwen2-VL-2B-Instruct-4bit"
model, processor = load(model_path)
config = load_config(model_path)

# Prepare input
image = ["http://images.cocodataset.org/val2017/000000039769.jpg"]
prompt = "Describe this image."

# Apply chat template
formatted_prompt = apply_chat_template(
    processor, config, prompt, num_images=len(image)
)

# Generate output
output = generate(model, processor, formatted_prompt, image, verbose=False)
print(output)
```

## Multi-Image Chat Support

MLX-VLM supports analyzing multiple images simultaneously with select models. This feature enables more complex visual reasoning tasks and comprehensive analysis across multiple images in a single conversation.

### Supported Models

The following models support multi-image chat:

1. Idefics 2
2. LLaVA (Interleave)
3. Qwen2-VL
4. Phi3-Vision
5. Pixtral

### Usage Examples

#### Python Script

```python
from mlx_vlm import load, generate
from mlx_vlm.prompt_utils import apply_chat_template
from mlx_vlm.utils import load_config

model_path = "mlx-community/Qwen2-VL-2B-Instruct-4bit"
model, processor = load(model_path)
config = load_config(model_path)

images = ["path/to/image1.jpg", "path/to/image2.jpg"]
prompt = "Compare these two images."

formatted_prompt = apply_chat_template(
    processor, config, prompt, num_images=len(images)
)

output = generate(model, processor, formatted_prompt, images, verbose=False)
print(output)
```

#### Command Line

```sh
python -m mlx_vlm.generate --model mlx-community/Qwen2-VL-2B-Instruct-4bit --max-tokens 100 --prompt "Compare these images" --image path/to/image1.jpg path/to/image2.jpg
```

## Video Understanding

MLX-VLM also supports video analysis such as captioning, summarization, and more, with select models.

### Supported Models

The following models support video chat:

1. Qwen2-VL
2. Qwen2.5-VL
3. Idefics3
4. LLaVA

With more coming soon.

### Usage Examples

#### Command Line
```sh
python -m mlx_vlm.video_generate --model mlx-community/Qwen2-VL-2B-Instruct-4bit --max-tokens 100 --prompt "Describe this video" --video path/to/video.mp4 --max-pixels 224 224 --fps 1.0
```


These examples demonstrate how to use multiple images with MLX-VLM for more complex visual reasoning tasks.

# Fine-tuning

MLX-VLM supports fine-tuning models with LoRA and QLoRA.

## LoRA & QLoRA

To learn more about LoRA, please refer to the [LoRA.md](./mlx_vlm/LORA.MD) file.
