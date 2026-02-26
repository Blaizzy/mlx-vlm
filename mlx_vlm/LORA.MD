# LoRA Training in MLX-VLM

## Overview

`lora.py` is a Python script for fine-tuning vision language models (VLMs) using Low-Rank Adaptation (LoRA or QLoRA). This script allows you to train the model on your custom dataset, adjusting various parameters through command-line arguments. It uses the MLX trainer backend for efficient training on Apple Silicon and other supported devices.

## Requirements

- Python 3.7+
- Required Python packages: `mlx-vlm`, `mlx`, `numpy`, `transformers`, `datasets`, `PIL`

## Trainer Backend

The script uses the MLX trainer backend which provides:
- Efficient training on Apple Silicon (M1/M2/M3/M4+)
- Automatic mixed precision training
- Gradient checkpointing to reduce memory usage
- Gradient accumulation for larger effective batch sizes
- Integration with Hugging Face datasets

## Supported Models
All models that are supported e.g. `Qwen2/3/3.5 VL`, `LLaVA`, `Deepseek-VL(-V2)`, `Mllama`, etc., exept: `Gemma3n`, `Qwen3 Omni`

## Usage

To use the script, run it from the command line with the desired arguments:

```
python lora.py --dataset /path/to/your/dataset [other options]
```

## Dataset Format

The dataset should be a Hugging Face dataset with an `images` column and a `messages` column. **The `messages` format must match the specific format that your model expects during inference.**

### Dataset Structure
```json
{
    "images": [...],  // Image list
    "messages": [...]  // Messages in model-specific format
}
```

### Message Formats by Model

#### Qwen3 / Qwen2 VL
```json
{
    "images": [image1, image2],
    "messages": [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image1},
                {"type": "text", "text": "What is in this image?"}
            ]
        },
        {
            "role": "assistant",
            "content": [
                {"type": "text", "text": "This image shows a cat sleeping on a couch."}
            ]
        }
    ]
}
```

#### Mllama (Llama-3.2-vision)
```json
{
    "images": [image1],
    "messages": [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image1},
                {"type": "text", "text": "Describe this image in detail."}
            ]
        },
        {
            "role": "assistant",
            "content": [
                {"type": "text", "text": "The image shows..."}
            ]
        }
    ]
}
```

#### Deepseek-VL-V2
```json
{
    "images": [image1],
    "messages": [
        {
            "role": "<|User|>",
            "content": "<image>\n<|ref|>What is this?<|/ref|>.",
            "images": [image1]
        },
        {
            "role": "<|Assistant|>",
            "content": "This is a detailed response."
        }
    ]
}
```

#### Pixtral / LLaVA
```json
{
    "images": [image1],
    "messages": [
        {
            "role": "user",
            "content": "<image>What is in this image?"
        },
        {
            "role": "assistant",
            "content": "This image shows..."
        }
    ]
}
```

### Dataset Transformation

If your dataset has `question` and `answer` columns with an `image`/`images` column, the script will automatically transform it to the appropriate message format (only works for `Qwen` and `deepseek`). You can also provide a custom prompt template using JSON:

```bash
python lora.py --dataset your_dataset \
    --custom-prompt-format '{"user": [{"type": "image","image":"{image}"}, {"type": "text","text":"{question}"}], "assistant": [{"type": "text","text":"{answer}"}]}'
```

## Arguments

### Model Arguments
- `--model-path`: Path to the pre-trained model (default: "mlx-community/Qwen2-VL-2B-Instruct-bf16")
- `--full-finetune`: Enable full weight fine-tuning instead of LoRA
- `--train-vision`: Unfreeze and train the vision modules alongside the language model

### Dataset Arguments
- `--dataset`: Path or Hugging Face dataset identifier (required)
- `--split`: Dataset split to use (default: "train")
- `--dataset-config`: Dataset configuration name (optional)
- `--image-resize-shape`: Resize images to specific shape, e.g., `--image-resize-shape 768 768`
- `--custom-prompt-format`: Custom JSON prompt template for dataset transformation

### Training Arguments
- `--learning-rate`: Learning rate for the optimizer (default: 2e-5)
- `--batch-size`: Batch size for training (default: 4)
- `--iters`: Total number of iterations/steps (default: 1000)
- `--epochs`: Number of epochs to train (overrides `--iters` if set)
- `--steps-per-report`: Report loss every n steps (default: 10)
- `--steps-per-eval`: Evaluation frequency in steps (default: 200)
- `--steps-per-save`: Save checkpoint every n steps (default: 100)
- `--val-batches`: Number of validation batches (default: 25)
- `--max-seq-length`: Maximum sequence length (default: 2048)
- `--grad-checkpoint`: Enable gradient checkpointing to reduce memory usage
- `--grad-clip`: Gradient clipping value (optional)
- `--train-on-completions`: Only compute loss on assistant responses
- `--gradient-accumulation-steps`: Accumulate gradients over n batches (default: 1)
- `--assistant-id`: Token ID for assistant role (default: 77091)

### LoRA Arguments
- `--lora-rank`: LoRA rank (default: 8)
- `--lora-alpha`: LoRA alpha scaling factor (default: 16)
- `--lora-dropout`: LoRA dropout probability (default: 0.0)

### Output Arguments
- `--output-path`: Path to save the trained adapter (default: "adapters.safetensors")
- `--adapter-path`: Path to resume training from a previously saved adapter

## Examples

### Basic LoRA Training with Qwen3-VL
Train Qwen3-VL with LoRA on a Hugging Face dataset:
```bash
python lora.py \
    --model-path mlx-community/Qwen3-VL-2B-Instruct-bf16 \
    --dataset your-huggingface-dataset-id \
    --batch-size 2 \
    --epochs 2 \
    --learning-rate 2e-5 \
    --output-path ./qwen3-lora-adapter.safetensors
```

### LoRA Training with Mllama (Llama-3.2-Vision)
Fine-tune Mllama with LoRA and gradient checkpointing:
```bash
python lora.py \
    --model-path mlx-community/Llama-3.2-11B-Vision-Instruct-bf16 \
    --dataset your-dataset-id \
    --batch-size 1 \
    --epochs 3 \
    --learning-rate 1e-5 \
    --grad-checkpoint \
    --gradient-accumulation-steps 4 \
    --output-path ./mllama-lora-adapter.safetensors
```

### Full Fine-tuning with Vision Training
Fully fine-tune Qwen3-VL including vision modules:
```bash
python lora.py \
    --model-path mlx-community/Qwen3-VL-2B-Instruct-bf16 \
    --dataset your-dataset-id \
    --full-finetune \
    --train-vision \
    --batch-size 1 \
    --epochs 1 \
    --learning-rate 5e-6 \
    --grad-checkpoint \
    --output-path ./qwen3-full-finetune.safetensors
```

### QLoRA Training (Quantized LoRA)
Use quantized model for lower memory consumption:
```bash
python lora.py \
    --model-path mlx-community/Qwen3-VL-2B-Instruct-4bit \
    --dataset your-dataset-id \
    --batch-size 4 \
    --epochs 2 \
    --learning-rate 2e-4 \
    --lora-rank 16 \
    --lora-alpha 32 \
    --output-path ./qwen3-qlora-adapter.safetensors
```

### Resume Training from Checkpoint
Continue training from a saved adapter:
```bash
python lora.py \
    --model-path mlx-community/Qwen3-VL-2B-Instruct-bf16 \
    --dataset your-dataset-id \
    --adapter-path ./qwen3-lora-adapter.safetensors \
    --epochs 2 \
    --learning-rate 1e-5 \
    --output-path ./qwen3-lora-adapter-v2.safetensors
```

### Training with Custom Image Resolution
Train with specific image resolution for your use case:
```bash
python lora.py \
    --model-path mlx-community/Llama-3.2-11B-Vision-Instruct-4bit \
    --dataset your-dataset-id \
    --image-resize-shape 512 512 \
    --batch-size 2 \
    --epochs 2 \
    --output-path ./mllama-custom-res-adapter.safetensors
```

## Python Code Examples

### Programmatic Training with Qwen3-VL

You can also run the training directly from Python by calling the main function with parsed arguments:

```python
import argparse
from mlx_vlm.lora import main

# Create arguments namespace
args = argparse.Namespace(
    model_path="mlx-community/Qwen3-VL-2B-Instruct-bf16",
    dataset="your-huggingface-dataset-id",
    split="train",
    dataset_config=None,
    batch_size=2,
    epochs=2,
    learning_rate=2e-5,
    iters=1000,
    steps_per_report=10,
    steps_per_eval=200,
    steps_per_save=100,
    val_batches=25,
    max_seq_length=2048,
    lora_rank=8,
    lora_alpha=16,
    lora_dropout=0.0,
    output_path="./qwen3-lora-adapter.safetensors",
    adapter_path=None,
    full_finetune=False,
    train_vision=False,
    grad_checkpoint=False,
    grad_clip=None,
    train_on_completions=False,
    gradient_accumulation_steps=1,
    assistant_id=77091,
    image_resize_shape=None,
    custom_prompt_format=None
)

# Run training
main(args)
```

### Training Mllama with Gradient Checkpointing

```python
import argparse
from mlx_vlm.lora import main

args = argparse.Namespace(
    model_path="mlx-community/Llama-3.2-11B-Vision-Instruct-4bit",
    dataset="your-dataset-id",
    split="train",
    dataset_config=None,
    batch_size=1,
    epochs=3,
    learning_rate=1e-5,
    iters=1000,
    steps_per_report=10,
    steps_per_eval=200,
    steps_per_save=100,
    val_batches=25,
    max_seq_length=2048,
    lora_rank=8,
    lora_alpha=16,
    lora_dropout=0.0,
    output_path="./mllama-lora-adapter.safetensors",
    adapter_path=None,
    full_finetune=False,
    train_vision=False,
    grad_checkpoint=True,  # Enable gradient checkpointing
    grad_clip=1.0,
    train_on_completions=False,
    gradient_accumulation_steps=4,
    assistant_id=77091,
    image_resize_shape=None,
    custom_prompt_format=None
)

main(args)
```

### Full Fine-tuning with Vision Training

```python
import argparse
from mlx_vlm.lora import main

args = argparse.Namespace(
    model_path="mlx-community/Qwen3-VL-2B-Instruct-bf16",
    dataset="your-dataset-id",
    split="train",
    dataset_config=None,
    batch_size=1,
    epochs=1,
    learning_rate=5e-6,
    iters=1000,
    steps_per_report=10,
    steps_per_eval=200,
    steps_per_save=100,
    val_batches=25,
    max_seq_length=2048,
    lora_rank=8,
    lora_alpha=16,
    lora_dropout=0.0,
    output_path="./qwen3-full-finetune.safetensors",
    adapter_path=None,
    full_finetune=True,  # Full weight fine-tuning
    train_vision=True,   # Train vision modules
    grad_checkpoint=True,
    grad_clip=None,
    train_on_completions=False,
    gradient_accumulation_steps=1,
    assistant_id=77091,
    image_resize_shape=None,
    custom_prompt_format=None
)

main(args)
```

### QLoRA Training (Programmatic)

```python
import argparse
from mlx_vlm.lora import main

# Using quantized model for QLoRA
args = argparse.Namespace(
    model_path="mlx-community/Qwen3-VL-2B-Instruct-4bit",  # Quantized model
    dataset="your-dataset-id",
    split="train",
    dataset_config=None,
    batch_size=4,
    epochs=2,
    learning_rate=2e-4,
    iters=1000,
    steps_per_report=10,
    steps_per_eval=200,
    steps_per_save=100,
    val_batches=25,
    max_seq_length=2048,
    lora_rank=16,      # Larger rank for better quality
    lora_alpha=32,
    lora_dropout=0.05,
    output_path="./qwen3-qlora-adapter.safetensors",
    adapter_path=None,
    full_finetune=False,
    train_vision=False,
    grad_checkpoint=False,
    grad_clip=None,
    train_on_completions=False,
    gradient_accumulation_steps=1,
    assistant_id=77091,
    image_resize_shape=None,
    custom_prompt_format=None
)

main(args)
```

### Resuming Training from Checkpoint

```python
import argparse
from mlx_vlm.lora import main

args = argparse.Namespace(
    model_path="mlx-community/Qwen3-VL-2B-Instruct-bf16",
    dataset="your-dataset-id",
    split="train",
    dataset_config=None,
    batch_size=2,
    epochs=2,
    learning_rate=1e-5,
    iters=1000,
    steps_per_report=10,
    steps_per_eval=200,
    steps_per_save=100,
    val_batches=25,
    max_seq_length=2048,
    lora_rank=8,
    lora_alpha=16,
    lora_dropout=0.0,
    output_path="./qwen3-lora-adapter-v2.safetensors",
    adapter_path="./qwen3-lora-adapter.safetensors",  # Resume from checkpoint
    full_finetune=False,
    train_vision=False,
    grad_checkpoint=False,
    grad_clip=None,
    train_on_completions=False,
    gradient_accumulation_steps=1,
    assistant_id=77091,
    image_resize_shape=None,
    custom_prompt_format=None
)

main(args)
```

### Custom Dataset Transformation with JSON Template

```python
import argparse
import json
from mlx_vlm.lora import main

# Define custom prompt format as JSON
custom_format = {
    "user": [
        {"type": "image", "image": "{image}"},
        {"type": "text", "text": "{question}"}
    ],
    "assistant": [
        {"type": "text", "text": "{answer}"}
    ]
}

args = argparse.Namespace(
    model_path="mlx-community/Qwen3-VL-2B-Instruct-bf16",
    dataset="your-dataset-id",
    split="train",
    dataset_config=None,
    batch_size=2,
    epochs=2,
    learning_rate=2e-5,
    iters=1000,
    steps_per_report=10,
    steps_per_eval=200,
    steps_per_save=100,
    val_batches=25,
    max_seq_length=2048,
    lora_rank=8,
    lora_alpha=16,
    lora_dropout=0.0,
    output_path="./qwen3-custom-format-adapter.safetensors",
    adapter_path=None,
    full_finetune=False,
    train_vision=False,
    grad_checkpoint=False,
    grad_clip=None,
    train_on_completions=False,
    gradient_accumulation_steps=1,
    assistant_id=77091,
    image_resize_shape=None,
    custom_prompt_format=json.dumps(custom_format)  # Pass as JSON string
)

main(args)
```

### Creating Training Dataset Programmatically

```python
from datasets import load_dataset, Dataset
from PIL import Image
import json

# Load and prepare dataset
def prepare_qwen3_dataset():
    # Example: Load from local images and create dataset
    dataset_dict = {
        "images": [],
        "messages": []
    }
    
    # Load your images and create message format
    for idx in range(num_samples):
        image = Image.open(f"path/to/image_{idx}.jpg")
        
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": "Describe this image"}
                ]
            },
            {
                "role": "assistant",
                "content": [
                    {"type": "text", "text": "Your annotation here"}
                ]
            }
        ]
        
        dataset_dict["images"].append(image)
        dataset_dict["messages"].append(messages)
    
    # Create Hugging Face dataset
    dataset = Dataset.from_dict(dataset_dict)
    
    # Push to Hugging Face Hub
    dataset.push_to_hub("your-username/your-dataset-name")
    
    return dataset

# Prepare and train
dataset = prepare_qwen3_dataset()

import argparse
from mlx_vlm.lora import main

args = argparse.Namespace(
    model_path="mlx-community/Qwen3-VL-2B-Instruct-bf16",
    dataset="your-username/your-dataset-name",
    split="train",
    dataset_config=None,
    batch_size=2,
    epochs=2,
    learning_rate=2e-5,
    iters=1000,
    steps_per_report=10,
    steps_per_eval=200,
    steps_per_save=100,
    val_batches=25,
    max_seq_length=2048,
    lora_rank=8,
    lora_alpha=16,
    lora_dropout=0.0,
    output_path="./qwen3-lora-adapter.safetensors",
    adapter_path=None,
    full_finetune=False,
    train_vision=False,
    grad_checkpoint=False,
    grad_clip=None,
    train_on_completions=False,
    gradient_accumulation_steps=1,
    assistant_id=77091,
    image_resize_shape=None,
    custom_prompt_format=None
)

main(args)
```

## Output

The script will display training progress and metrics at regular intervals (controlled by `--steps-per-report`). Output includes:
- Current training iteration and total iterations
- Loss value at each reporting step
- Running average loss
- Training speed (tokens/sec)
- Estimated time remaining

After training completes, the trained LoRA adapter is saved to the path specified by `--output-path`.

## Training Tips

### Memory Optimization
- Use `--grad-checkpoint` to reduce memory usage (slightly slower)
- Reduce `--batch-size` if running out of memory
- Use `--gradient-accumulation-steps` to simulate larger batch sizes
- Consider QLoRA with quantized models for constrained devices

### Convergence & Quality
- Start with lower learning rates (1e-5 to 2e-5) for LoRA training
- Increase `--lora-rank` (16-32) for more expressive adapters
- Use `--train-vision` only if vision quality matters for your task
- Monitor validation metrics with `--steps-per-eval`

### Hardware-Specific
- On Apple Silicon, MLX automatically optimizes memory and compute
- For larger models (>11B), enable gradient checkpointing
- Ensure `--batch-size` balances speed and memory usage

## QLoRA and Quantization

To use QLoRA (Quantized LoRA) for even more efficient training:
1. Use a quantized model checkpoint, e.g., `mlx-community/Qwen3-VL-2B-Instruct-4bit`
2. Run the training script normally with quantized model path
3. LoRA layers will be added in full precision for training
4. The resulting adapter is much smaller and faster to deploy

Example:
```bash
python lora.py \
    --model-path mlx-community/Qwen3-VL-2B-Instruct-4bit \
    --dataset your-dataset-id \
    --batch-size 4 \
    --epochs 2
```

## Notes

- Ensure you have sufficient disk space for your dataset and checkpoints
- Make sure you have the necessary permissions to read the dataset and write the output file
- Training times vary based on batch size, model size, dataset size, and hardware
- Use the same message format structure for your dataset as your target model expects

## Contributing

Feel free to submit issues or pull requests if you find any bugs or have suggestions for improvements.
