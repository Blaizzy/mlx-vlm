# LoRA Training Script

## Overview

`lora.py` is a Python script for fine-tuning a vision language models (VLMs) using Low-Rank Adaptation (LoRA or QLoRA). This script allows you to train the model on your custom dataset, adjusting various parameters through command-line arguments.

## Requirements

- Python 3.7+
- Required Python packages: `mlx-vlm`, `numpy`, `transformers`, `datasets`, `PIL`

## Supported Models
- Qwen2
- LLaVA (except for LLaVA-Next)
- Pixtral
- Idefics 2
- Deepseek-VL
- Paligemma

## Coming Soon
- LLaVA-Next
- Phi3_vision

## Usage

To use the script, run it from the command line with the desired arguments:

```
python lora.py --dataset /path/to/your/dataset [other options]
```

## Arguments

The script accepts the following command-line arguments:

- `--model_path`: Path to the pre-trained model (default: "mlx-community/nanoLLaVA-1.5-bf16")
- `--dataset`: Path to your dataset (required)
- `--learning_rate`: Learning rate for the optimizer (default: 1e-4)
- `--batch_size`: Batch size for training (default: 2)
- `--epochs`: Number of epochs to train (default: 1)
- `--steps`: Number of steps per epoch (default: 100)
- `--print_every`: Print loss every n steps (default: 10)
- `--output_path`: Path to save the trained adapter (default: "nanollava_lora_adapter.safetensors")

## Example

Here's an example of how to run the script with custom parameters:

```
python lora.py --dataset /path/to/your/dataset --model_path /path/to/your/model --epochs 2 --steps 200 --batch_size 4 --learning_rate 5e-5
```

This command will:
- Use the dataset at `/path/to/your/dataset`
- Use the model at `/path/to/your/model`
- Train for 2 epochs
- Perform 200 steps per epoch
- Use a batch size of 4
- Set the learning rate to 5e-5

## Output

The script will print the training loss at regular intervals (defined by `--print_every`). After training, it will save the LoRA adapter to the specified output path.

## Note

If you want to use QLoRA, you need to pass a pre-quantized model to the script using the `--model_path` argument (i.e. `mlx-community/nanoLLaVA-1.5-4bit`).
Make sure you have the necessary permissions to read the dataset and write the output file. Also, ensure that your system has sufficient computational resources to handle the specified batch size and model.

## Contributing

Feel free to submit issues or pull requests if you find any bugs or have suggestions for improvements.
