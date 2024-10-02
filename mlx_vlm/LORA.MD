# lora.py - NanoLLaVA LoRA Training Script

## Overview

`lora.py` is a Python script for fine-tuning a NanoLLaVA model using Low-Rank Adaptation (LoRA). This script allows you to train the model on your custom dataset, adjusting various parameters through command-line arguments.

## Requirements

- Python 3.7+
- MLX VLM library
- Required Python packages: `argparse`, `mlx_vlm`, `mlx`

## Supported Models
- Qwen2
- LLaVA (except for LLaVA-Next)
- Pixtral
- Idefics 2

## Coming Soon
- LLaVA-Next
- Phi3_vision
- Paligemma

Note: The script only works with model in full or half precision. Quantized models are not supported at the moment.

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
python lora.py --dataset /path/to/your/dataset --epochs 2 --steps 200 --batch_size 4 --learning_rate 5e-5
```

This command will:
- Use the dataset at `/path/to/your/dataset`
- Train for 2 epochs
- Perform 200 steps per epoch
- Use a batch size of 4
- Set the learning rate to 5e-5

## Output

The script will print the training loss at regular intervals (defined by `--print_every`). After training, it will save the LoRA adapter to the specified output path.

## Note

Make sure you have the necessary permissions to read the dataset and write the output file. Also, ensure that your system has sufficient computational resources to handle the specified batch size and model.

## Contributing

Feel free to submit issues or pull requests if you find any bugs or have suggestions for improvements.

## License

[Specify the license here, e.g., MIT, Apache 2.0, etc.]