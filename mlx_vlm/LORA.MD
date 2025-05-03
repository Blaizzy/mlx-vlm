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
- Mllama (Llama-3.2-vision)

## Coming Soon
- LLaVA-Next
- Phi3_vision

## Usage

To use the script, run it from the command line with the desired arguments:

```
python lora.py --dataset /path/to/your/dataset [other options]
```

## Dataset format

The dataset should be a Hugging Face dataset with a `images` column and a `messages` column.

```
{
    "images": ...,
    "messages": ...,
}
```

Support for other formats and column names will be added soon.

## Arguments

The script accepts the following command-line arguments:

- `--model-path`: Path to the pre-trained model (default: "mlx-community/Qwen2-VL-2B-Instruct-bf16")
- `--dataset`: Path to your dataset (required)
- `--learning-rate`: Learning rate for the optimizer (default: 1e-4)
- `--batch-size`: Batch size for training (default: 1)
- `--epochs`: Number of epochs to train (default: 1)
- `--steps`: Number of steps per epoch (default: 0)
- `--print-every`: Print loss every n steps (default: 10)
- `--adapter-path`: Load path to resume training from a previously saved adapter (default: None)
- `--output-path`: Path to save the trained adapter (default: "adapters.safetensors")

## Example

Here's an example of how to run the script with custom parameters:

```
python lora.py --dataset /path/to/your/dataset --model-path /path/to/your/model --epochs 2 --batch-size 4 --learning-rate 5e-5
```

## Output

The script will print the training loss at regular intervals (defined by `--print-every`). After training, it will save the LoRA adapter to the specified output path.

## Note

If you want to use QLoRA, you need to pass a pre-quantized model to the script using the `--model-path` argument (i.e. `mlx-community/Qwen2-VL-2B-Instruct-4bit`).
Make sure you have the necessary permissions to read the dataset and write the output file. Also, ensure that your system has sufficient computational resources to handle the specified batch size and model.

## Contributing

Feel free to submit issues or pull requests if you find any bugs or have suggestions for improvements.
