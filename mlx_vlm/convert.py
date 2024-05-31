"""

The `convert` module provides functionality to convert Hugging Face models into MLX (Machine Learning eXchange) format, which is suitable for deployment and use in different environments. It also supports model quantization to reduce the model size and potentially increase inference speed at the cost of precision. This module is designed to be used as part of the command-line interface (CLI) but can also be integrated into Python scripts for programmatic access.

#### Key features include:
- Load models from Hugging Face format, both from local directories and remote repositories.
- Save models into MLX format, which consist of the following files:
  - Model weights `.safetensors`
  - Configuration file `config.json`
  - Tokenizer files, if the model includes a tokenizer
- Option to quantize the model, reducing its memory footprint by approximating parameter values with fewer bits.
- Option to define custom quantization settings such as group size and number of bits per weight.
- Choose between different data types for saving parameters, with default as `float16`.
- Functionality to dequantize a previously quantized model, converting it back to its original form.
- Capability to upload the converted model to a specified Hugging Face repository.

The module consists of a configuration parser and a main function that handles the conversion process based on the provided command-line arguments. The parser defines and validates the commands and options available to the user when running the `convert` script from the command line.

#### Configuration Parser:
The `configure_parser` function generates an argument parser with the following options:
- `--hf-path` the input path to the Hugging Face model. This is a mandatory argument.
- `--mlx-path` the output path where the MLX model will be saved.
- `-q`/`--quantize` a flag to indicate if the model should be quantized.
- `--q-group-size` the group size for quantization.
- `--q-bits` the number of bits per weight for quantization.
- `--dtype` the datatype for storing the model's parameters. Optional if quantization is not applied.
- `--upload-repo` the Hugging Face repository where the converted model should be uploaded.
- `-d`/`--dequantize` a flag to indicate if a quantized model should be converted back to the non-quantized version.

#### Main Function:
The `main` function is the entry point of the module, responsible for parsing command-line arguments and invoking the `convert` function with the appropriate parameters to perform the model conversion.
"""

# Copyright © 2023-2024 Apple Inc.

import argparse

from .utils import convert


def configure_parser() -> argparse.ArgumentParser:
    """
    Configures and returns an argument parser for a command-line interface.
    This parser is designed to convert Hugging Face models into MLX format, providing
    options for the input and output paths, quantization, and upload configurations.
    It allows for the customization of quantization parameters including group size
    and bit-rate, as well as specifying the data type for saving model parameters.
    A dequantization option is also provided.

    Returns:
        (argparse.ArgumentParser):
             The configured argument parser with all the
            options for model conversion and handling.

    """
    parser = argparse.ArgumentParser(
        description="Convert Hugging Face model to MLX format"
    )

    parser.add_argument("--hf-path", type=str, help="Path to the Hugging Face model.")
    parser.add_argument(
        "--mlx-path", type=str, default="mlx_model", help="Path to save the MLX model."
    )
    parser.add_argument(
        "-q", "--quantize", help="Generate a quantized model.", action="store_true"
    )
    parser.add_argument(
        "--q-group-size", help="Group size for quantization.", type=int, default=64
    )
    parser.add_argument(
        "--q-bits", help="Bits per weight for quantization.", type=int, default=4
    )
    parser.add_argument(
        "--dtype",
        help="Type to save the parameters, ignored if -q is given.",
        type=str,
        choices=["float16", "bfloat16", "float32"],
        default="float16",
    )
    parser.add_argument(
        "--upload-repo",
        help="The Hugging Face repo to upload the model to.",
        type=str,
        default=None,
    )
    parser.add_argument(
        "-d",
        "--dequantize",
        help="Dequantize a quantized model.",
        action="store_true",
        default=False,
    )
    return parser


def main():
    """
    Configures and initializes the command-line argument parser for the conversion script.
    This function sets up the argument parser with options to convert a Hugging Face (HF) model to MLX format. It includes arguments for specifying the paths to the source HF model and the destination MLX model directory. Additional options are provided for quantization settings, parameter data type, and whether to upload the converted model to a Hugging Face repository.

    Returns:
        (argparse.ArgumentParser):
             An ArgumentParser object with configured arguments for the model conversion script.

    """
    parser = configure_parser()
    args = parser.parse_args()
    convert(**vars(args))


if __name__ == "__main__":
    main()
