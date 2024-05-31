"""

## Module Overview: `generate` Module

The `generate` module is designed to facilitate text generation from images using pre-trained models. It provides functionalities to parse command-line arguments, load models and processors, and run the generation process. The module's main function orchestrates the generation process by integrating various components.

**Components of the generate module:**

### 1. Argument Parsing:
* Responsible for parsing command-line arguments which configure the text generation process, including the model path, image URL or path, prompt, maximum tokens, temperature for sampling, and verbosity of the output.

### 2. Model and Processor Loading:
* Handles the retrieval and loading of the specified model and its associated processors. It includes loading the model configuration, image processor, and tokenizer.

### 3. Generation Utilities:
* Includes the `generate` function which executes the text generation process using the loaded model, prompt, and image.

### 4. Sampling Mechanism:
* Provides a `sample` function for generating tokens from logits using temperature-based sampling, enabling more diverse outputs when temperature is higher than 0.

### 5. Main Function:
* Serves as the entry point that brings together all components allowing the user to generate text from an image using a pre-trained model. It utilizes the parsed arguments, loads the necessary components, and performs the text generation task.
"""

import argparse
import codecs

import mlx.core as mx

from .prompt_utils import get_message_json
from .utils import generate, get_model_path, load, load_config, load_image_processor

MODEL_TYPE = ""


def parse_arguments():
    """
    Parses command line arguments for generating text from an image using a model.
    This function creates an argument parser to handle inputs for a script that generates text from
    an image. It defines, with defaults, the command line arguments for the path to the model,
    the image URL or path, the accompanying prompt, the maximum number of tokens to generate,
    the sampling temperature, and whether verbose output is desired.

    Returns:
        (argparse.Namespace):
             An object containing the parsed command line arguments with the
        (following attributes):
        - model (str):
             The path to the local model directory or Hugging Face repository.
        - image (str):
             The URL or path of the image to be processed.
        - prompt (str):
             The prompt message to be processed by the model.
        - max_tokens (int):
             The maximum number of tokens to generate in the output.
        - temp (float):
             The sampling temperature for token generation.
        - verbose (bool):
             A flag indicating if detailed output is to be provided.

    """
    parser = argparse.ArgumentParser(
        description="Generate text from an image using a model."
    )
    parser.add_argument(
        "--model",
        type=str,
        default="qnguyen3/nanoLLaVA",
        help="The path to the local model directory or Hugging Face repo.",
    )
    parser.add_argument(
        "--image",
        type=str,
        default="http://images.cocodataset.org/val2017/000000039769.jpg",
        help="URL or path of the image to process.",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="What are these?",
        help="Message to be processed by the model.",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=100,
        help="Maximum number of tokens to generate.",
    )
    parser.add_argument(
        "--temp", type=float, default=0.3, help="Temperature for sampling."
    )
    parser.add_argument(
        "--verbose",
        type=bool,
        help="Detailed output.",
        default=True,
    )
    return parser.parse_args()


def get_model_and_processors(model_path):
    """
    Fetches the model and its associated processors from the provided path.
    This function retrieves the model, tokenizer, image processor, and model configuration based on the provided model path. It resolves the model path using `get_model_path` function, loads the configuration using `load_config`, and then loads respective components: the model using `load`, the image processor using `load_image_processor`, and returns them together with the configuration.

    Args:
        model_path (str or Path):
             The path or repository identifier of the model.

    Returns:
        (tuple):
             A tuple containing four elements:
        (- nn.Module):
             The loaded model object.
        (- Union[PreTrainedTokenizer, PreTrainedTokenizerFast]):
             The tokenizer associated with the model.
        (- BaseImageProcessor or None):
             The image processor for the model, if available.
        (- dict):
             The configuration dictionary of the model.

    Raises:
        FileNotFoundError:
             If the model configuration or any required files are not found at the specified path.
        ValueError:
             If the model type is not supported or any other error occurs during loading of model components.

    """
    model_path = get_model_path(model_path)
    config = load_config(model_path)
    model, processor = load(model_path, {"trust_remote_code": True})
    image_processor = load_image_processor(model_path)
    return model, processor, image_processor, config


def sample(logits, temperature=0.0):
    """
    Computes a sample from the provided logits, optionally using temperature scaling.

    Args:
        logits (mxnet.ndarray.NDArray):
             A batch of logits from which to sample. Logits are typically unnormalized log probabilities.
        temperature (float, optional):
             A coefficient to scale the logits before applying softmax. If 0 (default), the function will return the argmax.

    Returns:
        (mxnet.ndarray.NDArray):
             The indices of the maximum logits if temperature is 0, or a sample drawn from the categorical distribution parameterized by the scaled logits otherwise.

    Raises:
        ValueError:
             If temperature is negative.

    """
    if temperature == 0:
        return mx.argmax(logits, axis=-1)
    else:
        return mx.random.categorical(logits * (1 / temperature))


def main():
    """
    Runs the complete process of image-based text generation using provided arguments.
    This function processes command-line arguments to define the behavior of text generation, loads the necessary model and processors, formats the prompt, runs the image-based generation, and handles the output.

    Args:
        None:
             All arguments are parsed from the command-line via `argparse`.

    Raises:
        ValueError:
             If either the `chat_template` or `tokenizer` attribute is missing in the processor, an error is raised.

    Returns:
        (None):
             The output generated text is either printed to the console or suppressed depending on the verbosity setting.

    """
    args = parse_arguments()
    model, processor, image_processor, config = get_model_and_processors(args.model)

    prompt = codecs.decode(args.prompt, "unicode_escape")

    if "chat_template" in processor.__dict__.keys():
        prompt = processor.apply_chat_template(
            [get_message_json(config["model_type"], prompt)],
            tokenize=False,
            add_generation_prompt=True,
        )

    elif "tokenizer" in processor.__dict__.keys():
        if model.config.model_type != "paligemma":
            prompt = processor.tokenizer.apply_chat_template(
                [get_message_json(config["model_type"], prompt)],
                tokenize=False,
                add_generation_prompt=True,
            )

    else:
        ValueError(
            "Error: processor does not have 'chat_template' or 'tokenizer' attribute."
        )

    output = generate(
        model,
        processor,
        args.image,
        prompt,
        image_processor,
        args.temp,
        args.max_tokens,
        args.verbose,
    )
    if not args.verbose:
        print(output)


if __name__ == "__main__":
    main()
