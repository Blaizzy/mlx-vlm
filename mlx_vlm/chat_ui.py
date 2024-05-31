"""

## Module Overview: chat_ui

This module represents a graphical user interface for generating text based on an image and prompt using machine learning models. The goal is to provide a seamless interaction where users can upload images, input prompts, and receive generated textual content that relates to the provided image.

### Key Components:
- `gradio` library: Used to create the web-based interface through which users interact with the machine learning model.
- `generate`: A function that integrates the model and the processor with the uploaded image and input text to produce the output content.
- `chat`: This function orchestrates the interaction flow, handling the uploaded image, user prompt, and generating responses.
- `demo`: An instance of `gr.ChatInterface` which acts as the main entry point for the UI. It defines interface elements like sliders for parameter adjustments and sets up the launch configuration.

### Functionalities:
- Prompt users to upload an image and provide a text prompt.
- Allow users to configure generation parameters such as temperature and max tokens through sliders.
- Consume and process user input using pre-loaded models and processors to generate textual responses.
- Present a conversational interface where the generated text is dynamically returned as a chat response, creating an interactive experience.

### Key Concepts:
- **Processor**: Handles text tokenization and detokenization, often specific to the model's expected input format.
- **Image Processor**: Transforms the uploaded image into the required format for the model.
- **Model**: The deep learning model that takes the processed input to generate text content taking the image context into account.

### Usage Scenario:
This module is ideal for creating a text-based conversation where the context of the conversation is driven by images. It may be used in various applications such as virtual assistants, automated image description software, or educational tools where visual prompts are translated into descriptive text.
"""

import argparse
from typing import Optional

import gradio as gr
import mlx.core as mx

from mlx_vlm import load

from .prompt_utils import get_message_json
from .utils import (
    generate_step,
    load,
    load_config,
    load_image_processor,
    prepare_inputs,
    sample,
)


def parse_arguments():
    """
    Parses the command-line arguments for the script.

    Returns:
        (argparse.Namespace):
             An object containing the parsed command-line options as attributes. Includes the 'model' argument specifying the path to the model.

    Raises:
        argparse.ArgumentError:
             If any problem is encountered during argument parsing.

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
    return parser.parse_args()


args = parse_arguments()
config = load_config(args.model)
model, processor = load(args.model, {"trust_remote_code": True})
image_processor = load_image_processor(args.model)


def generate(
    model,
    processor,
    image: str,
    prompt: str,
    image_processor=None,
    temp: float = 0.0,
    max_tokens: int = 100,
    repetition_penalty: Optional[float] = None,
    repetition_context_size: Optional[int] = None,
    top_p: float = 1.0,
):
    """
    Generates a sequence of tokens using a given model and processor, conditioned on an input image and prompt text.
    The function handles the input processing, including the loading of the image via the image processor, and tokenizing the prompt text. It then generates tokens step by step using the model's prediction output until the maximum number of tokens is reached or the end-of-sequence token is generated.

    Args:
        model (nn.Module):
             The model used to generate the token sequence.
        processor:
             The processor associated with the model, which handles the encoding of the prompt and the decoding of generated tokens.
        image (str):
             Path to the image or an URL of the image to be used as context for the text generation.
        prompt (str):
             Initial text prompt to condition the text generation.
        image_processor (optional):
             An object to process the input image. If None, the processor's built-in image processing capability is used.
        temp (float, optional):
             The temperature to use for controlling the randomness of the generation. Defaults to 0.0, meaning no randomness.
        max_tokens (int, optional):
             The maximum number of tokens to generate. Defaults to 100.
        repetition_penalty (Optional[float], optional):
             Penalty to apply for repetition of tokens within the sequence to discourage redundancy.
        repetition_context_size (Optional[int], optional):
             The context size to consider for repetition penalty. Only used if repetition_penalty is not None.
        top_p (float, optional):
             The nucleus sampling probability, controlling the randomness of token generation by focusing on top tokens. Defaults to 1.0.

    Returns:
        (Generator[str]):
             A generator that yields each segment of the generated text as it is produced.

    Raises:
        ValueError:
             If the model fails to generate an output for the tokens, or if required configurations are missing or incorrect.

    """
    if image_processor is not None:
        tokenizer = processor
    else:
        tokenizer = processor.tokenizer

    image_token_index = model.config.image_token_index
    input_ids, pixel_values, mask = prepare_inputs(
        image_processor, processor, image, prompt, image_token_index
    )
    logits, cache = model(input_ids, pixel_values, mask=mask)
    logits = logits[:, -1, :]
    y, _ = sample(logits, temp, top_p)

    detokenizer = processor.detokenizer
    detokenizer.reset()

    detokenizer.add_token(y.item())

    for (token, _), n in zip(
        generate_step(
            model.language_model,
            logits,
            mask,
            cache,
            temp,
            repetition_penalty,
            repetition_context_size,
            top_p,
        ),
        range(max_tokens),
    ):
        token = token.item()

        if token == tokenizer.eos_token_id:
            break

        detokenizer.add_token(token)
        detokenizer.finalize()
        yield detokenizer.last_segment


def chat(message, history, temperature, max_tokens):
    """
    Communicates with the model to process the image and text received in the message and generates the response based on the model's output.
    The function accepts the user message which contains an image and a text prompt, along with the conversation history, temperature for sampling, and the maximum number of tokens to be generated in the response. It then processes the message to conform to the model's expected input format, by using templates if necessary. The image is then prepared and along with the processed text, is forwarded to the model for generating the response. The response tokens are yielded incrementally as they are produced, allowing the caller to handle streaming generation.

    Args:
        message (dict):
             A dictionary containing the 'text' field with a text prompt and 'files' field with the image to be processed.
        history (list):
             A list of previous messages in the conversation (unused in this version).
        temperature (float):
             The temperature used for sampling the prediction. Lower temperatures result in less random completions.
        max_tokens (int):
             The maximum number of tokens to generate for the response.

    Raises:
        gr.Error:
             An error is raised if the message does not contain an image.

    Yields:
        str:
             The next chunk of the generated response as a string, which may include text and possibly image data.

    """
    chat = []
    if len(message["files"]) >= 1:
        chat.append(get_message_json(config["model_type"], message["text"]))
    else:
        raise gr.Error("Please upload an image. Text only chat is not supported.")

    files = message["files"][-1]
    if "chat_template" in processor.__dict__.keys():
        messages = processor.apply_chat_template(
            chat,
            tokenize=False,
            add_generation_prompt=True,
        )

    elif "tokenizer" in processor.__dict__.keys():
        if model.config.model_type != "paligemma":
            messages = processor.tokenizer.apply_chat_template(
                chat,
                tokenize=False,
                add_generation_prompt=True,
            )
        else:
            messages = message["text"]

    response = ""
    for chunk in generate(
        model,
        processor,
        files,
        messages,
        image_processor,
        temperature,
        max_tokens,
    ):
        response += chunk
        yield response


demo = gr.ChatInterface(
    fn=chat,
    title="MLX-VLM Chat UI",
    additional_inputs_accordion=gr.Accordion(
        label="⚙️ Parameters", open=False, render=False
    ),
    additional_inputs=[
        gr.Slider(
            minimum=0, maximum=1, step=0.1, value=0.1, label="Temperature", render=False
        ),
        gr.Slider(
            minimum=128,
            maximum=4096,
            step=1,
            value=200,
            label="Max new tokens",
            render=False,
        ),
    ],
    description=f"Now Running {args.model}",
    multimodal=True,
)

demo.launch(inbrowser=True)
