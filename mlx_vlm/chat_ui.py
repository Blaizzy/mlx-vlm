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
