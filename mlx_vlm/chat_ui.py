import argparse

import gradio as gr

from mlx_vlm import load

from .prompt_utils import apply_chat_template
from .utils import load, load_config, load_image_processor, stream_generate


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


def chat(message, history, temperature, max_tokens):
    chat = []
    if len(message["files"]) >= 1:
        chat.append({"role": "user", "content": message["text"]})
    else:
        raise gr.Error("Please upload an image. Text only chat is not supported.")

    file = message["files"][-1]
    if model.config.model_type != "paligemma":
        prompt = apply_chat_template(processor, config, chat)
    else:
        prompt = message.text

    response = ""
    for chunk in stream_generate(
        model, processor, file, prompt, image_processor, max_tokens, temp=temperature
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
