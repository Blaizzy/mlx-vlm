import argparse

import gradio as gr

from mlx_vlm import load

from .prompt_utils import get_chat_template, get_message_json
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
model, processor = load(args.model, processor_kwargs={"trust_remote_code": True})
image_processor = load_image_processor(args.model)


def chat(message, history, temperature, max_tokens):
    if config["model_type"] != "paligemma":
        chat_history = []
        for item in history:
            chat_history.append({"role": "user", "content": item[0]})
            if item[1] is not None:
                chat_history.append({"role": "assistant", "content": item[1]})

        chat_history.append({"role": "user", "content": message["text"]})

        messages = []
        for i, m in enumerate(chat_history):
            skip_token = True
            if i == len(chat_history) - 1 and m["role"] == "user":
                skip_token = False
            messages.append(
                get_message_json(
                    config["model_type"],
                    m["content"],
                    role=m["role"],
                    skip_image_token=skip_token,
                )
            )

        messages = get_chat_template(processor, messages, add_generation_prompt=True)

    else:
        messages = message["text"]

    files = ""
    if "files" in message and len(message["files"]) > 0:
        files = message["files"][-1]

    response = ""
    for chunk in stream_generate(
        model,
        processor,
        messages,
        files,
        max_tokens=max_tokens,
        temperature=temperature,
    ):
        response += chunk.text
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
