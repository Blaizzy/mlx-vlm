import argparse

import gradio as gr

from mlx_vlm import load

from .generate import stream_generate
from .prompt_utils import get_chat_template, get_message_json
from .utils import load_config, load_image_processor


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
# Use most of the viewport for conversation while leaving room for header/controls and avoid overshooting on short screens.
chatbot_height = "clamp(380px, calc(100vh - 320px), 820px)"


def chat(message, history, temperature, max_tokens):
    image_file = ""
    if "files" in message and len(message["files"]) > 0:
        image_file = message["files"][-1]

    num_images = 1 if image_file else 0

    if config["model_type"] != "paligemma":
        chat_history = []
        for item in history:
            # Handle new Gradio dict format
            if isinstance(item, dict):
                role = item.get("role", "user")
                content = item.get("content", "")
                # Extract text from multimodal content
                if isinstance(content, str):
                    pass  # Already a string
                elif isinstance(content, dict) and "text" in content:
                    content = content["text"]
                elif isinstance(content, list):
                    # Extract text from list of content items (multimodal)
                    text_parts = []
                    for c in content:
                        if isinstance(c, str):
                            text_parts.append(c)
                        elif isinstance(c, dict) and c.get("type") == "text":
                            text_parts.append(c.get("text", ""))
                    content = " ".join(text_parts) if text_parts else ""
                else:
                    content = ""
                # Skip stats lines from previous responses
                if role == "assistant" and isinstance(content, str) and content:
                    content = content.split("\n\n---\n")[0]
                if content:  # Only add non-empty messages
                    chat_history.append({"role": role, "content": content})
            # Handle old tuple format (user_msg, assistant_msg)
            elif isinstance(item, (list, tuple)):
                if isinstance(item[0], str):
                    chat_history.append({"role": "user", "content": item[0]})
                elif isinstance(item[0], dict) and "text" in item[0]:
                    chat_history.append({"role": "user", "content": item[0]["text"]})
                if item[1] is not None:
                    content = (
                        item[1].split("\n\n---\n")[0]
                        if isinstance(item[1], str)
                        else item[1]
                    )
                    chat_history.append({"role": "assistant", "content": content})

        chat_history.append({"role": "user", "content": message["text"]})

        messages = []
        for i, m in enumerate(chat_history):
            skip_token = True
            if i == len(chat_history) - 1 and m["role"] == "user" and image_file:
                skip_token = False
            messages.append(
                get_message_json(
                    config["model_type"],
                    m["content"],
                    role=m["role"],
                    skip_image_token=skip_token,
                    num_images=num_images if not skip_token else 0,
                )
            )

        messages = get_chat_template(processor, messages, add_generation_prompt=True)

    else:
        messages = message["text"]

    response = ""
    last_chunk = None
    for chunk in stream_generate(
        model,
        processor,
        messages,
        image=image_file,
        max_tokens=max_tokens,
        temperature=temperature,
    ):
        response += chunk.text
        last_chunk = chunk
        yield response

    # Append stats after generation completes
    if last_chunk is not None:
        stats = (
            f"\n\n---\n"
            f"<sub>📊 Prompt: {last_chunk.prompt_tokens} tokens @ {last_chunk.prompt_tps:.1f} t/s | "
            f"Generation: {last_chunk.generation_tokens} tokens @ {last_chunk.generation_tps:.1f} t/s | "
            f"Peak memory: {last_chunk.peak_memory:.2f} GB</sub>"
        )
        yield response + stats


chatbot = gr.Chatbot(height=chatbot_height, scale=1, render=False)

with gr.Blocks(fill_height=True) as demo:
    gr.Markdown(f"## MLX-VLM Chat UI — {args.model}")

    with gr.Row():
        temperature = gr.Slider(
            minimum=0, maximum=1, step=0.1, value=0.1, label="Temperature", scale=1
        )
        max_tokens = gr.Slider(
            minimum=128,
            maximum=4096,
            step=1,
            value=1024,
            label="Max new tokens",
            scale=1,
        )

    gr.ChatInterface(
        fn=chat,
        additional_inputs=[temperature, max_tokens],
        multimodal=True,
        fill_height=True,
        chatbot=chatbot,
    )

demo.launch(inbrowser=True)
