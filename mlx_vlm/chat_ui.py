import argparse
import threading

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
chatbot_height = "clamp(380px, calc(100vh - 400px), 820px)"

# Global flag for stopping generation
stop_generation = threading.Event()


def extract_image_from_message(message):
    """Extract image file path from various message formats."""
    # Handle dict with "files" key (standard multimodal input)
    if isinstance(message, dict):
        if "files" in message and message["files"]:
            img = message["files"][-1]
            # File might be a dict with "path" key or a string path
            if isinstance(img, dict) and "path" in img:
                return img["path"]
            elif isinstance(img, str):
                return img
        # Handle dict with "file" key (single file)
        if "file" in message and message["file"]:
            f = message["file"]
            if isinstance(f, dict) and "path" in f:
                return f["path"]
            elif isinstance(f, str):
                return f
    # Handle string path directly
    elif isinstance(message, str):
        return message if message else ""
    return ""


def extract_text_from_message(message):
    """Extract text content from various message formats."""
    if isinstance(message, str):
        return message
    if isinstance(message, dict):
        # Try "text" key first (standard multimodal input)
        if "text" in message:
            return message["text"] or ""
        # Try "content" key (chat message format)
        if "content" in message:
            content = message["content"]
            if isinstance(content, str):
                return content
            elif isinstance(content, list):
                # Extract text from list of content items
                text_parts = []
                for c in content:
                    if isinstance(c, str):
                        text_parts.append(c)
                    elif isinstance(c, dict) and c.get("type") == "text":
                        text_parts.append(c.get("text", ""))
                return " ".join(text_parts)
    return ""


def chat(
    message, history, temperature, max_tokens, top_p, repetition_penalty, system_prompt
):
    global stop_generation
    stop_generation.clear()

    image_file = extract_image_from_message(message)
    num_images = 1 if image_file else 0

    if config["model_type"] != "paligemma":
        chat_history = []

        # Add system prompt if provided
        if system_prompt and system_prompt.strip():
            chat_history.append({"role": "system", "content": system_prompt.strip()})

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

        chat_history.append(
            {"role": "user", "content": extract_text_from_message(message)}
        )

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
        messages = extract_text_from_message(message)

    response = ""
    last_chunk = None

    # Build generation kwargs
    gen_kwargs = {
        "max_tokens": max_tokens,
        "temperature": temperature,
    }

    # Only add optional params if they differ from defaults
    if top_p < 1.0:
        gen_kwargs["top_p"] = top_p
    if repetition_penalty != 1.0:
        gen_kwargs["repetition_penalty"] = repetition_penalty

    for chunk in stream_generate(
        model,
        processor,
        messages,
        image=image_file,
        **gen_kwargs,
    ):
        if stop_generation.is_set():
            response += "\n\n*[Generation stopped]*"
            yield response
            return

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


def stop_generating():
    """Set the stop flag to interrupt generation."""
    stop_generation.set()
    return gr.update(interactive=False)


def clear_chat():
    """Clear the chat history."""
    return [], None


# Create custom theme with dark mode support
theme = gr.themes.Soft(
    primary_hue="blue",
    secondary_hue="slate",
).set(
    body_background_fill="*neutral_50",
    body_background_fill_dark="*neutral_950",
    block_background_fill="*neutral_100",
    block_background_fill_dark="*neutral_900",
)

with gr.Blocks(fill_height=True, title=f"MLX-VLM Chat — {args.model}") as demo:
    gr.Markdown(f"## MLX-VLM Chat UI — {args.model}")

    # Main controls row
    with gr.Row():
        with gr.Column(scale=4):
            with gr.Accordion("⚙️ Generation Settings", open=False):
                with gr.Row():
                    temperature = gr.Slider(
                        minimum=0,
                        maximum=2,
                        step=0.05,
                        value=0.1,
                        label="Temperature",
                        info="Higher = more creative, lower = more focused",
                    )
                    max_tokens = gr.Slider(
                        minimum=128,
                        maximum=4096,
                        step=64,
                        value=1024,
                        label="Max Tokens",
                        info="Maximum length of response",
                    )
                with gr.Row():
                    top_p = gr.Slider(
                        minimum=0.1,
                        maximum=1.0,
                        step=0.05,
                        value=1.0,
                        label="Top-p (Nucleus Sampling)",
                        info="1.0 = disabled, lower = more focused",
                    )
                    repetition_penalty = gr.Slider(
                        minimum=1.0,
                        maximum=2.0,
                        step=0.05,
                        value=1.0,
                        label="Repetition Penalty",
                        info="1.0 = disabled, higher = less repetition",
                    )
                with gr.Row():
                    system_prompt = gr.Textbox(
                        label="System Prompt (optional)",
                        placeholder="You are a helpful assistant...",
                        lines=2,
                        max_lines=4,
                    )

        with gr.Column(scale=1, min_width=200):
            with gr.Row():
                stop_btn = gr.Button("⏹️ Stop", variant="stop", size="sm")
                clear_btn = gr.Button("🗑️ New Chat", variant="secondary", size="sm")

    # Chatbot component
    chatbot = gr.Chatbot(
        height=chatbot_height,
        scale=1,
        buttons=["copy", "copy_all"],
        placeholder="Upload an image and ask questions about it, or just chat!",
    )

    # Chat interface
    chat_interface = gr.ChatInterface(
        fn=chat,
        additional_inputs=[
            temperature,
            max_tokens,
            top_p,
            repetition_penalty,
            system_prompt,
        ],
        multimodal=True,
        fill_height=True,
        chatbot=chatbot,
    )

    # Connect buttons
    stop_btn.click(fn=stop_generating, outputs=[stop_btn])
    clear_btn.click(fn=clear_chat, outputs=[chatbot, chat_interface.textbox])


def main():
    demo.launch(inbrowser=True, theme=theme)


if __name__ == "__main__":
    main()
