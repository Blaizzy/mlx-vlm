import argparse
import gc
import json
import threading

import gradio as gr
import mlx.core as mx

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


# Global state for model
class ModelState:
    def __init__(self):
        self.model = None
        self.processor = None
        self.config = None
        self.image_processor = None
        self.current_model_name = None

    def load(self, model_name):
        """Load a model, clearing previous one from memory."""
        # Clear previous model from memory
        if self.model is not None:
            del self.model
            del self.processor
            del self.config
            del self.image_processor
            mx.metal.clear_cache()
            gc.collect()

        # Load new model
        self.config = load_config(model_name)
        self.model, self.processor = load(
            model_name, processor_kwargs={"trust_remote_code": True}
        )
        self.image_processor = load_image_processor(model_name)
        self.current_model_name = model_name


state = ModelState()

# Parse args and load initial model
args = parse_arguments()
state.load(args.model)

# Use most of the viewport for conversation
chatbot_height = "clamp(380px, calc(100vh - 450px), 820px)"

# Global flag for stopping generation
stop_generation = threading.Event()


def get_cached_vlm_models():
    """Scan HF cache for vision-capable models."""
    try:
        from huggingface_hub import scan_cache_dir

        vlm_models = []
        cache_info = scan_cache_dir()

        for repo in cache_info.repos:
            if repo.repo_type != "model":
                continue

            # Check for refs
            refs = getattr(repo, "refs", {})
            if not refs or "main" not in refs:
                # Try revisions instead
                revisions = getattr(repo, "revisions", None)
                if revisions:
                    for rev in revisions:
                        snapshot_path = getattr(rev, "snapshot_path", None)
                        if snapshot_path:
                            config_path = snapshot_path / "config.json"
                            if config_path.exists():
                                try:
                                    with open(config_path, "r") as f:
                                        config = json.load(f)
                                    if "vision_config" in config:
                                        vlm_models.append(repo.repo_id)
                                        break
                                except Exception:
                                    pass
                continue

            # Check config.json for vision_config
            main_ref = refs["main"]
            snapshot_path = getattr(main_ref, "snapshot_path", None)
            if snapshot_path:
                config_path = snapshot_path / "config.json"
                if config_path.exists():
                    try:
                        with open(config_path, "r") as f:
                            config = json.load(f)
                        if "vision_config" in config:
                            vlm_models.append(repo.repo_id)
                    except Exception:
                        pass

        # Ensure current model is in the list
        if state.current_model_name and state.current_model_name not in vlm_models:
            vlm_models.insert(0, state.current_model_name)

        return sorted(set(vlm_models))
    except Exception as e:
        print(f"Error scanning cache: {e}")
        # Return at least the current model
        return [state.current_model_name] if state.current_model_name else []


def load_model_by_name(model_name, progress=gr.Progress()):
    """Load a model and return status."""
    if not model_name:
        return "✓ Loaded", gr.update()

    if model_name == state.current_model_name:
        return "✓ Loaded", gr.update()

    try:
        progress(0.1, desc="Clearing memory...")
        progress(0.3, desc="Loading...")
        state.load(model_name)
        progress(1.0, desc="Done!")

        return "✓ Loaded", gr.update(value=[])
    except Exception as e:
        error_msg = str(e)
        # Truncate error for display
        short_err = error_msg[:60] + "..." if len(error_msg) > 60 else error_msg
        return f"⚠ {short_err}", gr.update()


def refresh_model_list():
    """Refresh the list of cached models."""
    models = get_cached_vlm_models()
    return gr.update(choices=models, value=state.current_model_name)


def extract_image_from_message(message):
    """Extract image file path from various message formats."""
    if isinstance(message, dict):
        if "files" in message and message["files"]:
            img = message["files"][-1]
            if isinstance(img, dict) and "path" in img:
                return img["path"]
            elif isinstance(img, str):
                return img
        if "file" in message and message["file"]:
            f = message["file"]
            if isinstance(f, dict) and "path" in f:
                return f["path"]
            elif isinstance(f, str):
                return f
    elif isinstance(message, str):
        return message if message else ""
    return ""


def extract_text_from_message(message):
    """Extract text content from various message formats."""
    if isinstance(message, str):
        return message
    if isinstance(message, dict):
        if "text" in message:
            return message["text"] or ""
        if "content" in message:
            content = message["content"]
            if isinstance(content, str):
                return content
            elif isinstance(content, list):
                text_parts = []
                for c in content:
                    if isinstance(c, str):
                        text_parts.append(c)
                    elif isinstance(c, dict) and c.get("type") == "text":
                        text_parts.append(c.get("text", ""))
                return " ".join(text_parts)
    return ""


def chat(
    message,
    history,
    temperature,
    max_tokens,
    top_p,
    repetition_penalty,
    system_prompt,
):
    global stop_generation
    stop_generation.clear()

    image_file = extract_image_from_message(message)
    num_images = 1 if image_file else 0

    if state.config["model_type"] != "paligemma":
        chat_history = []

        if system_prompt and system_prompt.strip():
            chat_history.append({"role": "system", "content": system_prompt.strip()})

        for item in history:
            if isinstance(item, dict):
                role = item.get("role", "user")
                content = item.get("content", "")
                if isinstance(content, str):
                    pass
                elif isinstance(content, dict) and "text" in content:
                    content = content["text"]
                elif isinstance(content, list):
                    text_parts = []
                    for c in content:
                        if isinstance(c, str):
                            text_parts.append(c)
                        elif isinstance(c, dict) and c.get("type") == "text":
                            text_parts.append(c.get("text", ""))
                    content = " ".join(text_parts) if text_parts else ""
                else:
                    content = ""
                if role == "assistant" and isinstance(content, str) and content:
                    content = content.split("\n\n---\n")[0]
                if content:
                    chat_history.append({"role": role, "content": content})
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
                    state.config["model_type"],
                    m["content"],
                    role=m["role"],
                    skip_image_token=skip_token,
                    num_images=num_images if not skip_token else 0,
                )
            )

        messages = get_chat_template(
            state.processor, messages, add_generation_prompt=True
        )

    else:
        messages = extract_text_from_message(message)

    response = ""
    last_chunk = None

    gen_kwargs = {
        "max_tokens": max_tokens,
        "temperature": temperature,
    }

    if top_p < 1.0:
        gen_kwargs["top_p"] = top_p
    if repetition_penalty != 1.0:
        gen_kwargs["repetition_penalty"] = repetition_penalty

    for chunk in stream_generate(
        state.model,
        state.processor,
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

# Get initial model list
initial_models = get_cached_vlm_models()

# JavaScript to toggle dark mode and set dark as default
dark_mode_js = """
() => {
    // Always set dark mode on load unless user explicitly chose light
    const savedTheme = localStorage.getItem('theme');
    const isDark = savedTheme !== 'light';
    document.body.classList.toggle('dark', isDark);
    return isDark ? '☀️' : '🌙';
}
"""

toggle_dark_js = """
() => {
    const isDark = document.body.classList.toggle('dark');
    localStorage.setItem('theme', isDark ? 'dark' : 'light');
    return isDark ? '☀️' : '🌙';
}
"""

# JavaScript to persist and restore selected model
save_model_js = """
(model_name) => {
    if (model_name) {
        localStorage.setItem('mlx_vlm_model', model_name);
    }
    return model_name;
}
"""

load_model_js = """
(server_model) => {
    const savedModel = localStorage.getItem('mlx_vlm_model');
    // Return saved model if available, otherwise use server's current model
    return savedModel || server_model;
}
"""

with gr.Blocks(fill_height=True, title="MLX-VLM Chat") as demo:
    gr.Markdown("## MLX-VLM Chat UI")

    # Model selector row
    with gr.Row():
        with gr.Column(scale=5):
            model_dropdown = gr.Dropdown(
                label="Model",
                choices=initial_models,
                value=state.current_model_name,
                show_label=True,
                allow_custom_value=True,
            )
        with gr.Column(scale=0):
            refresh_btn = gr.Button("🔄", size="sm", min_width=20, scale=0)
            theme_btn = gr.Button("☀️", size="sm", min_width=20, scale=0)
        with gr.Column(scale=5):
            model_status = gr.Textbox(
                value="✓ Loaded",
                label="Status",
                interactive=False,
            )

    # Main controls row
    with gr.Row():
        with gr.Column(scale=6):
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
            stop_btn = gr.Button("⏹️ Stop", variant="stop", size="sm")

    # Chatbot component
    chatbot = gr.Chatbot(
        height=chatbot_height,
        scale=1,
        buttons=["copy", "copy_all"],
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
        save_history=True,
    )

    # Connect model selector
    model_dropdown.change(
        fn=load_model_by_name,
        inputs=[model_dropdown],
        outputs=[model_status, chatbot],
    ).then(
        fn=None,
        inputs=[model_dropdown],
        js=save_model_js,
    )
    refresh_btn.click(
        fn=refresh_model_list,
        outputs=[model_dropdown],
    )

    # Connect theme toggle
    theme_btn.click(fn=None, js=toggle_dark_js, outputs=[theme_btn])

    # On page load: restore theme and model from localStorage
    demo.load(fn=None, js=dark_mode_js, outputs=[theme_btn])
    demo.load(
        fn=lambda: state.current_model_name,
        inputs=[],
        outputs=[model_dropdown],
        js=load_model_js,
    )

    # Connect control buttons
    stop_btn.click(fn=stop_generating, outputs=[stop_btn])


def main():
    demo.launch(inbrowser=True, theme=theme)


if __name__ == "__main__":
    main()
