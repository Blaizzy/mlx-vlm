import argparse
import codecs
import os
import sys
from typing import Dict, List

from rich import print as rprint
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.prompt import Prompt

from mlx_vlm import load
from mlx_vlm.generate import (
    DEFAULT_KV_GROUP_SIZE,
    DEFAULT_KV_QUANT_SCHEME,
    DEFAULT_MAX_TOKENS,
    DEFAULT_PREFILL_STEP_SIZE,
    DEFAULT_QUANTIZED_KV_START,
    DEFAULT_TEMPERATURE,
    DEFAULT_THINKING_END_TOKEN,
    DEFAULT_THINKING_START_TOKEN,
    PromptCacheState,
    stream_generate,
)
from mlx_vlm.prompt_utils import apply_chat_template
from mlx_vlm.utils import load_image
from mlx_vlm.vision_cache import VisionFeatureCache


class MLXVisionChat:
    def __init__(
        self,
        model_path: str = "mlx-community/idefics2-8b-chatty-4bit",
        temperature: float = 0.7,
        max_tokens: int = 1000,
        verbose: bool = False,
        **kwargs,
    ):
        self.console = Console()
        self.verbose = verbose
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.history: List[Dict] = []
        self.current_image = None
        self.current_image_path = None
        self.image_paths: List[str] = []
        self.vision_cache = VisionFeatureCache()
        self.prompt_cache_state = PromptCacheState()
        self.stream_kwargs = kwargs

        with self.console.status("[bold green]Loading model..."):
            self.model, self.processor = load(model_path)

        rprint("[bold green]Model loaded successfully![/bold green]")
        self.print_help()

    def print_help(self) -> None:
        """Print available commands."""
        help_text = """
[bold yellow]Available Commands:[/bold yellow]
• /image <path> - Load a new image for discussion
• /clear - Clear conversation history
• /help - Show this help message
• /exit - Exit the chat
• Any other input will be treated as a question or comment about the current image
        """
        rprint(Panel(help_text, title="Help", border_style="blue"))

    def process_image(self, image_path: str) -> bool:
        """Process an image and prepare it for the model. Returns True if successful."""
        try:
            if not os.path.exists(image_path):
                rprint(
                    f"[bold red]Error:[/bold red] Image file not found: {image_path}"
                )
                return False

            self.current_image = load_image(image_path)
            self.current_image_path = image_path
            if image_path not in self.image_paths:
                self.image_paths.append(image_path)
            rprint(f"[bold blue]Loaded image:[/bold blue] {image_path}")
            return True
        except Exception as e:
            rprint(f"[bold red]Error loading image:[/bold red] {str(e)}")
            return False

    def add_to_history(self, role: str, text: str) -> None:
        """Add a message to the conversation history."""
        content = [{"type": "text", "text": text}]
        self.history.append({"role": role, "content": content})

    def generate_response(self) -> str:
        """Generate a response from the model based on the conversation history."""
        chat_template_kwargs = {
            "enable_thinking": self.stream_kwargs.get("enable_thinking", False),
        }

        num_images = 1 if self.current_image_path else 0
        image = [self.current_image_path] if self.current_image_path else None

        prompt = apply_chat_template(
            self.processor,
            self.model.config,
            self.history,
            num_images=num_images,
            **chat_template_kwargs,
        )

        rprint("[bold green]Assistant:[/bold green]", end=" ", flush=True)

        text = ""
        for chunk in stream_generate(
            self.model,
            self.processor,
            prompt,
            image=image,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            vision_cache=self.vision_cache,
            prompt_cache_state=self.prompt_cache_state,
            **self.stream_kwargs,
        ):
            text += chunk.text
            if self.verbose:
                rprint(chunk.text, end="", flush=True)

        return text

    def handle_command(self, command: str, args: str) -> bool:
        """Handle special commands. Returns True if should continue chat, False if should exit."""
        if command == "/exit":
            rprint("[bold yellow]Goodbye![/bold yellow]")
            return False
        elif command == "/help":
            self.print_help()
        elif command == "/clear":
            self.history.clear()
            self.image_paths.clear()
            self.prompt_cache_state = PromptCacheState()
            rprint("[bold blue]Conversation history cleared.[/bold blue]")
        elif command == "/image":
            if not args:
                rprint("[bold red]Error:[/bold red] Please provide an image path")
                return True
            self.process_image(args.strip())
        else:
            rprint(f"[bold red]Unknown command:[/bold red] {command}")
        return True

    def chat_loop(self) -> None:
        """Main chat loop for interaction."""
        while True:
            try:
                user_input = Prompt.ask("\n[bold cyan]You[/bold cyan]").strip()

                # Handle commands
                if user_input.startswith("/"):
                    parts = user_input.split(maxsplit=1)
                    command = parts[0].lower()
                    args = parts[1] if len(parts) > 1 else ""
                    if not self.handle_command(command, args):
                        break
                    continue
                self.add_to_history("user", user_input)
                response = self.generate_response()

                if not self.verbose:
                    rprint(Panel(Markdown(response), border_style="green"))

                # Remove the eos token from the response
                response = response.replace("<end_of_utterance>", "")

                self.add_to_history("assistant", response)

            except KeyboardInterrupt:
                rprint(
                    "\n[bold yellow]Interrupted by user. Type /exit to quit.[/bold yellow]"
                )
                continue
            except Exception as e:
                rprint(f"[bold red]Error:[/bold red] {str(e)}")
                continue


def main():
    parser = argparse.ArgumentParser(description="MLX Vision Chat CLI")
    parser.add_argument(
        "--model",
        default="mlx-community/idefics2-8b-chatty-4bit",
        help="Path to the model or model identifier",
    )
    parser.add_argument("--verbose", action="store_false", help="Enable verbose output")
    parser.add_argument(
        "--temperature",
        type=float,
        default=DEFAULT_TEMPERATURE,
        help="Temperature for sampling.",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=DEFAULT_MAX_TOKENS,
        help="Maximum number of tokens to generate.",
    )
    parser.add_argument(
        "--resize-shape",
        type=int,
        nargs="+",
        default=None,
        help="Resize shape for the image.",
    )
    parser.add_argument(
        "--prefill-step-size",
        type=int,
        default=DEFAULT_PREFILL_STEP_SIZE,
        help="Number of tokens to process per prefill step.",
    )
    parser.add_argument(
        "--max-kv-size",
        type=int,
        default=None,
        help="Maximum KV size for the prompt cache.",
    )
    parser.add_argument(
        "--kv-bits",
        type=float,
        default=None,
        help="Number of bits to quantize the KV cache to.",
    )
    parser.add_argument(
        "--kv-group-size",
        type=int,
        default=DEFAULT_KV_GROUP_SIZE,
        help="Group size for uniform KV cache quantization.",
    )
    parser.add_argument(
        "--kv-quant-scheme",
        type=str,
        choices=("uniform", "turboquant"),
        default=DEFAULT_KV_QUANT_SCHEME,
        help="KV cache quantization backend.",
    )
    parser.add_argument(
        "--quantized-kv-start",
        type=int,
        default=DEFAULT_QUANTIZED_KV_START,
        help="Start index for the quantized KV cache.",
    )
    parser.add_argument(
        "--eos-tokens",
        type=str,
        nargs="+",
        default=None,
        help="EOS tokens to add to the tokenizer.",
    )
    parser.add_argument(
        "--skip-special-tokens",
        action="store_true",
        help="Skip special tokens in the detokenizer.",
    )
    parser.add_argument(
        "--enable-thinking",
        action="store_true",
        help="Enable thinking mode in the chat template.",
    )
    parser.add_argument(
        "--thinking-budget",
        type=int,
        default=None,
        help="Maximum number of thinking tokens before forcing end-of-thinking.",
    )
    parser.add_argument(
        "--thinking-start-token",
        type=str,
        default=DEFAULT_THINKING_START_TOKEN,
        help="Token that marks the start of a thinking block.",
    )
    parser.add_argument(
        "--thinking-end-token",
        type=str,
        default=DEFAULT_THINKING_END_TOKEN,
        help="Token that marks the end of a thinking block.",
    )

    args = parser.parse_args()

    # Build stream_generate kwargs matching generate.py's main()
    kwargs = {}

    if args.eos_tokens is not None:
        eos_tokens = []
        for token in args.eos_tokens:
            try:
                decoded_token = codecs.decode(token, "unicode_escape")
                eos_tokens.append(decoded_token)
            except (UnicodeDecodeError, UnicodeError):
                eos_tokens.append(token)
        kwargs["eos_tokens"] = eos_tokens

    if args.skip_special_tokens:
        kwargs["skip_special_tokens"] = args.skip_special_tokens

    # Thinking kwargs
    kwargs["enable_thinking"] = args.enable_thinking
    if args.thinking_budget is not None:
        kwargs["thinking_budget"] = args.thinking_budget
        kwargs["thinking_end_token"] = args.thinking_end_token
        if args.thinking_start_token is not None:
            kwargs["thinking_start_token"] = args.thinking_start_token

    # KV cache kwargs
    if args.max_kv_size is not None:
        kwargs["max_kv_size"] = args.max_kv_size
    if args.kv_bits is not None:
        kwargs["kv_bits"] = args.kv_bits
        kwargs["kv_group_size"] = args.kv_group_size
        kwargs["kv_quant_scheme"] = args.kv_quant_scheme
        kwargs["quantized_kv_start"] = args.quantized_kv_start

    if args.resize_shape is not None:
        kwargs["resize_shape"] = args.resize_shape
    if args.prefill_step_size is not None:
        kwargs["prefill_step_size"] = args.prefill_step_size

    try:
        chat = MLXVisionChat(
            model_path=args.model,
            temperature=args.temperature,
            max_tokens=args.max_tokens,
            verbose=args.verbose,
            **kwargs,
        )
        chat.chat_loop()
    except Exception as e:
        rprint(f"[bold red]Fatal error:[/bold red] {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()
