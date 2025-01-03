import argparse
import os
import sys
import time
from typing import Dict, List

import mlx.core as mx
from rich import print as rprint
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.prompt import Prompt

from mlx_vlm import load
from mlx_vlm.prompt_utils import get_message_json
from mlx_vlm.utils import generate_step, load_image


class MLXVisionChat:
    def __init__(
        self,
        model_path: str = "mlx-community/idefics2-8b-chatty-4bit",
        temperature: float = 0.7,
        max_tokens: int = 1000,
        verbose: bool = False,
    ):
        self.console = Console()
        self.verbose = verbose
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.history: List[Dict] = []
        self.current_image = None

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
        if self.current_image is None:
            return "Please load an image first using the /image command."

        messages = []
        for i, message in enumerate(self.history):
            skip_token = True
            if i == len(self.history) - 1 and message["role"] == "user":
                skip_token = False
            messages.append(
                get_message_json(
                    self.model.config.model_type,
                    message["content"][0]["text"],
                    role=message["role"],
                    skip_image_token=skip_token,
                )
            )

        text_prompt = self.processor.apply_chat_template(
            messages, add_generation_prompt=True
        )

        inputs = self.processor(
            text=[text_prompt],
            images=[self.current_image],
            padding=True,
            return_tensors="np",
        )

        pixel_values = mx.array(inputs["pixel_values"])
        input_ids = mx.array(inputs["input_ids"])
        mask = mx.array(inputs["attention_mask"])

        detokenizer = self.processor.detokenizer
        detokenizer.reset()

        tic = time.perf_counter()

        generator = generate_step(
            input_ids,
            self.model,
            pixel_values,
            mask,
            temp=self.temperature,
        )

        # Use print instead of rprint to avoid rich console's automatic newlines
        rprint("[bold green]Assistant:[/bold green]", end=" ", flush=True)
        for (token, prob), n in zip(generator, range(self.max_tokens)):
            if n == 0:
                prompt_time = time.perf_counter() - tic
                tic = time.perf_counter()

            if token == self.processor.tokenizer.eos_token_id and n > 0:
                break

            detokenizer.add_token(token)

            if self.verbose:
                rprint(detokenizer.last_segment, end="", flush=True)

        detokenizer.finalize()
        return detokenizer.text

    def handle_command(self, command: str, args: str) -> bool:
        """Handle special commands. Returns True if should continue chat, False if should exit."""
        if command == "/exit":
            rprint("[bold yellow]Goodbye![/bold yellow]")
            return False
        elif command == "/help":
            self.print_help()
        elif command == "/clear":
            self.history.clear()
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
                # Handle regular chat input
                if self.current_image is None:
                    rprint(
                        "[bold yellow]Please load an image first using the /image command[/bold yellow]"
                    )
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
        "--temperature", type=float, default=0.7, help="Temperature for the model"
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=1000,
        help="Maximum number of new tokens to generate",
    )

    args = parser.parse_args()

    try:
        chat = MLXVisionChat(
            model_path=args.model,
            temperature=args.temperature,
            max_tokens=args.max_tokens,
            verbose=args.verbose,
        )
        chat.chat_loop()
    except Exception as e:
        rprint(f"[bold red]Fatal error:[/bold red] {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()
