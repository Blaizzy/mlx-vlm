import argparse
import json
import platform
import subprocess
import sys
import textwrap
import time

import mlx.core as mx
import psutil
from rich.console import Console
from rich.panel import Panel
from tqdm import tqdm
from transformers import __version__ as transformers_version

from mlx_vlm import generate, load
from mlx_vlm.prompt_utils import apply_chat_template
from mlx_vlm.utils import load_config
from mlx_vlm.version import __version__

# Initialize console
console = Console()


def parse_args():
    parser = argparse.ArgumentParser(description="Test MLX-VLM models")
    parser.add_argument(
        "--models-file",
        type=str,
        required=True,
        help="Path to file containing model paths, one per line",
    )
    parser.add_argument(
        "--image", type=str, nargs="+", required=True, help="Path(s) to test image(s)"
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="Describe this image.",
        help="Vision-language prompt to test",
    )
    parser.add_argument(
        "--language-only-prompt",
        type=str,
        default="Hi, how are you?",
        help="Language-only prompt to test",
    )
    parser.add_argument(
        "--temperature", type=float, default=0.7, help="Sampling temperature"
    )
    parser.add_argument(
        "--max-tokens", type=int, default=100, help="Maximum tokens to generate"
    )
    parser.add_argument("--resize-shape", type=int, default=None, help="Resize shape")
    return parser.parse_args()


def get_device_info():
    # Disable tokenizers parallelism to avoid deadlocks after forking
    import os

    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    try:
        data = subprocess.check_output(
            ["system_profiler", "SPDisplaysDataType", "-json"], text=True
        )
        device_info = json.loads(data)
        return device_info
    except Exception as e:
        print(f"Could not retrieve GPU information: {e}")
        return None


def test_model_loading(model_path):
    try:
        console.print("[bold green]Loading model...")
        start_time = time.time()
        model, processor = load(model_path, trust_remote_code=True)
        config = load_config(model_path, trust_remote_code=True)
        end_time = time.time()
        console.print(
            f"[bold green]✓[/] Model loaded successfully in {end_time - start_time:.2f} seconds"
        )
        return model, processor, config, False
    except Exception as e:
        console.print(f"[bold red]✗[/] Failed to load model: {str(e)}")
        return None, None, None, True


def test_generation(
    model, processor, config, model_path, test_inputs, vision_language=True
):
    try:
        test_type = "vision-language" if vision_language else "language-only"
        console.print(f"[bold yellow]Testing {test_type} generation...")

        prompt = (
            test_inputs["prompt"]
            if vision_language
            else test_inputs["language_only_prompt"]
        )
        num_images = len(test_inputs["image"]) if vision_language else 0

        formatted_prompt = apply_chat_template(
            processor, config, prompt, num_images=num_images
        )

        generate_args = {
            "model": model,
            "processor": processor,
            "prompt": formatted_prompt,
            "verbose": True,
            **test_inputs["kwargs"],
        }
        if vision_language:
            generate_args["image"] = test_inputs["image"]

        output = generate(**generate_args)

        # Deepseek-vl2-tiny and ShowUI outputs are empty on VLM generation
        # Paligemma outputs are empty on language-only generation
        # So we skip the assertion for these models
        if (
            not any(x in model_path for x in ["deepseek-vl2-tiny", "ShowUI"])
            and vision_language
            or ("paligemma" not in model_path and not vision_language)
        ):
            assert isinstance(output, str) and len(output) > 0

        console.print(f"[bold green]✓[/] {test_type} generation successful")
        return False
    except Exception as e:
        console.print(f"[bold red]✗[/] {test_type} generation failed: {str(e)}")
        return True


def main():
    args = parse_args()

    # Load models list
    with open(args.models_file, "r", encoding="utf-8") as f:
        models = [line.strip() for line in f.readlines()]

    # Test inputs dictionary
    test_inputs = {
        "image": args.image,
        "prompt": args.prompt,
        "language_only_prompt": args.language_only_prompt,
        "kwargs": {
            "temp": args.temperature,
            "max_tokens": args.max_tokens,
            "resize_shape": (
                (args.resize_shape, args.resize_shape) if args.resize_shape else None
            ),
        },
    }

    results = []

    for model_path in tqdm(models):
        console.print(Panel(f"Testing {model_path}", style="bold blue"))

        # Run tests
        model, processor, config, error = test_model_loading(model_path)

        if not error and model:
            print("\n")
            # Test vision-language generation
            error |= test_generation(
                model, processor, config, model_path, test_inputs, vision_language=True
            )

            print("\n")

            # Clear cache and reset peak memory for next test
            mx.metal.clear_cache()
            mx.metal.reset_peak_memory()

            # Test language-only generation
            error |= test_generation(
                model, processor, config, model_path, test_inputs, vision_language=False
            )
            print("\n")

        console.print("[bold blue]Cleaning up...")
        del model, processor
        mx.metal.clear_cache()
        mx.metal.reset_peak_memory()
        console.print("[bold green]✓[/] Cleanup complete\n")
        results.append(
            f"[bold {'green' if not error else 'red'}]{'✓' if not error else '✗'}[/] {model_path}"
        )

    print("\n")
    success = all(result.startswith("[bold green]") for result in results)
    panel_style = "bold green" if success else "bold red"
    console.print(Panel("\n".join(results), title="Results", style=panel_style))
    console.print(
        f"[bold {'green' if success else 'red'}]{'All' if success else 'Some'} models tested {'successfully' if success else 'failed to test'}"
    )

    print("\n")
    device_info = get_device_info()
    console.print(
        Panel(
            title="System Information",
            renderable=textwrap.dedent(
                f"""{platform.machine() == 'arm64' and f'''
            MAC OS:       v{platform.mac_ver()[0]}
            Python:       v{sys.version.split()[0]}
            MLX:          v{mx.__version__}
            MLX-VLM:      v{__version__}
            Transformers: v{transformers_version}

            Hardware:
            • Chip:       {device_info['SPDisplaysDataType'][0]['_name']}
            • RAM:        {psutil.virtual_memory().total / (1024 ** 3):.1f} GB
            • CPU Cores:  {psutil.cpu_count(logical=False)}
            • GPU Cores:  {device_info['SPDisplaysDataType'][0]['sppci_cores']}
            ''' or 'Not running on Apple Silicon'}"""
            ),
            style="bold blue",
        )
    )


if __name__ == "__main__":
    main()
