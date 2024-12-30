import argparse
import codecs

from .prompt_utils import apply_chat_template
from .utils import generate, get_model_path, load, load_config, load_image_processor

DEFAULT_MODEL_PATH = "mlx-community/nanoLLaVA-1.5-8bit"
DEFAULT_IMAGE = []
DEFAULT_PROMPT = "What are these?"
DEFAULT_MAX_TOKENS = 256
DEFAULT_TEMP = 0.5
DEFAULT_TOP_P = 1.0
DEFAULT_SEED = 0


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Generate text from an image using a model."
    )
    parser.add_argument(
        "--model",
        type=str,
        default=DEFAULT_MODEL_PATH,
        help="The path to the local model directory or Hugging Face repo.",
    )
    parser.add_argument(
        "--adapter-path",
        type=str,
        default=None,
        help="The path to the adapter weights.",
    )
    parser.add_argument(
        "--image",
        type=str,
        nargs="+",
        default=DEFAULT_IMAGE,
        help="URL or path of the image to process.",
    )
    parser.add_argument(
        "--resize-shape",
        type=int,
        nargs=2,
        default=None,
        help="Resize shape for the image.",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default=DEFAULT_PROMPT,
        help="Message to be processed by the model.",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=DEFAULT_MAX_TOKENS,
        help="Maximum number of tokens to generate.",
    )
    parser.add_argument(
        "--temp", type=float, default=DEFAULT_TEMP, help="Temperature for sampling."
    )
    parser.add_argument("--verbose", action="store_false", help="Detailed output.")
    return parser.parse_args()


def get_model_and_processors(model_path, adapter_path):
    model_path = get_model_path(model_path)
    config = load_config(model_path, trust_remote_code=True)
    model, processor = load(
        model_path, adapter_path=adapter_path, lazy=False, trust_remote_code=True
    )
    return model, processor, config


def main():
    args = parse_arguments()
    if isinstance(args.image, str):
        args.image = [args.image]

    model, processor, config = get_model_and_processors(args.model, args.adapter_path)

    prompt = codecs.decode(args.prompt, "unicode_escape")

    prompt = apply_chat_template(processor, config, prompt, num_images=len(args.image))

    kwargs = {}
    if args.resize_shape is not None:
        assert (
            len(args.resize_shape) == 2
        ), "Resize shape must be a tuple of two integers"
        kwargs["resize_shape"] = args.resize_shape

    output = generate(
        model,
        processor,
        prompt,
        image=args.image,
        temp=args.temp,
        max_tokens=args.max_tokens,
        verbose=args.verbose,
        **kwargs,
    )
    if not args.verbose:
        print(output)


if __name__ == "__main__":
    main()
