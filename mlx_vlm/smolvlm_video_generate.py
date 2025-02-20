import argparse
import logging

import mlx.core as mx

from .utils import generate, load

# This is a proof-of-concept script for video generation with SmolVLM2.

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler())


def main():
    parser = argparse.ArgumentParser(description="Video Description CLI")
    parser.add_argument(
        "--video", type=str, required=True, help="Path to the video file"
    )
    parser.add_argument(
        "--max-frames", type=int, default=None, help="Maximum number of frames"
    )
    parser.add_argument(
        "--prompt", default="Describe this video.", help="Text prompt for the model"
    )
    parser.add_argument("--system", type=str, required=False, help="System prompt")
    parser.add_argument(
        "--temp", type=float, default=0.7, help="Temperature for generation"
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=100,
        help="Maximum number of tokens to generate",
    )
    parser.add_argument(
        "--model",
        default="HuggingFaceTB/SmolVLM2-500M-Video-Instruct-mlx",
        help="Select the model to use",
    )
    parser.add_argument("--verbose", action="store_false", help="Print verbose output")

    args = parser.parse_args()

    print(f"\033[32mLoading model:\033[0m {args.model}")
    model, processor = load(args.model)

    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "video",
                    "path": args.video,
                },
                {"type": "text", "text": args.prompt},
            ],
        }
    ]
    if args.system:
        messages.insert(
            0,
            {
                "role": "system",
                "content": [
                    {"type": "text", "text": args.video},
                ],
            },
        )

    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="np",
    )

    input_ids = mx.array(inputs["input_ids"])
    pixel_values = mx.array(inputs["pixel_values"][0])
    pixel_values = mx.expand_dims(pixel_values, 0)
    mask = mx.array(inputs["attention_mask"])
    pixel_mask = mx.array(inputs["pixel_attention_mask"])

    logger.info("\033[32mGenerating response...\033[0m")

    kwargs = {}
    kwargs["input_ids"] = input_ids
    kwargs["pixel_values"] = pixel_values
    kwargs["mask"] = mask
    kwargs["pixel_mask"] = pixel_mask
    kwargs["temp"] = args.temp
    kwargs["max_tokens"] = args.max_tokens

    response = generate(
        model,
        processor,
        prompt="",
        verbose=args.verbose,
        **kwargs,
    )

    if not args.verbose:
        print(response)


if __name__ == "__main__":
    main()
