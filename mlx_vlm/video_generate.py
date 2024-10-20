import argparse
import sys
import time

import mlx.core as mx
from qwen_vl_utils import process_vision_info

from mlx_vlm import load
from mlx_vlm.utils import generate_step

# This is a beta version of the video generation script.
# It is not fully tested and may not work as expected.

AVAILABLE_MODELS = [
    "mlx-community/Qwen2-VL-2B-Instruct-8bit",
    "mlx-community/Qwen2-VL-2B-Instruct-4bit",
    "mlx-community/Qwen2-VL-2B-Instruct-bf16",
    "mlx-community/Qwen2-VL-7B-Instruct-8bit",
    "mlx-community/Qwen2-VL-7B-Instruct-4bit",
    "mlx-community/Qwen2-VL-7B-Instruct-bf16",
    "mlx-community/Qwen2-VL-72B-Instruct-8bit",
    "mlx-community/Qwen2-VL-72B-Instruct-4bit",
]


def generate(
    model,
    processor,
    input_ids,
    pixel_values,
    mask,
    temp=0.1,
    max_tokens=100,
    verbose=True,
    **kwargs,
):
    # Initialize timing and detokenizer
    tic = time.perf_counter()

    tokenizer = processor.tokenizer
    detokenizer = processor.detokenizer
    detokenizer.reset()

    token_count = 0
    prompt_time = 0
    for (token, _), n in zip(
        generate_step(input_ids, model, pixel_values, mask, temp=temp, **kwargs),
        range(max_tokens),
    ):
        if n == 0:
            prompt_time = time.perf_counter() - tic
            tic = time.perf_counter()
        if token == tokenizer.eos_token_id:
            break
        detokenizer.add_token(token)
        if verbose:
            print(detokenizer.last_segment, end="", flush=True)
        token_count += 1

    detokenizer.finalize()

    if verbose:
        print(detokenizer.last_segment, flush=True)
        gen_time = time.perf_counter() - tic
        print("=" * 10)
        if token_count == 0:
            print("No tokens generated for this prompt")
            return
        prompt_tps = input_ids.size / prompt_time
        gen_tps = (token_count - 1) / gen_time

        print(f"Prompt: {prompt_tps:.3f} tokens-per-sec")
        print(f"Generation: {gen_tps:.3f} tokens-per-sec")

    return detokenizer.text


def validate_model(model):
    if model not in AVAILABLE_MODELS:
        print(f"Error: The model '{model}' is not in the list of available models.")
        print("Available models are:")
        for m in AVAILABLE_MODELS:
            print(f"  - {m}")
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(description="Video Description CLI")
    parser.add_argument("--video", required=True, help="Path to the video file")
    parser.add_argument(
        "--max-pixels",
        type=int,
        nargs=2,
        default=224 * 224,
        help="Maximum number of pixels",
    )
    parser.add_argument("--fps", type=float, default=1.0, help="Frames per second")
    parser.add_argument(
        "--prompt", default="Describe this video.", help="Text prompt for the model"
    )
    parser.add_argument(
        "--temp", type=float, default=0.5, help="Temperature for generation"
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=100,
        help="Maximum number of tokens to generate",
    )
    parser.add_argument(
        "--model", default=AVAILABLE_MODELS[0], help="Select the model to use"
    )
    parser.add_argument("--verbose", action="store_false", help="Print verbose output")

    args = parser.parse_args()

    # Validate the model
    validate_model(args.model)

    print(f"\033[32mLoading model:\033[0m {args.model}")
    model, processor = load(args.model)

    max_pixels = args.max_pixels[0] * args.max_pixels[1]

    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "video",
                    "video": args.video,
                    "max_pixels": max_pixels,
                    "fps": args.fps,
                },
                {"type": "text", "text": args.prompt},
            ],
        }
    ]

    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )

    if args.verbose:
        print("=" * 10)
        print("\033[32mVideo:\033[0m", args.video, "\n")
        print("\033[32mPrompt:\033[0m", text)

    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="np",
    )

    input_ids = mx.array(inputs["input_ids"])
    pixel_values = mx.array(inputs["pixel_values_videos"])
    mask = mx.array(inputs["attention_mask"])
    image_grid_thw = mx.array(inputs["video_grid_thw"])
    kwargs = {
        "image_grid_thw": image_grid_thw,
    }

    print("\033[32mGenerating video description...\033[0m")
    response = generate(
        model,
        processor,
        input_ids,
        pixel_values,
        mask,
        temp=args.temp,
        max_tokens=args.max_tokens,
        verbose=args.verbose,
        **kwargs,
    )

    if not args.verbose:
        print(response)


if __name__ == "__main__":
    main()
