import argparse
import logging
from typing import List

import cv2
import mlx.core as mx
import numpy as np
from PIL import Image

from .generate import generate
from .utils import load, process_inputs_with_fallback

# This is a proof-of-concept script for video generation with SmolVLM2.

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler())


def extract_video_frames(video_path: str, max_frames: int = 50) -> List[Image.Image]:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = int(cap.get(cv2.CAP_PROP_FPS)) or 1

    # Sample at ~1fps
    frame_indices = list(range(0, total_frames, fps))

    if max_frames is not None and len(frame_indices) > max_frames:
        indices = np.linspace(0, len(frame_indices) - 1, max_frames, dtype=int)
        frame_indices = [frame_indices[i] for i in indices]

    frames = []
    for frame_idx in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if ret:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(Image.fromarray(frame))

    cap.release()
    if not frames:
        raise ValueError("No frames read from the video.")
    return frames


def main():
    parser = argparse.ArgumentParser(description="Video Description CLI")
    parser.add_argument(
        "--video", type=str, required=True, help="Path to the video file"
    )
    parser.add_argument(
        "--max-frames", type=int, default=50, help="Maximum number of frames"
    )
    parser.add_argument(
        "--prompt", default="Describe this video.", help="Text prompt for the model"
    )
    parser.add_argument("--system", type=str, required=False, help="System prompt")
    parser.add_argument(
        "--temperature", type=float, default=0.7, help="Temperature for generation"
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

    # Ensure processor has chat_template (may only be on tokenizer)
    if getattr(processor, "chat_template", None) is None and hasattr(
        processor, "tokenizer"
    ):
        processor.chat_template = getattr(processor.tokenizer, "chat_template", None)

    # Extract video frames as images
    frames = extract_video_frames(args.video, max_frames=args.max_frames)
    logger.info(f"Extracted {len(frames)} frames from video")

    # Build messages with image tokens for each frame
    image_tokens = [{"type": "image"} for _ in frames]
    messages = [
        {
            "role": "user",
            "content": [
                *image_tokens,
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
                    {"type": "text", "text": args.system},
                ],
            },
        )

    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )

    inputs = process_inputs_with_fallback(
        processor,
        images=frames,
        prompts=text,
        audio=None,
    )

    input_ids = mx.array(inputs["input_ids"])
    pixel_values = mx.array(inputs["pixel_values"])
    mask = mx.array(inputs["attention_mask"])

    logger.info("\033[32mGenerating response...\033[0m")

    kwargs = {}
    kwargs["input_ids"] = input_ids
    kwargs["pixel_values"] = pixel_values
    kwargs["mask"] = mask
    if "pixel_attention_mask" in inputs:
        kwargs["pixel_mask"] = mx.array(inputs["pixel_attention_mask"])
    kwargs["temperature"] = args.temperature
    kwargs["max_tokens"] = args.max_tokens

    # Pass through any extra processor outputs
    for key, value in inputs.items():
        if key not in [
            "input_ids",
            "pixel_values",
            "attention_mask",
            "pixel_attention_mask",
        ] and not isinstance(value, (str, list)):
            kwargs[key] = mx.array(value)

    response = generate(
        model,
        processor,
        prompt=text,
        verbose=args.verbose,
        **kwargs,
    )

    if not args.verbose:
        print(response)


if __name__ == "__main__":
    main()
