"""Run extraction models from the command line."""

import argparse
import json
from pathlib import Path

import numpy as np

from .extraction import extract
from .utils import load, load_image


def _stack_images(paths):
    """Load image paths into a single (H, W, 3) or (T, H, W, 3) uint8 array."""
    frames = [np.asarray(load_image(str(path)).convert("RGB")) for path in paths]
    shapes = {frame.shape for frame in frames}
    if len(shapes) > 1:
        raise ValueError(f"Images must share a shape, got {sorted(shapes)}")
    return frames[0] if len(frames) == 1 else np.stack(frames)


def _read_video(path, max_frames):
    """Read an (T, H, W, 3) uint8 array from a video file."""
    from .models.video_depth_anything.generate import read_video_frames

    frames = read_video_frames(str(path), max_len=max_frames)
    return frames[0] if isinstance(frames, tuple) else frames


def _manifest(task, outputs):
    """Describe named outputs without materializing them into JSON."""
    described = {}
    for name, value in outputs.items():
        array = np.asarray(value)
        described[name] = {"shape": list(array.shape), "dtype": str(array.dtype)}
    return {"task": task, "outputs": described}


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Predict structured outputs with MLX-VLM extraction models"
    )
    parser.add_argument("--model", required=True, help="Model path or Hugging Face ID")
    source = parser.add_mutually_exclusive_group()
    source.add_argument(
        "--image", type=Path, action="append", help="Image path, repeatable for frames"
    )
    source.add_argument("--video", type=Path, help="Video path (requires cv2)")
    parser.add_argument(
        "--task", default=None, help="Task to request when a model serves several"
    )
    parser.add_argument(
        "--prompt", default=None, help="Text prompt, for models that detect by concept"
    )
    parser.add_argument(
        "--list-tasks", action="store_true", help="Print the model's tasks and exit"
    )
    parser.add_argument(
        "--max-frames", type=int, default=-1, help="Cap frames read from --video"
    )
    parser.add_argument(
        "--output", type=Path, default=None, help="Write named arrays to this .npz"
    )
    args = parser.parse_args(argv)

    if not args.list_tasks and args.image is None and args.video is None:
        parser.error("one of --image or --video is required")

    model, processor = load(args.model)
    tasks = getattr(model, "extraction_types", ())
    if not tasks:
        parser.error(f"{args.model} does not support extraction prediction")
    if args.list_tasks:
        print(json.dumps({"tasks": list(tasks)}, indent=2))
        return
    try:
        inputs = (
            _read_video(args.video, args.max_frames)
            if args.video is not None
            else _stack_images(args.image)
        )
    except (OSError, ValueError, ImportError) as error:
        parser.error(str(error))

    try:
        extra = {"text_prompt": args.prompt} if args.prompt is not None else {}
        outputs = extract(model, processor, inputs, task=args.task, **extra)
    except ValueError as error:
        parser.error(str(error))

    task = args.task if args.task is not None else tasks[0]
    if args.output is not None:
        np.savez(args.output, **{k: np.asarray(v) for k, v in outputs.items()})
    print(json.dumps(_manifest(task, outputs), indent=2))


if __name__ == "__main__":
    main()
