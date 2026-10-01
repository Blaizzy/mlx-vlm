"""Run extraction models from the command line."""

import argparse
import ast
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


def _as_array(value):
    """Return value as a numpy array, or None if it is not array-shaped."""
    try:
        array = np.asarray(value)
    except Exception:
        return None
    return None if array.dtype == object else array


def _manifest(task, outputs):
    """Describe named outputs without materializing them into JSON."""
    described = {}
    for name, value in outputs.items():
        array = _as_array(value)
        if array is None:
            described[name] = {"type": type(value).__name__, "array": False}
        else:
            described[name] = {"shape": list(array.shape), "dtype": str(array.dtype)}
    return {"task": task, "outputs": described}


def _parse_settings(pairs, parser):
    """Turn repeated NAME=VALUE arguments into keyword arguments."""
    settings = {}
    for item in pairs or []:
        name, separator, raw = item.partition("=")
        name = name.strip()
        if not separator or not name:
            parser.error(f"--set expects NAME=VALUE, got {item!r}")
        if name in settings:
            parser.error(f"--set {name} was given more than once")
        lowered = raw.strip().lower()
        if lowered in ("true", "false"):
            settings[name] = lowered == "true"
        elif lowered in ("none", "null"):
            settings[name] = None
        else:
            try:
                settings[name] = ast.literal_eval(raw)
            except (ValueError, SyntaxError):
                settings[name] = raw
    return settings


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
        "--set",
        action="append",
        metavar="NAME=VALUE",
        help="Extra keyword for the model, repeatable (--set score_threshold=0.5)",
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
        extra = _parse_settings(args.set, parser)
        if args.prompt is not None:
            extra["text_prompt"] = args.prompt
        outputs = extract(model, processor, inputs, task=args.task, **extra)
    except ValueError as error:
        parser.error(str(error))
    except TypeError as error:
        # A mistyped --set name reaches the predictor as an unknown keyword.
        if "unexpected keyword argument" not in str(error):
            raise
        parser.error(f"{error}; this model does not take that --set name")

    task = args.task if args.task is not None else tasks[0]
    if args.output is not None:
        # Object arrays save but cannot be read back with the default np.load,
        # so only array-shaped outputs are written.
        arrays = {k: a for k, v in outputs.items() if (a := _as_array(v)) is not None}
        skipped = sorted(set(outputs) - set(arrays))
        args.output.parent.mkdir(parents=True, exist_ok=True)
        np.savez(args.output, **arrays)
        if skipped:
            print(f"not array-shaped, omitted from {args.output.name}: {skipped}")
    print(json.dumps(_manifest(task, outputs), indent=2))


if __name__ == "__main__":
    main()
