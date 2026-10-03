"""Run extraction models from the command line."""

import argparse
import ast
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps

from .extraction import METADATA, describe_outputs, extract
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


def _image_suffixes():
    """Every file extension the installed Pillow can open."""
    Image.init()
    return {
        ext for ext, fmt in Image.registered_extensions().items() if fmt in Image.OPEN
    }


def _load_input_file(path, parser):
    """Read an array-valued model input from a file."""
    path = Path(path)
    if not path.exists():
        parser.error(f"--set-file path does not exist: {path}")
    suffix = path.suffix.lower()
    if suffix == ".npy":
        return np.load(path, allow_pickle=False)
    if suffix == ".npz":
        with np.load(path, allow_pickle=False) as bundle:
            names = list(bundle.keys())
            if len(names) != 1:
                parser.error(
                    f"{path.name} holds {names}; point --set-file at a .npy instead"
                )
            return bundle[names[0]]
    if suffix == ".json":
        return json.loads(path.read_text())
    if suffix in _image_suffixes():
        # Keep the file's own mode: load_image forces RGB, which turns a
        # grayscale mask into three channels and drops an RGBA cutout's alpha.
        with Image.open(path) as image:
            return np.asarray(ImageOps.exif_transpose(image))
    parser.error(f"unsupported --set-file type {suffix or path.name!r}")


def _parse_input_files(pairs, parser):
    """Turn repeated NAME=PATH arguments into array keyword arguments."""
    loaded = {}
    for item in pairs or []:
        name, separator, raw = item.partition("=")
        name = name.strip()
        if not separator or not name or not raw:
            parser.error(f"--set-file expects NAME=PATH, got {item!r}")
        if name in loaded:
            parser.error(f"--set-file {name} was given more than once")
        loaded[name] = _load_input_file(raw.strip(), parser)
    return loaded


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


def _manifest(task, outputs):
    """Describe named outputs without materializing them into JSON."""
    return {"task": task, **describe_outputs(outputs)}


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
        "--set-file",
        action="append",
        metavar="NAME=PATH",
        help="Array keyword read from a file, repeatable (--set-file mask=m.png)",
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
        for name, value in _parse_input_files(args.set_file, parser).items():
            if name in extra:
                parser.error(f"{name} given by both --set and --set-file")
            extra[name] = value
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
        args.output.parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            args.output,
            **{k: np.asarray(v) for k, v in outputs.items() if k != METADATA},
        )

    print(json.dumps(_manifest(task, outputs), indent=2))


if __name__ == "__main__":
    main()
