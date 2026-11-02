"""Image-to-3D generation through the shared generation CLI and Python API."""

import json
from pathlib import Path

from ..utils import get_model_path


def load_3d_model(model, *, revision=None, force_download=False, **kwargs):
    """Load a supported image-to-3D model from a local path or Hugging Face."""
    path = get_model_path(
        str(Path(model).expanduser()),
        revision=revision,
        force_download=force_download,
        allow_patterns=["config.json"],
        local_files_only=kwargs.get("local_files_only", False),
    )
    config = json.loads((path / "config.json").read_text())
    if config.get("denoiser", {}).get("name") != "LatentSeqMMFlowModel":
        raise ValueError(f"Unsupported 3D generation model: {model}")

    from ..models.idol import IdolPipeline

    return IdolPipeline.from_pretrained(
        model, revision=revision, force_download=force_download, **kwargs
    )


def generate_3d(model, image, *, output_path=None, guidance=None, **kwargs):
    """Generate one object from an image or image list and optionally save it."""
    if guidance is not None:
        kwargs["cfg_strength"] = guidance
    result = model(image, **kwargs)
    if output_path is not None:
        result.save(output_path)
    return result


def run_3d_generation_cli(args):
    images = args.image or []
    if not images:
        raise ValueError("3D generation requires at least one --image")
    incompatible = [
        f"--{name.replace('_', '-')}"
        for name in (
            "audio",
            "video",
            "chat",
            "adapter_path",
            "draft_model",
            "size",
            "processor_kwargs",
        )
        if getattr(args, name, None)
    ]
    if args.task != "generate":
        incompatible.append("--task edit")
    if incompatible:
        raise ValueError(f"3D generation does not support: {', '.join(incompatible)}")

    kwargs = dict(args.gen_kwargs)
    for name in ("seed", "steps", "guidance"):
        value = getattr(args, name)
        if value is not None:
            kwargs[name] = value
    kwargs.setdefault("verbose", args.verbose)
    output = (
        Path(args.output).expanduser()
        if args.output
        else (Path("outputs") / f"object-{kwargs.get('seed', 42)}.ply")
    )
    if output.suffix.lower() != ".ply":
        raise ValueError("3D generation requires a .ply --output path")
    model = load_3d_model(
        args.model, revision=args.revision, force_download=args.force_download
    )
    image = images[0] if len(images) == 1 else images
    result = generate_3d(model, image, output_path=output, **kwargs)
    print(f"Saved {result.path}")
