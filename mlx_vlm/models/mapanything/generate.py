"""MapAnything command line: images -> metric point cloud and per-view geometry.

Usage:
    python -m mlx_vlm.models.mapanything.generate \\
        --model facebook/map-anything --images path/to/folder --output out

``--model`` takes an MLX model directory or Hub repo, or an official
``facebook/map-anything*`` repo, converted in memory. Writes
``predictions.npz`` (every output, per view) and ``points.ply`` (the masked
points colored by the input images).
"""

import argparse
import json
import os
import time
from pathlib import Path
from typing import Dict, List

import mlx.core as mx
import numpy as np

from ...utils import get_model_path
from .config import ModelConfig
from .convert import load_official
from .mapanything import Model
from .processing_mapanything import MapAnythingProcessor


def load(model: str, dtype: str = "float16"):
    """(model, processor) from an MLX model directory or repo, or from an
    official MapAnything repo converted in memory (trunk in ``dtype``)."""
    path = get_model_path(model)
    config = json.loads((path / "config.json").read_text())
    if "model_type" in config:
        from ... import load as load_mlx

        return load_mlx(str(path))
    config, weights = load_official(path, dtype)
    net = Model(ModelConfig.from_dict(config))
    net.load_weights(list(net.sanitize(weights).items()))
    return net, MapAnythingProcessor()


def write_ply(path: Path, points: np.ndarray, colors: np.ndarray) -> None:
    """Binary little-endian PLY with float32 xyz and uint8 rgb."""
    vertex = np.empty(
        len(points),
        dtype=[
            ("x", "<f4"),
            ("y", "<f4"),
            ("z", "<f4"),
            ("r", "u1"),
            ("g", "u1"),
            ("b", "u1"),
        ],
    )
    for i, axis in enumerate("xyz"):
        vertex[axis] = points[:, i]
    for i, channel in enumerate("rgb"):
        vertex[channel] = colors[:, i]
    header = (
        "ply\nformat binary_little_endian 1.0\n"
        f"element vertex {len(points)}\n"
        "property float x\nproperty float y\nproperty float z\n"
        "property uchar red\nproperty uchar green\nproperty uchar blue\n"
        "end_header\n"
    )
    with open(path, "wb") as f:
        f.write(header.encode())
        f.write(vertex.tobytes())


def point_cloud(predictions: List[Dict[str, mx.array]]):
    """Masked world points (N, 3) and uint8 colors (N, 3) of all views."""
    points, colors = [], []
    for pred in predictions:
        pts = np.array(pred["pts3d"]).reshape(-1, 3)
        rgb = np.array(pred["img_no_norm"]).reshape(-1, 3)
        keep = np.array(pred["mask"]).reshape(-1) if "mask" in pred else slice(None)
        points.append(pts[keep])
        colors.append((rgb[keep] * 255).round().astype(np.uint8))
    return np.concatenate(points), np.concatenate(colors)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="facebook/map-anything")
    parser.add_argument(
        "--images", nargs="+", required=True, help="A folder or image files"
    )
    parser.add_argument("--output", default="mapanything_output")
    parser.add_argument(
        "--dtype", default="float16", choices=["float16", "bfloat16", "float32"]
    )
    parser.add_argument("--resize-mode", default="fixed_mapping")
    parser.add_argument("--size", type=int, default=None)
    parser.add_argument("--stride", type=int, default=1, help="Use every n-th image")
    parser.add_argument(
        "--no-mask", action="store_true", help="Keep ambiguous and edge pixels"
    )
    parser.add_argument(
        "--confidence-percentile",
        type=float,
        default=None,
        help="Drop the least confident pixels",
    )
    parser.add_argument("--multiview-confidence", action="store_true")
    parser.add_argument(
        "--tf32",
        action="store_true",
        help="Allow TF32 float32 matmuls (faster, less exact)",
    )
    args = parser.parse_args()
    os.environ.setdefault("MLX_ENABLE_TF32", "1" if args.tf32 else "0")

    model, processor = load(args.model, args.dtype)
    if args.resize_mode != "fixed_mapping" or args.size is not None:
        processor = MapAnythingProcessor(resize_mode=args.resize_mode, size=args.size)
    images = (
        args.images[0]
        if len(args.images) == 1 and Path(args.images[0]).is_dir()
        else args.images
    )

    start = time.perf_counter()
    views = processor.load_images(images, stride=args.stride)
    predictions = model.infer(
        views,
        apply_mask=not args.no_mask,
        apply_confidence_mask=args.confidence_percentile is not None,
        confidence_percentile=args.confidence_percentile or 10,
        use_multiview_confidence=args.multiview_confidence,
    )
    mx.eval(predictions)
    elapsed = time.perf_counter() - start
    height, width = views[0]["img"].shape[1:3]
    print(f"{len(views)} views at {width}x{height} in {elapsed:.2f} s")

    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    arrays = {
        f"{key}_{i}": np.array(value)
        for i, pred in enumerate(predictions)
        for key, value in pred.items()
    }
    np.savez(out / "predictions.npz", **arrays)
    points, colors = point_cloud(predictions)
    write_ply(out / "points.ply", points, colors)
    print(
        f"Wrote {out / 'predictions.npz'} and {out / 'points.ply'} ({len(points)} points)"
    )


if __name__ == "__main__":
    main()
