"""Camera and depth reconstruction from an image sequence with VGGT-Omega.

``VGGTOmegaPredictor.infer`` builds one lazy MLX graph from image decoding
to world points; nothing runs until an output is read (``mx.eval``,
``np.array``). The CLI writes the predictions as ``.npz`` and, optionally, a
colored point cloud as ``.ply``:

    python -m mlx_vlm.models.vggt_omega.generate \
        --model mlx-community/VGGT-Omega-1B-512-bf16 \
        --video clip.mp4 --fps 1 --output scene.npz --ply scene.ply
"""

import argparse
import time
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Union

import mlx.core as mx
import numpy as np

from ..video_depth_anything.generate import read_video_frames
from .geometry import encoding_to_camera, unproject_depth
from .processing_vggt_omega import ImageLike, VGGTOmegaProcessor

_IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff"}


def list_images(sources: Iterable[str]) -> List[str]:
    """Expand directories to their sorted image files."""
    paths = []
    for source in sources:
        path = Path(source)
        if path.is_dir():
            paths += sorted(
                str(p) for p in path.iterdir() if p.suffix.lower() in _IMAGE_SUFFIXES
            )
        else:
            paths.append(str(source))
    return paths


def read_video(
    source, fps: float = 1.0, max_frames: Optional[int] = None
) -> List[np.ndarray]:
    """Sample RGB frames from a video at about ``fps`` frames per second,
    as the reference demo does."""
    frames, _ = read_video_frames(str(source), max_frames or -1, fps)
    return list(frames)


class VGGTOmegaPredictor:
    def __init__(self, model, processor: Optional[VGGTOmegaProcessor] = None):
        self.model = model
        self.processor = processor or VGGTOmegaProcessor(
            image_resolution=model.config.image_resolution,
            patch_size=model.config.patch_size,
        )

    def infer(self, images: Sequence[ImageLike]) -> Dict[str, mx.array]:
        """Reconstruct one sequence; the first image is the reference frame.

        Returns lazy float32 arrays for the S frames at the model input size
        (H, W):
            ``extrinsics`` (S, 3, 4) camera-from-world, OpenCV convention
            ``intrinsics`` (S, 3, 3) in pixels
            ``depth`` (S, H, W), ``depth_conf`` (S, H, W)
            ``world_points`` (S, H, W, 3) unprojected depth
            ``pose_enc`` (S, 9), ``images`` (S, H, W, 3) in [0, 1],
            ``camera_and_register_tokens`` (S, 17, 2048)
        and ``text_alignment_embedding`` / ``text_alignment_token`` for the
        text-aligned model.
        """
        pixel_values = self.processor(images)["pixel_values"]
        out = {k: v[0] for k, v in self.model(pixel_values[None]).items()}
        if "depth" in out:
            out["depth"] = out["depth"][..., 0]
        if "pose_enc" in out:
            size = pixel_values.shape[1:3]
            extrinsics, intrinsics = encoding_to_camera(out["pose_enc"], size)
            out["extrinsics"], out["intrinsics"] = extrinsics, intrinsics
            if "depth" in out:
                out["world_points"] = unproject_depth(
                    out["depth"], extrinsics, intrinsics
                )
        return out


def point_cloud(
    predictions: Dict[str, Union[mx.array, np.ndarray]],
    conf_percentile: float = 20.0,
    max_points: Optional[int] = 1_000_000,
    seed: int = 0,
):
    """Points and uint8 colors whose confidence is above the given
    percentile, as in the reference demo; subsampled to ``max_points``."""
    points = np.asarray(predictions["world_points"]).reshape(-1, 3)
    colors = np.asarray(predictions["images"]).reshape(-1, 3)
    conf = np.asarray(predictions["depth_conf"]).reshape(-1)
    keep = np.isfinite(points).all(axis=1) & np.isfinite(conf)
    if conf_percentile > 0 and keep.any():
        keep &= conf >= np.percentile(conf[keep], conf_percentile)
    keep &= conf > 1e-5
    index = np.flatnonzero(keep)
    if max_points is not None and len(index) > max_points:
        index = np.random.default_rng(seed).choice(index, max_points, replace=False)
    colors = np.clip(colors[index] * 255, 0, 255).astype(np.uint8)
    return points[index], colors


def write_ply(path, points: np.ndarray, colors: np.ndarray):
    """Binary little-endian PLY with float xyz and uchar RGB."""
    vertex = np.empty(
        len(points),
        dtype=[("xyz", "<f4", 3), ("rgb", "u1", 3)],
    )
    vertex["xyz"], vertex["rgb"] = points, colors
    header = (
        "ply\nformat binary_little_endian 1.0\n"
        f"element vertex {len(points)}\n"
        "property float x\nproperty float y\nproperty float z\n"
        "property uchar red\nproperty uchar green\nproperty uchar blue\n"
        "end_header\n"
    )
    with open(path, "wb") as out:
        out.write(header.encode())
        out.write(vertex.tobytes())


def main():
    parser = argparse.ArgumentParser(
        description="VGGT-Omega camera and depth reconstruction."
    )
    parser.add_argument("--model", default="mlx-community/VGGT-Omega-1B-512-bf16")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--images", nargs="+", help="Image files or directories")
    source.add_argument("--video", help="Video file")
    parser.add_argument("--fps", type=float, default=1.0, help="Video sampling rate")
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--resolution", type=int, default=None)
    parser.add_argument("--mode", choices=["balanced", "max_size"], default=None)
    parser.add_argument("--output", default="predictions.npz")
    parser.add_argument("--ply", default=None, help="Also write a point cloud")
    parser.add_argument("--conf-percentile", type=float, default=20.0)
    parser.add_argument("--max-points", type=int, default=1_000_000)
    args = parser.parse_args()

    from ...utils import get_model_path, load_model

    model_path = get_model_path(args.model)
    model = load_model(model_path)
    overrides = {
        k: v
        for k, v in (("image_resolution", args.resolution), ("mode", args.mode))
        if v is not None
    }
    processor = VGGTOmegaProcessor.from_pretrained(model_path, **overrides)
    predictor = VGGTOmegaPredictor(model, processor)

    if args.video:
        images = read_video(args.video, args.fps, args.max_frames)
    else:
        images = list_images(args.images)[: args.max_frames]
    start = time.perf_counter()
    predictions = predictor.infer(images)
    mx.eval(predictions)
    elapsed = time.perf_counter() - start
    S, H, W = predictions["depth"].shape
    print(
        f"{S} frames at {W}x{H}: {elapsed:.2f} s, "
        f"peak memory {mx.get_peak_memory() / 1e9:.2f} GB"
    )

    arrays = {k: np.array(v) for k, v in predictions.items()}
    np.savez(args.output, **arrays)
    print(f"Saved predictions to {args.output}")
    if args.ply:
        points, colors = point_cloud(arrays, args.conf_percentile, args.max_points)
        write_ply(args.ply, points, colors)
        print(f"Saved {len(points)} points to {args.ply}")


if __name__ == "__main__":
    main()
