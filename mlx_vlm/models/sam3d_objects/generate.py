"""CLI for single requests or a JSONL stream of image-to-3D requests."""

import argparse
import asyncio
import json
import sys
from pathlib import Path

import mlx.core as mx
import numpy as np
from mlx.utils import tree_flatten
from PIL import Image

from .pipeline import Pipeline, Request


def read_image(path):
    """Read an image as an HWC uint8 MLX array; RGBA when it has transparency."""
    with Image.open(path) as image:
        mode = "RGBA" if image.has_transparency_data else "RGB"
        return mx.array(np.asarray(image.convert(mode)))


def read_mask(path):
    """Read a grayscale mask, or the alpha channel of an RGBA cutout, as HW booleans."""
    with Image.open(path) as image:
        channel = 3 if image.has_transparency_data else 0
        pixels = np.asarray(image.convert("RGBA"))
    return mx.array(pixels[..., channel] > 0)


def write_gaussians(path, gaussian):
    """Write the standard Gaussian-splat PLY fields (log-scales/logit opacity)."""
    opacity = mx.clip(gaussian["opacities"], 1e-7, 1 - 1e-7)
    values = mx.concatenate(
        [
            gaussian["positions"],
            mx.zeros_like(gaussian["positions"]),
            gaussian["sh_dc"],
            mx.log(opacity / (1 - opacity)),
            mx.log(gaussian["scales"]),
            gaussian["rotations"],
        ],
        axis=-1,
    ).astype(mx.float32)
    fields = [
        "x",
        "y",
        "z",
        "nx",
        "ny",
        "nz",
        "f_dc_0",
        "f_dc_1",
        "f_dc_2",
        "opacity",
        "scale_0",
        "scale_1",
        "scale_2",
        "rot_0",
        "rot_1",
        "rot_2",
        "rot_3",
    ]
    header = (
        "ply\nformat binary_little_endian 1.0\n"
        + f"element vertex {values.shape[0]}\n"
        + "".join(f"property float {f}\n" for f in fields)
        + "end_header\n"
    )
    with open(path, "wb") as out:
        out.write(header.encode())
        out.write(np.array(values, dtype="<f4").tobytes())


def write_obj(path, mesh):
    """Write vertices with RGB colors and 1-based triangle faces."""
    vertices = mx.concatenate(
        [mesh["vertices"], mesh["vertex_colors"][:, :3]], axis=-1
    ).astype(mx.float32)
    with open(path, "w") as out:
        np.savetxt(out, np.array(vertices), fmt="v %.8g %.8g %.8g %.8g %.8g %.8g")
        np.savetxt(out, np.array(mesh["faces"]) + 1, fmt="f %d %d %d")


def _request(values, defaults):
    image = read_image(values["image"])
    mask = read_mask(values["mask"]) if values.get("mask") else None
    pointmap = mx.load(values["pointmap"])["points"] if values.get("pointmap") else None
    return Request(
        image,
        mask,
        pointmap,
        request_id=str(values.get("id", "0")),
        seed=int(values.get("seed", defaults.seed)),
        ss_steps=defaults.ss_steps,
        slat_steps=defaults.slat_steps,
        formats=tuple(defaults.formats),
        estimate_depth=not defaults.no_depth,
    )


def _save(result, root):
    root.mkdir(parents=True, exist_ok=True)
    tensors = {k: v for k, v in tree_flatten(result) if isinstance(v, mx.array)}
    mx.save_safetensors(str(root / "result.safetensors"), tensors)
    for kind in ("gaussian", "gaussian_4"):
        if kind in result:
            write_gaussians(root / f"{kind}.ply", result[kind])
    if "mesh" in result:
        write_obj(root / "mesh.obj", result["mesh"])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, help="Local converted BF16 bundle")
    parser.add_argument(
        "--image", help="RGB or RGBA image in any Pillow-readable format"
    )
    parser.add_argument(
        "--mask", help="Object mask image; RGBA cutouts use their alpha channel"
    )
    parser.add_argument(
        "--pointmap", help="Safetensors file with an HWC 'points' array"
    )
    parser.add_argument(
        "--jsonl",
        action="store_true",
        help="Read JSON requests from stdin and stream JSON progress to stdout",
    )
    parser.add_argument("--output", required=True, help="Output directory")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--ss-steps", type=int)
    parser.add_argument("--slat-steps", type=int)
    parser.add_argument(
        "--formats",
        nargs="+",
        choices=("gaussian", "gaussian_4", "mesh"),
        default=["gaussian", "mesh"],
    )
    parser.add_argument(
        "--no-depth",
        action="store_true",
        help="Use trained point-map dropout when no map is supplied",
    )
    args = parser.parse_args()
    if not args.jsonl and not args.image:
        parser.error("Provide --image or --jsonl")
    pipeline = Pipeline.from_pretrained(args.model)
    if not args.jsonl:
        request = _request(
            {"image": args.image, "mask": args.mask, "pointmap": args.pointmap}, args
        )
        for event in pipeline.stream(request):
            print(
                json.dumps(
                    {
                        "id": event.request_id,
                        "stage": event.stage,
                        "step": event.step,
                        "total_steps": event.total_steps,
                    }
                ),
                flush=True,
            )
            if event.stage == "complete":
                _save(event.data, Path(args.output))
        return

    async def run():
        async def requests():
            while line := await asyncio.to_thread(sys.stdin.readline):
                if line.strip():
                    yield _request(json.loads(line), args)

        request_index = 0
        async for event in pipeline.astream(requests()):
            record = {
                "id": event.request_id,
                "stage": event.stage,
                "step": event.step,
                "total_steps": event.total_steps,
            }
            if event.stage == "complete":
                # Input IDs are labels, never filesystem paths.
                output = Path(args.output) / f"{request_index:06d}"
                await asyncio.to_thread(_save, event.data, output)
                record["output"] = str(output)
                request_index += 1
            print(json.dumps(record), flush=True)

    asyncio.run(run())


if __name__ == "__main__":
    main()
