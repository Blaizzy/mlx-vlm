"""Refine a coarse mesh with Hi3D v1.0: image + coarse mesh -> detailed mesh.

Pass ``--mesh`` with an existing coarse mesh, or ``--sam3d-model`` to build
one from the image with SAM 3D Objects first.
"""

import argparse
import json
import time
from pathlib import Path

import mlx.core as mx
from PIL import Image

from .mesh import coarse_from_sam3d, read_mesh, write_obj


def sam3d_coarse_mesh(model_path, image_path, mask_path=None, seed=0):
    """Coarse mesh of the masked object from a SAM 3D Objects bundle."""
    from ..sam3d_objects.generate import read_image, read_mask
    from ..sam3d_objects.pipeline import Pipeline

    pipeline = Pipeline.from_pretrained(model_path)
    result = pipeline.generate(
        read_image(image_path),
        read_mask(mask_path or image_path),
        formats=("mesh",),
        seed=seed,
    )
    vertices, faces = coarse_from_sam3d(result)
    del pipeline, result
    mx.clear_cache()
    return vertices, faces


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, help="Local converted Hi3D bundle")
    parser.add_argument("--image", required=True, help="Object image; alpha marks it")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--mesh", help="Coarse mesh (OBJ, or any trimesh format)")
    source.add_argument(
        "--sam3d-model", help="SAM 3D Objects bundle for the coarse mesh"
    )
    parser.add_argument("--mask", help="Object mask for SAM 3D (default: image alpha)")
    parser.add_argument(
        "--save-coarse", help="Also write the SAM 3D coarse mesh (.obj)"
    )
    parser.add_argument("--output", required=True, help="Refined mesh (.obj)")
    parser.add_argument(
        "--resolution", type=int, default=1024, choices=(512, 1024, 1536)
    )
    parser.add_argument("--steps", type=int, default=None, help="Heun steps per pass")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    from .hi3d import Model

    report = (lambda message: None) if args.quiet else (lambda m: print(m, flush=True))
    timings = {}
    if args.sam3d_model:
        report("generating the coarse mesh with SAM 3D Objects")
        start = time.perf_counter()
        vertices, faces = sam3d_coarse_mesh(
            args.sam3d_model, args.image, args.mask, args.seed
        )
        timings["coarse_mesh"] = time.perf_counter() - start
        if args.save_coarse:
            write_obj(args.save_coarse, vertices, faces)
    else:
        vertices, faces = read_mesh(args.mesh)
    start = time.perf_counter()
    model = Model.from_pretrained(args.model)
    timings["load"] = time.perf_counter() - start
    result = model(
        Image.open(args.image),
        vertices,
        faces,
        resolution=args.resolution,
        steps=args.steps,
        seed=args.seed,
        progress=report,
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    write_obj(output, result.vertices, result.faces)
    timings.update(result.timings)
    summary = {
        "vertices": int(len(result.vertices)),
        "faces": int(len(result.faces)),
        "timings": {k: round(v, 2) for k, v in timings.items()},
        "peak_memory_gb": round(mx.get_peak_memory() / 1e9, 2),
    }
    print(json.dumps(summary))


if __name__ == "__main__":
    main()
