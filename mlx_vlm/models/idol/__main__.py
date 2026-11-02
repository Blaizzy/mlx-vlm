import argparse
from pathlib import Path

from .pipeline import IdolPipeline
from .weights import resolve_model


def main():
    parser = argparse.ArgumentParser(
        description="Idol: image to 3D Gaussian splats using MLX"
    )
    parser.add_argument("--model", "--weights", default="speridlabs/idol")
    parser.add_argument("--image", type=Path, help="Input image or directory")
    parser.add_argument("--output", "--out", type=Path, default=Path("outputs/idol"))
    parser.add_argument("--mode", choices=("synthetic", "real"), default="synthetic")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--cfg-strength", type=float, default=3.0)
    parser.add_argument("--rescale-t", type=float, default=3.0)
    parser.add_argument("--num-gaussians", type=int, default=262144)
    parser.add_argument("--crop-margin", type=float)
    parser.add_argument(
        "--dinov3", help="DINOv3 Hugging Face repository or local directory"
    )
    parser.add_argument("--flux2-vae", help="FLUX.2 repository or local VAE directory")
    parser.add_argument(
        "--dinov3-dtype", choices=("bf16", "fp32", "fp16"), default="bf16"
    )
    parser.add_argument(
        "--no-background-removal",
        action="store_true",
        help="Require an input with transparency",
    )
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument(
        "--download-only",
        action="store_true",
        help="Download the Idol weights without loading the pipeline",
    )
    parser.add_argument(
        "--batch-cfg",
        action="store_true",
        help="Batch conditional and unconditional predictions (uses more memory)",
    )
    args = parser.parse_args()
    if args.download_only:
        print(resolve_model(args.model, local_files_only=args.local_files_only))
        return
    if args.image is None:
        parser.error("--image is required unless --download-only is used")
    if args.image.is_dir():
        images = sorted(
            p
            for p in args.image.iterdir()
            if p.suffix.lower() in (".png", ".jpg", ".jpeg", ".webp")
        )
        if not images:
            parser.error(f"No images found in {args.image}")
    else:
        images = [args.image]
    if len(images) > 1 and args.output.suffix.lower() == ".ply":
        parser.error("Use an output directory when processing multiple images")
    pipeline = IdolPipeline.from_pretrained(
        args.model,
        dinov3=args.dinov3,
        flux2_vae=args.flux2_vae,
        dinov3_dtype=args.dinov3_dtype,
        background_removal=not args.no_background_removal,
        local_files_only=args.local_files_only,
    )
    for image in images:
        result = pipeline(
            image,
            mode=args.mode,
            seed=args.seed,
            steps=args.steps,
            cfg_strength=args.cfg_strength,
            rescale_t=args.rescale_t,
            num_gaussians=args.num_gaussians,
            crop_margin=args.crop_margin,
            verbose=True,
            batch_cfg=args.batch_cfg,
        )
        destination = (
            args.output
            if args.output.suffix.lower() == ".ply"
            else args.output / f"{image.stem}.ply"
        )
        result.save(destination)
        print(destination)


if __name__ == "__main__":
    main()
