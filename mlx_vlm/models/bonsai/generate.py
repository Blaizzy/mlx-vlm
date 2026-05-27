from __future__ import annotations

import argparse
import random
from pathlib import Path

from mlx_vlm.models.bonsai.config import parse_size
from mlx_vlm.models.bonsai.pipeline import BonsaiImage


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate an image with the standalone Bonsai MLX model.")
    parser.add_argument("--variant", default="ternary", help="Model variant: ternary or binary.")
    parser.add_argument("--model-path", type=Path, default=None, help="Existing local model snapshot.")
    parser.add_argument("--models-dir", type=Path, default=None, help="Directory for downloaded models.")
    parser.add_argument("--no-download", action="store_true", help="Do not download missing weights.")
    parser.add_argument("-p", "--prompt", required=True, help="Text prompt.")
    parser.add_argument("--seed", type=int, default=None, help="Generation seed.")
    parser.add_argument("--steps", type=int, default=4, help="Inference steps.")
    parser.add_argument("--size", type=parse_size, default=(512, 512), help="Image size as WIDTHxHEIGHT.")
    parser.add_argument("--output", type=Path, default=Path("outputs/bonsai.png"), help="Output PNG path.")
    parser.add_argument("--guidance", type=float, default=1.0, help="Classifier-free guidance.")
    parser.add_argument("--max-sequence-length", type=int, default=512)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    width, height = args.size
    seed = args.seed if args.seed is not None else random.randrange(2**32)
    model = BonsaiImage.from_pretrained(
        args.variant,
        model_path=args.model_path,
        models_dir=args.models_dir,
        download=not args.no_download,
        max_sequence_length=args.max_sequence_length,
    )
    image = model.generate(
        args.prompt,
        seed=seed,
        steps=args.steps,
        width=width,
        height=height,
        guidance=args.guidance,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    image.save(args.output)
    print(f"saved {args.output} seed={seed} size={width}x{height}")


if __name__ == "__main__":
    main()
