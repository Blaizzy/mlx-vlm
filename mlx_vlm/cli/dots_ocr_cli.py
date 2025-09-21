from __future__ import annotations

import argparse
import os

from PIL import Image

from mlx_vlm.convert.convert_dots_ocr import cli_convert
from mlx_vlm.models.dots_ocr.dots_ocr import DotsOCRConfig, DotsOCRForCausalLM_MLX


def load_pdf_images(pdf_path: str, dpi: int = 200) -> list[Image.Image]:
    try:
        from pdf2image import convert_from_path
    except ImportError as exc:
        raise RuntimeError(
            "pdf2image is required for --pdf support; install via `pip install pdf2image`"
        ) from exc

    pages = convert_from_path(pdf_path, dpi=dpi)
    return [page.convert("RGB") for page in pages]


def main() -> None:
    parser = argparse.ArgumentParser("mlx-vlm-dots-ocr")
    parser.add_argument("--weights-root", required=True, help="Path to dots.ocr HF folder")
    parser.add_argument(
        "--out",
        default="weights/dots_ocr_vision.npz",
        help="NPZ destination for vision weights",
    )
    parser.add_argument("--pdf", help="Optional PDF path to encode")
    parser.add_argument("--dpi", type=int, default=200)
    args = parser.parse_args()

    out_dir = os.path.dirname(args.out)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    cli_convert(args.weights_root, args.out)

    if not args.pdf:
        return

    pages = load_pdf_images(args.pdf, dpi=args.dpi)
    if not pages:
        print("[vision] no pages extracted from PDF; skipping encode")
        return

    cfg = DotsOCRConfig({"vision_config": {"num_layers": 2}})
    adapter = DotsOCRForCausalLM_MLX(cfg)
    print("[load]", adapter.load_vision_npz(args.out))

    tokens, grids = adapter.encode_images(pages)
    print(
        f"[vision] pages={len(pages)} tokens={tokens.shape} first_grid={grids[0]}"
    )
    print("NOTE: text decoding not yet wired; future tasks will add it.")


if __name__ == "__main__":
    main()
