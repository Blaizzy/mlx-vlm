from __future__ import annotations

import argparse
from typing import Sequence

import mlx.core as mx
import numpy as np
from PIL import Image

from mlx_vlm.models.dots_ocr.dots_ocr import (
    DotsOCRConfig,
    DotsOCRForCausalLM_MLX,
    splice_image_tokens,
    splice_image_tokens_multi,
)


def _load_or_random(path: str | None, fallback_size: Sequence[int]) -> Image.Image:
    if path:
        return Image.open(path).convert("RGB")
    width, height = fallback_size
    return Image.fromarray((np.random.rand(height, width, 3) * 255).astype("uint8"))


def _split_tokens(tokens: mx.array, grids: list[tuple[int, int, int]]):
    splits = []
    offset = 0
    for _, h, w in grids:
        count = (h * w) // 4
        splits.append(tokens[offset : offset + count])
        offset += count
    return splits


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Load vision NPZ, encode images, and compute splice lengths."
    )
    parser.add_argument("--npz", default="weights/dots_ocr_vision.npz")
    parser.add_argument("--multi", action="store_true")
    parser.add_argument("--img1", default=None)
    parser.add_argument("--img2", default=None)
    parser.add_argument("--image-token-id", type=int, default=151652)
    args = parser.parse_args()

    cfg = DotsOCRConfig({"vision_config": {"num_layers": 2}})
    adapter = DotsOCRForCausalLM_MLX(cfg)

    report = adapter.load_vision_npz(args.npz)
    print("[load]", report)

    image1 = _load_or_random(args.img1, (520, 360))

    if not args.multi:
        tokens, grids = adapter.encode_images([image1])
        print("[encode-single] tokens", tuple(tokens.shape), "grids", grids)

        input_ids = mx.array([10, 20, args.image_token_id, 30], dtype=mx.int32)
        pos, fused = splice_image_tokens(input_ids, args.image_token_id, tokens)
        print("[splice-single] pos", pos, "fused_len", fused)
    else:
        image2 = _load_or_random(args.img2, (400, 256))
        tokens, grids = adapter.encode_images([image1, image2])
        print("[encode-multi] tokens", tuple(tokens.shape), "grids", grids)

        chunks = _split_tokens(tokens, grids)
        input_ids = mx.array(
            [999, args.image_token_id, 42, args.image_token_id, 7], dtype=mx.int32
        )
        pos, fused = splice_image_tokens_multi(
            input_ids, args.image_token_id, chunks
        )
        print("[splice-multi] pos", pos, "fused_len", fused)


if __name__ == "__main__":
    main()
