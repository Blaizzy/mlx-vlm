from __future__ import annotations

import time

import numpy as np
from PIL import Image
import mlx.core as mx

from mlx_vlm.models.dots_ocr.dots_ocr import DotsOCRConfig, DotsOCRForCausalLM_MLX


def main() -> None:
    cfg = DotsOCRConfig({"vision_config": {"num_layers": 6}})
    adapter = DotsOCRForCausalLM_MLX(cfg)
    adapter.load_vision_npz("weights/dots_ocr_vision.npz")

    images = [
        Image.fromarray((np.random.rand(672, 672, 3) * 255).astype("uint8")),
        Image.fromarray((np.random.rand(512, 768, 3) * 255).astype("uint8")),
    ]

    start = time.time()
    tokens, grids = adapter.encode_images(images)
    mx.eval(tokens)
    elapsed = time.time() - start
    print(
        f"[bench] tokens={tokens.shape[0]} dim={tokens.shape[1]} "
        f"images={len(images)} time={elapsed:.3f}s"
    )


if __name__ == "__main__":
    main()
