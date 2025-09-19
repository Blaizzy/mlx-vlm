import os
import tempfile

import numpy as np
import mlx.core as mx
from PIL import Image

from mlx_vlm.models.dots_ocr.dots_ocr import (
    DotsOCRConfig,
    DotsOCRForCausalLM_MLX,
    splice_image_tokens,
)


def test_adapter_encode_single_and_multi():
    cfg = DotsOCRConfig({"vision_config": {"num_layers": 2}})
    adapter = DotsOCRForCausalLM_MLX(cfg)

    img1 = Image.fromarray((np.random.rand(380, 512, 3) * 255).astype("uint8"))
    img2 = Image.fromarray((np.random.rand(256, 400, 3) * 255).astype("uint8"))

    vt1, g1 = adapter.encode_images([img1])
    assert len(g1) == 1
    assert vt1.shape[1] == cfg.vision.embed_dim

    vt2, g2 = adapter.encode_images([img1, img2])
    assert len(g2) == 2
    assert vt2.shape[1] == cfg.vision.embed_dim


def test_splice_single_placeholder():
    ids = mx.array([10, 20, 151652, 30], dtype=mx.int32)
    vision_tokens = mx.zeros((96, 1536))
    pos, fused_len = splice_image_tokens(ids, 151652, vision_tokens)
    assert pos == 2
    assert fused_len == len(ids) - 1 + 96


def test_adapter_loads_npz_into_vision():
    cfg = DotsOCRConfig({"vision_config": {"num_layers": 1}})
    adapter = DotsOCRForCausalLM_MLX(cfg)

    with tempfile.TemporaryDirectory() as tmp:
        npz_path = os.path.join(tmp, "vision.npz")
        np.savez(
            npz_path,
            **{
                "vision.patch.proj.weight": np.zeros((1536, 3, 14, 14), dtype=np.float32),
                "vision.patch.norm.weight": np.ones((1536,), dtype=np.float32),
                "vision.blocks.0.attn.qkv.weight": np.zeros((1536 * 3, 1536), dtype=np.float32),
                "vision.blocks.0.attn.proj.weight": np.zeros((1536, 1536), dtype=np.float32),
                "vision.blocks.0.mlp.fc1.weight": np.zeros((4224, 1536), dtype=np.float32),
                "vision.blocks.0.mlp.fc2.weight": np.zeros((1536, 4224), dtype=np.float32),
                "vision.blocks.0.mlp.fc3.weight": np.zeros((4224, 1536), dtype=np.float32),
                "vision.blocks.0.norm1.weight": np.ones((1536,), dtype=np.float32),
                "vision.blocks.0.norm2.weight": np.ones((1536,), dtype=np.float32),
                "vision.post.weight": np.ones((1536,), dtype=np.float32),
                "vision.merger.ln.weight": np.ones((1536,), dtype=np.float32),
                "vision.merger.mlp.0.weight": np.zeros((1536, 1536 * 4), dtype=np.float32),
                "vision.merger.mlp.0.bias": np.zeros((1536,), dtype=np.float32),
                "vision.merger.mlp.2.weight": np.zeros((1536, 1536), dtype=np.float32),
                "vision.merger.mlp.2.bias": np.zeros((1536,), dtype=np.float32),
            },
        )

        report = adapter.load_vision_npz(npz_path)
        assert report["loaded"] >= 12
        assert report["missing"] == 0
