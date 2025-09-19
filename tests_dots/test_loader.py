import os
import tempfile

import mlx.core as mx
import numpy as np

from mlx_vlm.models.dots_ocr.dots_ocr import DotsOCRConfig
from mlx_vlm.models.dots_ocr.dots_vision import DotsVisionTransformer_MLX
from mlx_vlm.models.dots_ocr.weight_loader import load_npz_into_vision


def test_loader_assigns_and_forward_pass():
    cfg = DotsOCRConfig({"vision_config": {"num_layers": 1}})
    model = DotsVisionTransformer_MLX(cfg)

    with tempfile.TemporaryDirectory() as td:
        path = os.path.join(td, "weights.npz")
        dim = 1536
        hidden = 4224

        np.savez(
            path,
            **{
                "vision.patch.proj.weight": np.zeros((dim, 3, 14, 14), dtype=np.float32),
                "vision.patch.norm.weight": np.ones((dim,), dtype=np.float32),
                "vision.blocks.0.attn.qkv.weight": np.zeros((dim * 3, dim), dtype=np.float32),
                "vision.blocks.0.attn.proj.weight": np.zeros((dim, dim), dtype=np.float32),
                "vision.blocks.0.mlp.fc1.weight": np.zeros((hidden, dim), dtype=np.float32),
                "vision.blocks.0.mlp.fc2.weight": np.zeros((dim, hidden), dtype=np.float32),
                "vision.blocks.0.mlp.fc3.weight": np.zeros((hidden, dim), dtype=np.float32),
                "vision.blocks.0.norm1.weight": np.ones((dim,), dtype=np.float32),
                "vision.blocks.0.norm2.weight": np.ones((dim,), dtype=np.float32),
                "vision.post.weight": np.ones((dim,), dtype=np.float32),
                "vision.merger.ln.weight": np.ones((dim,), dtype=np.float32),
                "vision.merger.mlp.0.weight": np.zeros((dim, dim * 4), dtype=np.float32),
                "vision.merger.mlp.0.bias": np.zeros((dim,), dtype=np.float32),
                "vision.merger.mlp.2.weight": np.zeros((dim, dim), dtype=np.float32),
                "vision.merger.mlp.2.bias": np.zeros((dim,), dtype=np.float32),
            },
        )

        report = load_npz_into_vision(model, path)
        assert report["loaded"] >= 12
        assert report["missing"] == 0
        assert model.patch.proj.weight.shape == (dim, 14, 14, 3)

        x = mx.random.uniform(shape=(1, 3, 224, 224))
        grid = [[1, 16, 16]]
        out = model(x, grid)
        assert out.shape == (64, dim)
