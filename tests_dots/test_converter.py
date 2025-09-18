import os
import tempfile

import numpy as np
from numpy.lib import npyio
from safetensors.numpy import save_file

from mlx_vlm.convert.convert_dots_ocr import (
    convert_dir_or_file_to_npz,
    list_vision_keys,
)


def test_list_vision_keys_minimal_tmpfile():
    with tempfile.TemporaryDirectory() as td:
        path = os.path.join(td, "fake.safetensors")
        save_file(
            {
                "vision_tower.patch_embed.patchifier.proj.weight": np.zeros(
                    (1536, 3, 14, 14), dtype=np.float32
                ),
                "vision_tower.blocks.0.attn.qkv.weight": np.zeros(
                    (1536 * 3, 1536), dtype=np.float32
                ),
                "text_decoder.dummy.weight": np.zeros((4, 4), dtype=np.float32),
            },
            path,
        )
        keys = list_vision_keys(path)
        assert any(k.endswith("proj.weight") for k in keys)
        assert all(k.startswith("vision_tower") for k in keys)


def test_convert_minimal_mapping_npz_roundtrip():
    with tempfile.TemporaryDirectory() as td:
        st = os.path.join(td, "fake.safetensors")
        save_file(
            {
                "vision_tower.patch_embed.patchifier.proj.weight": np.zeros(
                    (1536, 3, 14, 14), dtype=np.float32
                ),
                "vision_tower.patch_embed.patchifier.norm.weight": np.ones(
                    (1536,), dtype=np.float32
                ),
                "vision_tower.blocks.0.attn.qkv.weight": np.zeros(
                    (1536 * 3, 1536), dtype=np.float32
                ),
                "vision_tower.blocks.0.attn.proj.weight": np.zeros(
                    (1536, 1536), dtype=np.float32
                ),
                "vision_tower.blocks.0.mlp.fc1.weight": np.zeros(
                    (4224, 1536), dtype=np.float32
                ),
                "vision_tower.blocks.0.mlp.fc2.weight": np.zeros(
                    (1536, 4224), dtype=np.float32
                ),
                "vision_tower.blocks.0.mlp.fc3.weight": np.zeros(
                    (4224, 1536), dtype=np.float32
                ),
                "vision_tower.blocks.0.norm1.weight": np.ones((1536,), dtype=np.float32),
                "vision_tower.blocks.0.norm2.weight": np.ones((1536,), dtype=np.float32),
                "vision_tower.post_trunk_norm.weight": np.ones((1536,), dtype=np.float32),
                "vision_tower.merger.ln_q.weight": np.ones((1536,), dtype=np.float32),
                "vision_tower.merger.mlp.0.weight": np.zeros(
                    (1536, 1536 * 4), dtype=np.float32
                ),
                "vision_tower.merger.mlp.0.bias": np.zeros((1536,), dtype=np.float32),
                "vision_tower.merger.mlp.2.weight": np.zeros(
                    (1536, 1536), dtype=np.float32
                ),
                "vision_tower.merger.mlp.2.bias": np.zeros((1536,), dtype=np.float32),
                "vision_tower.patch_embed.patchifier.proj.bias": np.zeros(
                    (1536,), dtype=np.float32
                ),
            },
            st,
        )
        out_npz = os.path.join(td, "out.npz")
        info = convert_dir_or_file_to_npz(st, out_npz)
        assert os.path.exists(out_npz)
        assert info["tensors"] >= 10
        npz: npyio.NpzFile = np.load(out_npz)
        assert "vision.patch.proj.weight" in npz.files
        assert "vision.patch.norm.weight" in npz.files
        assert "vision.blocks.0.attn.qkv.weight" in npz.files
        assert "vision.post.weight" in npz.files
        assert "vision.merger.mlp.0.bias" in npz.files
