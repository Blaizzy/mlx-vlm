import os
import tempfile

import numpy as np
from safetensors.numpy import save_file

from mlx_vlm.convert.convert_dots_ocr import list_vision_keys


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
