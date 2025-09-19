import numpy as np
from PIL import Image

from mlx_vlm.models.dots_ocr.dots_ocr import DotsOCRConfig, DotsOCRForCausalLM_MLX


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
