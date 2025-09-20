from mlx_vlm.models.dots_ocr.tokenizer import SimpleTokenizer, render_chat


def test_simple_tokenizer_image_and_words():
    tokenizer = SimpleTokenizer(image_token_id=151652)
    ids = tokenizer.encode("hi there <image> now")
    assert 151652 in ids
    assert len(ids) == 4

    rendered = render_chat("hello <image>")
    assert "<image>" in rendered

import numpy as np
from PIL import Image

from mlx_vlm.models.dots_ocr.dots_ocr import DotsOCRConfig, DotsOCRForCausalLM_MLX


def test_prepare_and_generate_stub_single_image():
    cfg = DotsOCRConfig({"vision_config": {"num_layers": 2}})
    adapter = DotsOCRForCausalLM_MLX(cfg)

    image = Image.fromarray((np.random.rand(320, 480, 3) * 255).astype("uint8"))
    result = adapter.generate("Question: <image> describe.", [image], npz_path=None)

    assert "fused_len" in result
    assert result["tokens_shape"][1] == cfg.vision.embed_dim
