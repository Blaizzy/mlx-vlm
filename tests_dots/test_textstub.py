from mlx_vlm.models.dots_ocr.tokenizer import (
    SimpleTokenizer,
    render_chat,
    try_hf_tokenizer,
)


def test_simple_tokenizer_image_and_words():
    tokenizer = SimpleTokenizer(image_token_id=151652)
    ids = tokenizer.encode("hi there <image> now")
    assert 151652 in ids
    assert len(ids) == 4

    rendered = render_chat("hello <image>")
    assert "<image>" in rendered

import numpy as np
from PIL import Image
import mlx.core as mx

from mlx_vlm.models.dots_ocr.dots_ocr import DotsOCRConfig, DotsOCRForCausalLM_MLX
from mlx_vlm.text.embedding_fuser import fuse_embeddings_from_image_tokens


def test_prepare_and_generate_stub_single_image():
    cfg = DotsOCRConfig({"vision_config": {"num_layers": 2}})
    adapter = DotsOCRForCausalLM_MLX(cfg)

    image = Image.fromarray((np.random.rand(320, 480, 3) * 255).astype("uint8"))
    result = adapter.generate("Question: <image> describe.", [image], npz_path=None)

    assert "fused_len" in result
    assert result["tokens_shape"][1] == cfg.vision.embed_dim


def test_try_hf_tokenizer_skip_if_missing():
    maybe_tok = try_hf_tokenizer("nonexistent-model-xyz")
    assert maybe_tok is None or (isinstance(maybe_tok, tuple) and len(maybe_tok) == 2)


def test_qwen_loader_imports():
    try:
        from mlx_vlm.text.mlx_qwen_loader import MLXQwen, QwenLoadOpts  # noqa: F401
    except Exception:
        import pytest

        pytest.skip("mlx-lm not installed")


def test_embedding_fuser_shapes():
    vocab, hidden = 32000, 1024
    embedding = mx.random.uniform(shape=(vocab, hidden))
    input_ids = np.array([10, 151652, 20, 30], dtype=np.int32)
    projected = mx.random.uniform(shape=(96, hidden))

    fused, pos = fuse_embeddings_from_image_tokens(
        embedding, input_ids, 151652, projected
    )

    assert fused.shape == (len(input_ids) - 1 + projected.shape[0], hidden)
    assert pos == 1
