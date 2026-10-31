"""Text-only rope deltas must be relative to each row's unpadded length."""

from types import SimpleNamespace

import mlx.core as mx

from mlx_vlm.models.qwen3_vl.config import TextConfig
from mlx_vlm.models.qwen3_vl.language import LanguageModel


def _language_model():
    args = TextConfig(
        model_type="qwen3_vl",
        num_hidden_layers=1,
        hidden_size=32,
        intermediate_size=64,
        num_attention_heads=2,
        rms_norm_eps=1e-6,
        vocab_size=16,
        num_key_value_heads=2,
        head_dim=16,
        rope_theta=10000.0,
        max_position_embeddings=128,
        rope_scaling={"type": "default", "mrope_section": [2, 3, 3]},
    )
    config = SimpleNamespace(
        vision_config=SimpleNamespace(spatial_merge_size=2),
        image_token_id=100,
        video_token_id=101,
        vision_start_token_id=102,
    )
    return LanguageModel(args, config)


def test_text_only_rope_deltas_ignore_left_padding():
    lm = _language_model()
    # Row 0 is left-padded by 3 tokens, row 1 has no padding.
    ids = mx.ones((2, 5), dtype=mx.int32)
    mask = mx.array([[0, 0, 0, 1, 1], [1, 1, 1, 1, 1]])

    _, deltas = lm.get_rope_index(ids, attention_mask=mask)

    # BatchKVCache offsets already exclude padding, so decode position is
    # offset + delta == number of real tokens seen so far.
    assert deltas.tolist() == [[0], [0]]
