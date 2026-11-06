import pytest
from transformers.models.smolvlm.configuration_smolvlm import SmolVLMVisionConfig

from mlx_vlm.models.idefics3.vision import EncoderLayer
from mlx_vlm.models.smolvlm.config import VisionConfig


def test_missing_vision_heads_matches_transformers_checkpoint_default():
    # The 2.2B checkpoint omits hidden_size and num_attention_heads.
    checkpoint = {"image_size": 384, "patch_size": 14, "intermediate_size": 4304}
    config = VisionConfig.from_dict(checkpoint)
    reference = SmolVLMVisionConfig(**checkpoint)
    assert config.hidden_size == reference.hidden_size == 1152
    assert config.num_attention_heads == reference.num_attention_heads == 16


def test_explicit_vision_head_count_is_preserved():
    config = VisionConfig.from_dict({"hidden_size": 64, "num_attention_heads": 4})
    layer = EncoderLayer(config)
    assert layer.self_attn.num_heads == 4
    assert layer.self_attn.scale == pytest.approx((64 / 4) ** -0.5)


def test_default_vision_head_count_reaches_attention_layer():
    config = VisionConfig.from_dict({"hidden_size": 64, "intermediate_size": 64})
    layer = EncoderLayer(config)
    assert layer.self_attn.num_heads == 16
    assert layer.self_attn.scale == pytest.approx((64 / 16) ** -0.5)
