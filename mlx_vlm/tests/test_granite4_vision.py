import mlx.core as mx
import numpy as np

from mlx_vlm.models.granite4_vision.config import ModelConfig, TextConfig, VisionConfig
from mlx_vlm.models.granite4_vision.language import MLP, LanguageModel, SharedMLP


def tiny_text_config_moehybrid():
    return TextConfig(
        model_type="granitemoehybrid",
        vocab_size=64,
        hidden_size=16,
        num_hidden_layers=2,
        intermediate_size=32,
        shared_intermediate_size=32,
        num_attention_heads=4,
        num_key_value_heads=2,
        embedding_multiplier=1.0,
        attention_multiplier=0.25,
        residual_multiplier=1.0,
        logits_scaling=1.0,
    )


def tiny_text_config_granite():
    return TextConfig(
        model_type="granite",
        vocab_size=64,
        hidden_size=16,
        num_hidden_layers=2,
        intermediate_size=32,
        shared_intermediate_size=32,
        num_attention_heads=4,
        num_key_value_heads=2,
        embedding_multiplier=1.0,
        attention_multiplier=0.25,
        residual_multiplier=1.0,
        logits_scaling=1.0,
    )


def tiny_vision_config():
    return VisionConfig(
        hidden_size=16,
        num_hidden_layers=1,
        num_attention_heads=4,
        intermediate_size=32,
        image_size=32,
        patch_size=16,
    )


def tiny_model_config(text_config):
    return ModelConfig(
        text_config=text_config,
        vision_config=tiny_vision_config(),
        image_token_index=63,
        vocab_size=64,
    )


# --- MLP variant selection ---


def test_use_shared_mlp_granitemoehybrid():
    cfg = tiny_text_config_moehybrid()
    assert cfg.use_shared_mlp is True


def test_use_shared_mlp_granite():
    cfg = tiny_text_config_granite()
    assert cfg.use_shared_mlp is False


def test_transformer_block_uses_shared_mlp_for_moehybrid():
    from mlx_vlm.models.granite4_vision.language import TransformerBlock

    cfg = tiny_text_config_moehybrid()
    block = TransformerBlock(cfg)
    assert hasattr(block, "shared_mlp")
    assert not hasattr(block, "mlp")


def test_transformer_block_uses_mlp_for_granite():
    from mlx_vlm.models.granite4_vision.language import TransformerBlock

    cfg = tiny_text_config_granite()
    block = TransformerBlock(cfg)
    assert hasattr(block, "mlp")
    assert not hasattr(block, "shared_mlp")


# --- Forward pass shape checks ---


def test_language_model_forward_moehybrid():
    cfg = tiny_text_config_moehybrid()
    model = LanguageModel(cfg)
    model.eval()

    input_ids = mx.zeros((1, 4), dtype=mx.int32)
    out = model(input_ids)
    assert out.logits.shape == (1, 4, cfg.vocab_size)
    assert np.isfinite(np.array(out.logits)).all()


def test_language_model_forward_granite():
    cfg = tiny_text_config_granite()
    model = LanguageModel(cfg)
    model.eval()

    input_ids = mx.zeros((1, 4), dtype=mx.int32)
    out = model(input_ids)
    assert out.logits.shape == (1, 4, cfg.vocab_size)
    assert np.isfinite(np.array(out.logits)).all()


# --- MLP unit tests ---


def test_shared_mlp_output_shape():
    cfg = tiny_text_config_moehybrid()
    mlp = SharedMLP(cfg)
    x = mx.random.normal((2, 5, cfg.hidden_size))
    out = mlp(x)
    assert out.shape == (2, 5, cfg.hidden_size)


def test_mlp_output_shape():
    cfg = tiny_text_config_granite()
    mlp = MLP(cfg)
    x = mx.random.normal((2, 5, cfg.hidden_size))
    out = mlp(x)
    assert out.shape == (2, 5, cfg.hidden_size)
