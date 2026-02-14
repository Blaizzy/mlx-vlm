import mlx.core as mx

from mlx_vlm.models import minicpmo


def _tiny_config():
    text_config = minicpmo.TextConfig(
        model_type="minicpmo",
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        rms_norm_eps=1e-6,
        vocab_size=256,
        num_key_value_heads=4,
        head_dim=16,
        rope_theta=10000.0,
        max_position_embeddings=2048,
    )
    vision_config = minicpmo.VisionConfig(
        model_type="siglip_vision_model",
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_channels=3,
        image_size=28,
        patch_size=14,
    )
    return minicpmo.ModelConfig(
        text_config=text_config,
        vision_config=vision_config,
        query_num=4,
    )


def test_minicpmo_config_from_root_fields():
    cfg = {
        "model_type": "minicpmo",
        "hidden_size": 4096,
        "intermediate_size": 12288,
        "num_hidden_layers": 36,
        "num_attention_heads": 32,
        "rms_norm_eps": 1e-6,
        "vocab_size": 151936,
        "num_key_value_heads": 8,
        "head_dim": 128,
        "rope_theta": 1000000.0,
        "max_position_embeddings": 40960,
        "query_num": 64,
        "vision_config": {
            "model_type": "siglip",
            "hidden_size": 1152,
            "intermediate_size": 4304,
            "num_hidden_layers": 27,
            "num_attention_heads": 16,
            "num_channels": 3,
            "image_size": 448,
            "patch_size": 14,
        },
    }
    model_config = minicpmo.ModelConfig.from_dict(cfg)
    assert model_config.text_config.hidden_size == 4096
    assert model_config.vision_config.model_type == "siglip_vision_model"
    assert model_config.query_num == 64


def test_minicpmo_sanitize_key_mapping_and_qkv_split():
    model = minicpmo.Model(_tiny_config())
    weights = {
        "llm.model.embed_tokens.weight": mx.zeros((10, 10)),
        "llm.lm_head.weight": mx.zeros((10, 10)),
        "vpm.embeddings.patch_embedding.weight": mx.zeros((8, 3, 14, 14)),
        "resampler.attn.in_proj_weight": mx.zeros((192, 64)),
        "resampler.attn.in_proj_bias": mx.zeros((192,)),
        "apm.conv1.weight": mx.zeros((1, 1)),
    }

    sanitized = model.sanitize(weights)
    assert "language_model.model.embed_tokens.weight" in sanitized
    assert "language_model.lm_head.weight" in sanitized
    assert "vision_tower.embeddings.patch_embedding.weight" in sanitized
    assert "apm.conv1.weight" not in sanitized

    assert "resampler.attn.q_proj.weight" in sanitized
    assert "resampler.attn.k_proj.weight" in sanitized
    assert "resampler.attn.v_proj.weight" in sanitized
    assert "resampler.attn.q_proj.bias" in sanitized
    assert "resampler.attn.k_proj.bias" in sanitized
    assert "resampler.attn.v_proj.bias" in sanitized


def test_minicpmo_sanitize_audio_conv_layout():
    model = minicpmo.Model(_tiny_config())
    weights = {
        "apm.conv1.weight": mx.zeros((8, 80, 3)),
        "apm.conv2.weight": mx.zeros((8, 8, 3)),
    }

    sanitized = model.sanitize(weights)
    assert sanitized["audio_tower.conv1.weight"].shape == (8, 3, 80)
    assert sanitized["audio_tower.conv2.weight"].shape == (8, 3, 8)
