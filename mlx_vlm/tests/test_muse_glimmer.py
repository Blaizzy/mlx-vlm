import mlx.core as mx
import mlx.nn as nn
import numpy as np
from PIL import Image

from mlx_vlm.models.cache import KVCache, RotatingKVCache
from mlx_vlm.models.muse_glimmer import Model, ModelConfig, TextConfig, VisionConfig
from mlx_vlm.models.muse_glimmer.language import CenteredRMSNorm, NormedEmbedding
from mlx_vlm.models.muse_glimmer.processing_muse_glimmer import (
    MuseGlimmerImageProcessor,
    smart_resize,
)


def tiny_config():
    text = TextConfig(
        vocab_size=64,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=4,
        max_position_embeddings=128,
        sliding_window=8,
        layer_types=["sliding_attention", "full_attention"],
        layer_rope_theta=[10000.0, 0],
    )
    vision = VisionConfig(
        hidden_size=8,
        intermediate_size=16,
        num_attention_heads=2,
        num_hidden_layers=2,
        patch_size=2,
        patch_temporal=2,
        merge_size=2,
        pos_emb_height=4,
        pos_emb_width=4,
        max_position_embeddings=16,
        layer_types=["window_attention", "full_attention"],
    )
    return ModelConfig(
        text_config=text,
        vision_config=vision,
        image_token_id=7,
        video_token_id=6,
        out_hidden_size=32,
        projector_hidden_size=16,
    )


def test_config_matches_local_global_patterns():
    text = TextConfig(num_hidden_layers=8)
    assert text.layer_types == [
        "sliding_attention",
        "sliding_attention",
        "sliding_attention",
        "full_attention",
        "sliding_attention",
        "sliding_attention",
        "sliding_attention",
        "full_attention",
    ]
    assert text.layer_rope_theta[3] == 0
    assert text.layer_rope_theta[7] == 0

    vision = VisionConfig(num_hidden_layers=6)
    assert vision.layer_types == [
        "window_attention",
        "window_attention",
        "window_attention",
        "full_attention",
        "window_attention",
        "full_attention",
    ]


def test_centered_rms_norm_uses_one_plus_checkpoint_weight():
    norm = CenteredRMSNorm(2, eps=0.0)
    norm.weight = mx.array([0.0, 1.0])
    output = norm(mx.array([[3.0, 4.0]]))
    expected_base = np.array([[3.0, 4.0]]) / np.sqrt((9.0 + 16.0) / 2.0)
    np.testing.assert_allclose(
        np.asarray(output), expected_base * np.array([[1.0, 2.0]]), rtol=1e-6
    )


def test_image_processor_patch_layout_and_grid():
    assert smart_resize(28, 56, patch_size=28, max_tokens=4096) == (28, 56)
    processor = MuseGlimmerImageProcessor(
        patch_size=14,
        temporal_patch_size=2,
        merge_size=2,
        max_image_tokens=4096,
    )
    output = processor(Image.new("RGB", (28, 28), (255, 0, 0)))
    assert output["pixel_values"].shape == (4, 1176)
    assert output["image_grid_thw"].tolist() == [[1, 2, 2]]
    # Temporal copies are adjacent within each flattened patch.
    first = output["pixel_values"][0].reshape(2, 3, 14, 14)
    np.testing.assert_array_equal(first[0], first[1])


def test_tiny_text_and_multimodal_forward():
    mx.random.seed(0)
    model = Model(tiny_config())

    text_output = model(mx.array([[1, 2, 3]]))
    mx.eval(text_output.logits)
    assert text_output.logits.shape == (1, 3, 64)
    assert bool(mx.isfinite(text_output.logits).all().item())

    # A 2x2 raw patch grid is pixel-shuffled into one visual token.
    pixels = mx.zeros((4, 2 * 3 * 2 * 2), dtype=mx.float32)
    grid = mx.array([[1, 2, 2]])
    embeddings = model.get_input_embeddings(
        mx.array([[1, 7, 2]]), pixels, image_grid_thw=grid
    ).inputs_embeds
    mx.eval(embeddings)
    assert embeddings.shape == (1, 3, 16)
    assert bool(mx.isfinite(embeddings).all().item())


def test_quantization_preserves_normalized_embedding():
    config = tiny_config()
    config.text_config.hidden_size = 32
    model = Model(config)
    embedding = model.language_model.model.embed_tokens

    assert isinstance(embedding, NormedEmbedding)
    assert not model.quant_predicate("language_model.model.embed_tokens", embedding)
    assert model.quant_predicate(
        "language_model.model.layers.0.self_attn.q_proj",
        model.language_model.model.layers[0].self_attn.q_proj,
    )

    nn.quantize(
        model.language_model,
        group_size=32,
        bits=4,
        class_predicate=lambda path, module: (
            hasattr(module, "to_quantized")
            and model.quant_predicate(path, module)
            and module.weight.shape[-1] % 32 == 0
        ),
    )

    assert isinstance(model.language_model.model.embed_tokens, NormedEmbedding)
    assert isinstance(
        model.language_model.model.layers[0].self_attn.q_proj,
        nn.QuantizedLinear,
    )


def test_cache_matches_layer_attention_type():
    caches = Model(tiny_config()).make_cache()
    assert isinstance(caches[0], RotatingKVCache)
    assert caches[0].max_size == 8
    assert isinstance(caches[1], KVCache)


def test_checkpoint_prefixes_are_sanitized_to_native_modules():
    model = Model(tiny_config())
    weights = model.sanitize(
        {
            "model.language_model.layers.0.self_attn.q_proj.weight": mx.zeros((16, 16)),
            "model.vision_tower.ln_pre.weight": mx.ones((8,)),
            "lm_head.weight": mx.zeros((64, 16)),
        }
    )
    assert "language_model.model.layers.0.self_attn.q_proj.weight" in weights
    assert "vision_tower.ln_pre.weight" in weights
    assert "language_model.lm_head.weight" in weights
