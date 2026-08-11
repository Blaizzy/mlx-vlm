import mlx.core as mx
import mlx.nn as nn
import numpy as np
from PIL import Image

from mlx_vlm.models.cache import KVCache, RotatingKVCache
from mlx_vlm.models.muse_glimmer import Model, ModelConfig, TextConfig, VisionConfig
from mlx_vlm.models.muse_glimmer.language import (
    CenteredRMSNorm,
    RMSNormNoScale,
    TextRotaryEmbedding,
    _scale_queries,
)
from mlx_vlm.models.muse_glimmer.muse_glimmer import masked_scatter
from mlx_vlm.models.muse_glimmer.processing_muse_glimmer import (
    MuseGlimmerImageProcessor,
    smart_resize,
)
from mlx_vlm.models.muse_glimmer.vision import _window_index, apply_rotary, rotate_half


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


def test_centered_rms_norm_preserves_transformers_fp32_operation_order():
    norm = CenteredRMSNorm(4, eps=1e-6)
    norm.weight = (mx.arange(4, dtype=mx.float32) * 0.031 - 0.2).astype(mx.bfloat16)
    inputs = (
        (mx.arange(4, dtype=mx.float32) * 0.37 - 1.13).reshape(1, 4).astype(mx.bfloat16)
    )

    inputs32 = inputs.astype(mx.float32)
    variance = mx.mean(mx.square(inputs32), axis=-1, keepdims=True)
    expected = inputs32 * mx.rsqrt(variance + norm.eps)
    expected = expected * (1.0 + norm.weight.astype(mx.float32))
    expected = expected.astype(inputs.dtype)
    output = norm(inputs)
    mx.eval(output, expected)

    assert bool(mx.array_equal(output, expected).item())


def test_unscaled_rms_norm_preserves_transformers_fp32_operation_order():
    norm = RMSNormNoScale(eps=1e-6)
    inputs = mx.array([[0.1, 1.0, 3.0, 9.0]], dtype=mx.bfloat16)

    inputs32 = inputs.astype(mx.float32)
    mean_squared = mx.mean(mx.square(inputs32), axis=-1, keepdims=True) + norm.eps
    expected = (inputs32 * mx.power(mean_squared, -0.5)).astype(inputs.dtype)
    output = norm(inputs)
    mx.eval(output, expected)

    assert bool(mx.array_equal(output, expected).item())


def test_query_scale_uses_transformers_fp32_operation_order():
    inputs = mx.array([0.1, 1.0, 9.0], dtype=mx.bfloat16)
    output = _scale_queries(inputs, 3.87)
    expected = (inputs.astype(mx.float32) * 3.87).astype(inputs.dtype)
    bf16_scalar_result = inputs * 3.87
    mx.eval(output, expected, bf16_scalar_result)

    assert bool(mx.array_equal(output, expected).item())
    assert not bool(mx.array_equal(output, bf16_scalar_result).item())


def test_text_rotary_embedding_uses_fp32_frequencies():
    inputs = (mx.arange(24, dtype=mx.float32) / 7).reshape(1, 2, 3, 4)
    inputs = inputs.astype(mx.bfloat16)
    rope = TextRotaryEmbedding(dim=4, base=10000.0)

    positions = mx.arange(2, 5, dtype=mx.float32)
    frequencies = 1.0 / (10000.0 ** (mx.arange(0, 4, 2, dtype=mx.float32) / 4))
    angles = positions[:, None] * frequencies[None]
    angles = mx.concatenate([angles, angles], axis=-1)
    cos = mx.cos(angles).astype(inputs.dtype)[None, None]
    sin = mx.sin(angles).astype(inputs.dtype)[None, None]
    rotated = mx.concatenate([-inputs[..., 2:], inputs[..., :2]], axis=-1)
    expected = inputs * cos + rotated * sin
    output = rope(inputs, offset=2)
    mx.eval(output, expected)

    assert bool(mx.array_equal(output, expected).item())


def test_compiled_vision_rotary_matches_fp32_reference():
    q = mx.arange(48, dtype=mx.bfloat16).reshape(1, 2, 2, 12) / 48
    k = q + 0.25
    cos = mx.cos(mx.arange(24, dtype=mx.float32).reshape(2, 12) / 24)
    sin = mx.sin(mx.arange(24, dtype=mx.float32).reshape(2, 12) / 24)

    actual_q, actual_k = apply_rotary(q, k, cos, sin)
    expanded_cos = cos[None, :, None, :]
    expanded_sin = sin[None, :, None, :]
    q32, k32 = q.astype(mx.float32), k.astype(mx.float32)
    expected_q = (q32 * expanded_cos + rotate_half(q32) * expanded_sin).astype(q.dtype)
    expected_k = (k32 * expanded_cos + rotate_half(k32) * expanded_sin).astype(k.dtype)
    mx.eval(actual_q, actual_k, expected_q, expected_k)

    assert bool(mx.array_equal(actual_q, expected_q).item())
    assert bool(mx.array_equal(actual_k, expected_k).item())


def test_window_index_skips_identity_reordering():
    identity_index, identity_cu = _window_index([[1, 4, 4]], window_size=4)
    assert identity_index is None
    assert identity_cu == [0, 16]

    window_index, window_cu = _window_index([[1, 4, 8]], window_size=4)
    mx.eval(window_index)
    assert window_cu == [0, 16, 32]
    assert sorted(window_index.tolist()) == list(range(32))


def test_masked_scatter_stays_in_mlx():
    inputs = mx.arange(12).reshape(1, 3, 4)
    mask = mx.broadcast_to(mx.array([[[False], [True], [False]]]), inputs.shape)
    source = mx.array([[20, 21, 22, 23]])
    output = masked_scatter(inputs, mask, source)
    mx.eval(output)
    assert output.tolist() == [[[0, 1, 2, 3], [20, 21, 22, 23], [8, 9, 10, 11]]]


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


def test_quantization_keeps_embedding_normalization_outside_embedding():
    config = tiny_config()
    config.text_config.hidden_size = 32
    model = Model(config)

    nn.quantize(
        model.language_model,
        group_size=32,
        bits=4,
        class_predicate=lambda path, module: (
            hasattr(module, "to_quantized") and module.weight.shape[-1] % 32 == 0
        ),
    )

    text_model = model.language_model.model
    assert isinstance(text_model.embed_tokens, nn.QuantizedEmbedding)
    assert isinstance(text_model.embed_norm, RMSNormNoScale)
    assert isinstance(
        text_model.layers[0].self_attn.q_proj,
        nn.QuantizedLinear,
    )

    input_ids = mx.array([[1, 2, 3]])
    normalized = text_model.embed_norm(text_model.embed_tokens(input_ids))
    multimodal_path = model.get_input_embeddings(input_ids).inputs_embeds
    direct_path = text_model(input_ids)
    embedded_path = text_model(None, inputs_embeds=normalized)
    mx.eval(normalized, multimodal_path, direct_path, embedded_path)

    np.testing.assert_allclose(multimodal_path, normalized, rtol=1e-5, atol=1e-7)
    np.testing.assert_allclose(direct_path, embedded_path, rtol=1e-5, atol=1e-7)


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
