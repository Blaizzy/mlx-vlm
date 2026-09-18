"""Offline LLaDA-Image checks; no checkpoint downloads or PyTorch dependency."""

import importlib
import json
from types import SimpleNamespace
from unittest.mock import Mock

import mlx.core as mx
import numpy as np
import pytest
from mlx.utils import tree_flatten
from numpy.testing import assert_allclose, assert_array_equal

from mlx_vlm.generate.edit_image import ImageEditRequest
from mlx_vlm.generate.image import ImageGenerationRequest
from mlx_vlm.models.llada2_moe.config import ModelConfig as TextConfig
from mlx_vlm.models.llada_image.conditioning import (
    QueryFormer,
    TextProjection,
    format_prompt,
    query_attention_mask,
)
from mlx_vlm.models.llada_image.config import (
    LLaDAImageTransformerConfig,
    validate_dimensions,
)
from mlx_vlm.models.llada_image.model import LLaDAImageGenerationModel
from mlx_vlm.models.llada_image.pipeline import LLaDAImagePipeline
from mlx_vlm.models.llada_image.scheduler import LLaDAImageScheduler
from mlx_vlm.models.llada_image.text_encoder import LLaDAImageTextEncoder, TextAttention
from mlx_vlm.models.llada_image.transformer import (
    LLaDAImageTransformer,
    padded_positions,
    sanitize_transformer_weights,
)
from mlx_vlm.models.llada_image.weights import (
    apply_weights,
    load_safetensors,
    restore_fp8_weights,
    sanitize_text_encoder_weights,
)


def tiny_config():
    return LLaDAImageTransformerConfig(
        dim=24,
        n_heads=2,
        n_layers=2,
        n_refiner_layers=1,
        in_channels=8,
        cap_feat_dim=12,
        semantic_feat_dim=16,
        axes_dims=(4, 4, 4),
    )


def _linear(x, layer):
    return x @ np.array(layer.weight).T + np.array(layer.bias)


def _norm(x, eps=1e-6, layer_norm=False):
    if layer_norm:
        x = x - x.mean(axis=-1, keepdims=True)
    return x / np.sqrt((x * x).mean(axis=-1, keepdims=True) + eps)


def _gelu(x):
    return 0.5 * x * (1 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * x**3)))


def _attention(q, k, v, heads, mask=None):
    batch, length, dim = q.shape
    q, k, v = [
        x.reshape(batch, -1, heads, dim // heads).transpose(0, 2, 1, 3)
        for x in (q, k, v)
    ]
    scores = q @ k.transpose(0, 1, 3, 2) / np.sqrt(dim // heads)
    if mask is not None:
        scores = np.where(mask[:, None, None], scores, -np.inf)
    probabilities = np.exp(scores - scores.max(axis=-1, keepdims=True))
    probabilities /= probabilities.sum(axis=-1, keepdims=True)
    return (probabilities @ v).transpose(0, 2, 1, 3).reshape(batch, length, dim)


def test_queryformer_matches_numpy_reference():
    config = {
        "hidden_size": 16,
        "num_queries": 3,
        "num_attention_heads": 2,
        "num_hidden_layers": 2,
        "intermediate_size": 32,
    }
    model = QueryFormer(config)
    model.meta_queries = mx.random.normal(model.meta_queries.shape)
    for block in model.query_blocks:
        block.cross_attn.in_proj_weight = mx.random.normal((48, 16)) * 0.1
    x = np.random.default_rng(7).normal(size=(1, 5, 16)).astype(np.float32)
    mask = np.array([[True, True, True, False, False]])
    expected = np.array(model.meta_queries)[None]
    for block in model.query_blocks:
        queries = _norm(expected, layer_norm=True)
        context = _norm(x, layer_norm=True)
        wq, wk, wv = np.split(np.array(block.cross_attn.in_proj_weight), 3)
        bq, bk, bv = np.split(np.array(block.cross_attn.in_proj_bias), 3)
        attended = _attention(
            queries @ wq.T + bq, context @ wk.T + bk, context @ wv.T + bv, 2, mask
        )
        normalized = _norm(
            queries + _linear(attended, block.cross_attn.out_proj), layer_norm=True
        )
        expected = normalized + _linear(
            _gelu(_linear(normalized, block.mlp.fc1)), block.mlp.fc2
        )
    actual = model(mx.array(x), mx.array(mask))
    assert_allclose(np.array(actual), expected, atol=2e-6, rtol=2e-5)
    x[:, 3:] *= 100
    assert_allclose(
        np.array(model(mx.array(x), mx.array(mask))), np.array(actual), atol=2e-6
    )


def test_projection_matches_numpy_reference():
    model = TextProjection(
        {
            "hidden_size": 16,
            "projection_dim": 12,
            "num_attention_heads": 2,
            "num_hidden_layers": 2,
            "intermediate_size": 32,
        }
    )
    x = np.random.default_rng(5).normal(size=(1, 5, 16)).astype(np.float32)
    expected = x.copy()
    for block in model.layers:
        normed = _norm(expected)
        q = _norm(_linear(normed, block.self_attn.q_proj).reshape(1, 5, 2, 8)).reshape(
            1, 5, 16
        )
        k = _norm(_linear(normed, block.self_attn.k_proj).reshape(1, 5, 2, 8)).reshape(
            1, 5, 16
        )
        v = _linear(normed, block.self_attn.v_proj)
        expected = expected + _linear(_attention(q, k, v, 2), block.self_attn.out_proj)
        expected = expected + _linear(
            _gelu(_linear(_norm(expected), block.mlp.fc1)), block.mlp.fc2
        )
    expected = _linear(expected, model.projector)
    assert_allclose(np.array(model(mx.array(x))), expected, atol=2e-6, rtol=2e-5)


def test_query_mask_and_prompt_template():
    mask = np.array(query_attention_mask(2, 3))[0, 0]
    assert_array_equal(mask[:2, :2], True)
    assert_array_equal(mask[:2, 2:], False)
    assert_array_equal(mask[2:], True)
    assert (
        format_prompt("  a cat  ")
        == "<role>HUMAN</role> Generate an image: a cat\n<role>ASSISTANT</role>\n<IMAGE1>"
    )
    assert "Generate an image.\n" in format_prompt(None)
    assert "Generate an image: \n" in format_prompt("")


def test_rotary_padding_uses_zero_positions():
    caption, image = (np.array(x)[0] for x in padded_positions(7, 3, 4))
    assert_array_equal(caption[:7, 0], np.arange(1, 8))
    assert_array_equal(caption[7:], 0)
    assert_array_equal(image[:12, 0], 33)
    assert_array_equal(image[:12, 1], np.repeat(np.arange(3), 4))
    assert_array_equal(image[:12, 2], np.tile(np.arange(4), 3))
    assert_array_equal(image[12:], 0)


@pytest.mark.parametrize("dtype", [mx.float32, mx.bfloat16])
def test_transformer_forward_preserves_dtype_and_has_no_norm_weights(dtype):
    model = LLaDAImageTransformer(tiny_config())
    model.set_dtype(dtype)
    output = model(
        mx.ones((1, 8, 3, 4), dtype), mx.array([0.5], dtype), mx.ones((1, 7, 12), dtype)
    )
    assert output.shape == (1, 8, 3, 4)
    assert output.dtype == dtype
    assert bool(mx.isfinite(output).all())
    assert not any("norm" in key for key, _ in tree_flatten(model.parameters()))


def test_stacked_experts_load_strictly_without_lm_head():
    config = TextConfig(
        vocab_size=32,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=1,
        num_experts=4,
        num_experts_per_tok=2,
        n_group=2,
        topk_group=1,
        moe_intermediate_size=8,
    )
    model = LLaDAImageTextEncoder(config)
    raw = {}
    for key, value in tree_flatten(model.parameters()):
        key = "model.language_model." + key
        if ".switch_mlp." in key:
            key = key.replace(".switch_mlp.", ".experts.").removesuffix(".weight")
        raw[key] = value
    raw["model.lm_head.weight"] = mx.zeros((32, 16))
    weights = sanitize_text_encoder_weights(raw)
    apply_weights(model, weights)
    assert weights["layers.1.mlp.switch_mlp.gate_proj.weight"].shape == (4, 8, 16)
    output = model(mx.array([[1, 2, 3]]), mask=query_attention_mask(2, 1))
    assert bool(mx.isfinite(output).all())
    weights.pop("norm.weight")
    with pytest.raises(ValueError):
        apply_weights(model, weights)


def test_transformer_weight_mapping_retains_editing_conditioning():
    raw = {
        "all_x_embedder.1-1.weight": mx.zeros((24, 8)),
        "all_final_layer.1-1.adaLN_modulation.1.weight": mx.zeros((24, 24)),
        "t_embedder.mlp.2.bias": mx.zeros((24,)),
        "semantic_embedder.1.weight": mx.zeros((24, 16)),
        "sigvq_pad_token": mx.zeros((1, 24)),
        "unexpected.weight": mx.zeros((1,)),
    }
    assert set(sanitize_transformer_weights(raw)) == {
        "x_embedder.weight",
        "final_layer.adaLN_modulation.0.weight",
        "t_embedder.linear2.bias",
        "semantic_embedder.1.weight",
        "sigvq_pad_token",
        "unexpected.weight",
    }


def test_shard_index_missing_file_fails_before_inference(tmp_path):
    (tmp_path / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {"a": "missing.safetensors"}})
    )
    with pytest.raises((ValueError, RuntimeError, OSError)):
        load_safetensors(tmp_path)


def test_scheduler_matches_flow_matching_equations():
    scheduler = LLaDAImageScheduler({"use_uniform_sigmas": True})
    assert_allclose(np.array(scheduler.sigmas(4)), [1, 0.9, 0.75, 0.5, 0], atol=1e-7)
    sample, prediction, noise = [
        mx.array(x) for x in ([1.0, 2.0], [2.0, 1.0], [-1.0, 1.0])
    ]
    actual = scheduler.step(prediction, sample, 0.75, 0.5, noise=noise)
    assert_allclose(np.array(actual), [-0.75, 1.125])
    # The terminal stochastic step must contain no residual noise.
    assert_allclose(
        np.array(scheduler.step(prediction, sample, 0.5, 0, noise=noise)), [0, 1.5]
    )
    scheduler.stochastic_sampling = False
    assert_allclose(
        np.array(scheduler.step(prediction, sample, 0.75, 0.5)), [0.5, 1.75]
    )
    with pytest.raises(ValueError):
        scheduler.sigmas(0)


@pytest.mark.parametrize("size", [(0, 16), (16, 15), (17, 32)])
def test_invalid_image_dimensions(size):
    with pytest.raises(ValueError):
        validate_dimensions(*size)


def test_metadata_dispatch_works_for_arbitrary_repository_name(tmp_path, monkeypatch):
    image = importlib.import_module("mlx_vlm.generate.image")
    (tmp_path / "model_index.json").write_text(
        json.dumps({"_class_name": "LLaDAImagePipeline"})
    )
    assert (
        image.image_generation_model_class(str(tmp_path)) is LLaDAImageGenerationModel
    )
    assert image.is_image_generation_model(str(tmp_path))
    monkeypatch.setattr(
        image, "_resolve_image_model_path", lambda *args, **kwargs: tmp_path
    )
    factory = Mock(return_value="loaded")
    monkeypatch.setattr(LLaDAImageGenerationModel, "from_model_id", factory)
    assert image.load_image_generation_model("someone/arbitrary-mirror") == "loaded"
    factory.assert_called_once_with("someone/arbitrary-mirror", model_path=tmp_path)


def test_public_generation_defaults_and_modes(tmp_path):
    pipeline = SimpleNamespace(
        model_path=tmp_path,
        scheduler=LLaDAImageScheduler({"use_uniform_sigmas": True}),
        generate_array=Mock(return_value=mx.zeros((16, 32, 3), dtype=mx.uint8)),
        tokenize=lambda _: [1, 2, 3],
    )
    model = LLaDAImageGenerationModel(pipeline, "mirror/checkpoint")
    result = model.generate(ImageGenerationRequest("a cat", width=32, height=16))
    assert (result.steps, result.guidance, result.family, result.prompt_tokens) == (
        4,
        1.0,
        "llada_image",
        3,
    )
    assert result.array.shape == (16, 32, 3)
    assert pipeline.generate_array.call_args.kwargs["negative_prompt"] is None
    result = model.generate(
        ImageGenerationRequest("a cat", extra={"generation_mode": "vq"})
    )
    assert result.metadata["generation_mode"] == "vq"
    assert pipeline.generate_array.call_args.kwargs["generation_mode"] == "vq"
    with pytest.raises(ValueError, match="edit_image"):
        model.generate(
            ImageGenerationRequest("a cat", extra={"generation_mode": "editing"})
        )


def test_pipeline_staging_cache_and_seed_reproducibility(tmp_path, monkeypatch):
    module = importlib.import_module("mlx_vlm.models.llada_image.pipeline")
    monkeypatch.setattr(module, "validate_model_layout", lambda path: path)
    monkeypatch.setattr(module, "read_config", lambda _: {"use_uniform_sigmas": True})
    tokenizer = SimpleNamespace(encode=lambda text, **kwargs: [1, 2, len(text) % 31])
    monkeypatch.setattr(
        module.AutoTokenizer, "from_pretrained", lambda *a, **k: tokenizer
    )
    events = []

    def text_encoder(_):
        events.append("text")
        mx.random.seed(101)
        model = LLaDAImageTextEncoder(
            TextConfig(
                vocab_size=32,
                hidden_size=16,
                intermediate_size=32,
                num_hidden_layers=1,
                num_attention_heads=2,
                num_key_value_heads=1,
            )
        )
        model.set_dtype(mx.bfloat16)
        return model

    def queryformer(_):
        model = QueryFormer(
            {
                "hidden_size": 16,
                "num_queries": 3,
                "num_attention_heads": 2,
                "num_hidden_layers": 1,
                "intermediate_size": 32,
            }
        )
        model.set_dtype(mx.bfloat16)
        return model

    def projection(_):
        model = TextProjection(
            {
                "hidden_size": 16,
                "projection_dim": 12,
                "num_attention_heads": 2,
                "num_hidden_layers": 1,
                "intermediate_size": 32,
            }
        )
        model.set_dtype(mx.bfloat16)
        return model

    def transformer(_):
        events.append("transformer")
        assert (
            pipeline.text_encoder
            is pipeline.queryformer
            is pipeline.text_projection
            is None
        )
        mx.random.seed(102)
        model = LLaDAImageTransformer(
            LLaDAImageTransformerConfig(
                dim=24,
                n_heads=2,
                n_layers=1,
                n_refiner_layers=1,
                in_channels=128,
                cap_feat_dim=12,
                axes_dims=(4, 4, 4),
            )
        )
        model.set_dtype(mx.bfloat16)
        return model

    def vae(_):
        events.append("vae")
        assert pipeline.transformer is None
        return SimpleNamespace(
            decode_packed_latents=lambda x: mx.repeat(
                mx.repeat(x[:, :3], 16, axis=2), 16, axis=3
            )
        )

    for name, loader in (
        ("text_encoder", text_encoder),
        ("queryformer", queryformer),
        ("text_projection", projection),
        ("transformer", transformer),
        ("vae", vae),
    ):
        monkeypatch.setattr(module, "load_" + name, loader)
    pipeline = LLaDAImagePipeline(tmp_path)
    assert events == []
    first = pipeline.generate_array(
        "a cat", width=16, height=16, steps=2, seed=42, guidance=2
    )
    assert events == ["text", "transformer", "vae"]
    assert set(pipeline.prompt_cache) == {"a cat", None}
    again = pipeline.generate_array(
        "a cat", width=16, height=16, steps=2, seed=42, guidance=2
    )
    assert_array_equal(np.array(first), np.array(again))
    assert events.count("text") == 1
    pipeline.generate_array("a different prompt", width=16, height=16, steps=1, seed=42)
    assert events.count("text") == 2
    assert len(pipeline.prompt_cache) == 2


def test_base_schedule_is_not_silently_loaded_as_turbo():
    base = LLaDAImageScheduler({"shift": 1.0, "stochastic_sampling": False})
    assert (base.default_steps, base.default_guidance) == (50, 5)
    assert not base.stochastic_sampling
    # Values from the reference's float64 schedule, rounded to FP32 for Diffusers.
    assert_allclose(
        np.array(base.sigmas(4)),
        [0.99989225, 0.86497145, 0.65944823, 0.39515334, 0],
        atol=1e-7,
    )


def test_base_defaults_and_native_edit_dispatch(tmp_path):
    edit = importlib.import_module("mlx_vlm.generate.edit_image")
    (tmp_path / "model_index.json").write_text(
        json.dumps({"_class_name": "LLaDAImagePipeline"})
    )
    assert edit.image_edit_model_class(str(tmp_path)) is LLaDAImageGenerationModel
    assert edit.is_image_edit_model(str(tmp_path))
    pipeline = SimpleNamespace(
        model_path=tmp_path,
        scheduler=LLaDAImageScheduler({"shift": 1, "stochastic_sampling": False}),
        edit_array=Mock(return_value=mx.zeros((64, 96, 3), mx.uint8)),
        tokenize=lambda _: [1],
    )
    model = LLaDAImageGenerationModel(pipeline, "renamed/base")
    result = model.edit(ImageEditRequest("make it blue", ("source.png",)))
    assert (result.width, result.height, result.steps, result.guidance) == (
        96,
        64,
        50,
        5,
    )
    assert result.variant == "llada-image"
    assert result.metadata["generation_mode"] == "editing"
    with pytest.raises(ValueError, match="exactly one"):
        model.edit(ImageEditRequest("edit", ("one.png", "two.png")))


@pytest.mark.parametrize("mode", ["vq", "editing"])
def test_semantic_and_source_conditioning_affect_prediction(mode):
    model = LLaDAImageTransformer(tiny_config())
    x = mx.random.normal((1, 8, 3, 4))
    caption = mx.random.normal((1, 7, 12))
    semantic = mx.random.normal((1, 5, 16))
    time = mx.array([0.8])
    kwargs = {"source_latents": x} if mode == "editing" else {}
    actual = model(x, time, caption, semantic=semantic, **kwargs)
    no_semantic = model(x, time, caption, semantic=semantic[:, :0], **kwargs)
    assert bool(mx.isfinite(actual).all())
    assert float(mx.max(mx.abs(actual - no_semantic))) > 1e-4
    if mode == "editing":
        changed = model(x, time, caption, semantic=semantic, source_latents=-x)
        assert float(mx.max(mx.abs(actual - changed))) > 1e-4


def test_fp8_block_scales_and_fused_transformer_weights():
    shape = (129, 130)
    raw = mx.full(shape, 56, mx.uint8)  # E4M3 1.0
    scales = mx.array([[1, 2], [4, 8]], mx.float32)
    config = {
        "quantization_config": {
            "quant_method": "fp8",
            "fmt": "e4m3",
            "weight_block_size": [128, 128],
        }
    }
    restored = restore_fp8_weights(
        {"linear.weight": raw, "linear.weight_scale_inv": scales}, config
    )
    weight = restored["linear.weight"]
    assert weight.dtype == mx.bfloat16
    expected = np.repeat(np.repeat(np.array(scales), 128, axis=0), 128, axis=1)[
        :129, :130
    ]
    assert_array_equal(np.array(weight.astype(mx.float32)), expected)
    packed = mx.arange(24).reshape(6, 4)
    split = sanitize_transformer_weights(
        {
            "layers.0.attention.to_qkv.weight": packed,
            "layers.0.feed_forward.w13.weight": packed,
        }
    )
    assert_array_equal(
        np.array(split["layers.0.attention.to_k.weight"]), np.array(packed[2:4])
    )
    assert_array_equal(
        np.array(split["layers.0.feed_forward.w3.weight"]), np.array(packed[3:])
    )
    with pytest.raises(ValueError, match="missing its block scales"):
        restore_fp8_weights({"linear.weight": raw}, config)


def test_fp8_expert_scales_preserve_expert_dimension():
    key = "model.language_model.layers.1.mlp.experts.gate_proj"
    raw = mx.full((2, 128, 256), 56, mx.uint8)
    scales = mx.array([[[1, 2]], [[4, 8]]], mx.float32)
    config = {
        "llada_fp8_experts": {
            "enabled": True,
            "format": "e4m3fn",
            "weight_granularity": "per_expert_2d_block",
            "weight_block_size": [128, 128],
        }
    }
    restored = restore_fp8_weights({key: raw, key + "_scale": scales}, config)
    weights = sanitize_text_encoder_weights(restored)
    actual = weights["layers.1.mlp.switch_mlp.gate_proj.weight"]
    assert actual.shape == raw.shape
    assert_array_equal(
        np.array(actual[:, 0, ::128].astype(mx.float32)), [[1, 2], [4, 8]]
    )
    with pytest.raises(ValueError, match="block shapes"):
        restore_fp8_weights({key: raw, key + "_scale": scales[:1]}, config)


def test_sigvq_half_pixel_positions_and_token_projection():
    from mlx_vlm.models.llada_image.sigvq import SigVQ, resize_positions

    positions = mx.arange(16, dtype=mx.float32)[:, None]
    actual = resize_positions(positions, 2, 2)
    assert_array_equal(np.array(actual[0, :, 0]), [2.5, 4.5, 10.5, 12.5])
    config = {
        "image_size": 16,
        "patch_size": 4,
        "hidden_size": 8,
        "intermediate_size": 16,
        "num_hidden_layers": 1,
        "num_attention_heads": 2,
        "codebook_size": 16,
        "codebook_embed_dim": 8,
        "semantic_embed_dim": 12,
    }
    model = SigVQ(config)
    pixels = mx.random.normal((1, 8, 12, 3))
    ids = model.encode(pixels)
    assert ids.shape == (1, 6)
    assert bool(mx.all(ids < 16))
    features = model(pixels=pixels)
    assert features.shape == (1, 6, 12)
    assert_allclose(np.array(features), np.array(model(token_ids=ids)))


def test_vq_sampler_block_boundaries_cfg_positions_and_token_count():
    from mlx_vlm.models.llada_image.vq_sampling import generate_vq_tokens

    calls = []

    def encoder(ids, *, mask, position_ids):
        calls.append((np.array(ids), np.array(mask), np.array(position_ids)))
        return mx.zeros((*ids.shape, 1))

    def head(hidden):
        logits = mx.zeros((*hidden.shape[:2], 8))
        logits[0, :, 2] = 1
        logits[1, :, 3] = 1
        return logits

    result = generate_vq_tokens(
        encoder,
        head,
        [1] * 37,
        [1] * 30,
        35,
        image_token_offset=0,
        codebook_size=8,
        threshold=0.999,
    )
    assert_array_equal(np.array(result), np.full((1, 35), 2))
    ids, mask, positions = calls[0]
    assert ids.shape == (2, 64)
    assert_array_equal(mask[0, 0, :32, 32:], False)
    assert_array_equal(mask[0, 0, 32:, :], True)
    assert_array_equal(mask[1, :, :, :7], False)
    assert_array_equal(positions[1], np.maximum(np.arange(64) - 7, 0))
    assert calls[-1][0].shape == (2, 96)
    assert_array_equal(calls[-1][0][0, :37], 1)


def test_text_attention_explicit_positions_and_padding_match_unpadded_input():
    config = TextConfig(hidden_size=32, num_attention_heads=4, num_key_value_heads=2)
    attention = TextAttention(config, 0)
    x = mx.random.normal((1, 5, 32))
    expected = attention(x)
    padded = mx.pad(x, [(0, 0), (3, 0), (0, 0)])
    positions = mx.maximum(mx.arange(8) - 3, 0)[None]
    mask = (mx.arange(8) >= 3)[None, None, None, :]
    actual = attention(padded, mask=mask, position_ids=positions)
    assert_allclose(np.array(actual[:, 3:]), np.array(expected), atol=1e-6)


def _bf16(x):
    bits = np.asarray(x, dtype=np.float32).view(np.uint32)
    rounded = bits + 0x7FFF + ((bits >> 16) & 1)
    return (rounded & np.uint32(0xFFFF0000)).view(np.float32)


def test_text_rotary_matches_reference_bf16_rounding():
    config = TextConfig(hidden_size=32, num_attention_heads=4, num_key_value_heads=2)
    attention = TextAttention(config, 0)
    rng = np.random.default_rng(9)
    query = _bf16(rng.normal(size=(1, 4, 80, 8)))
    key = _bf16(rng.normal(size=(1, 2, 80, 8)))
    positions = np.arange(80, dtype=np.float32)[:, None]
    frequencies = config.rope_theta ** (-np.arange(0, 4, 2, dtype=np.float32) / 4)
    angles = np.tile(positions * frequencies, (1, 2))
    cos, sin = _bf16(np.cos(angles)), _bf16(np.sin(angles))

    def rotate(x):
        first, second = np.split(x[..., :4], 2, axis=-1)
        rotated = _bf16(
            _bf16(x[..., :4] * cos)
            + _bf16(np.concatenate([-second, first], axis=-1) * sin)
        )
        return np.concatenate([rotated, x[..., 4:]], axis=-1)

    actual = attention._apply_rope(
        mx.array(query).astype(mx.bfloat16), mx.array(key).astype(mx.bfloat16)
    )
    for output, expected in zip(actual, (rotate(query), rotate(key))):
        assert_array_equal(np.array(output.astype(mx.float32)), expected)
