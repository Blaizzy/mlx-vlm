import copy
import json
from dataclasses import asdict
from unittest.mock import patch

import mlx.core as mx
import numpy as np
import pytest
from mlx import nn
from mlx.utils import tree_flatten
from transformers import AutoProcessor

from mlx_vlm.models import prism_hadamard_qwen35, qwen3_5
from mlx_vlm.models.prism_hadamard_qwen35 import Model, ModelConfig
from mlx_vlm.models.prism_hadamard_qwen35.prism_hadamard_qwen35 import (
    HadamardQuantizedEmbedding,
    HadamardQuantizedLinear,
    hadamard_transform,
)
from mlx_vlm.prompt_utils import apply_chat_template
from mlx_vlm.utils import get_model_and_args, load_model


def _rotation(width, block):
    h = np.ones((1, 1), dtype=np.float32)
    while len(h) < block:
        h = np.block([[h, h], [h, -h]])
    return np.kron(np.eye(width // block, dtype=np.float32), h / np.sqrt(block))


@pytest.mark.parametrize("dtype", [mx.float16, mx.float32])
@pytest.mark.parametrize("block", [512, 1024])
def test_signed_hadamard_matches_dense_reference(dtype, block):
    rng = np.random.default_rng(7)
    width = 2 * block
    x = mx.array(rng.normal(size=(2, 3, width)), dtype=dtype)
    signs = mx.array(rng.choice([-1.0, 1.0], size=width), dtype=mx.float32)
    rotation = _rotation(width, block)
    expected = (np.asarray(x).astype(np.float32) * np.asarray(signs)) @ rotation
    actual = hadamard_transform(x, block, signs)
    restored = hadamard_transform(actual, block, signs, inverse=True)
    assert actual.dtype == dtype
    np.testing.assert_allclose(actual, expected, atol=2e-3, rtol=1e-3)
    np.testing.assert_allclose(restored, x, atol=2e-3, rtol=1e-3)


@pytest.mark.parametrize("block", [0, 512])
def test_packed_projections_and_embedding_match_unrotated_dense_weights(block):
    mx.random.seed(3)
    width, rows = 1024, 8
    layer = HadamardQuantizedLinear(width, rows, block)
    layer.weight, layer.scales, layer.biases = mx.quantize(
        mx.random.normal((rows, width)), group_size=128, bits=2
    )
    if block:
        layer.signs = mx.where(mx.arange(width) % 3 == 0, -1.0, 1.0)
    dense = np.asarray(
        mx.dequantize(layer.weight, layer.scales, layer.biases, group_size=128, bits=2)
    )
    rotation = _rotation(width, block) if block else np.eye(width)
    signs = np.asarray(layer.signs) if block else np.ones(width)
    unrotated = (dense @ rotation) * signs
    x = mx.random.normal((2, 3, width))
    np.testing.assert_allclose(
        layer(x), np.asarray(x) @ unrotated.T, atol=1e-4, rtol=1e-4
    )

    embedding = HadamardQuantizedEmbedding(width, rows, block)
    embedding.load_weights(tree_flatten(layer.parameters()))
    indices = mx.array([[0, 5], [2, 0]])
    expected = (dense.astype(np.float16).astype(np.float32) @ rotation) * signs
    actual = embedding(indices)
    assert actual.dtype == mx.float16
    np.testing.assert_allclose(
        actual, expected[np.asarray(indices)], atol=2e-3, rtol=1e-3
    )
    np.testing.assert_allclose(embedding.as_linear(x), layer(x), atol=1e-5)


@pytest.fixture
def packed_checkpoint(tmp_path):
    text = qwen3_5.TextConfig(
        model_type="qwen3_5_text",
        hidden_size=512,
        intermediate_size=1024,
        linear_num_value_heads=4,
        linear_num_key_heads=1,
        linear_key_head_dim=128,
        linear_value_head_dim=128,
        linear_conv_kernel_dim=4,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=1,
        head_dim=128,
        rms_norm_eps=1e-6,
        vocab_size=64,
        max_position_embeddings=1024,
        full_attention_interval=2,
        rope_parameters={
            "type": "default",
            "rope_theta": 10000000,
            "partial_rotary_factor": 0.25,
            "mrope_section": [4, 6, 6],
        },
    )
    vision = qwen3_5.VisionConfig(
        depth=1,
        hidden_size=32,
        intermediate_size=64,
        out_hidden_size=512,
        num_heads=4,
        patch_size=16,
        num_position_embeddings=16,
    )
    source = qwen3_5.Model(
        qwen3_5.ModelConfig(
            text_config=text,
            vision_config=vision,
            model_type="qwen3_5",
            image_token_id=60,
            video_token_id=61,
            vision_start_token_id=62,
            vision_end_token_id=63,
        )
    )
    weights = dict(tree_flatten(source.parameters()))
    records = []
    for path, module in source.language_model.named_modules():
        if not isinstance(module, (nn.Linear, nn.Embedding)):
            continue
        if path.endswith(("in_proj_a", "in_proj_b")):
            continue
        key = "language_model." + path
        raw = weights.pop(key + ".weight")
        arrays = mx.quantize(raw, group_size=128, bits=2)
        for suffix, value in zip(("weight", "scales", "biases"), arrays):
            weights[key + "." + suffix] = value
        weights[key + ".signs"] = mx.where(mx.arange(raw.shape[1]) % 3 == 0, -1.0, 1.0)
        records.append(
            {
                "path": path,
                "block": 512,
                "embedding": isinstance(module, nn.Embedding),
                "dtype": "float16",
            }
        )
    config = asdict(source.config)
    config.update(
        model_type="prism_hadamard_qwen35",
        schema_version=2,
        modules=records,
        tensor_namespace="mlx-vlm-qwen3_5",
        base_model_type="qwen3_5",
        gdn_activation_layout="grouped",
        quantization={"bits": 2, "group_size": 128, "mode": "affine"},
    )
    (tmp_path / "config.json").write_text(json.dumps(config))
    mx.save_safetensors(str(tmp_path / "model.safetensors"), weights)
    return tmp_path, config, weights


def test_standard_loader_preserves_packed_weights_and_cached_decode(packed_checkpoint):
    path, config, weights = packed_checkpoint
    module, model_type = get_model_and_args(config)
    assert module is prism_hadamard_qwen35
    assert model_type == "prism_hadamard_qwen35"
    model = load_model(path)
    lm = model.language_model
    assert isinstance(lm.model.embed_tokens, HadamardQuantizedEmbedding)
    assert isinstance(lm.lm_head, HadamardQuantizedLinear)
    assert isinstance(model.vision_tower.blocks[0].attn.qkv, nn.Linear)
    assert isinstance(lm.layers[0].linear_attn.in_proj_a, nn.Linear)
    for key, value in tree_flatten(model.parameters()):
        assert mx.array_equal(value, weights[key]).item(), key

    tokens = mx.array([[1, 2, 3, 4]])
    full = lm(tokens).logits
    cache = lm.make_cache()
    prefill = lm(tokens[:, :3], cache=cache).logits
    decode = lm(tokens[:, 3:], cache=cache).logits
    mx.eval(full, prefill, decode)
    assert full.shape == (1, 4, 64)
    assert mx.all(mx.isfinite(full)).item()
    np.testing.assert_allclose(prefill, full[:, :3], atol=3e-3, rtol=3e-3)
    np.testing.assert_allclose(decode, full[:, 3:], atol=3e-3, rtol=3e-3)

    # The same entry point merges image features and supplies multimodal RoPE.
    image_tokens = mx.array([[1, 62, 60, 63, 2]])
    features = model.get_input_embeddings(
        image_tokens,
        mx.zeros((4, 3 * 2 * 16 * 16)),
        image_grid_thw=mx.array([[1, 2, 2]]),
    )
    result = lm(
        image_tokens,
        inputs_embeds=features.inputs_embeds,
        position_ids=features.position_ids,
    ).logits
    assert result.shape == (1, 5, 64)
    assert mx.all(mx.isfinite(result)).item()


def test_processor_and_multimodal_prompt_registration(packed_checkpoint):
    path, config, _ = packed_checkpoint
    with patch.object(
        prism_hadamard_qwen35.Qwen3VLProcessor, "from_pretrained"
    ) as factory:
        assert AutoProcessor.from_pretrained(path) is factory.return_value
        factory.assert_called_once()
    messages = apply_chat_template(
        None, config, "Describe.", num_images=2, video="clip.mp4", return_messages=True
    )
    assert [entry["type"] for entry in messages[0]["content"]] == [
        "image",
        "image",
        "video",
        "text",
    ]


@pytest.mark.parametrize(
    "field,value",
    [
        ("schema_version", 1),
        ("tensor_namespace", "gguf"),
        ("gdn_activation_layout", "ungrouped"),
        ("modules", []),
        ("quantization", {"bits": 4, "group_size": 128, "mode": "affine"}),
    ],
)
def test_rejects_incompatible_pack_config(packed_checkpoint, field, value):
    _, config, _ = packed_checkpoint
    config[field] = value
    with pytest.raises(ValueError):
        ModelConfig.from_dict(config)


@pytest.mark.parametrize("mutation", ["duplicate", "kind", "dtype", "block", "path"])
def test_rejects_invalid_module_manifest(packed_checkpoint, mutation):
    _, config, _ = packed_checkpoint
    config = copy.deepcopy(config)
    if mutation == "duplicate":
        config["modules"].append(config["modules"][0])
    else:
        field, value = {
            "kind": ("embedding", not config["modules"][0]["embedding"]),
            "dtype": ("dtype", "bfloat16"),
            "block": ("block", 2048),
            "path": ("path", "missing"),
        }[mutation]
        config["modules"][0][field] = value
    with pytest.raises(ValueError):
        Model(ModelConfig.from_dict(config))


@pytest.mark.parametrize("missing", [True, False])
def test_rejects_missing_or_invalid_signs(packed_checkpoint, missing):
    _, config, weights = packed_checkpoint
    key = "language_model.model.embed_tokens.signs"
    if missing:
        del weights[key]
    else:
        weights[key] = mx.zeros_like(weights[key])
    with pytest.raises(ValueError, match="sign"):
        Model(ModelConfig.from_dict(config)).sanitize(weights)
