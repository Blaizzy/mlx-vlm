"""Step 5 checkpoint arithmetic, sparse caching and real NVFP4 offload parity."""

import json
from dataclasses import asdict

import mlx.core as mx
import numpy as np
import pytest
from mlx.utils import tree_flatten, tree_map

from mlx_vlm.models.step5_vl import Model, ModelConfig
from mlx_vlm.models.step5_vl.language import Attention, round_e4m3
from mlx_vlm.moe_offload import repack
from mlx_vlm.tests.test_models import model_case
from mlx_vlm.utils import load_model


def tiny_config():
    return ModelConfig.from_dict(model_case("step5_vl")["config"])


@pytest.mark.parametrize("dtype", [mx.float32, mx.bfloat16])
def test_sparse_chunked_and_decode_parity(dtype):
    mx.random.seed(12)
    model = Model(tiny_config())
    model.update(tree_map(lambda p: p.astype(dtype), model.parameters()))
    tokens = mx.array([[1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21]])
    full = model(tokens).logits
    cache = model.language_model.make_cache()
    chunks = []
    for start, stop in [(0, 3), (3, 7), (7, 10), (10, 11)]:
        chunks.append(model(tokens[:, start:stop], cache=cache).logits)
    joined = mx.concatenate(chunks, axis=1)
    assert mx.all(mx.isfinite(joined)).item()
    assert mx.allclose(full, joined, atol=0.02 if dtype == mx.bfloat16 else 1e-5).item()
    # Later input cannot influence earlier sparse-region selection.
    changed = model(tokens.at[:, 7:].add(30)).logits
    assert mx.allclose(full[:, :7], changed[:, :7], atol=1e-5).item()


def test_sparse_mask_ties_current_region_and_causality():
    attention = Attention(tiny_config().text_config, 1)
    attention.sparse_indexer_w.weight = mx.zeros_like(attention.sparse_indexer_w.weight)
    mask = np.asarray(attention._sparse_mask(mx.ones((1, 11, 64)), None, 0))
    # Zero scores choose the earliest two completed regions for every provider.
    expected = np.zeros((11, 11), dtype=bool)
    for query in range(11):
        for key in range(query + 1):
            expected[query, key] = key // 2 < 2 or key // 2 == query // 2
    np.testing.assert_array_equal(mask, np.broadcast_to(expected, (1, 4, 11, 11)))


def test_e4m3_rounding_midpoints_and_subnormals():
    values = mx.array([0, 2**-10, 3 * 2**-10, 1.0625, 1.1875, -1.1875, 448])
    np.testing.assert_array_equal(
        np.asarray(round_e4m3(values)), [0, 0, 2**-8, 1, 1.25, -1.25, 448]
    )


def test_zero_centered_norms_are_shifted_once():
    model = Model(tiny_config())
    raw = {
        "model.norm.weight": mx.zeros(64),
        "model.layers.0.input_layernorm.weight": mx.zeros(64),
        "model.layers.1.self_attn.sparse_indexer_q_norm.weight": mx.zeros(16),
        "model.layers.1.self_attn.sparse_indexer_k_norm.weight": mx.ones(16),
    }
    sanitized = model.sanitize(raw)
    assert all(mx.all(v == 1).item() for v in sanitized.values())
    again = model.sanitize(sanitized)
    assert all(mx.array_equal(again[k], v).item() for k, v in sanitized.items())


@pytest.mark.parametrize("dtype", [mx.float32, mx.bfloat16])
@pytest.mark.parametrize("scalar_scale", [False, True])
def test_raw_nvfp4_repack_inference_and_eviction(tmp_path, dtype, scalar_scale):
    mx.random.seed(42)
    config = tiny_config()
    model = Model(config)
    raw = {}
    for name, value in tree_flatten(model.parameters()):
        value = value.astype(dtype)
        name = name.removeprefix("language_model.")
        if ".mlp.switch_mlp." in name:
            prefix = name.removesuffix(".weight").replace(".mlp.switch_mlp.", ".moe.")
            scale = (
                mx.array(0.137)
                if scalar_scale
                else mx.array([0.137, 0.348, 0.779, 0.29])
            )
            divisor = scale if scalar_scale else scale[:, None, None]
            packed, block = mx.quantize(value / divisor, mode="nvfp4")
            raw[prefix + ".weight"] = packed.view(mx.uint8)
            raw[prefix + ".weight_scale"] = block
            raw[prefix + ".weight_scale_2"] = scale
            raw[prefix + ".input_scale"] = mx.ones_like(scale)
        else:
            name = name.replace(".mlp.gate.gate.", ".moe.gate.")
            name = name.replace(".mlp.gate.router_bias", ".moe.router_bias")
            name = name.replace(".mlp.share_expert.", ".share_expert.")
            if (
                name.startswith("model.")
                and name.endswith("norm.weight")
                and not name.endswith("sparse_indexer_k_norm.weight")
            ):
                value = value.astype(mx.float32) - 1
            raw[name] = value
    build, out = tmp_path / "raw", tmp_path / "offload"
    build.mkdir()
    saved = asdict(config)
    saved["quantization_config"] = {"quant_method": "modelopt", "quant_algo": "NVFP4"}
    (build / "config.json").write_text(json.dumps(saved))
    mx.save_safetensors(str(build / "model.safetensors"), raw)
    resident = load_model(build)
    repack(str(build), str(out))
    offloaded = load_model(out, expert_cache_gb=0.00001)
    tokens = mx.array([[1, 3, 5, 7, 9, 11, 13]])
    expected = resident(tokens).logits
    actual = offloaded(tokens).logits
    assert mx.all(mx.isfinite(actual)).item()
    assert mx.allclose(
        actual, expected, atol=0.02 if dtype == mx.bfloat16 else 2e-5
    ).item()
    cache = offloaded.language_model.make_cache()
    resident_cache = resident.language_model.make_cache()
    offloaded(tokens[:, :2], cache=cache)
    resident(tokens[:, :2], cache=resident_cache)
    decoded = offloaded(tokens[:, 2:3], cache=cache).logits
    reference = resident(tokens[:, 2:3], cache=resident_cache).logits
    assert mx.allclose(
        decoded, reference, atol=0.02 if dtype == mx.bfloat16 else 2e-5
    ).item()
    assert offloaded.moe_offload_store.stats()["evictions"] > 0
    expert = mx.load(str(out / "experts/layer_0001.safetensors"))
    assert "e0.gate_proj.weight_scale_2" in expert
    assert expert["e0.gate_proj.weight"].dtype == mx.uint32
