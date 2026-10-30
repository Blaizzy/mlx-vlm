import json
import math
from dataclasses import asdict

import mlx.core as mx
import numpy as np
import pytest
from mlx.utils import tree_flatten

from mlx_vlm.models.mistral_large4.config import TextConfig
from mlx_vlm.models.mistral_large4.language import (
    LanguageModel,
    LatentCache,
    SoftmaxScaling,
    rotate,
)


def tiny_config(**kwargs):
    config = dict(
        hidden_size=32,
        intermediate_size=64,
        moe_intermediate_size=32,
        num_hidden_layers=8,
        num_attention_heads=2,
        num_key_value_heads=2,
        kv_lora_rank=16,
        qk_rope_head_dim=8,
        qk_nope_head_dim=8,
        v_head_dim=16,
        n_routed_experts=8,
        index_n_heads=2,
        index_head_dim=16,
        index_topk=4,
        vocab_size=50,
        kv_source_layers=[0, 1, 2, 3, 4, 5, 6, 3],
        indexer_types=[
            "full",
            "full",
            "shared",
            "full",
            "full",
            "shared",
            "full",
            "shared",
        ],
    )
    config.update(kwargs)
    return TextConfig(**config)


def test_mistral_large4_cached_dsa_and_shared_kv_match_full_sequence():
    mx.random.seed(17)
    model = LanguageModel(tiny_config())
    model.eval()
    ids = mx.array([[1, 2, 7, 5, 9, 13, 11], [2, 4, 6, 8, 10, 12, 14]])
    expected = model(ids).logits
    cache = model.make_cache()
    chunks = [
        model(ids[:, a:b], cache=cache).logits for a, b in [(0, 3), (3, 5), (5, 7)]
    ]
    actual = mx.concatenate(chunks, axis=1)
    np.testing.assert_allclose(
        np.array(actual), np.array(expected), atol=3e-6, rtol=3e-5
    )
    assert all(c.offset == 7 for c in cache)
    assert cache[7].keys is None  # This layer reuses layer 3's compressed KV.
    assert cache[3].keys.shape == (2, 1, 7, 16)
    assert cache[0].index_keys.shape == (2, 7, 16)
    assert cache[2].index_keys is None  # Indexer sharing carries only indices.


def test_mistral_large4_cache_restore_and_trim():
    model = LanguageModel(tiny_config())
    model.eval()
    ids = mx.array([[1, 2, 3, 4, 5, 6]])
    cache = model.make_cache()
    model(ids, cache=cache)
    restored = [LatentCache.from_state(c.state, c.meta_state) for c in cache]
    for c in restored:
        assert c.trim(2) == 2
    actual = model(ids[:, 4:], cache=restored).logits
    expected = model(ids).logits[:, 4:]
    np.testing.assert_allclose(
        np.array(actual), np.array(expected), atol=3e-6, rtol=3e-5
    )


def test_mistral_large4_rotary_pair_layout_and_position_scaling():
    x = mx.array([[[[1.0, 2.0, 3.0, 4.0]]]])
    cos, sin = mx.array([[[[0.0, 1.0]]]]), mx.array([[[[1.0, 0.0]]]])
    np.testing.assert_array_equal(
        np.array(rotate(x, cos, sin)), [[[[-2.0, 3.0, 1.0, 4.0]]]]
    )
    scaling = SoftmaxScaling(2)
    scaling.s = mx.array([0.25, -0.5])
    scaling.b = mx.array([1.0, 2.0])
    positions = mx.array([[0, 3]])
    actual = scaling(mx.ones((1, 2, 2, 3)), positions)
    expected = np.array(
        [[[1.0, 1.0 + 0.25 * math.log(4)], [2.0, 2.0 - 0.5 * math.log(4)]]]
    )[..., None]
    np.testing.assert_allclose(
        np.array(actual), np.broadcast_to(expected, (1, 2, 2, 3)), atol=1e-6
    )


def test_mistral_large4_nvfp4_global_scales_preserve_weight_values():
    # Nibble 2 is FP4 1.0, E4M3 byte 56 is 1.0; global scales differ by expert.
    from mlx_vlm.models.switch_layers import QuantizedSwitchLinear
    from mlx_vlm.quantization.nvfp4 import ScaledQuantizedSwitchLinear

    linear = ScaledQuantizedSwitchLinear(
        QuantizedSwitchLinear(
            16, 4, 2, bias=False, group_size=16, bits=4, mode="nvfp4"
        ),
        mx.array([0.25, 2.0], mx.float32),
        "weight_scale_2",
    )
    linear.weight = mx.full((2, 4, 2), 0x22222222, mx.uint32)
    linear.scales = mx.full((2, 4, 1), 56, mx.uint8)
    x = mx.ones((2, 1, 16), mx.bfloat16)
    actual = linear(x, mx.array([0, 1]))
    np.testing.assert_array_equal(
        np.array(actual.astype(mx.float32)), [[[4.0] * 4], [[32.0] * 4]]
    )


def test_mistral_large4_partitions_allow_three_ranks_and_reject_wrong_shards():
    class Group:
        def __init__(self, rank):
            self._rank = rank

        def rank(self):
            return self._rank

        def size(self):
            return 3

    model = LanguageModel(tiny_config())
    original = model.layers[1].feed_forward.experts.w1.weight
    model.shard(Group(1))
    moe = model.layers[1].feed_forward
    assert (moe.start, moe.end) == (2, 5)
    np.testing.assert_array_equal(
        np.array(moe.experts.w1.weight), np.array(original[2:5])
    )
    with pytest.raises(ValueError, match="partition"):
        model.shard(Group(2))


def test_mistral_large4_invalid_kv_sharing_fails_at_configuration():
    with pytest.raises(ValueError, match="rotary layout"):
        tiny_config(kv_source_layers=[0, 1, 2, 3, 4, 5, 6, 0])
    with pytest.raises(ValueError, match="Invalid KV source"):
        tiny_config(kv_source_layers=[0, 1, 2, 3, 4, 5, 6, 8])


@pytest.mark.parametrize("partition", [None, (2, 5)])
def test_mistral_large4_native_config_keeps_generation_defaults(
    tmp_path, monkeypatch, partition
):
    from mlx_vlm.models.mistral_large4 import config
    from mlx_vlm.utils import load_config

    (tmp_path / "params.json").write_text(
        json.dumps({"lerope": True, "ssmax_scaling": True, "dsa": {"topk": 4}})
    )
    (tmp_path / "generation_config.json").write_text(
        json.dumps({"eos_token_id": [2, 3], "temperature": 0.7})
    )
    monkeypatch.setattr(
        config,
        "config_from_params",
        lambda params: {
            "model_type": "mistral_large4",
            "eos_token_id": 2,
            "text_config": {},
        },
    )
    if partition is not None:
        (tmp_path / "rank.json").write_text(
            json.dumps({"expert_start": partition[0], "expert_end": partition[1]})
        )
    result = load_config(tmp_path)
    assert result["model_type"] == "mistral_large4"
    assert result["eos_token_id"] == [2, 3]
    assert result["temperature"] == 0.7
    if partition is not None:
        assert result["text_config"] == {
            "expert_start": partition[0],
            "expert_end": partition[1],
        }


def test_mistral_large4_shared_loader_native_weights_and_roundtrip(tmp_path):
    from mlx_vlm.models.mistral_large4 import Model, ModelConfig, VisionConfig
    from mlx_vlm.quantization.nvfp4 import ScaledQuantizedSwitchLinear
    from mlx_vlm.utils import load_model

    mx.random.seed(31)
    args = ModelConfig(
        text_config=tiny_config(),
        vision_config=VisionConfig(
            hidden_size=16,
            intermediate_size=32,
            num_hidden_layers=1,
            num_attention_heads=2,
            head_dim=8,
            image_size=8,
            patch_size=2,
        ),
        spatial_merge_size=2,
    )
    reference = Model(args)
    native = {}
    direct = {
        "language_model.model.embed_tokens.weight": "tok_embeddings.weight",
        "language_model.model.norm.weight": "norm.weight",
        "language_model.model.rotary_emb.alpha": "lerope.alpha",
        "language_model.lm_head.weight": "output.weight",
        "multi_modal_projector.norm.weight": "pre_mm_projector_norm.weight",
        "multi_modal_projector.patch_merger.merging_layer.weight": "patch_merger.merging_layer.weight",
        "multi_modal_projector.linear_1.weight": "vision_language_adapter.w_in.weight",
        "multi_modal_projector.linear_2.weight": "vision_language_adapter.w_out.weight",
    }
    expected = {}
    for key, value in tree_flatten(reference.parameters()):
        name = direct.get(key, key)
        if key.startswith("language_model.model.layers."):
            name = key.removeprefix("language_model.model.")
            if not name.startswith("layers.0."):
                name = name.replace(".feed_forward.", ".")
        elif key.startswith("vision_tower.vision_model."):
            name = key.replace("vision_tower.vision_model.", "vision_encoder.")
            for dest, source in (
                ("q_proj", "wq"),
                ("k_proj", "wk"),
                ("v_proj", "wv"),
                ("o_proj", "wo"),
                ("gate_proj", "w1"),
                ("down_proj", "w2"),
                ("up_proj", "w3"),
            ):
                name = name.replace(f".{dest}.", f".{source}.")
        value = value.astype(mx.bfloat16)
        if ".experts." in key:
            prefix, projection = name.split(".experts.")
            decoded = []
            for expert in range(args.text_config.n_routed_experts):
                global_scale = float(expert + 1)
                packed, scales = mx.quantize(
                    value[expert] / (expert + 1), group_size=16, bits=4, mode="nvfp4"
                )
                base = f"{prefix}.experts.{expert}.{projection.removesuffix('.weight')}"
                native[f"{base}.weight"] = packed.view(mx.uint8)
                native[f"{base}.weight_scale"] = scales
                native[f"{base}.weight_scale_2"] = mx.array([global_scale])
                native[f"{base}.input_scale"] = mx.array(1.0)
                decoded.append(
                    (
                        mx.dequantize(
                            packed, scales, group_size=16, bits=4, mode="nvfp4"
                        )
                        * (expert + 1)
                    ).astype(mx.bfloat16)
                )
            expected[key] = mx.stack(decoded)
        else:
            if name.endswith(".wq.weight"):
                # Per-tensor FP8 alongside NVFP4 routed experts.
                scale = mx.array(0.5, mx.float32)
                packed = mx.to_fp8(value.astype(mx.float32) / scale)
                native[name] = packed
                native[name.removesuffix(".weight") + ".weight_scale"] = scale
                value = (mx.from_fp8(packed, dtype=mx.float32) * scale).astype(
                    mx.bfloat16
                )
            else:
                native[name] = (
                    value.transpose(0, 3, 1, 2)
                    if name.endswith("patch_conv.weight")
                    else value
                )
            expected[key] = value
    native["mtp.extra.weight"] = mx.ones((2, 2))
    reference.load_weights(list(expected.items()))
    reference.eval()
    config = asdict(args)
    config["text_config"]["nvfp4"] = True
    config["quantization_config"] = {
        "quant_method": "modelopt",
        "quant_algo": "MIXED_PRECISION",
    }
    (tmp_path / "config.json").write_text(json.dumps(config))
    mx.save_safetensors(str(tmp_path / "model.safetensors"), native)
    model = load_model(tmp_path)
    assert isinstance(
        model.language_model.layers[1].feed_forward.experts.w1,
        ScaledQuantizedSwitchLinear,
    )
    ids = mx.array([[1, 2, 3, 4]])
    actual = model(ids).logits
    np.testing.assert_allclose(
        np.array(actual.astype(mx.float32)),
        np.array(reference(ids).logits.astype(mx.float32)),
        atol=0.06,
        rtol=0.06,
    )
    cache = model.language_model.make_cache()
    cached = mx.concatenate(
        [model(ids[:, :2], cache=cache).logits, model(ids[:, 2:], cache=cache).logits],
        axis=1,
    )
    np.testing.assert_allclose(
        np.array(cached.astype(mx.float32)),
        np.array(actual.astype(mx.float32)),
        atol=0.06,
        rtol=0.06,
    )
    saved = dict(tree_flatten(model.parameters()))
    sanitized = model.sanitize(dict(saved))
    assert saved.keys() == sanitized.keys()
    for key in saved:
        np.testing.assert_array_equal(
            np.array(saved[key].astype(mx.float32)),
            np.array(sanitized[key].astype(mx.float32)),
        )
    # Native staging uses global expert IDs, including nonzero starting ranks.
    config["text_config"].update(expert_start=2, expert_end=5)
    (tmp_path / "config.json").write_text(json.dumps(config))
    partition = load_model(tmp_path)
    for key, value in tree_flatten(partition.parameters()):
        if ".experts." in key:
            np.testing.assert_array_equal(
                np.array(value.astype(mx.float32)),
                np.array(saved[key][2:5].astype(mx.float32)),
            )
    config["text_config"].update(expert_start=0, expert_end=8)
    config["quantization_config"] = config["quantization"] = {
        "group_size": 16,
        "bits": 4,
        "mode": "nvfp4",
    }
    (tmp_path / "config.json").write_text(json.dumps(config))
    mx.save_safetensors(str(tmp_path / "model.safetensors"), saved)
    restored = load_model(tmp_path)
    np.testing.assert_array_equal(
        np.array(restored(ids).logits.astype(mx.float32)),
        np.array(actual.astype(mx.float32)),
    )


def test_modelopt_fp8_block_scales_use_shared_conversion():
    from mlx_vlm.quantization.checkpoint import transform_quantized_weights

    values = mx.arange(256 * 256).reshape(256, 256) % 17 - 8
    packed = mx.to_fp8(values.astype(mx.float32))
    scales = mx.array([0.25, 0.5, 1.0, 2.0]).reshape(2, 1, 2, 1)
    expected = (values.reshape(2, 128, 2, 128) * scales).reshape(256, 256)
    weights, _ = transform_quantized_weights(
        {"linear.weight": packed, "linear.weight_scale": scales},
        {"quant_method": "modelopt", "quant_algo": "MIXED_PRECISION"},
    )
    assert set(weights) == {"linear.weight"}
    np.testing.assert_array_equal(
        np.array(weights["linear.weight"].astype(mx.float32)), np.array(expected)
    )
