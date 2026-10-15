"""Nemotron 3.5 Super VL and Bolt checkpoint and inference contracts."""

import json
from types import SimpleNamespace
from unittest.mock import Mock

import mlx.core as mx
import mlx.nn as nn
import numpy as np
import pytest
from mlx.utils import tree_flatten
from PIL import Image
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from transformers import PreTrainedTokenizerFast

from mlx_vlm.generate.video import processor_handles_video
from mlx_vlm.models.nemotron_h_nano_omni import Model, ModelConfig
from mlx_vlm.models.nemotron_h_nano_omni.nemotron_h_nano_omni import (
    compute_retention_mask,
)
from mlx_vlm.prompt_utils import apply_chat_template
from mlx_vlm.utils import get_model_and_args, load_model, load_processor, prepare_inputs


@pytest.fixture
def config():
    # The released config has no num_hidden_layers, and uses tagged infinity.
    return {
        "model_type": "nemotron_h_omni",
        "img_context_token_id": 18,
        "projector_hidden_size": 32,
        "vit_hidden_size": 16,
        "ps_version": "v2",
        "video_pruning_rate": 0.7,
        "llm_config": {
            "model_type": "nemotron_h",
            "vocab_size": 128,
            "hidden_size": 32,
            "intermediate_size": 32,
            "max_position_embeddings": 2048,
            "num_attention_heads": 4,
            "num_key_value_heads": 2,
            "attention_bias": False,
            "head_dim": 8,
            "mamba_num_heads": 4,
            "mamba_head_dim": 16,
            "mamba_proj_bias": False,
            "ssm_state_size": 8,
            "conv_kernel": 4,
            "n_groups": 1,
            "mlp_bias": False,
            "layer_norm_epsilon": 1e-5,
            "use_bias": False,
            "use_conv_bias": True,
            "layers_block_type": ["mamba", "moe", "attention"],
            "time_step_limit": [0.0, {"__float__": "Infinity"}],
            "num_nextn_predict_layers": 1,
            "mtp_layers_block_type": ["attention", "moe"],
            "moe_latent_size": 32,
            "moe_intermediate_size": 32,
            "moe_shared_expert_intermediate_size": 32,
            "n_routed_experts": 2,
            "n_shared_experts": 1,
            "num_experts_per_tok": 1,
            "n_group": 1,
            "topk_group": 1,
            "norm_topk_prob": True,
            "routed_scaling_factor": 1.0,
        },
        "vision_config": {
            "hidden_size": 16,
            "intermediate_size": 32,
            "num_hidden_layers": 1,
            "num_attention_heads": 4,
            "image_size": 32,
            "patch_size": 16,
            "max_resolution": 64,
            "video_temporal_patch_size": 2,
            "args": {
                "teachers": [{"name": 0}, {"name": 1}, {"name": 2}],
                "cls_token_per_teacher": True,
                "register_multiple": 10,
            },
        },
    }


@pytest.fixture
def bolt_config(config):
    config["vision_final_layernorm"] = False
    text = config["llm_config"]
    text["model_type"] = "nemotron_h_puzzle"
    text["num_nextn_predict_layers"] = 0
    text["mtp_layers_block_type"] = []
    text.pop("layers_block_type")
    # Bolt stores these only on individual MoE blocks.
    for key in ("n_routed_experts", "moe_intermediate_size", "num_experts_per_tok"):
        text.pop(key)
    text["block_configs"] = [
        {"block_type": "mamba"},
        {
            "block_type": "moe",
            "n_routed_experts": 2,
            "moe_intermediate_size": 32,
            "num_experts_per_tok": 1,
        },
        {"block_type": "attention"},
        {
            "block_type": "moe",
            "n_routed_experts": 3,
            "moe_intermediate_size": 64,
            "num_experts_per_tok": 2,
        },
    ]
    return config


def test_puzzle_layer_configs(bolt_config):
    from mlx_vlm.models.nemotron_h.config import ModelConfig as TextConfig

    raw = bolt_config["llm_config"]
    original = json.dumps(raw)
    # The block list takes precedence, as in NVIDIA's Puzzle config.
    parsed = TextConfig.from_dict({**raw, "layers_block_type": ["attention"]})
    assert parsed.hybrid_override_pattern == ["M", "E", "*", "E"]
    assert parsed.num_hidden_layers == 4
    assert parsed.n_routed_experts is None
    assert parsed.for_layer(1).n_routed_experts == 2
    assert parsed.for_layer(3).n_routed_experts == 3
    assert parsed.for_layer(1).moe_intermediate_size == 32
    assert parsed.for_layer(3).moe_intermediate_size == 64
    assert parsed.for_layer(3).num_experts_per_tok == 2
    assert json.dumps(raw) == original
    _, family = get_model_and_args({"model_type": "nemotron_h_puzzle"})
    assert family == "nemotron_h"


@pytest.mark.parametrize("vision_norm,mtp_depth", [(True, 0), (False, 1)])
def test_explicit_vision_norm_overrides_mtp(config, vision_norm, mtp_depth):
    config["vision_final_layernorm"] = vision_norm
    config["llm_config"]["num_nextn_predict_layers"] = mtp_depth
    model = Model(ModelConfig.from_dict(config))
    assert isinstance(model.vision_final_layernorm, nn.LayerNorm) == vision_norm
    assert model.config.to_dict()["vision_final_layernorm"] == vision_norm


def test_bolt_checkpoint_conversion_and_multimodal_forward(tmp_path, bolt_config):
    from mlx_vlm.quant_utils import quantize_model
    from mlx_vlm.speculative.drafters.mtp_split import detect_mtp_splitter
    from mlx_vlm.utils import save_weights

    mx.random.seed(123)
    reference = Model(ModelConfig.from_dict(bolt_config))
    reference.eval()
    source = {}
    for key, weight in tree_flatten(reference.parameters()):
        if ".switch_mlp." in key:
            prefix, projection = key.split(".switch_mlp.")
            projection = projection.replace("fc1", "up_proj").replace(
                "fc2", "down_proj"
            )
            for expert, value in enumerate(weight):
                source[f"{prefix}.experts.{expert}.{projection}"] = value
        else:
            if "conv1d.weight" in key:
                weight = weight.moveaxis(1, 2)
            source[key.replace("mlp1.layers.", "mlp1.")] = weight
    (tmp_path / "config.json").write_text(json.dumps(bolt_config))
    mx.save_safetensors(str(tmp_path / "model.safetensors"), source)
    assert detect_mtp_splitter(tmp_path) is None
    loaded = load_model(tmp_path)
    for idx, experts, width, topk in [(1, 2, 32, 1), (3, 3, 64, 2)]:
        moe = loaded.language_model.layers[idx].mixer
        assert moe.switch_mlp.fc1.weight.shape == (experts, width, 32)
        assert moe.switch_mlp.fc2.weight.shape == (experts, 32, width)
        assert moe.gate.weight.shape == (experts, 32)
        assert moe.gate.top_k == topk
    ids = mx.array([[1, 18, 3]])
    pixels = mx.random.normal((1, 3, 32, 32))
    expected = reference(ids, pixels).logits
    np.testing.assert_allclose(
        np.array(loaded(ids, pixels).logits), np.array(expected), atol=1e-5
    )
    features = loaded.get_input_embeddings(ids, pixels).inputs_embeds
    cache = loaded.language_model.make_cache()
    reference_cache = reference.language_model.make_cache()
    for i in range(ids.shape[1]):
        actual = loaded.language_model(
            ids[:, i : i + 1], inputs_embeds=features[:, i : i + 1], cache=cache
        ).logits
        expected_step = reference.language_model(
            ids[:, i : i + 1],
            inputs_embeds=features[:, i : i + 1],
            cache=reference_cache,
        ).logits
        np.testing.assert_allclose(np.array(actual), np.array(expected_step), atol=1e-5)
    video_ids = mx.array([[1, 18, 18, 3]])
    video = mx.random.normal((4, 3, 32, 32))
    np.testing.assert_allclose(
        np.array(loaded(video_ids, pixel_values_videos=video).logits),
        np.array(reference(video_ids, pixel_values_videos=video).logits),
        atol=1e-5,
    )
    quantized, quant_config = quantize_model(
        loaded,
        bolt_config,
        32,
        4,
        quant_predicate=lambda path, module: path.startswith("language_model."),
    )
    output = tmp_path / "4bit"
    save_weights(output, quantized)
    (output / "config.json").write_text(json.dumps(quant_config))
    reloaded = load_model(output)
    assert reloaded.language_model.layers[3].mixer.config.n_routed_experts == 3
    np.testing.assert_array_equal(
        np.array(reloaded(ids, pixels).logits), np.array(quantized(ids, pixels).logits)
    )


def test_conversion_can_rewrite_readonly_hub_metadata(
    tmp_path, bolt_config, monkeypatch
):
    from importlib import import_module

    conversion = import_module("mlx_vlm.convert")

    source, output = tmp_path / "source", tmp_path / "output"
    source.mkdir()
    for name, content in (("config.json", bolt_config), ("tokenizer.json", {})):
        file = source / name
        file.write_text(json.dumps(content))
        file.chmod(0o444)

    def save_processor(path):
        tokenizer_file = path / "tokenizer.json"
        assert tokenizer_file.stat().st_mode & 0o200
        tokenizer_file.write_text(json.dumps({"saved": True}))

    model = Model(ModelConfig.from_dict(bolt_config))
    processor = SimpleNamespace(save_pretrained=save_processor)
    monkeypatch.setattr(
        conversion,
        "fetch_from_hub",
        lambda *args, **kwargs: (model, bolt_config, processor),
    )
    conversion.convert(str(source), mlx_path=str(output), trust_remote_code=False)
    assert json.loads((output / "tokenizer.json").read_text()) == {"saved": True}
    assert (
        json.loads((output / "config.json").read_text())["llm_config"]["block_configs"]
        == bolt_config["llm_config"]["block_configs"]
    )
    assert not (source / "tokenizer.json").stat().st_mode & 0o200


@pytest.mark.parametrize("text_key", ["llm_config", "text_config"])
def test_released_config(config, text_key):
    config[text_key] = config.pop("llm_config")
    module, family = get_model_and_args(config)
    parsed = module.ModelConfig.from_dict(config)
    assert family == "nemotron_h_nano_omni"
    assert parsed.text_config.num_hidden_layers == 3
    assert parsed.text_config.hybrid_override_pattern == ["M", "E", "*"]
    assert parsed.text_config.time_step_limit == (0, float("inf"))
    assert "num_hidden_layers" not in config[text_key]
    assert isinstance(Model(parsed).vision_final_layernorm, nn.LayerNorm)


def test_mixed_checkpoint_strict_load_and_forward(tmp_path, config):
    mx.random.seed(7)
    reference = Model(ModelConfig.from_dict(config))
    reference.eval()
    dense = dict(tree_flatten(reference.parameters()))
    source = {}
    for key, weight in dense.items():
        if ".switch_mlp." in key:
            prefix, projection = key.split(".switch_mlp.")
            projection = projection.replace("fc1", "up_proj").replace(
                "fc2", "down_proj"
            )
            decoded = []
            for expert in range(weight.shape[0]):
                scale = 0.137 * (expert + 1)
                packed, scales = mx.quantize(
                    weight[expert] / scale, group_size=16, bits=4, mode="nvfp4"
                )
                path = f"{prefix}.experts.{expert}.{projection.removesuffix('.weight')}"
                source[f"{path}.weight"] = packed.view(mx.uint8)
                source[f"{path}.weight_scale"] = scales
                source[f"{path}.weight_scale_2"] = mx.array(scale)
                source[f"{path}.input_scale"] = mx.array(1.0)
                decoded.append(
                    mx.dequantize(packed, scales, group_size=16, bits=4, mode="nvfp4")
                    * scale
                )
            dense[key] = mx.stack(decoded)
            continue
        if key == "language_model.lm_head.weight":
            # E4M3 bytes 56 and 64 represent 1 and 2, respectively.
            values = mx.arange(weight.size).reshape(weight.shape) % 2
            source[key] = (56 + 8 * values).astype(mx.uint8)
            source[key.removesuffix("weight") + "weight_scale"] = mx.array(0.5)
            source[key.removesuffix("weight") + "input_scale"] = mx.array(1.0)
            dense[key] = (0.5 + 0.5 * values).astype(mx.bfloat16)
            continue
        if "conv1d.weight" in key:
            weight = weight.moveaxis(1, 2)
        key = key.replace("mlp1.layers.", "mlp1.")
        if key.startswith("vision_final_layernorm."):
            key = "vision_projector." + key
        source[key] = weight
    source["language_model.mtp.unused.weight"] = mx.ones((2, 2))
    source["vision_model.summary_idxs"] = mx.array([0, 1])
    config["quantization_config"] = {
        "quant_method": "modelopt_mixed",
        "quant_algo": "MIXED_PRECISION",
    }
    (tmp_path / "config.json").write_text(json.dumps(config))
    mx.save_safetensors(str(tmp_path / "model.safetensors"), source)
    reference.load_weights(list(dense.items()))
    loaded = load_model(tmp_path)
    ids = mx.array([[1, 18, 4]])
    pixels = mx.random.normal((1, 3, 32, 32))
    expected = reference(ids, pixels).logits
    actual = loaded(ids, pixels).logits
    np.testing.assert_allclose(
        np.array(actual), np.array(expected), atol=1e-3, rtol=1e-3
    )
    cache = loaded.language_model.make_cache()
    features = loaded.get_input_embeddings(ids, pixels).inputs_embeds
    for i in range(ids.shape[1]):
        step = loaded.language_model(
            ids[:, i : i + 1], inputs_embeds=features[:, i : i + 1], cache=cache
        ).logits
        np.testing.assert_allclose(
            np.array(step), np.array(actual[:, i : i + 1]), atol=2e-3, rtol=2e-3
        )


@pytest.mark.parametrize("quantized", [False, True])
def test_native_mtp_load_split_and_forward(tmp_path, config, quantized):
    from mlx_vlm.speculative.drafters import load_drafter
    from mlx_vlm.speculative.drafters.nemotron_h_mtp import Model as DraftModel
    from mlx_vlm.speculative.drafters.nemotron_h_mtp import ModelConfig as DraftConfig
    from mlx_vlm.speculative.drafters.nemotron_h_mtp.split import NemotronHMTPSplitter

    mx.random.seed(42)
    target = Model(ModelConfig.from_dict(config))
    reference = DraftModel(DraftConfig(text_config=config["llm_config"]))
    expected = dict(tree_flatten(reference.parameters()))
    source = {}
    for key, weight in expected.items():
        if ".switch_mlp." not in key:
            source["language_model.mtp." + key] = weight
            continue
        prefix, projection = key.split(".switch_mlp.")
        projection = projection.replace("fc1", "up_proj").replace("fc2", "down_proj")
        decoded = []
        for expert, value in enumerate(weight):
            path = f"language_model.mtp.{prefix}.experts.{expert}.{projection}"
            if quantized:
                scale = 0.137 * (expert + 1)
                packed, scales = mx.quantize(value / scale, mode="nvfp4")
                source[path] = packed.view(mx.uint8)
                source[path.removesuffix("weight") + "weight_scale"] = scales
                source[path.removesuffix("weight") + "weight_scale_2"] = mx.array(scale)
                source[path.removesuffix("weight") + "input_scale"] = mx.array(1.0)
                value = (
                    mx.dequantize(packed, scales, mode="nvfp4", dtype=mx.float32)
                    * scale
                )
            else:
                source[path] = value
            decoded.append(value)
        expected[key] = mx.stack(decoded)
    if quantized:
        config["quantization_config"] = {
            "quant_method": "modelopt_mixed",
            "quant_algo": "MIXED_PRECISION",
        }
        source["language_model.mtp.layers.0.mixer.k_proj.k_scale"] = mx.array(1.0)
    reference.load_weights(list(expected.items()))
    (tmp_path / "config.json").write_text(json.dumps(config))
    mx.save_safetensors(str(tmp_path / "mtp.safetensors"), source)
    weight_map = dict.fromkeys(source, "mtp.safetensors")
    # Loading the drafter must not open the much larger target shards.
    weight_map["language_model.backbone.embeddings.weight"] = "absent.safetensors"
    (tmp_path / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": weight_map})
    )
    draft, kind = load_drafter(str(tmp_path))
    assert kind == "mtp"
    ids = mx.array([[1, 2, 3]])
    hidden = target.language_model(ids, return_hidden=True).hidden_states[-1]
    reference.reset(target)
    expected_hidden, _ = reference._forward_tokens(ids, hidden, mx.int32)
    draft.reset(target)
    actual_hidden, _ = draft._forward_tokens(ids, hidden, mx.int32)
    np.testing.assert_allclose(
        np.array(actual_hidden), np.array(expected_hidden), atol=1e-4, rtol=1e-4
    )
    # Cached singleton forwards must match a causal block.
    draft.reset(target)
    steps = [
        draft._forward_tokens(ids[:, i : i + 1], hidden[:, i : i + 1], mx.int32)[0]
        for i in range(ids.shape[1])
    ]
    np.testing.assert_allclose(
        np.array(mx.concatenate(steps, axis=1)), np.array(actual_hidden), atol=1e-4
    )
    output = NemotronHMTPSplitter().split(str(tmp_path), str(tmp_path / "draft"))
    reloaded, kind = load_drafter(str(output))
    reloaded.reset(target)
    roundtrip, _ = reloaded._forward_tokens(ids, hidden, mx.int32)
    np.testing.assert_array_equal(np.array(roundtrip), np.array(actual_hidden))
    source.pop("language_model.mtp.layers.0.eh_proj.weight")
    mx.save_safetensors(str(tmp_path / "mtp.safetensors"), source)
    weight_map.pop("language_model.mtp.layers.0.eh_proj.weight")
    (tmp_path / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": weight_map})
    )
    with pytest.raises(ValueError, match="Missing"):
        load_drafter(str(tmp_path))


@pytest.mark.parametrize("media", ["text", "image", "video"])
@pytest.mark.parametrize("dtype", [mx.float32, mx.bfloat16])
@pytest.mark.parametrize("temperature", [0, 0.7])
def test_mtp_generation_matches_autoregressive(
    config, media, dtype, temperature, monkeypatch
):
    from mlx_vlm.generate.ar import generate_step
    from mlx_vlm.speculative.drafters.nemotron_h_mtp import Model as DraftModel
    from mlx_vlm.speculative.drafters.nemotron_h_mtp import ModelConfig as DraftConfig

    mx.random.seed(19)
    model = Model(ModelConfig.from_dict(config))
    draft = DraftModel(DraftConfig(text_config=config["llm_config"]))
    model.set_dtype(dtype)
    draft.set_dtype(dtype)
    model.eval()
    draft.eval()
    prefill = Mock(wraps=draft.prefill_from_target_hidden)
    monkeypatch.setattr(draft, "prefill_from_target_hidden", prefill)
    ids = mx.array([[1, 2, 3]])
    pixels = None
    kwargs = {}
    if media == "image":
        ids = mx.array([[1, 18, 3]])
        pixels = mx.random.normal((1, 3, 32, 32))
    elif media == "video":
        ids = mx.array([[1, 18, 18, 18, 3]])
        kwargs["pixel_values_videos"] = mx.random.normal((6, 3, 32, 32))
        embeds = model.get_input_embeddings(ids, **kwargs)
        assert embeds.input_ids.tolist() == [[1, 18, 3]]
    common = dict(
        max_tokens=12, temperature=temperature, seed=19, prefill_step_size=2, **kwargs
    )
    expected = [t for t, _ in generate_step(ids, model, pixels, None, **common)]
    actual = [
        t
        for t, _ in generate_step(
            ids,
            model,
            pixels,
            None,
            draft_model=draft,
            draft_kind="mtp",
            draft_block_size=4,
            **common,
        )
    ]
    assert actual == expected
    assert draft.draft_lens and any(
        a < d for a, d in zip(draft.accept_lens, draft.draft_lens)
    )
    prompt_ids, prompt_hidden = prefill.call_args.args[:2]
    expected_ids = embeds.input_ids if media == "video" else ids
    assert prompt_ids.tolist() == expected_ids.tolist()
    assert prompt_hidden.shape[1] == expected_ids.shape[1]


@pytest.mark.parametrize("accepted", [0, 1, 3])
def test_mtp_mamba_rollback_matches_accepted_prefix(config, accepted):
    mx.random.seed(71)
    # Exercise the same Metal state replay used by the released model.
    config["llm_config"]["ssm_state_size"] = 32
    model = Model(ModelConfig.from_dict(config)).language_model
    model.eval()
    ordinary, speculative = model.make_cache(), model.make_cache()
    prompt = mx.array([[1, 2, 3]])
    model(prompt, cache=ordinary)
    model(prompt, cache=speculative)
    block = mx.array([[4, 5, 6, 7]])
    hidden, _, states = model.speculative_verify_hidden(block, speculative)
    mx.eval(hidden)
    model.rollback_speculative_cache(speculative, states, accepted, 4)
    for i in range(accepted + 1):
        model(block[:, i : i + 1], cache=ordinary)
    for actual, expected in zip(speculative, ordinary):
        for a, e in zip(actual.state, expected.state):
            np.testing.assert_allclose(np.array(a), np.array(e), atol=1e-5, rtol=1e-5)
    probe = mx.array([[8]])
    np.testing.assert_allclose(
        np.array(model(probe, cache=speculative).logits),
        np.array(model(probe, cache=ordinary).logits),
        atol=1e-5,
        rtol=1e-5,
    )


@pytest.mark.parametrize(
    "rate,expected",
    [
        (0.0, [True] * 6),
        (0.5, [True, True, False, True, False, False]),
        (1.0, [True, True, False, False, False, False]),
    ],
)
def test_video_retention_order_and_first_tubelet(rate, expected):
    # The second token changes most in frame 2; equal scores preserve order.
    features = mx.array(
        [[[1.0, 0.0], [1.0, 0.0]], [[1.0, 0.0], [-1.0, 0.0]], [[1.0, 0.0], [-1.0, 0.0]]]
    )
    assert compute_retention_mask(features, rate).tolist() == expected
    assert compute_retention_mask(mx.zeros((1, 2, 4)), rate).tolist() == [True, True]


def test_vision_normalization_and_video_pruning(config):
    model = Model(ModelConfig.from_dict(config))
    features = mx.random.normal((1, 4, 16))
    # The projector must normalize BEFORE spatially merging the four patches.
    expected = model.mlp1(model.vision_final_layernorm(features).reshape(1, 1, 64))
    np.testing.assert_allclose(
        np.array(model._project_vision_features(features, 2, 2)),
        np.array(expected),
        rtol=1e-5,
        atol=1e-5,
    )
    # Five rectangular frames pad to three tubelets, two tokens per tubelet.
    pixels = mx.random.normal((5, 3, 32, 64))
    ids = mx.array([[1, 18, 18, 2, 18, 18, 3, 18, 18, 4]])
    full = model.extract_video_feature(pixels)
    assert full.shape == (3, 2, 32)
    keep = np.flatnonzero(np.array(compute_retention_mask(full, 0.7)))
    actual = model.get_input_embeddings(ids, pixel_values_videos=pixels).inputs_embeds
    assert actual.shape == (1, 6, 32)
    unpruned = model.get_input_embeddings(
        ids, pixel_values_videos=pixels, video_pruning_rate=0
    ).inputs_embeds
    assert unpruned.shape == (1, 10, 32)
    assert model.video_pruning_rate == 0.7
    with pytest.raises(ValueError, match="video_pruning_rate"):
        model.get_input_embeddings(
            ids, pixel_values_videos=pixels, video_pruning_rate=-1
        )
    np.testing.assert_allclose(
        np.array(actual[0, 1:3]), np.array(full.reshape(-1, 32)[mx.array(keep)])
    )
    expected_text = model.language_model.get_input_embeddings(mx.array([[1, 2, 3, 4]]))
    np.testing.assert_allclose(
        np.array(actual[:, [0, 3, 4, 5]]), np.array(expected_text)
    )
    images = [mx.zeros((3, 32, 32)), mx.zeros((3, 32, 64))]
    assert model.get_input_embeddings(
        mx.array([[1, 18, 18, 18, 2]]), images
    ).inputs_embeds.shape == (1, 5, 32)


def test_native_processor_image_video_and_metadata(tmp_path, config):
    (tmp_path / "config.json").write_text(json.dumps(config))
    (tmp_path / "processor_config.json").write_text(
        json.dumps({"processor_class": "NemotronH_Omni_Reasoning_V3Processor"})
    )
    (tmp_path / "preprocessor_config.json").write_text(
        json.dumps(
            {
                "norm_mean": [0.5] * 3,
                "norm_std": [0.5] * 3,
                "min_num_patches": 4,
                "max_num_patches": 16,
                "video_target_num_patches": 8,
            }
        )
    )
    vocab = {str(i): i for i in range(21)}
    for index, token in [
        (0, "[UNK]"),
        (5, "seconds"),
        (18, "<image>"),
        (19, "<img>"),
        (20, "</img>"),
    ]:
        del vocab[str(index)]
        vocab[token] = index
    backend = Tokenizer(WordLevel(vocab, unk_token="[UNK]"))
    backend.pre_tokenizer = Whitespace()
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=backend,
        unk_token="[UNK]",
        eos_token="[UNK]",
        pad_token="[UNK]",
        additional_special_tokens=["<image>", "<img>", "</img>"],
    )
    tokenizer.save_pretrained(tmp_path)
    processor = load_processor(tmp_path, add_detokenizer=False, trust_remote_code=False)
    assert processor_handles_video(processor)
    messages = apply_chat_template(
        processor, config, "Describe.", num_images=1, return_messages=True
    )
    assert messages[0]["content"][0] == {"type": "image"}
    messages = apply_chat_template(
        processor, config, "Describe.", video="clip.mp4", return_messages=True
    )
    assert messages[0]["content"][0]["type"] == "video"
    image = Image.new("RGB", (32, 32), "red")
    inputs = prepare_inputs(processor, prompts="<image>Describe.", images=[image])
    assert int(mx.sum(inputs["input_ids"] == 18)) == int(inputs["num_tokens"].sum())
    frames = np.zeros((5, 3, 32, 64), dtype=np.uint8)
    inputs = prepare_inputs(processor, prompts="<video>Describe.", videos=[frames])
    assert inputs["pixel_values_videos"].shape[0] == 5
    assert int(mx.sum(inputs["input_ids"] == 18)) == 3 * int(inputs["num_tokens"][0])
    assert (
        int(mx.sum(inputs["input_ids"] == 5)) == 5
    )  # Per-frame timestamps survived the loader.
    assert processor.image_processor._is_video_mode is False
    with pytest.raises(ValueError, match="one video"):
        prepare_inputs(processor, prompts="<video><video>", videos=[frames, frames])
