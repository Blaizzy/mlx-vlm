"""Tiny Muse Spark model, checkpoint, processor and audio regression tests."""

import copy
import json
from pathlib import Path
from types import SimpleNamespace

import mlx.core as mx
import numpy as np
import pytest
from mlx.utils import tree_flatten, tree_map
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from transformers import PreTrainedTokenizerFast

from mlx_vlm.models import muse_spark
from mlx_vlm.models.cache import BatchKVCache, KVCache, RotatingKVCache
from mlx_vlm.models.muse_spark.image_processing import (
    ImageProcessor,
    _find_resize_target,
)
from mlx_vlm.models.muse_spark.processing_muse_spark import MuseSparkFeatureExtractor
from mlx_vlm.models.muse_spark.sanitize import _ZERO_CENTERED_NORMS
from mlx_vlm.prompt_utils import apply_chat_template
from mlx_vlm.utils import (
    get_model_and_args,
    load_model,
    load_model_weights,
    prepare_inputs,
    process_inputs,
)

_CASE = next(
    case
    for case in json.loads(Path(__file__).with_name("model_cases.json").read_text())[
        "cases"
    ]
    if case["id"] == "muse_spark"
)
CONFIG = {
    key: value
    for key, value in _CASE["config"].items()
    if key not in ("vision_config", "audio_config", "projector_hidden_size")
}
VISION_CONFIG = _CASE["config"]["vision_config"]
AUDIO_CONFIG = _CASE["config"]["audio_config"]


def create_model(vision=False, audio=False):
    mx.random.seed(7)
    config = copy.deepcopy(CONFIG)
    if vision:
        config.update(vision_config=VISION_CONFIG, projector_hidden_size=48)
    if audio:
        config["audio_config"] = AUDIO_CONFIG
        config["audio_token_id"] = 254
    model = muse_spark.Model(muse_spark.ModelConfig.from_dict(config))
    model.eval()
    return model


def upstream_weights(model):
    weights = {}
    for key, value in tree_flatten(model.parameters()):
        if key.startswith("language_model.model."):
            key = "model.language_model." + key[len("language_model.model.") :]
        elif key.startswith("language_model.lm_head."):
            key = key[len("language_model.") :]
        else:
            key = "model." + key
        weights[key] = value
    return weights


def hf_checkpoint(model):
    weights = upstream_weights(model)
    args = model.config.text_config
    q, kv = (
        args.num_attention_heads * args.head_dim,
        args.num_key_value_heads * args.head_dim,
    )
    hf = {}

    def rotate_half_rows(value, dim):
        shape = value.shape
        return value.reshape(-1, dim // 2, 2, *shape[1:]).swapaxes(1, 2).reshape(shape)

    for key, value in weights.items():
        if key.endswith("self_attn.qkvg_proj.weight"):
            parts = mx.split(value, [q, q + kv, q + 2 * kv])
            for name, part in zip(("q", "k", "v", "gate"), parts):
                if name in ("q", "k"):
                    part = rotate_half_rows(part, args.head_dim)
                hf[key.replace("qkvg_proj", name + "_proj")] = part
        elif ".attn.qkv_proj." in key:
            for name, part in zip("qkv", mx.split(value, 3)):
                if name in "qk":
                    part = rotate_half_rows(part, model.config.vision_config.head_dim)
                hf[key.replace("qkv_proj", name + "_proj")] = part
        elif key.endswith(("_alpha", "_beta")):
            name = "post_attention" if ".add_attn_" in key else "post_feedforward"
            hf[key.rsplit(".", 1)[0] + f".{name}_residual_gate.gate"] = mx.zeros(
                value.shape
            )
        elif key.endswith(_ZERO_CENTERED_NORMS):
            hf[key] = value - 1
        elif key.endswith("mlp.experts.gate_proj.weight"):
            up = weights[key.replace("gate_proj", "up_proj")]
            hf[key.replace("gate_proj.weight", "gate_up_proj")] = mx.concatenate(
                [value, up], axis=1
            )
        elif key.endswith("mlp.experts.up_proj.weight"):
            continue
        elif key.endswith("mlp.experts.down_proj.weight"):
            hf[key.replace("down_proj.weight", "down_proj")] = value
        elif key.startswith("model.audio_tower.") and key.endswith(".conv.weight"):
            hf[key] = value.transpose(0, 2, 1)
        else:
            hf[key] = value
    return hf


def create_processor():
    vocab = {
        "[UNK]": 0,
        "[PAD]": 1,
        "<|patch|>": 255,
        "<|audio|>": 254,
        "<|video|>": 253,
    }
    for token in (
        "<|image_start|>",
        "<|image_end|>",
        "<|soft_audio_start|>",
        "<|soft_audio_end|>",
        "<|vid_start|>",
        "<|vid_end|>",
        "<|vid_frame_separator|>",
    ):
        vocab[token] = len(vocab)
    for index in set(range(256)) - set(vocab.values()):
        vocab[f"unused_{index}"] = index
    backend = Tokenizer(WordLevel(vocab, unk_token="[UNK]"))
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=backend, unk_token="[UNK]", pad_token="[PAD]"
    )
    tokenizer.add_special_tokens(
        {
            "additional_special_tokens": [
                token for token in vocab if token.startswith("<|")
            ]
        }
    )
    tokenizer.chat_template = "{% for message in messages %}{% for part in message['content'] %}{% if part['type'] == 'image' %}<|patch|>{% elif part['type'] == 'audio' %}<|audio|>{% elif part['type'] == 'video' %}<|video|>{% else %}{{ part['text'] }}{% endif %}{% endfor %}{% endfor %}"
    return muse_spark.MuseSparkProcessor(
        image_processor=ImageProcessor(
            patch_size=2, downsample_factor=2, max_num_image_tokens=16
        ),
        tokenizer=tokenizer,
        feature_extractor=MuseSparkFeatureExtractor(samples_per_token=4),
        video_processor=muse_spark.MuseSparkVideoProcessor(
            patch_size=2, temporal_patch_size=2, merge_size=2, max_video_frame_tokens=16
        ),
        chat_template=tokenizer.chat_template,
    )


def test_loader_registration():
    for model_type in ("muse_spark", "muse_spark_open"):
        module, _ = get_model_and_args({"model_type": model_type})
        assert module.Model is muse_spark.Model
    model = create_model()
    assert model.quantization_path_aliases("language_model.lm_head") == ("lm_head",)


@pytest.mark.parametrize("dtype", [mx.float32, mx.bfloat16])
def test_forward_and_cached_decode(dtype):
    model = create_model()
    model.update(tree_map(lambda p: p.astype(dtype), model.parameters()))
    tokens = mx.random.randint(0, 200, (2, 20))
    full = model(tokens).logits
    cache = model.make_cache()
    steps = [
        model(tokens[:, :6], cache=cache).logits,
        model(tokens[:, 6:12], cache=cache).logits,
    ]
    steps += [model(tokens[:, i : i + 1], cache=cache).logits for i in range(12, 20)]
    assert full.dtype == mx.float32
    assert mx.all(mx.isfinite(full))
    assert mx.allclose(mx.concatenate(steps, axis=1), full, atol=3e-3, rtol=3e-3)
    assert [type(c) for c in cache] == [RotatingKVCache] * 3 + [KVCache]


def test_left_padded_batch():
    model = create_model()
    ids = mx.array([[0, 0, 1, 2, 3], [4, 5, 6, 7, 8]])
    cache = [BatchKVCache(left_padding=mx.array([2, 0])) for _ in model.layers]
    batch = model(ids, cache=cache).logits
    single = model(ids[:1, 2:]).logits
    assert mx.allclose(batch[:1, 2:], single, atol=3e-3, rtol=3e-3)


def test_logits_to_keep_projects_only_the_requested_positions():
    model = create_model()
    ids = mx.array([[1, 2, 3]])
    full = model(ids).logits
    final = model(ids, logits_to_keep=1).logits
    assert final.shape == (1, 1, 256)
    assert mx.allclose(final, full[:, -1:], atol=1e-5)


@pytest.mark.parametrize("mask_kind", ["padding", "additive"])
def test_direct_attention_mask_preserves_causality_and_sliding_window(mask_kind):
    model = create_model()
    ids = mx.array([[0, 0] + list(range(1, 14))])
    mask = mx.array([[0, 0] + [1] * 13])
    if mask_kind == "additive":
        mask = mx.where(mask[:, None, None, :], 0.0, -float("inf"))
    batch = model(ids, mask=mask).logits[:, 2:]
    single = model(ids[:, 2:]).logits
    assert mx.allclose(batch, single, atol=3e-3, rtol=3e-3)


def test_tensor_sharding_preserves_projection_segments():
    for rank in (0, 1):
        model = create_model()
        attn = model.layers[0].self_attn
        projections = mx.split(attn.qkvg_proj.weight, [64, 96, 128])
        expected = mx.concatenate(
            [
                part[rank * (part.shape[0] // 2) : (rank + 1) * (part.shape[0] // 2)]
                for part in projections
            ]
        )
        model.shard(SimpleNamespace(size=lambda: 2, rank=lambda rank=rank: rank))
        assert mx.array_equal(model.layers[0].self_attn.qkvg_proj.weight, expected)
        assert model.layers[0].self_attn.n_heads == 2
        assert model.layers[0].self_attn.n_kv_heads == 1
        assert model.layers[0].mlp.experts.gate_proj.weight.shape == (4, 16, 32)
        assert model.layers[0].mlp.experts.down_proj.weight.shape == (4, 32, 16)


def test_pipeline_local_cache_types():
    for rank, cache_types in (
        (0, [RotatingKVCache, KVCache]),
        (1, [RotatingKVCache, RotatingKVCache]),
    ):
        model = create_model()
        model.language_model.model.pipeline(
            SimpleNamespace(size=lambda: 2, rank=lambda rank=rank: rank)
        )
        assert [type(cache) for cache in model.make_cache()] == cache_types


def test_hf_sanitize_round_trip():
    model = create_model(vision=True, audio=True)
    expected = dict(tree_flatten(model.parameters()))
    sanitized = model.sanitize(hf_checkpoint(model))
    assert sanitized.keys() == expected.keys()
    model.load_weights(list(sanitized.items()), strict=True)
    for key, value in expected.items():
        assert mx.allclose(sanitized[key], value, atol=1e-5), key
    again = model.sanitize(sanitized)
    for key in expected:
        assert mx.array_equal(again[key], sanitized[key]), key


def test_load_upstream_conversion_without_audio(tmp_path):
    model = create_model(vision=True, audio=True)
    weights = {
        k: v
        for k, v in upstream_weights(model).items()
        if not k.startswith("model.audio_tower.")
    }
    sanitized = model.sanitize(weights)
    load_model_weights(model, {}, sanitized, tmp_path, declared_keys=set(weights))
    assert model.audio_tower is None
    with pytest.raises(ValueError, match="audio-inclusive"):
        model.encode_audio(mx.zeros((1, 4)))


@pytest.mark.parametrize("scale_layout", ["per_expert", "per_half"])
def test_modelopt_nvfp4_load(tmp_path, scale_layout):
    model = create_model(vision=True, audio=True)
    config = copy.deepcopy(CONFIG)
    config.update(
        vision_config=VISION_CONFIG,
        projector_hidden_size=48,
        audio_config=AUDIO_CONFIG,
        audio_token_id=254,
    )
    config["quantization_config"] = {"quant_method": "modelopt", "quant_algo": "NVFP4"}
    weights = hf_checkpoint(model)
    for idx in range(CONFIG["text_config"]["num_hidden_layers"]):
        prefix = f"model.language_model.layers.{idx}.mlp.experts."
        gate_up = weights[prefix + "gate_up_proj"]
        down = weights[prefix + "down_proj"]
        for name, value in (("gate_up_proj", gate_up), ("down_proj", down)):
            packed, scales = mx.quantize(value, group_size=16, bits=4, mode="nvfp4")
            weights[prefix + name] = packed.view(mx.uint8)
            weights[prefix + name + "_weight_scale"] = scales
            global_shape = (
                (value.shape[0], 2)
                if name == "gate_up_proj" and scale_layout == "per_half"
                else (value.shape[0],)
            )
            weights[prefix + name + "_weight_scale_2"] = mx.ones(global_shape)
            weights[prefix + name + "_input_scale"] = mx.ones(global_shape)
    mx.save_safetensors(str(tmp_path / "model.safetensors"), weights)
    (tmp_path / "config.json").write_text(json.dumps(config))
    loaded = load_model(tmp_path)
    from mlx_vlm.quantization.nvfp4 import ScaledQuantizedSwitchLinear

    assert isinstance(
        loaded.layers[0].mlp.experts.gate_proj, ScaledQuantizedSwitchLinear
    )
    assert loaded.layers[0].mlp.experts.gate_proj.mode == "nvfp4"
    assert mx.all(mx.isfinite(loaded(mx.array([[1, 2]])).logits))


def test_processor_save_and_load(tmp_path):
    from transformers import AutoProcessor

    processor = create_processor()
    processor.config = muse_spark.ModelConfig.from_dict(_CASE["config"])
    processor.save_pretrained(tmp_path)
    (tmp_path / "config.json").write_text(json.dumps(_CASE["config"]))
    loaded = AutoProcessor.from_pretrained(tmp_path)
    assert isinstance(loaded, muse_spark.MuseSparkProcessor)
    assert loaded.feature_extractor.samples_per_token == 4
    assert loaded.image_processor.max_num_image_tokens == 16
    assert loaded.chat_template == processor.chat_template
    audio = loaded("<|audio|>", audio=[np.ones(5)])
    assert int((audio["input_ids"] == 254).sum()) == 2


def test_standard_load_and_multimodal_generate(tmp_path):
    from mlx_vlm import generate, load

    model = create_model(vision=True, audio=True)
    processor = create_processor()
    processor.save_pretrained(tmp_path)
    (tmp_path / "config.json").write_text(json.dumps(_CASE["config"]))
    mx.save_safetensors(str(tmp_path / "model.safetensors"), hf_checkpoint(model))
    loaded, processor = load(tmp_path)
    prompt = apply_chat_template(
        processor, loaded.config, "describe", num_images=1, num_audios=1
    )
    result = generate(
        loaded,
        processor,
        prompt,
        image=[np.zeros((8, 12, 3), dtype=np.uint8)],
        audio=[np.ones(5, dtype=np.float32)],
        max_tokens=2,
        verbose=False,
    )
    assert result.generation_tokens == 2


def test_text_only_checkpoint_does_not_create_optional_towers(tmp_path):
    model = create_model()
    (tmp_path / "config.json").write_text(json.dumps(CONFIG))
    mx.save_safetensors(str(tmp_path / "model.safetensors"), upstream_weights(model))
    loaded = load_model(tmp_path)
    assert loaded.audio_tower is None and loaded.vision_tower is None


def test_global_scale_and_expert_half_mapping():
    model = create_model()
    prefix = "model.language_model.layers.1.mlp.experts."
    factors = mx.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]])
    mapped = model.sanitize({prefix + "gate_up_proj_weight_scale_2": factors})
    base = "language_model.model.layers.1.mlp.experts."
    assert mx.array_equal(mapped[base + "gate_proj.weight_scale_2"], factors[:, 0])
    assert mx.array_equal(mapped[base + "up_proj.weight_scale_2"], factors[:, 1])
    mapped = model.sanitize({prefix + "gate_proj.global_scale": factors[:, 0] * 2688})
    assert mx.array_equal(mapped[base + "gate_proj.weight_scale_2"], factors[:, 0])


def test_image_embeddings_and_chunked_prefill():
    model = create_model(vision=True)
    image = mx.random.normal((3, 12, 16))
    ids = mx.array([[1, 2] + [255] * 12 + [3, 4]])
    embeds = model.get_input_embeddings(ids, [image]).inputs_embeds
    assert mx.allclose(embeds[0, 2:14], model.encode_image(image))
    assert mx.array_equal(embeds[0, :2], model.language_model.model.embed(ids)[0, :2])
    full = model(ids, inputs_embeds=embeds).logits
    cache = model.make_cache()
    chunks = [
        model(ids[:, :7], inputs_embeds=embeds[:, :7], cache=cache).logits,
        model(ids[:, 7:], inputs_embeds=embeds[:, 7:], cache=cache).logits,
    ]
    assert mx.allclose(mx.concatenate(chunks, axis=1), full, atol=3e-3, rtol=3e-3)
    with pytest.raises(ValueError, match="tokens and features"):
        model.get_input_embeddings(mx.array([[255]]), [image])


def test_audio_streaming_and_causality():
    model = create_model(audio=True).audio_tower
    wave = mx.random.normal((2, 40))
    full = model(wave)
    cache = muse_spark.AudioConvCache()
    streamed = mx.concatenate(
        [
            model(wave[:, :12], cache),
            model(wave[:, 12:16], cache),
            model(wave[:, 16:], cache),
        ],
        axis=1,
    )
    assert mx.allclose(streamed, full, atol=1e-5, rtol=1e-5)
    assert mx.allclose(model(wave[:, :20]), full[:, :5], atol=1e-5)
    with pytest.raises(ValueError, match="multiple"):
        model(wave[:, :19])


def test_variable_length_audio_and_mixed_embeddings():
    model = create_model(vision=True, audio=True)
    extractor = MuseSparkFeatureExtractor(samples_per_token=4)
    audio = extractor(
        [np.ones(5), np.ones(13)], sampling_rate=24000, return_tensors="mlx"
    )
    features = model.encode_audio(audio["input_values"], audio["input_values_mask"])
    expected = mx.concatenate(
        [
            model.encode_audio(audio["input_values"][:1, :8]),
            model.encode_audio(audio["input_values"][1:]),
        ]
    )
    assert features.shape == (6, 64)
    assert mx.allclose(features, expected, atol=1e-5)
    image = mx.zeros((3, 4, 4))
    ids = mx.array([[1, 255, 254, 254, 2, 254, 254, 254, 254]])
    embeds = model.get_input_embeddings(ids, [image], **audio).inputs_embeds
    assert mx.allclose(embeds[0, 2:4], features[:2])
    assert mx.allclose(embeds[0, 5:], features[2:])
    assert mx.allclose(embeds[0, 1:2], model.encode_image(image))
    result = model(ids, [image], **audio).logits
    assert result.shape == (1, 9, 256)
    assert mx.all(mx.isfinite(result))


def test_audio_matches_independent_reference():
    # PyTorch is optional and is used only for numerical validation.
    torch = pytest.importorskip("torch")
    F = torch.nn.functional
    model = create_model(audio=True).audio_tower
    arrays = {
        key: torch.from_numpy(np.array(value))
        for key, value in tree_flatten(model.parameters())
    }
    wave = np.random.default_rng(0).normal(size=(2, 32)).astype(np.float32)
    x = torch.from_numpy(wave)[..., None]
    config = model.config

    def linear(x, prefix):
        return F.linear(x, arrays[prefix + ".weight"], arrays[prefix + ".bias"])

    def norm(x, prefix):
        return F.rms_norm(
            x, (x.shape[-1],), arrays[prefix + ".weight"], config.rms_norm_eps
        )

    for idx, patch in enumerate(config.patch_sizes):
        prefix = f"layers.{idx}."
        x = x.reshape(x.shape[0], -1, x.shape[-1] * patch)
        x = norm(
            linear(x, prefix + "patch_embedder.proj"), prefix + "patch_embedder.norm"
        )
        for block in range(config.depths[idx]):
            p = prefix + f"conv_blocks.{block}"
            conv = F.conv1d(
                F.pad(x.transpose(1, 2), (config.conv_kernel_size - 1, 0)),
                arrays[p + ".conv.weight"].transpose(1, 2),
                arrays[p + ".conv.bias"],
                groups=x.shape[-1],
            ).transpose(1, 2)
            x = x + linear(
                F.gelu(linear(norm(conv, p + ".norm"), p + ".mlp.fc1")), p + ".mlp.fc2"
            )
    expected = norm(linear(norm(x, "norm"), "proj"), "post_proj_norm")
    np.testing.assert_allclose(
        np.array(model(mx.array(wave))), expected.numpy(), atol=2e-5, rtol=2e-5
    )


def test_processor_image_audio_batch_and_prompt():
    processor = create_processor()
    prompt = apply_chat_template(
        processor, {"model_type": "muse_spark"}, "describe", num_images=1, num_audios=2
    )
    assert prompt.count("<|patch|>") == 1 and prompt.count("<|audio|>") == 2
    inputs = prepare_inputs(
        processor,
        images=[np.zeros((8, 12, 3), dtype=np.uint8)],
        audio=[np.ones(5), np.ones(13)],
        prompts=prompt,
    )
    ids = inputs["input_ids"]
    assert int((ids == 255).sum()) == 6
    assert int((ids == 254).sum()) == 6
    assert inputs["input_values"].shape == (2, 16)
    assert inputs["audio_lengths"].tolist() == [5, 13]
    assert inputs["pixel_values"][0].shape == (3, 8, 12)
    batch = process_inputs(
        processor, ["<|audio|>", "<|audio|>"], audio=[np.ones(5), np.ones(13)]
    )
    assert batch["input_ids"].shape[0] == 2
    assert int((batch["input_ids"][0] == 254).sum()) == 2
    assert int((batch["input_ids"][1] == 254).sum()) == 4
    with pytest.raises(ValueError, match="placeholders"):
        processor("<|audio|>", audio=[np.ones(5), np.ones(13)])
    with pytest.raises(ValueError, match="24000"):
        processor("<|audio|>", audio=np.ones(5), sampling_rate=16000)


def test_resize_bounds_and_channel_cleanup():
    size, grid = _find_resize_target(
        100, 50000, patch_h=28, patch_w=28, min_tokens=1, max_tokens=16
    )
    assert grid[0] * grid[1] <= 16
    assert size == (grid[0] * 28, grid[1] * 28)
    processor = create_processor()
    assert (
        processor.clean_output(
            "to=self<|message|>reason<|eom|><|start|>assistant to=user<|message|>answer<|eot|>"
        )
        == "answer"
    )
