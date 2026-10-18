"""Rotating-cache correctness with token IDs and tiny, randomly initialized models."""

import importlib
import json
from dataclasses import fields
from itertools import product
from pathlib import Path

import mlx.core as mx
import numpy as np
import pytest

from mlx_vlm.models.base import cache_position_offset, create_attention_mask
from mlx_vlm.models.cache import (
    BatchKVCache,
    BatchRotatingKVCache,
    BufferedRotatingKVCache,
    KVCache,
    RotatingKVCache,
    create_causal_mask,
)


@pytest.mark.parametrize("size", [2, 8, 255, 256, 257])
@pytest.mark.parametrize("keep_kind", ["none", "prefix", "all_but_one"])
def test_rotating_cache_visible_tokens(size, keep_kind):
    keep = {"none": 0, "prefix": min(4, size - 1), "all_but_one": size - 1}[keep_kind]
    cache = RotatingKVCache(size, keep)
    retained, start = [], 0
    # Exercise allocation, rotation, and concatenation in both directions.
    for n in [1] * (size + 3) + [7, 1, size + 5, 1, 3]:
        mask = cache.make_mask(n)
        kv = mx.arange(start, start + n).reshape(1, 1, n, 1)
        keys, _ = cache.update_and_fetch(kv, kv)
        ids = keys[0, 0, :, 0].tolist()
        rows = mask.tolist() if isinstance(mask, mx.array) else None
        for row in range(n):
            # Independent reference: evict the oldest token outside the prefix.
            retained.append(start + row)
            if len(retained) > size:
                del retained[keep]
            visible = [
                token
                for col, token in enumerate(ids)
                if (token <= start + row if rows is None else rows[row][col])
            ]
            assert sorted(visible) == retained
        start += n


def test_rotating_cache_explicit_windows():
    for keep, history, n, window, return_array in product(
        [0, 1, 3], [0, 3, 4, 9], [2, 6], [1, 2, 3, 7], [False, True]
    ):
        cache = RotatingKVCache(4, keep)
        kv = mx.zeros((1, 1, 1, 1))
        for _ in range(history):
            cache.update_and_fetch(kv, kv)
        actual = cache.make_mask(n, window, return_array)
        offset = min(3, history)
        if offset + n > window or return_array:
            expected = create_causal_mask(n, offset, window_size=window)
            assert mx.array_equal(actual, expected)
        else:
            assert actual == "causal"


@pytest.mark.parametrize("container", [None, list, tuple])
def test_rotating_cache_mask_dispatch(container):
    cache = RotatingKVCache(8, keep=2)
    wrapped = container([cache]) if container in (list, tuple) else cache
    mask = create_attention_mask(mx.zeros((1, 20, 8)), wrapped)
    assert mask.shape == (20, 20)
    assert mx.all(mask[2:, :2])
    assert mask[-1].sum().item() == 8


@pytest.mark.parametrize("kind", [KVCache, RotatingKVCache, BufferedRotatingKVCache])
def test_cache_position_offset_is_absolute(kind):
    cache = kind() if kind is KVCache else kind(8, keep=2)
    for _ in range(21):
        kv = mx.zeros((1, 1, 1, 8))
        cache.update_and_fetch(kv, kv)
    assert cache_position_offset(cache) == 21
    if kind is not KVCache:
        assert cache._idx != 21


@pytest.mark.parametrize("rotating", [False, True])
def test_cache_position_offset_includes_batch_padding(rotating):
    cache = (
        BatchRotatingKVCache(8, left_padding=[2, 3])
        if rotating
        else BatchKVCache(left_padding=[2, 3])
    )
    for _ in range(21):
        kv = mx.zeros((2, 1, 1, 8))
        cache.update_and_fetch(kv, kv)
    assert cache_position_offset(cache) == 21
    assert cache.offset.tolist() == [19, 18]


LANGUAGE_FAMILIES = [
    "qwen2",
    "qwen3",
    "qwen3_5",
    "qwen2_vl",
    "qwen2_5_vl",
    "qwen3_vl",
    "qwen3_vl_moe",
    "qwen3_omni_moe",
    "glm4v",
    "glm4v_moe",
    "glm_ocr",
    "ernie4_5_moe_vl",
    "paddleocr_vl",
    "falcon_ocr",
    "hunyuan_vl",
    "mllama",
    "moondream3",
    "minimax_m3_vl",
]


def tiny_language_model(name):
    config_module = importlib.import_module(f"mlx_vlm.models.{name}.config")
    language_module = importlib.import_module(f"mlx_vlm.models.{name}.language")
    config_type = getattr(config_module, "TextConfig", config_module.ModelConfig)
    values = dict(
        linear_num_value_heads=4,
        linear_num_key_heads=2,
        linear_key_head_dim=8,
        linear_value_head_dim=8,
        linear_conv_kernel_dim=4,
        full_attention_interval=1,
        rotary_dim=8,
        dense_intermediate_size=64,
        shared_intermediate_size=32,
        num_local_experts=2,
        moe_layer_freq=[0, 1],
        model_type=name,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        attention_head_dim=8,
        rms_norm_eps=1e-5,
        vocab_size=64,
        org_vocab_size=64,
        max_position_embeddings=256,
        rope_theta=10000.0,
        tie_word_embeddings=False,
        partial_rotary_factor=1.0,
        rope_dim=8,
        prefix_attn=0,
        cross_attention_layers=[],
        norm_topk_prob=True,
        num_experts=2,
        decoder_sparse_step=1,
        mlp_only_layers=[],
        num_experts_per_tok=1,
        moe_intermediate_size=32,
        n_group=1,
        topk_group=1,
        n_shared_experts=1,
        n_routed_experts=2,
        routed_scaling_factor=1.0,
        first_k_dense_replace=1,
        use_qk_norm=True,
        attention_bias=False,
        moe_num_experts=2,
        moe_k=1,
        moe_layer_start_index=1,
        moe_layer_end_index=2,
        moe_start_layer=1,
        mrope_section=[1, 1, 2],
    )
    rope = {"type": "default", "rope_type": "default", "mrope_section": [1, 1, 2]}
    if name not in ("qwen2", "qwen3", "mllama", "moondream3", "falcon_ocr"):
        values["rope_scaling"] = rope
    if name in ("glm_ocr", "qwen3_5"):
        values["rope_parameters"] = {
            **rope,
            "partial_rotary_factor": 1.0,
            "rope_theta": 10000.0,
        }
    if name == "hunyuan_vl":
        values["rope_scaling"] = {"type": "xdrope", "xdrope_section": [1, 1, 1, 1]}
    if name == "moondream3":
        values["num_key_value_heads"] = 4
    accepted = {f.name for f in fields(config_type)}
    config = config_type(**{k: v for k, v in values.items() if k in accepted})
    if name in ("qwen2", "qwen3", "mllama", "moondream3", "minimax_m3_vl"):
        return language_module.LanguageModel(config)
    vision_values = dict(
        model_type=name,
        depth=1,
        hidden_size=32,
        intermediate_size=64,
        num_heads=4,
        patch_size=2,
    )
    vision = config_module.VisionConfig(
        **{
            k: v
            for k, v in vision_values.items()
            if k in {f.name for f in fields(config_module.VisionConfig)}
        }
    )
    outer_type = (
        config_module.ThinkerConfig
        if name == "qwen3_omni_moe"
        else config_module.ModelConfig
    )
    extra = (
        {"audio_config": config_module.AudioConfig()}
        if name == "qwen3_omni_moe"
        else {}
    )
    outer = outer_type(
        text_config=config, vision_config=vision, model_type=name, **extra
    )
    return (
        language_module.LanguageModel(outer)
        if name == "hunyuan_vl"
        else language_module.LanguageModel(config, outer)
    )


@pytest.mark.parametrize("name", LANGUAGE_FAMILIES)
@pytest.mark.parametrize("keep", [0, 2])
def test_rotating_cache_random_model_chunk_parity(name, keep):
    mx.random.seed(17)
    model = tiny_language_model(name)
    model.eval()
    mx.eval(model.parameters())
    ids = (mx.arange(41)[None] * 7 + 3) % 64

    def forward(schedule):
        for attr in ("_rope_deltas", "_position_ids"):
            if hasattr(model, attr):
                setattr(model, attr, None)
        caches = [RotatingKVCache(8, keep=keep) for _ in model.layers]
        result, start = [], 0
        for n in schedule:
            logits = model(ids[:, start : start + n], cache=caches).logits
            mx.eval(logits, [c.state for c in caches])
            result.append(logits)
            start += n
        assert start == ids.shape[1]
        return np.array(mx.concatenate(result, axis=1))

    expected = forward([1] * 41)
    # Include a chunk larger than the cache and multi-token continuation after
    # single-token decoding has wrapped the storage cursor.
    for schedule in ([3] * 13 + [1, 1], [17, 7, 8, 8, 1], [1] * 19 + [5, 8, 8, 1]):
        np.testing.assert_allclose(forward(schedule), expected, atol=1e-4, rtol=1e-4)


@pytest.mark.parametrize(
    "name",
    [
        "qwen2_vl",
        "qwen2_5_vl",
        "qwen3_vl",
        "qwen3_vl_moe",
        "qwen3_omni_moe",
        "qwen3_5",
        "falcon_ocr",
    ],
)
def test_rotated_model_slices_full_prompt_metadata(name):
    mx.random.seed(19)
    model = tiny_language_model(name)
    model.eval()
    ids = (mx.arange(24)[None] * 7 + 3) % 64
    positions = mx.broadcast_to(mx.arange(24)[None, None, :], (3, 1, 24))
    positions = positions + mx.array([3, 7, 11])[:, None, None]
    if name == "falcon_ocr":
        positions = positions[0]
    metadata = {"position_ids": positions}
    if name in ("qwen3_vl", "qwen3_vl_moe", "qwen3_omni_moe"):
        metadata["visual_pos_masks"] = mx.ones((1, 24), mx.bool_)
        metadata["deepstack_visual_embeds"] = (
            mx.random.normal((1, 24, 2, 32))
            if name == "qwen3_omni_moe"
            else [mx.random.normal((24, 32)) for _ in range(2)]
        )

    def forward(full_metadata):
        for attr in ("_rope_deltas", "_position_ids"):
            if hasattr(model, attr):
                setattr(model, attr, None)
        cache = [RotatingKVCache(8, keep=2) for _ in model.layers]
        for start in range(21):
            out = model(ids[:, start : start + 1], cache=cache)
            mx.eval(out.logits)
        assert cache[0].offset == 21 and cache[0]._idx != 21
        kwargs = dict(metadata)
        if not full_metadata:
            kwargs["position_ids"] = positions[..., 21:24]
            if "visual_pos_masks" in kwargs:
                kwargs["visual_pos_masks"] = kwargs["visual_pos_masks"][:, 21:24]
                embeds = kwargs["deepstack_visual_embeds"]
                kwargs["deepstack_visual_embeds"] = (
                    [e[21:24] for e in embeds]
                    if isinstance(embeds, list)
                    else embeds[:, 21:24]
                )
        if name == "falcon_ocr":
            # Falcon's full-prompt metadata is passed with precomputed embeddings.
            kwargs["inputs_embeds"] = model.model.embed_tokens(ids[:, 21:24])
            if not full_metadata:
                # This wrapper accepts full position IDs only; compare the
                # internally selected positions against a direct model call.
                mask = create_attention_mask(kwargs["inputs_embeds"], cache)
                return np.array(
                    model.lm_head(
                        model.model(
                            ids[:, 21:24],
                            inputs_embeds=kwargs["inputs_embeds"],
                            position_ids=positions[..., 21:24],
                            mask=mask,
                            cache=cache,
                        )
                    )
                )
        return np.array(model(ids[:, 21:24], cache=cache, **kwargs).logits)

    np.testing.assert_allclose(forward(True), forward(False), atol=1e-4, rtol=1e-4)


def tiny_text_config(name):
    data = json.loads(Path(__file__).with_name("model_cases.json").read_text())
    case = next(
        c
        for c in data["cases"]
        if c["id"] == f"TestGetInputEmbeddings.{name}_input_embeddings"
    )
    module = importlib.import_module(f"mlx_vlm.models.{name}.config")
    return module.TextConfig(**case["config"]["text_config"])


def test_gemma3n_rotated_per_layer_inputs():
    from mlx_vlm.models.gemma3n.language import LanguageModel

    mx.random.seed(23)
    config = tiny_text_config("gemma3n")
    config.sliding_window = 8
    model = LanguageModel(config)
    # The constructor initializes this gate to zero, which would hide a bad
    # per-layer-input slice in an otherwise randomly initialized model.
    for layer in model.layers:
        layer.altup.correct_output_scale = mx.random.uniform(
            low=0.5, high=1.5, shape=(config.hidden_size,)
        )
    model.eval()
    ids = (mx.arange(41)[None] * 7 + 3) % config.vocab_size
    per_layer_inputs = model.model.get_per_layer_inputs(ids)

    def forward(step, full_metadata):
        cache, outputs = model.make_cache(), []
        for start in range(0, 41, step):
            metadata = (
                per_layer_inputs
                if full_metadata
                else per_layer_inputs[:, start : start + step]
            )
            out = model(
                ids[:, start : start + step], cache=cache, per_layer_inputs=metadata
            )
            mx.eval(out.logits)
            outputs.append(out.logits)
        return np.array(mx.concatenate(outputs, axis=1))

    expected = forward(1, False)
    np.testing.assert_allclose(forward(3, True), expected, atol=1e-4, rtol=1e-4)


def test_florence2_rotated_decoder_positions_and_mask():
    from mlx_vlm.models.florence2.language import Florence2Decoder

    mx.random.seed(29)
    config = tiny_text_config("florence2")
    decoder = Florence2Decoder(config)
    decoder.eval()
    encoder_states = mx.random.normal((1, 4, config.d_model))
    embeddings = mx.random.normal((1, 41, config.d_model))

    def forward(step):
        cache = [[RotatingKVCache(8, keep=2), KVCache()] for _ in decoder.layers]
        outputs = []
        for start in range(0, 41, step):
            out = decoder(
                inputs_embeds=embeddings[:, start : start + step],
                encoder_hidden_states=encoder_states,
                cache=cache,
            )
            mx.eval(out)
            outputs.append(out)
        return np.array(mx.concatenate(outputs, axis=1))

    np.testing.assert_allclose(forward(3), forward(1), atol=1e-4, rtol=1e-4)
