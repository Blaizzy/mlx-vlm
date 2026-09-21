"""JSON model contracts, checkpoint loading, sanitization, and document layout."""

from __future__ import annotations

import copy
import importlib
import inspect
import json
import logging
import math
import struct
import tempfile
import textwrap
import unittest
from contextlib import contextmanager
from operator import attrgetter
from pathlib import Path
from types import SimpleNamespace
from types import SimpleNamespace as NS
from unittest.mock import MagicMock, patch

import mlx.core as mx
import mlx.nn as nn
import numpy as np
import pytest
from mlx.utils import tree_flatten, tree_map

from mlx_vlm.models.base import InputEmbeddingsFeatures
from mlx_vlm.models.cache import make_prompt_cache
from mlx_vlm.utils import (
    _drop_modules_without_weights,
    _load_safetensors,
    get_model_and_args,
    get_model_path,
    load,
    load_config,
    load_model,
)

# Shared model contracts


def capture_positions(
    language_model, hidden_size, inputs, *, idx, offset, fa_idx=None, **kwargs
):
    class Recorder:
        embed_tokens = NS(as_linear=lambda x: x)

        def __call__(self, inputs, position_ids=None, **kwargs):
            self.positions = position_ids
            return mx.zeros((*inputs.shape, hidden_size))

    recorder = Recorder()
    if fa_idx is not None:
        recorder.fa_idx = fa_idx
    language_model.model = recorder
    language_model.lm_head = lambda x: x
    language_model(inputs, cache=[NS(_idx=idx, offset=mx.array(offset))], **kwargs)
    return recorder.positions


class ModelChecks:
    """Reusable assertions; each JSON case constructs fresh configs and models."""

    def forward_cache(self, model, vocab_size, *, chunk_sizes=()):
        model.eval()
        mx.eval(model.parameters())
        ids = mx.array([[1, 5, 9, 13, 2, 7, 11, 3]])
        assert model(ids).logits.shape == (1, 8, vocab_size)
        if chunk_sizes:
            for dtype in (mx.float32, mx.float16):
                model.update(tree_map(lambda p: p.astype(dtype), model.parameters()))
                self.prefill_parity(model, ids, {}, chunk_sizes)
            return
        cache = model.language_model.make_cache()
        model(ids[:, :-1], cache=cache)
        assert model(ids[:, -1:], cache=cache).logits.shape == (1, 1, vocab_size)

    def assert_close(self, actual, expected, *, logits=False):
        assert actual.shape == expected.shape
        assert mx.all(mx.isfinite(actual)).item()
        tolerance = 5e-3 if actual.dtype == mx.float16 else 1e-4
        # One-token attention uses a different Metal kernel from full prefill.
        if logits:
            tolerance = max(tolerance, 2e-3)
        np.testing.assert_allclose(
            np.array(actual.astype(mx.float32)),
            np.array(expected.astype(mx.float32)),
            atol=tolerance,
            rtol=tolerance,
        )

    def prefill_parity(self, model, ids, media, chunk_sizes):
        """Compare full, cached and chunked prefill, then decode one new token."""
        language = model.language_model
        features = model.get_input_embeddings(ids, **media).to_dict()
        embeds = features.pop("inputs_embeds")
        reference = model(ids, cache=make_prompt_cache(language), **media).logits
        if ids.shape[0] > 1:
            solo = []
            for row in range(ids.shape[0]):
                row_media = {
                    key: value[row * count : (row + 1) * count]
                    for key, value in media.items()
                    for count in [value.shape[0] // ids.shape[0]]
                }
                solo.append(
                    model(
                        ids[row : row + 1],
                        cache=make_prompt_cache(language),
                        **row_media,
                    ).logits
                )
            self.assert_close(reference, mx.concatenate(solo), logits=True)
        token = mx.full((ids.shape[0], 1), 7, mx.int32)
        extended = model(
            mx.concatenate([ids, token], axis=1),
            cache=make_prompt_cache(language),
            **media,
        ).logits[:, -1:]
        for size in (ids.shape[1] - 1, *chunk_sizes):
            cache, chunks = make_prompt_cache(language), []
            for start in range(0, ids.shape[1], size):
                stop = min(start + size, ids.shape[1])
                chunks.append(
                    language(
                        ids[:, start:stop],
                        inputs_embeds=embeds[:, start:stop],
                        cache=cache,
                        **features,
                    ).logits
                )
            self.assert_close(mx.concatenate(chunks, axis=1), reference, logits=True)
            decode = {k: v for k, v in features.items() if k != "position_ids"}
            self.assert_close(
                language(token, cache=cache, **decode).logits, extended, logits=True
            )

    def multimodal(
        self,
        model,
        config,
        *,
        image_grid,
        video_grid,
        chunk_sizes,
        batch_sizes=(1, 2),
        layouts=(("image",), ("video",), ("image", "video"), ("video", "image")),
    ):
        """Image/video fusion and DeepStack contracts across compatible models."""
        core = getattr(model, "thinker", model)
        config = getattr(config, "thinker_config", config)
        vision, vc = core.vision_tower, config.vision_config
        layers, width = len(vc.deepstack_visual_indexes), vc.out_hidden_size
        assert layers > 0, "The multimodal case must enable DeepStack layers"
        patch_width = vc.in_channels * vc.temporal_patch_size * vc.patch_size**2
        tokens = {"image": config.image_token_id, "video": config.video_token_id}
        grids = {"image": image_grid, "video": video_grid}
        model.eval()
        for dtype in (mx.float32, mx.float16):
            model.update(tree_map(lambda p: p.astype(dtype), model.parameters()))
            for layout in (tuple(layout) for layout in layouts):
                for batch in batch_sizes:
                    rows, media, encoded = [], {}, {}
                    for row in range(batch):
                        order = layout if row == 0 else layout[::-1]
                        ids = [1, 2]
                        for kind in order:
                            count = math.prod(grids[kind]) // vc.spatial_merge_size**2
                            ids += [config.vision_start_token_id] + [
                                tokens[kind]
                            ] * count
                            ids += [config.vision_end_token_id]
                        rows.append(ids + [3, 4, 5])
                    ids = mx.array(rows, mx.int32)
                    for kind in layout:
                        pixels = mx.random.normal(
                            (batch * math.prod(grids[kind]), patch_width)
                        ).astype(dtype)
                        grid = mx.array([grids[kind]] * batch)
                        media[
                            "pixel_values" if kind == "image" else "pixel_values_videos"
                        ] = pixels
                        media[kind + "_grid_thw"] = grid
                        encoded[kind] = vision(pixels, grid)
                    features = model.get_input_embeddings(ids, **media)
                    assert isinstance(features, InputEmbeddingsFeatures)
                    mask = (ids == tokens["image"]) | (ids == tokens["video"])
                    assert mx.array_equal(features.visual_pos_masks, mask).item()
                    residuals = features.deepstack_visual_embeds
                    if isinstance(residuals, (list, tuple)):
                        residuals = mx.stack(residuals)
                    assert residuals is not None and residuals.dtype == dtype
                    expected = mx.zeros((*ids.shape, layers, width), dtype)
                    expected_embeds = model.language_model.model.embed_tokens(ids)
                    for kind in layout:
                        positions = mx.array(
                            [
                                i
                                for i, t in enumerate(ids.flatten().tolist())
                                if t == tokens[kind]
                            ],
                            mx.uint32,
                        )
                        embeddings, layer_features = encoded[kind]
                        assert len(layer_features) == layers
                        flat = expected.reshape(-1, layers, width)
                        flat[positions] = mx.stack(list(layer_features), axis=1)
                        expected = flat.reshape(expected.shape)
                        flat = expected_embeds.reshape(-1, width)
                        flat[positions] = embeddings
                        expected_embeds = flat.reshape(expected_embeds.shape)
                    self.assert_close(features.inputs_embeds, expected_embeds)
                    if residuals.ndim == 4:
                        self.assert_close(residuals, expected)
                    else:
                        positions = mx.array(
                            [i for i, v in enumerate(mask.flatten().tolist()) if v],
                            mx.uint32,
                        )
                        self.assert_close(
                            residuals,
                            expected.reshape(-1, layers, width)[positions].transpose(
                                1, 0, 2
                            ),
                        )
                    for layer in range(layers):
                        value = (
                            residuals[:, :, layer]
                            if residuals.ndim == 4
                            else residuals[layer]
                        )
                        injected = model.language_model.model._deepstack_process(
                            mx.zeros_like(features.inputs_embeds), mask, value
                        )
                        self.assert_close(injected, expected[:, :, layer])
                    self.prefill_parity(model, ids, media, chunk_sizes)

    def input_embeddings(self, model, model_name):
        result = model.get_input_embeddings(input_ids=mx.array([[1, 2, 3, 4, 5]]))
        assert isinstance(result, InputEmbeddingsFeatures), model_name
        assert result.inputs_embeds is not None

    def request_positions(self, model):
        language = model.language_model
        positions, deltas = mx.array([[[9, 9, 9]]], mx.int32), mx.array(
            [[99]], mx.int32
        )
        language._position_ids, language._rope_deltas = positions, deltas
        result = model.get_input_embeddings(input_ids=mx.array([[1, 2, 3]], mx.int32))
        assert result.position_ids is not None and result.rope_deltas is not None
        assert result.position_ids.shape == (1, 3)
        assert result.position_ids.tolist() == [[0, 1, 2]]
        assert result.rope_deltas.tolist() == [[0]]
        assert mx.array_equal(language._position_ids, positions).item()
        assert mx.array_equal(language._rope_deltas, deltas).item()

    def chunked_positions(self, model):
        language = model.language_model
        width = language.args.hidden_size
        positions = mx.arange(15, dtype=mx.int32).reshape(3, 1, 5)
        actual = capture_positions(
            language,
            width,
            mx.array([[7, 8]], dtype=mx.int32),
            idx=2,
            offset=2,
            fa_idx=0,
            inputs_embeds=mx.zeros((1, 2, width), dtype=mx.float32),
            position_ids=positions,
        )
        assert actual.shape == (3, 1, 2)
        assert actual.tolist() == positions[:, :, 2:4].tolist()

    def language(self, model, config, *, num_layers=None):
        if num_layers is None:
            num_layers = config.num_hidden_layers
        assert model.model_type == config.model_type
        assert len(model.layers) == num_layers
        for dtype in (mx.float32, mx.float16):
            model.update(tree_map(lambda p: p.astype(dtype), model.parameters()))
            logits = model(mx.array([[0, 1]])).logits
            assert logits.shape == (1, 2, config.vocab_size) and logits.dtype == dtype
            logits = model(
                mx.argmax(logits[0, -1:, :], keepdims=True), cache=None
            ).logits
            assert logits.shape == (1, 1, config.vocab_size) and logits.dtype == dtype

    def projector(
        self, model, input_width, output_width, *, grid_hw=None, downsample_ratio=1
    ):
        batch = math.prod(grid_hw) if grid_hw else 1
        tokens = (
            math.prod(math.ceil(n / downsample_ratio) for n in grid_hw)
            if grid_hw
            else 1
        )
        kwargs = dict(zip(("n_h", "n_w"), grid_hw)) if grid_hw else {}
        for dtype in (mx.float32, mx.float16):
            model.update(tree_map(lambda p: p.astype(dtype), model.parameters()))
            inputs = mx.random.uniform(shape=(batch, input_width), dtype=dtype)
            output = model(mx.array(inputs), **kwargs)
            assert output.shape == (tokens, output_width) and output.dtype == dtype

    def vision(
        self,
        model,
        model_type,
        width,
        channels,
        image_size,
        vision_feature_layer=-2,
        channel_first=False,
        **kwargs,
    ):
        if model_type == "llama4_vision_model":
            width = kwargs.pop("projector_output_dim", width)
        batch = kwargs.pop("batch_size", 1)
        flat = (
            "qwen2_5_vl qwen3_5 qwen3_5_moe qwen4_exp "
            "glm4v_moe glm4v hunyuan_vl siglip2_vision_model"
        ).split()
        shape = (
            image_size
            if len(image_size) > 2 or model_type in flat
            else (
                (batch, channels, *image_size)
                if channel_first
                else (batch, *image_size, channels)
            )
        )
        parameters = inspect.signature(model.__call__).parameters
        for dtype in (mx.float32, mx.float16):
            model.update(tree_map(lambda p: p.astype(dtype), model.parameters()))
            if model_type is not None:
                assert model.model_type == model_type
            inputs = mx.random.uniform(shape=shape)
            if "image_masks" in parameters:
                inputs = inputs.transpose(0, 3, 1, 2)
                kwargs["image_masks"] = mx.ones((batch, channels, image_size[0]))
            options = (
                {"output_hidden_states": True}
                if "output_hidden_states" in parameters
                else {}
            )
            hidden = model(inputs.astype(dtype), **options, **kwargs)
            if vision_feature_layer is not None:
                hidden = hidden[vision_feature_layer]
            assert hidden.shape[1 if channel_first else -1] == width
            assert hidden.dtype == dtype

    def _assert_audio_features(self, features, shape, dtype):
        mx.eval(features)
        assert features.shape == shape
        assert features.dtype == dtype
        assert mx.all(mx.isfinite(features)).item()

    def audio(self, model, config, model_name, *, frames=32, lengths=None):
        if model_name not in {"inkling", "gemma3n", "gemma4", "gemma4_unified"}:
            raise ValueError(f"Unsupported audio model: {model_name}")
        audio_config = config.audio_config
        text_width = config.text_config.hidden_size
        lengths = [frames, frames // 2] if lengths is None else lengths
        assert lengths and all((0 <= n <= frames for n in lengths))
        batch = len(lengths)
        valid_mask = mx.arange(frames)[None, :] < mx.array(lengths)[:, None]
        components = [
            getattr(model, name, None) for name in ("audio_tower", "embed_audio")
        ]
        for dtype in (mx.float32, mx.float16):
            for component in components:
                if component is not None:
                    component.eval()
                    component.update(
                        tree_map(lambda p: p.astype(dtype), component.parameters())
                    )

            if model_name == "inkling":
                bins = audio_config.n_mel_bins
                inputs = mx.arange(batch * frames * bins, dtype=mx.int32)
                inputs = (
                    inputs.reshape(batch, frames, bins) % audio_config.mel_vocab_size
                )
                features = model.audio_tower(inputs)
                shape, output_dtype = (batch, frames, text_width), dtype
            elif model_name == "gemma4_unified":
                inputs = mx.random.normal(
                    (batch, frames, audio_config.output_proj_dims)
                ).astype(dtype)
                projected = model.get_audio_features(inputs)
                self._assert_audio_features(
                    projected, (batch * frames, text_width), dtype
                )
                features = model.get_audio_features(inputs, valid_mask)
                shape, output_dtype = (sum(lengths), text_width), dtype
            else:
                is_gemma3n = model_name == "gemma3n"
                # Gemma 4's subsampling projection requires 128 input mel bins.
                bins = audio_config.input_feat_size if is_gemma3n else 128
                inputs = mx.random.normal((batch, frames, bins)).astype(dtype)
                encoded, mask = model.audio_tower(inputs, ~valid_mask)
                stride = (
                    math.prod(s[0] for s in audio_config.sscp_conv_stride_size)
                    * max(1, audio_config.conf_reduction_factor)
                    if is_gemma3n
                    else 4
                )
                steps = (frames + stride - 1) // stride
                width = (
                    getattr(audio_config, "output_proj_dims", None)
                    or audio_config.hidden_size
                )
                # Both Gemma encoders currently promote float16 inputs to float32.
                output_dtype = mx.float32
                self._assert_audio_features(
                    encoded, (batch, steps, width), output_dtype
                )
                assert mask.shape == (batch, steps)
                assert mask.dtype == mx.bool_
                assert mx.array_equal(mask, ~valid_mask[:, ::stride]).item()
                assert mx.all(mx.where(mask[..., None], encoded == 0, True)).item()
                features = (
                    model.embed_audio(inputs_embeds=encoded)
                    if is_gemma3n
                    else model.embed_audio(encoded)
                )
                shape = (batch, steps, text_width)
            self._assert_audio_features(features, shape, output_dtype)

    def mrope_cache_index(self, language_model, hidden_size):
        """Use the Python cache index (10), avoiding a GPU sync on offset (3)."""
        language_model._rope_deltas = mx.array([[0]])
        language_model._position_ids = None
        positions = capture_positions(
            language_model, hidden_size, mx.array([[5]]), idx=10, offset=3
        )
        assert positions is not None
        assert tuple(positions.shape) in {(1, 1), (3, 1, 1)}
        assert positions.reshape(-1)[0].item() == 10

    def mrope_deltas(self, language_model, hidden_size):
        """The request's delta (5) must override stale model state (99)."""
        language_model._rope_deltas = mx.array([[99]])
        language_model._position_ids = None
        positions = capture_positions(
            language_model,
            hidden_size,
            mx.array([[7]]),
            idx=10,
            offset=3,
            fa_idx=0,
            rope_deltas=mx.array([[5]]),
        )
        assert positions is not None
        assert tuple(positions.shape) == (3, 1, 1)
        assert positions[0, 0, 0].item() == 15


CONFIG_TYPES = {
    "text_config": "TextConfig",
    "vision_config": "VisionConfig",
    "projector_config": "ProjectorConfig",
    "perceiver_config": "PerceiverConfig",
    "audio_config": "AudioConfig",
    "thinker_config": "ThinkerConfig",
    "talker_config": "TalkerConfig",
    "code_predictor_config": "CodePredictorConfig",
    "code2wav_config": "Code2WavConfig",
    "vit_config": "config.VitConfig",
    "adapter_config": "config.AdapterConfig",
}
DATA = json.loads(
    Path(__file__).with_name("model_cases.json").read_text(encoding="utf-8")
)
if DATA["version"] != 2:
    raise ValueError(f"Unsupported model_cases.json version: {DATA['version']}")

TINY_DEFAULTS = DATA["tiny_defaults"]
TINY_MODELS = DATA["shared_configs"]


def build_config(module, values, config_type="ModelConfig"):
    """Construct the model family's config classes from ordinary nested data."""
    fields = copy.deepcopy(values)
    for name, value in fields.items():
        if name in CONFIG_TYPES and isinstance(value, dict):
            fields[name] = build_config(module, value, CONFIG_TYPES[name])
    return attrgetter(config_type)(module)(**fields)


def first_attribute(obj, *names):
    for name in names:
        if hasattr(obj, name):
            return getattr(obj, name)
    raise AttributeError(f"{type(obj).__name__} has none of {names}")


def check_arguments(kind, case, model, config):
    """Keep shared component selection and dimension wiring in Python."""
    name = case["module"]
    # Phi3-V keeps language dimensions on the outer config.
    core_config = getattr(config, "thinker_config", config)
    text = (
        config if name == "phi3_v" else getattr(core_config, "text_config", core_config)
    )
    if kind == "forward_cache":
        return (model, text.vocab_size), case.get("forward_cache", {})
    if kind == "multimodal":
        return (model, config), case["multimodal"]
    if kind == "input_embeddings":
        return (model, name), {}
    if kind == "audio":
        return (model, config, name), case.get("audio", {})
    if kind in {"request_positions", "chunked_positions"}:
        return (model,), {}
    if kind in {"language", "mrope_cache_index", "mrope_deltas"}:
        if kind == "language":
            options = (
                {}
                if hasattr(text, "num_hidden_layers")
                else {"num_layers": text.n_layers}
            )
            return (model.language_model, text), options
        return (model.language_model, text.hidden_size), {}
    if kind == "projector":
        projector = attrgetter(case.get("projector_path", "multi_modal_projector"))(
            model
        )
        if name in ("deepseek_v4", "deepseek_v41"):
            return (
                projector,
                first_attribute(config, "vision_dim", "vision_hidden_size"),
                config.hidden_size,
            ), {
                "grid_hw": case["vision"]["grid_hw"],
                "downsample_ratio": config.vision_downsample_ratio,
            }
        return (projector, config.vision_config.hidden_size, text.hidden_size), {}
    if kind == "vision":
        vision = attrgetter(case.get("vision_path", "vision_tower"))(model)
        options = case.get("vision", {})
        if name in ("deepseek_v4", "deepseek_v41"):
            return (
                vision,
                None,
                first_attribute(config, "vision_dim", "vision_hidden_size"),
                3,
                tuple(options["input_shape"]),
            ), {
                "vision_feature_layer": options["feature_layer"],
                "n_h": options["grid_hw"][0],
                "n_w": options["grid_hw"][1],
            }
        vc = config.vision_config
        image_size = options.get("input_shape")
        if image_size is None:
            image_size = (vc.image_size, vc.image_size)
        width = first_attribute(
            vc, "out_hidden_size", "hidden_size", "d_model", "width", "text_hidden_size"
        )
        # Molmo's hidden_size describes its projector; use d_model for vision.
        if name == "molmo":
            width = vc.d_model
        channels = first_attribute(vc, "num_channels", "in_channels")
        kwargs = {}
        if "feature_layer" in options:
            kwargs["vision_feature_layer"] = options["feature_layer"]
        if "channel_first" in options:
            kwargs["channel_first"] = options["channel_first"]
        if "grid_thw" in options:
            kwargs["grid_thw"] = mx.array(
                options["grid_thw"],
                dtype=getattr(mx, options.get("grid_dtype", "int64")),
            )
        if vc.model_type == "llama4_vision_model":
            kwargs["projector_output_dim"] = vc.projector_output_dim
        # MiniMax's vision wrapper does not expose model_type.
        model_type = None if name == "minimax_m3_vl" else vc.model_type
        return (vision, model_type, width, channels, tuple(image_size)), kwargs
    raise ValueError(f"Unknown model check: {kind}")


@pytest.mark.parametrize("case", DATA["cases"], ids=lambda case: case["id"])
def test_model_contract(case):
    if "multimodal" in case["checks"]:
        mx.random.seed(17)
    module = importlib.import_module("mlx_vlm.models." + case["module"])
    config = build_config(module, case["config"])
    model = module.Model(config)
    checks = ModelChecks()
    for kind in case["checks"]:
        args, kwargs = check_arguments(kind, case, model, config)
        getattr(checks, kind)(*args, **kwargs)


@pytest.mark.parametrize("name", DATA["dense"])
def test_dense_model(name):
    module = importlib.import_module("mlx_vlm.models." + name)
    config = DATA["dense"][name]
    model = module.Model(module.ModelConfig.from_dict(copy.deepcopy(config)))
    ModelChecks().forward_cache(model, config["vocab_size"])


def tiny_config(family, profile=None, **overrides):
    """Build a fresh tiny config, optionally selecting a named test profile."""
    case = TINY_MODELS[family]
    fields = TINY_DEFAULTS | case["config"] | case.get("profiles", {}).get(profile, {})
    module = importlib.import_module("mlx_vlm.models." + case["module"])
    return build_config(module, fields | overrides, case["config_type"])


_DEEPSEEK_V41_BASE = {
    "vocab_size": 64,
    "hidden_size": 16,
    "num_hidden_layers": 1,
    "num_nextn_predict_layers": 0,
    "num_attention_heads": 2,
    "head_dim": 32,
    "qk_rope_head_dim": 8,
    "q_lora_rank": 8,
    "o_lora_rank": 4,
    "o_groups": 1,
    "sliding_window": 4,
    "compress_ratios": [0],
    "kv_source_layer_ids": [],
    "index_source_layer_ids": [],
    "index_n_heads": 2,
    "index_head_dim": 32,
    "index_topk": 4,
    "moe_intermediate_size": 8,
    "n_routed_experts": 4,
    "num_experts_per_tok": 2,
    "hc_mult": 2,
    "hc_sinkhorn_iters": 2,
    "dspark_target_layer_ids": [],
    "engram_layer_ids": [],
    "engram_num_embeddings": [256],
    "engram_max_ngram_size": 3,
    "engram_vocab_size": 512,
    "engram_n_heads": 2,
    "engram_head_dim": 8,
    "engram_compressed_vocab_size": 7,
}

_DEEPSEEK_V41_PROFILES = {
    "sparse": {
        "num_hidden_layers": 4,
        "compress_ratios": [2, 2, 1, 0],
        "kv_source_layer_ids": [0],
        "index_source_layer_ids": [0, 1, 2],
        "candidate_source_layer_id": 1,
        "candidate_topk_blocks": 2,
        "candidate_block_size": 2,
    },
    "stack": {
        "hidden_size": 32,
        "num_hidden_layers": 5,
        "compress_ratios": [0, 2, 2, 1, 0],
        "max_position_embeddings": 64,
        "sliding_window": 8,
        "qk_rope_head_dim": 4,
        "o_lora_rank": 8,
        "o_groups": 2,
        "moe_intermediate_size": 16,
        "kv_source_layer_ids": [1, 3],
        "index_source_layer_ids": [1, 2],
        "candidate_source_layer_id": 1,
        "candidate_topk_blocks": 4,
        "candidate_block_size": 2,
        "dspark_target_layer_ids": [3],
        "engram_layer_ids": [1],
        "engram_num_embeddings": [1024],
    },
    "vision": {
        "vision_hidden_size": 8,
        "vision_num_layers": 1,
        "vision_num_heads": 2,
        "vision_intermediate_size": 16,
        "vision_patch_size": 14,
        "vision_downsample_ratio": 3,
        "vision_min_pixels": 1,
        "vision_max_image_tokens": 1024,
    },
    "splice": {
        "vision_hidden_size": 16,
        "vision_num_layers": 1,
        "vision_num_heads": 2,
        "vision_intermediate_size": 32,
        "vision_patch_size": 2,
        "vision_downsample_ratio": 3,
    },
}


def deepseek_v41_config(profile=None, **overrides):
    """Tiny deepseek_v41 config for its own op/sanitize/engram/vision tests.

    Deliberately not a ``shared_configs`` entry: ``test_speculative`` parametrizes
    over every shared family and deepseek_v41 ships no speculative drafter, so it
    would enroll in contracts it cannot satisfy.
    """
    module = importlib.import_module("mlx_vlm.models.deepseek_v41")
    fields = (
        TINY_DEFAULTS | _DEEPSEEK_V41_BASE | _DEEPSEEK_V41_PROFILES.get(profile, {})
    )
    return build_config(module, fields | overrides, "ModelConfig")


def test_gemma3_can_preserve_caller_supplied_embedding_scale():
    case = next(case for case in DATA["cases"] if case["module"] == "gemma3")
    module = importlib.import_module(f"mlx_vlm.models.{case['module']}.language")
    config = build_config(
        module,
        case["config"]["text_config"]
        | TINY_DEFAULTS
        | dict(intermediate_size=32, head_dim=8, sliding_window_pattern=1),
        "TextConfig",
    )
    model = module.Gemma3Model(config, scale_inputs_embeds=False)
    capture = MagicMock(side_effect=lambda inputs, *_: inputs)
    model.layers, model.norm = [capture], nn.Identity()
    expected = mx.ones((1, 1, config.hidden_size))
    output = model(None, inputs_embeds=mx.array(expected))

    assert mx.array_equal(capture.call_args.args[0], expected)
    assert mx.array_equal(output, expected)


# Document layout


class TestPPDocLayoutV3(unittest.TestCase):
    def test_pp_doclayout_v3_config_routes(self):
        from mlx_vlm.models import pp_doclayout_v3
        from mlx_vlm.utils import get_model_and_args

        model_class, model_type = get_model_and_args(
            config={"model_type": "pp_doclayout_v3"}
        )
        assert model_class is pp_doclayout_v3
        assert model_type == "pp_doclayout_v3"
        assert pp_doclayout_v3.ModelConfig().num_labels == 25

        cfg = pp_doclayout_v3.ModelConfig.from_dict(
            {
                "model_type": "pp_doclayout_v3",
                "num_labels": 37,
                "id2label": {"0": "Question", "1": "Paragraph"},
                "backbone_config": {"model_type": "hgnet_v2"},
            }
        )
        assert cfg.id2label == {0: "Question", 1: "Paragraph"}
        assert cfg.num_queries == 300
        assert cfg.decoder_layers == 6

    def test_pp_doclayout_v3_decode_order(self):
        import mlx.core as mx

        from mlx_vlm.models.pp_doclayout_v3.decoder import decode_order

        # Chain 0 -> 1 -> 2 with strong pairwise scores.
        scores = mx.array([[-1e4, 5.0, 5.0], [-1e4, -1e4, 5.0], [-1e4, -1e4, -1e4]])
        assert decode_order(scores).tolist() == [0, 1, 2]
        # Reference formula cross-check on random input.
        rng_scores = mx.random.normal((7, 7))
        mx.eval(rng_scores)
        got = decode_order(rng_scores).tolist()
        import numpy as np

        arr = np.array(rng_scores.tolist())
        s = 1.0 / (1.0 + np.exp(-arr))
        votes = np.triu(s, 1).sum(0) + np.tril(1.0 - s.T, -1).sum(0)
        assert got == np.argsort(votes).tolist()

    def test_pp_doclayout_v3_mask_to_box(self):
        import mlx.core as mx

        from mlx_vlm.models.pp_doclayout_v3.decoder import mask_to_box_coordinate

        mask = mx.zeros((1, 2, 8, 10))
        mask[0, 0, 2:5, 3:7] = 1.0
        mx.eval(mask)
        boxes = mask_to_box_coordinate(mask)
        mx.eval(boxes)
        # x in [3,7), y in [2,5) over W=10,H=8 -> cxcywh
        self.assertAlmostEqual(float(boxes[0, 0, 0]), 0.5, places=5)
        self.assertAlmostEqual(float(boxes[0, 0, 1]), 0.4375, places=5)
        self.assertAlmostEqual(float(boxes[0, 0, 2]), 0.4, places=5)
        self.assertAlmostEqual(float(boxes[0, 0, 3]), 0.375, places=5)
        # Empty mask -> zeros.
        assert float(boxes[0, 1].sum()) == 0.0

    def test_pp_doclayout_v3_bilinear_upsample(self):
        import mlx.core as mx

        from mlx_vlm.models.pp_doclayout_v3.encoder import upsample_bilinear2x

        x = mx.array([[[[0.0], [1.0]], [[2.0], [3.0]]]])
        mx.eval(x)
        y = upsample_bilinear2x(x)
        mx.eval(y)
        # align_corners=False exact values: edges replicate, interior lerps.
        assert tuple(y.shape) == (1, 4, 4, 1)
        self.assertAlmostEqual(float(y[0, 0, 0, 0]), 0.0, places=5)
        self.assertAlmostEqual(float(y[0, 0, 1, 0]), 0.25, places=5)
        self.assertAlmostEqual(float(y[0, 1, 0, 0]), 0.5, places=5)
        self.assertAlmostEqual(float(y[0, 1, 1, 0]), 0.75, places=5)
        self.assertAlmostEqual(float(y[0, 3, 3, 0]), 3.0, places=5)

    def test_pp_doclayout_conversion_without_torch(self):
        import json
        import tempfile
        from pathlib import Path
        from unittest.mock import patch

        import mlx.core as mx

        from mlx_vlm.models.pp_doclayout_v3.convert import convert

        for dtype in (mx.float32, mx.bfloat16):
            with self.subTest(dtype=dtype), tempfile.TemporaryDirectory() as tmp:
                src = Path(tmp) / "source"
                src.mkdir()
                config = {"model_type": "pp_doclayout_v3"}
                (src / "config.json").write_text(json.dumps(config))
                weight = mx.arange(48).reshape(2, 3, 2, 4).astype(dtype)
                mx.save_safetensors(
                    str(src / "model.safetensors"),
                    {
                        "model.backbone.model.embedder.stem1.convolution.weight": weight,
                        "model.denoising_class_embed.weight": mx.ones((2, 2)),
                        "model.backbone.model.embedder.stem1.normalization.num_batches_tracked": mx.array(
                            0
                        ),
                    },
                )
                with (
                    patch.dict(
                        "sys.modules", {"torch": None, "safetensors.torch": None}
                    ),
                    patch("mlx_vlm.models.pp_doclayout_v3.convert._verify") as verify,
                ):
                    out = convert(str(src), str(Path(tmp) / "converted"))
                    again = convert(
                        str(out), str(Path(tmp) / "reconverted"), "bfloat16"
                    )
                assert verify.call_count == 2
                key = "backbone.embedder.stem1.conv.weight"
                expected = weight.transpose(0, 2, 3, 1)
                converted = mx.load(str(out / "model.safetensors"))
                reconverted = mx.load(str(again / "model.safetensors"))
                assert set(converted) == {key}
                assert set(reconverted) == {key}
                assert converted[key].dtype == mx.float32
                assert reconverted[key].dtype == mx.bfloat16
                assert converted[key].tolist() == expected.tolist()
                assert reconverted[key].tolist() == expected.tolist()
                assert json.loads((out / "config.json").read_text()) == config


# Loading and utility contracts


def test_load_config_applies_generation_config_sampling_defaults(tmp_path):
    generation_config = {
        "eos_token_id": [2, 3],
        "do_sample": True,
        "temperature": 1.0,
        "top_p": 0.95,
        "top_k": 64,
        "max_new_tokens": 4096,
    }
    (tmp_path / "config.json").write_text(
        json.dumps({"model_type": "demo", "eos_token_id": 1}), encoding="utf-8"
    )
    (tmp_path / "generation_config.json").write_text(
        json.dumps(generation_config), encoding="utf-8"
    )

    config = load_config(tmp_path)

    assert config["generation_config"] == generation_config
    assert config["eos_token_id"] == [2, 3]
    assert config["do_sample"] is True
    assert config["temperature"] == 1.0
    assert config["top_p"] == 0.95
    assert config["top_k"] == 64
    assert "max_new_tokens" not in config


def test_get_model_path_downloads_jsonl_tokenizers(monkeypatch, tmp_path):
    captured = {}

    def fake_snapshot_download(**kwargs):
        captured.update(kwargs)
        return str(tmp_path)

    monkeypatch.setattr("mlx_vlm.utils.snapshot_download", fake_snapshot_download)

    assert get_model_path("org/model") == tmp_path
    assert "*.jsonl" in captured["allow_patterns"]


def test_load_passes_revision():
    model_mock = MagicMock()
    model_mock.config = MagicMock(eos_token_id=None)
    processor_mock = MagicMock()

    with (
        patch("mlx_vlm.utils.get_model_path") as mock_get_model_path,
        patch("mlx_vlm.utils.load_model", return_value=model_mock),
        patch("mlx_vlm.utils.load_processor", return_value=processor_mock),
        patch("mlx_vlm.utils.load_image_processor", return_value=None),
    ):
        mock_get_model_path.return_value = Path("/tmp/model")

        model, processor = load("repo", revision="abc")

        assert model is model_mock
        assert processor is processor_mock
        mock_get_model_path.assert_called_with(
            "repo", revision="abc", force_download=False
        )


def test_get_model_and_args_rejects_unknown_text_configs():
    with pytest.raises(ValueError):
        get_model_and_args({"model_type": "unknown_text_arch"})


class TestDropModulesWithoutWeights:
    class ParameterlessHelper(nn.Module):
        pass

    class FakeModel(nn.Module):
        def __init__(self, config=None):
            super().__init__()
            self.config = config
            self.language_model = nn.Linear(2, 2, bias=False)
            self.vision_tower = nn.Linear(2, 2, bias=True)
            self.parameterless_helper = (
                TestDropModulesWithoutWeights.ParameterlessHelper()
            )

    def test_keeps_module_declared_in_manifest_but_not_loaded(self):
        # Manifest declares vision weights but none loaded -> keep, strict fails (#1963).
        model = self.FakeModel()
        weights = {"language_model.weight": mx.zeros((2, 2))}
        declared = {"language_model.weight", "vision_tower.weight", "vision_tower.bias"}

        _drop_modules_without_weights(model, weights, declared)

        assert model.vision_tower is not None
        with pytest.raises(ValueError, match="Missing"):
            model.load_weights(list(weights.items()), strict=True)

    def test_drops_module_absent_from_manifest(self, caplog):
        # The manifest also omits vision -> an intentional text-only conversion.
        model = self.FakeModel()
        weights = {"language_model.weight": mx.zeros((2, 2))}
        declared = {"language_model.weight"}

        with caplog.at_level(logging.WARNING):
            _drop_modules_without_weights(model, weights, declared)

        assert model.vision_tower is None
        assert "vision_tower" in caplog.text


def test_load_safetensors_reinterprets_f8_e8m0_header(tmp_path):
    path = tmp_path / "model.safetensors"
    header = {"weight": {"dtype": "F8_E8M0", "shape": [1], "data_offsets": [0, 1]}}
    header_bytes = json.dumps(header, separators=(",", ":")).encode("utf-8")
    path.write_bytes(struct.pack("<Q", len(header_bytes)) + header_bytes + b"\x00")

    loaded = {"weight": mx.array([1], dtype=mx.uint8)}

    def fake_mx_load(file_path):
        current = json.loads(path.read_bytes()[8 : 8 + len(header_bytes)])
        if current["weight"]["dtype"] == "F8_E8M0":
            raise RuntimeError("unsupported dtype F8_E8M0")
        assert current["weight"]["dtype"] == "U8"
        return loaded

    with patch("mlx_vlm.utils.mx.load", side_effect=fake_mx_load):
        assert _load_safetensors(str(path)) is loaded

    restored = json.loads(path.read_bytes()[8 : 8 + len(header_bytes)])
    assert restored["weight"]["dtype"] == "F8_E8M0"


class _CheckpointConfig:
    @classmethod
    def from_dict(cls, config):
        return cls()


class _CheckpointModel(nn.Module):
    def __init__(self, config, **modules):
        super().__init__()
        self.config = config
        for name, module in modules.items():
            setattr(self, name, module)

    def load_weights(self, weights, strict=True):
        self.loaded_weights, self.loaded_strict = weights, strict


@contextmanager
def _checkpoint_loading(config, model_class, weights, *, side_effect=None):
    with (
        patch("mlx_vlm.utils.load_config", return_value=config),
        patch("mlx_vlm.utils.glob.glob", return_value=["/tmp/model/model.safetensors"]),
        patch("mlx_vlm.utils._load_safetensors", return_value=weights),
        patch(
            "mlx_vlm.utils.get_model_and_args",
            return_value=(
                SimpleNamespace(ModelConfig=_CheckpointConfig, Model=model_class),
                config["model_type"],
            ),
        ),
        patch("mlx_vlm.utils.nn.quantize", side_effect=side_effect) as quantize,
    ):
        yield quantize


def test_load_model_uses_deepseek_v4_fp8_quantization_config():

    quantization = {
        "group_size": 64,
        "bits": 8,
        "mode": "affine",
        "language_model.weight": {"group_size": 64, "bits": 8, "mode": "affine"},
    }
    with (
        patch(
            "mlx_vlm.models.deepseek_v4.language.make_quantization_config",
            return_value=quantization,
        ) as make_quantization_config,
        _checkpoint_loading(
            {
                "model_type": "deepseek_v4",
                "quantization_config": {"quant_method": "fp8"},
            },
            lambda config: _CheckpointModel(
                config, language_model=nn.Linear(2, 2, bias=False)
            ),
            {},
        ) as quantize,
    ):
        model = load_model(Path("/tmp/model"), lazy=True)
    make_quantization_config.assert_called_once_with(model)
    quantize.assert_called_once()
    assert quantize.call_args.kwargs["group_size"] == 64
    assert quantize.call_args.kwargs["bits"] == 8
    assert quantize.call_args.kwargs["mode"] == "affine"


def test_load_model_transforms_fine_grained_fp8_by_format():

    source_config = {
        "model_type": "future_compatible_model",
        "quantization_config": {
            "quant_method": "fp8",
            "fmt": "e4m3",
            "weight_block_size": [128, 128],
        },
    }
    with _checkpoint_loading(
        source_config,
        lambda config: _CheckpointModel(config, proj=nn.Linear(128, 128, bias=False)),
        {
            "proj.weight": mx.zeros((128, 128), dtype=mx.uint8),
            "proj.weight_scale_inv": mx.ones((1, 1), dtype=mx.bfloat16),
        },
    ) as quantize:
        model = load_model(Path("/tmp/model"), lazy=True)
    quantize.assert_called_once()
    assert quantize.call_args.kwargs["group_size"] == 32
    assert quantize.call_args.kwargs["bits"] == 8
    assert quantize.call_args.kwargs["mode"] == "mxfp8"
    loaded = dict(model.loaded_weights)
    assert loaded["proj.weight"].dtype == mx.uint32
    assert loaded["proj.scales"].dtype == mx.uint8
    assert "proj.weight_scale_inv" not in loaded


def test_load_model_quantizes_projector_with_scales_when_skip_vision():

    def model(config):
        projector = nn.Module()
        projector.linear_1 = nn.Linear(64, 64, bias=False)
        return _CheckpointModel(
            config,
            vision_tower=nn.Linear(64, 64, bias=False),
            multi_modal_projector=projector,
            language_model=nn.Linear(64, 64, bias=False),
        )

    weights = {
        "language_model.weight": mx.zeros((64, 16), dtype=mx.uint32),
        "language_model.scales": mx.zeros((64, 1), dtype=mx.float16),
        "multi_modal_projector.linear_1.weight": mx.zeros((64, 16), dtype=mx.uint32),
        "multi_modal_projector.linear_1.scales": mx.zeros((64, 1), dtype=mx.float16),
        "vision_tower.weight": mx.zeros((64, 64), dtype=mx.float16),
    }
    selected = {}

    def fake_quantize(model, *args, **kwargs):
        predicate = kwargs["class_predicate"]
        selected["language"] = predicate("language_model", model.language_model)
        selected["projector"] = predicate(
            "multi_modal_projector.linear_1", model.multi_modal_projector.linear_1
        )
        selected["vision"] = predicate("vision_tower", model.vision_tower)

    with _checkpoint_loading(
        {
            "model_type": "kimi_vl",
            "quantization": {"group_size": 64, "bits": 8},
            "vision_config": {"skip_vision": True},
        },
        model,
        weights,
        side_effect=fake_quantize,
    ):
        load_model(Path("/tmp/model"), lazy=True)
    assert selected == {"language": True, "projector": True, "vision": False}


class _LazyPathLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.engram = nn.Module()
        self.engram.embed = nn.Embedding(8, 4)
        self.dense = nn.Linear(4, 4, bias=False)


class _LazyPathModel(nn.Module):
    """Two tables, one declared sparsely-gathered.

    Nested under ``layers`` so the flattened paths carry the leading dot the
    real markers rely on (``language_model.layers.N.engram.embed.weight``); a
    flat module would make the marker match nothing and the skip assertion
    pass vacuously.
    """

    lazy_parameter_paths = (".engram.embed.",)

    def __init__(self):
        super().__init__()
        self.layers = [_LazyPathLayer()]


def _materialized_paths(model):
    """Paths whose arrays are handed to mx.eval during materialization."""
    from mlx_vlm import utils as utils_module

    by_id = {
        id(v): p for p, v in tree_flatten(model.parameters()) if isinstance(v, mx.array)
    }
    return [
        by_id[id(array)]
        for array in utils_module._eager_parameters(model)
        if id(array) in by_id
    ]


def test_materialize_parameters_skips_declared_lazy_paths():
    """Declared paths are skipped; the undeclared table still materializes.

    The second assertion is the control: without it this would pass even if
    nothing were materialized at all.
    """
    seen = _materialized_paths(_LazyPathModel())
    assert not any(".engram.embed." in p for p in seen)
    assert any("dense." in p for p in seen)


def test_materialize_parameters_materializes_everything_without_a_declaration():
    """Control for the opt-out: no declaration means no skipping.

    Declared via a subclass rather than instance assignment: ``nn.Module`` is a
    dict subclass, so assigning the attribute on an instance writes into the
    dict while normal attribute lookup still finds the class attribute.
    """

    class _Undeclared(_LazyPathModel):
        lazy_parameter_paths = ()

    seen = _materialized_paths(_Undeclared())
    assert any(".engram.embed." in p for p in seen)


def test_deepseek_v41_declares_its_engram_tables_lazy():
    from mlx_vlm.models import deepseek_v41

    assert ".engram.embed." in deepseek_v41.Model.lazy_parameter_paths


# Local Python model files

MODEL_PY = textwrap.dedent("""
    import mlx.core as mx
    import mlx.nn as nn


    class ModelConfig:
        # Deliberately minimal: exposing text_config/vision_config attributes
        # opts in to update_module_configs, which then requires TextConfig /
        # VisionConfig classes in this module. A model_file module controls
        # both sides of that contract.
        def __init__(self, model_type="custom"):
            self.model_type = model_type

        @classmethod
        def from_dict(cls, params):
            return cls(model_type=params.get("model_type", "custom"))


    class Model(nn.Module):
        loaded_via_model_file = True

        def __init__(self, config):
            super().__init__()
            self.config = config
            self.proj = nn.Linear(4, 4, bias=False)

        def __call__(self, x):
            return self.proj(x)
    """)


def _write_checkpoint(path, config_extra=None):
    config = {"model_type": "does-not-exist-in-registry", "model_file": "model.py"}
    config.update(config_extra or {})
    (path / "config.json").write_text(json.dumps(config), encoding="utf-8")
    (path / "model.py").write_text(MODEL_PY, encoding="utf-8")
    mx.save_safetensors(
        str(path / "model.safetensors"),
        {"proj.weight": mx.zeros((4, 4))},
        metadata={"format": "mlx"},
    )


def test_load_model_uses_checkpoint_model_file(tmp_path):
    _write_checkpoint(tmp_path)

    model = load_model(tmp_path)

    # the class must come from the checkpoint's model.py, not the registry
    # (the registry would have raised: model_type does not exist there)
    assert getattr(model, "loaded_via_model_file", False) is True
    assert model.proj.weight.shape == (4, 4)


def test_missing_model_file_raises_clearly(tmp_path):
    _write_checkpoint(tmp_path)
    (tmp_path / "model.py").unlink()

    with pytest.raises(FileNotFoundError, match="model_file"):
        load_model(tmp_path)


# Patch embedding layouts


class TestDeepseekV41Aligner(unittest.TestCase):
    def test_deepseek_v41_aligner_reorders_channel_major_and_pads(self):
        """The reference groups each channel's r*r neighbours together.

        Its kernel reads `channel = d / 9`, `within = d % 9`, and contributes
        zero where a neighbour falls outside the grid. Laying the vector out
        position-major instead is a permutation of the same values, so nothing
        fails and the features stay plausible while the model cannot read them.
        """
        from mlx_vlm.models.deepseek_v41.vision import Aligner

        config = deepseek_v41_config("vision")
        aligner = Aligner(config)
        mx.eval(aligner.parameters())

        r = config.vision_downsample_ratio
        d = config.vision_hidden_size
        for n_h, n_w in ((3, 3), (4, 5), (7, 2)):
            x = mx.random.normal((n_h * n_w, d))
            source = x.reshape(n_h, n_w, d)
            h_out, w_out = -(-n_h // r), -(-n_w // r)
            want = mx.zeros((h_out * w_out, d * r * r)).tolist()
            for row in range(h_out * w_out):
                by, bx = divmod(row, w_out)
                for col in range(d * r * r):
                    channel, within = divmod(col, r * r)
                    sy, sx = by * r + within // r, bx * r + within % r
                    want[row][col] = (
                        float(source[sy, sx, channel].item())
                        if sy < n_h and sx < n_w
                        else 0.0
                    )
            expected = aligner.w2(aligner.act(aligner.w1(mx.array(want))))
            got = aligner(x, n_h, n_w)
            mx.eval(expected, got)
            self.assertEqual(got.shape, (h_out * w_out, config.hidden_size))
            diff = mx.abs(got.astype(mx.float32) - expected.astype(mx.float32))
            mx.eval(diff)
            self.assertLess(float(mx.max(diff).item()), 1e-4, f"{n_h}x{n_w}")

    def test_deepseek_v41_image_span_matches_the_aligner_row_count(self):
        """Every reserved IMAGE slot must have an aligner row behind it.

        The gather clamps out-of-range indices, so a span wider than the
        feature grid silently repeats the last feature instead of failing.
        """
        from PIL import Image

        from mlx_vlm.models.deepseek_v41.processing_deepseek_v41 import (
            IMAGE,
            load_image,
            prepare_vl_inputs,
        )

        config = deepseek_v41_config("vision")
        r = config.vision_downsample_ratio
        for size in ((896, 896), (640, 480), (128, 900), (42, 42)):
            image = Image.new("RGB", size, (10, 120, 200))
            _, n_vit_h, n_vit_w, _, _ = load_image(image, config)
            _, types, spans = prepare_vl_inputs(
                [1, config.image_token_id, 2], [image], config
            )
            rows = (-(-n_vit_h // r)) * (-(-n_vit_w // r))
            self.assertEqual(types.count(IMAGE), rows, f"{size}")
            self.assertEqual(len(spans), 1)
            self.assertEqual(spans[0].patches.shape[0], n_vit_h * n_vit_w)


QWEN_PATCH_EMBED_KEY = "model.visual.patch_embed.proj.weight"


QWEN_SANITIZED_KEY = "vision_tower.patch_embed.proj.weight"


def _qwen_patch_model(
    in_channels=3, temporal_patch_size=2, patch_size=4, hidden_size=8
):
    qwen = importlib.import_module("mlx_vlm.models.qwen3_5")
    text_config = qwen.TextConfig(
        model_type="qwen3_5_text",
        hidden_size=32,
        intermediate_size=64,
        linear_num_value_heads=4,
        linear_num_key_heads=2,
        linear_key_head_dim=8,
        linear_value_head_dim=8,
        linear_conv_kernel_dim=4,
        num_hidden_layers=2,
        num_attention_heads=2,
        rms_norm_eps=1e-6,
        vocab_size=64,
        num_key_value_heads=1,
        max_position_embeddings=128,
        full_attention_interval=2,
        head_dim=16,
    )
    vision_config = qwen.VisionConfig(
        model_type="qwen3_5",
        depth=1,
        hidden_size=hidden_size,
        intermediate_size=16,
        out_hidden_size=32,
        num_heads=1,
        in_channels=in_channels,
        patch_size=patch_size,
        temporal_patch_size=temporal_patch_size,
        spatial_merge_size=1,
        num_position_embeddings=4,
    )
    config = qwen.ModelConfig(
        text_config=text_config, vision_config=vision_config, model_type="qwen3_5"
    )
    return qwen.Model(config), vision_config


def test_patch_embed_is_transposed_from_ncdhw_to_ndhwc():
    """Qwen3.5 stores the Conv3d patch embed as NCDHW; MLX expects NDHWC."""
    model, vision_config = _qwen_patch_model()
    expected = model.vision_tower.patch_embed.proj.weight.shape

    ncdhw = mx.zeros(
        (
            vision_config.hidden_size,
            vision_config.in_channels,
            vision_config.temporal_patch_size,
            vision_config.patch_size,
            vision_config.patch_size,
        ),
        dtype=mx.bfloat16,
    )
    sanitized = model.sanitize({QWEN_PATCH_EMBED_KEY: ncdhw})

    assert sanitized[QWEN_SANITIZED_KEY].shape == expected


def test_glm_quantized_head_sanitization_loads_strictly():
    module = importlib.import_module("mlx_vlm.models.glm5_next")
    model = module.Model(
        module.ModelConfig(
            text_config=tiny_config("glm", hidden_size=32),
            vision_config=module.VisionConfig(
                depth=1,
                hidden_size=16,
                intermediate_size=32,
                out_hidden_size=32,
                projection_intermediate_size=32,
                num_heads=2,
                image_size=8,
                patch_size=2,
            ),
        )
    )
    nn.quantize(
        model,
        group_size=32,
        bits=4,
        class_predicate=lambda path, _: path.endswith("language_model.lm_head"),
    )
    checkpoint = dict(tree_flatten(model.parameters()))
    head = {
        f"lm_head.{suffix}": checkpoint.pop(f"language_model.lm_head.{suffix}")
        for suffix in ("weight", "scales", "biases")
    }
    sanitized = model.sanitize(head)
    assert sanitized.keys() == {
        f"language_model.lm_head.{suffix}" for suffix in ("weight", "scales", "biases")
    }
    again = model.sanitize(sanitized)
    assert again.keys() == sanitized.keys()
    for key, value in sanitized.items():
        assert mx.array_equal(again[key], value).item()
    model.load_weights(list(model.sanitize(checkpoint | head).items()), strict=True)


class TestMistralLarge3(unittest.TestCase):
    """Mistral Large 3: params.json config mapping and native->4-bit load."""

    DIMS = dict(
        H=128,
        DI=128,
        MI=64,
        NH=2,
        NP=64,
        RP=64,
        VH=64,
        QL=64,
        KL=64,
        V=128,
        VHID=64,
        VNH=2,
        VI=64,
        PATCH=14,
        IMG=28,
        SM=2,
        NL=5,
        NE=8,
        NS=1,
        DENSE=3,
    )

    def _config(self):
        from mlx_vlm.models.mistral_large3 import ModelConfig

        d = self.DIMS
        return ModelConfig(
            text_config=dict(
                model_type="mistral_large3",
                vocab_size=d["V"],
                hidden_size=d["H"],
                intermediate_size=d["DI"],
                moe_intermediate_size=d["MI"],
                num_hidden_layers=d["NL"],
                num_attention_heads=d["NH"],
                num_key_value_heads=d["NH"],
                q_lora_rank=d["QL"],
                kv_lora_rank=d["KL"],
                qk_nope_head_dim=d["NP"],
                qk_rope_head_dim=d["RP"],
                v_head_dim=d["VH"],
                n_routed_experts=d["NE"],
                n_shared_experts=d["NS"],
                num_experts_per_tok=4,
                first_k_dense_replace=d["DENSE"],
                rms_norm_eps=1e-6,
                rope_theta=1e4,
            ),
            vision_config=dict(
                model_type="pixtral",
                hidden_size=d["VHID"],
                num_hidden_layers=d["VNH"],
                num_attention_heads=d["VNH"],
                head_dim=d["VHID"] // d["VNH"],
                intermediate_size=d["VI"],
                image_size=d["IMG"],
                patch_size=d["PATCH"],
                rope_theta=1e4,
            ),
            image_token_id=10,
            spatial_merge_size=d["SM"],
            multimodal_projector_bias=False,
            vocab_size=d["V"],
        )

    def _native_source(self):
        d = self.DIMS
        H, DI, MI, NH, NP, RP, VH, QL, KL, V = (
            d["H"],
            d["DI"],
            d["MI"],
            d["NH"],
            d["NP"],
            d["RP"],
            d["VH"],
            d["QL"],
            d["KL"],
            d["V"],
        )
        VHID, VI, SM = d["VHID"], d["VI"], d["SM"]

        def z(*s):
            return (mx.random.normal(s) * 0.02).astype(mx.bfloat16)

        w = {
            "tok_embeddings.weight": z(V, H),
            "norm.weight": z(H),
            "output.weight": z(V, H),
        }
        qhd = NP + RP
        for l in range(d["NL"]):
            p = f"layers.{l}"
            w[f"{p}.attention_norm.weight"] = z(H)
            w[f"{p}.ffn_norm.weight"] = z(H)
            w[f"{p}.attention.wq_a.weight"] = z(QL, H)
            w[f"{p}.attention.q_a_norm.weight"] = z(QL)
            w[f"{p}.attention.wq_b.weight"] = z(NH * qhd, QL)
            w[f"{p}.attention.wkv_a_with_mqa.weight"] = z(KL + RP, H)
            w[f"{p}.attention.kv_a_norm.weight"] = z(KL)
            w[f"{p}.attention.wkv_b.weight"] = z(NH * (NP + VH), KL)
            w[f"{p}.attention.wo.weight"] = z(H, NH * VH)
            if l < d["DENSE"]:
                w[f"{p}.feed_forward.w1.weight"] = z(DI, H)
                w[f"{p}.feed_forward.w2.weight"] = z(H, DI)
                w[f"{p}.feed_forward.w3.weight"] = z(DI, H)
            else:
                w[f"{p}.gate.weight"] = z(d["NE"], H)
                for e in range(d["NE"]):
                    w[f"{p}.experts.{e}.w1.weight"] = z(MI, H)
                    w[f"{p}.experts.{e}.w2.weight"] = z(H, MI)
                    w[f"{p}.experts.{e}.w3.weight"] = z(MI, H)
                w[f"{p}.shared_experts.w1.weight"] = z(MI, H)
                w[f"{p}.shared_experts.w2.weight"] = z(H, MI)
                w[f"{p}.shared_experts.w3.weight"] = z(MI, H)
        w["vision_encoder.patch_conv.weight"] = z(VHID, 3, d["PATCH"], d["PATCH"])
        w["vision_encoder.ln_pre.weight"] = z(VHID)
        for n in range(d["VNH"]):
            p = f"vision_encoder.transformer.layers.{n}"
            for wk in ("wq", "wk", "wv", "wo"):
                w[f"{p}.attention.{wk}.weight"] = z(VHID, VHID)
            w[f"{p}.attention_norm.weight"] = z(VHID)
            w[f"{p}.feed_forward.w1.weight"] = z(VI, VHID)
            w[f"{p}.feed_forward.w2.weight"] = z(VHID, VI)
            w[f"{p}.feed_forward.w3.weight"] = z(VI, VHID)
            w[f"{p}.ffn_norm.weight"] = z(VHID)
        w["pre_mm_projector_norm.weight"] = z(VHID)
        w["patch_merger.merging_layer.weight"] = z(VHID, VHID * SM * SM)
        w["vision_language_adapter.w_in.weight"] = z(H, VHID)
        w["vision_language_adapter.w_out.weight"] = z(H, H)
        return w

    def test_config_from_params_maps_native_fields(self):
        from mlx_vlm.models.mistral_large3 import ModelConfig
        from mlx_vlm.models.mistral_large3.config import config_from_params

        params = dict(
            dim=7168,
            hidden_dim=16384,
            n_layers=61,
            n_heads=128,
            n_kv_heads=128,
            q_lora_rank=1536,
            kv_lora_rank=512,
            qk_nope_head_dim=128,
            qk_rope_head_dim=64,
            v_head_dim=128,
            norm_eps=1e-6,
            rope_theta=1e4,
            vocab_size=131072,
            max_position_embeddings=294912,
            moe=dict(
                expert_hidden_dim=4096,
                num_experts=128,
                num_shared_experts=1,
                num_experts_per_tok=4,
                first_k_dense_replace=3,
            ),
            vision_encoder=dict(
                hidden_size=1664,
                num_hidden_layers=48,
                num_attention_heads=16,
                intermediate_size=8192,
                image_size=1540,
                patch_size=14,
                rope_theta=1e4,
                image_token_id=10,
                spatial_merge_size=2,
                adapter_bias=False,
            ),
        )
        cfg = ModelConfig(**config_from_params(params))
        assert cfg.text_config.n_routed_experts == 128
        assert cfg.text_config.moe_intermediate_size == 4096
        assert cfg.text_config.first_k_dense_replace == 3
        assert cfg.text_config.q_lora_rank == 1536
        assert cfg.vision_config.hidden_size == 1664
        assert cfg.vision_config.head_dim == 1664 // 16
        assert cfg.image_token_id == 10

    def test_native_4bit_roundtrip_loads_and_runs(self):
        cfg = self._config()

        def quantizable(name):
            if not name.endswith(".weight") or name.startswith(
                (
                    "vision_encoder.",
                    "patch_merger.",
                    "vision_language_adapter.",
                    "pre_mm_projector",
                )
            ):
                return False
            if ".attention." in name:
                return any(
                    f".{p}.weight" in name
                    for p in ("wq_a", "wq_b", "wkv_a_with_mqa", "wkv_b", "wo")
                )
            return name.endswith((".w1.weight", ".w2.weight", ".w3.weight"))

        converted = {}
        for name, w in self._native_source().items():
            if quantizable(name):
                qw, sc, bi = mx.quantize(w, group_size=64, bits=4)
                base = name[: -len(".weight")]
                converted[name] = qw
                converted[base + ".scales"] = sc
                converted[base + ".biases"] = bi
            else:
                converted[name] = w

        from mlx_vlm.models.mistral_large3 import Model

        model = Model(cfg)
        weights = model.sanitize(converted)
        nn.quantize(
            model,
            group_size=64,
            bits=4,
            class_predicate=lambda p, m: hasattr(m, "to_quantized")
            and f"{p}.scales" in weights,
        )
        expected = set(dict(tree_flatten(model.parameters())))
        got = set(weights)
        assert expected == got, (expected - got, got - expected)
        model.load_weights(list(weights.items()), strict=True)

        out = model(mx.array([[1, 2, 3, 4]]), pixel_values=None)
        logits = out.logits if hasattr(out, "logits") else out
        assert tuple(logits.shape) == (1, 4, self.DIMS["V"])

    def test_sanitize_is_idempotent(self):
        from mlx_vlm.models.mistral_large3 import Model

        model = Model(self._config())
        final = dict(tree_flatten(model.parameters()))
        assert model.sanitize(final).keys() == final.keys()


class TestMoondream2Sanitize(unittest.TestCase):
    """Weight-key remapping for the moondream2 port."""

    def _tiny_model(self):
        from mlx_vlm.models.moondream2 import Model, ModelConfig

        config = ModelConfig.from_dict(
            {
                "model_type": "moondream2",
                "text_config": {
                    "num_hidden_layers": 1,
                    "hidden_size": 64,
                    "intermediate_size": 128,
                    "num_attention_heads": 2,
                    "num_key_value_heads": 2,
                    "vocab_size": 128,
                },
                "vision_config": {
                    "num_hidden_layers": 1,
                    "hidden_size": 64,
                    "intermediate_size": 128,
                    "num_attention_heads": 2,
                },
            }
        )
        return Model(config)

    def test_layout_is_remapped(self):
        model = self._tiny_model()
        sanitized = model.sanitize(
            {
                "model.text.wte": mx.zeros((1,)),
                "model.text.blocks.0.attn.qkv.weight": mx.zeros((1,)),
                "model.text.post_ln.weight": mx.zeros((1,)),
                "model.text.lm_head.weight": mx.zeros((1,)),
                "model.vision.patch_emb.weight": mx.zeros((1,)),
                "model.vision.blocks.0.ln1.weight": mx.zeros((1,)),
                "model.vision.proj_mlp.fc1.weight": mx.zeros((1,)),
                "model.region.coord_decoder.fc1.weight": mx.zeros((1,)),
            }
        )
        self.assertIn("text.model.embed_tokens.weight", sanitized)
        self.assertIn("text.model.layers.0.attn.qkv.weight", sanitized)
        self.assertIn("text.model.post_ln.weight", sanitized)
        self.assertIn("text.lm_head.weight", sanitized)
        self.assertIn("vision.encoder.patch_emb.weight", sanitized)
        self.assertIn("vision.encoder.blocks.0.ln1.weight", sanitized)
        self.assertIn("vision.proj_mlp.fc1.weight", sanitized)
        self.assertNotIn("model.region.coord_decoder.fc1.weight", sanitized)


class TestMoondream3Sanitize(unittest.TestCase):
    """sanitize must be idempotent so already-converted mlx quants load."""

    def _tiny_model(self):
        from mlx_vlm.models.moondream3 import Model, ModelConfig

        config = ModelConfig.from_dict(
            {
                "model_type": "moondream3",
                "text_config": {
                    "num_hidden_layers": 1,
                    "hidden_size": 64,
                    "intermediate_size": 128,
                    "num_attention_heads": 2,
                    "num_key_value_heads": 2,
                    "head_dim": 32,
                    "vocab_size": 128,
                    "num_experts": 2,
                    "num_experts_per_tok": 1,
                    "moe_intermediate_size": 32,
                    "moe_start_layer": 1,
                },
                "vision_config": {
                    "num_hidden_layers": 1,
                    "hidden_size": 64,
                    "intermediate_size": 128,
                    "num_attention_heads": 2,
                },
            }
        )
        return Model(config)

    def test_already_converted_keys_pass_through(self):
        model = self._tiny_model()
        keys = {
            "text.model.blocks.0.attn.qkv.weight": mx.zeros((1,)),
            "text.lm_head.weight": mx.zeros((1,)),
            "vision.encoder.blocks.0.ln1.weight": mx.zeros((1,)),
            "vision.proj_mlp.fc1.weight": mx.zeros((1,)),
        }
        once = model.sanitize(dict(keys))
        self.assertEqual(set(once), set(keys))
        twice = model.sanitize(once)
        self.assertEqual(set(twice), set(once))

    def test_raw_keys_are_remapped(self):
        model = self._tiny_model()
        sanitized = model.sanitize(
            {
                "model.text.blocks.0.attn.qkv.weight": mx.zeros((1,)),
                "model.vision.blocks.0.ln1.weight": mx.zeros((1,)),
            }
        )
        self.assertIn("text.model.blocks.0.attn.qkv.weight", sanitized)
        self.assertIn("vision.encoder.blocks.0.ln1.weight", sanitized)


# DeepSeek-V4.1 engram, vision splice and checkpoint keys


class TestDeepseekV41Config(unittest.TestCase):
    def test_deepseek_v41_config_defaults(self):
        from mlx_vlm.models import deepseek_v41

        config = deepseek_v41.ModelConfig()
        self.assertEqual(config.model_type, "deepseek_v41")
        self.assertEqual(
            len(config.compress_ratios),
            config.num_hidden_layers + config.num_nextn_predict_layers,
        )
        self.assertEqual(config.compress_ratios[2], 2)
        self.assertEqual(config.compress_ratios[20], 1)
        self.assertEqual(config.compress_ratios[40:], [0, 0, 0])
        self.assertEqual(config.kv_source_layer_ids, [2, 8, 14, 20])

        loaded = deepseek_v41.ModelConfig.from_dict(
            {"model_type": "deepseek_v41", "eos_token_id": 1}
        )
        self.assertEqual(loaded.eos_token_id, 1)


class TestDeepseekV41EndToEnd(unittest.TestCase):
    @staticmethod
    def _config():
        return deepseek_v41_config(
            "sparse",
            num_hidden_layers=2,
            compress_ratios=[2, 1],
            engram_layer_ids=[1],
            dspark_target_layer_ids=[1],
            dspark_block_size=2,
        )

    def test_deepseek_v41_full_forward(self):
        """Shapes come from the JSON contract; this pins the extras it cannot.

        A fresh cache has to reproduce the prefill it replaces, which only says
        anything once the head is not its zero initialization.
        """
        from mlx_vlm.models import deepseek_v41

        model = deepseek_v41.Model(self._config())
        model.language_model.head.weight = (
            mx.random.normal(model.language_model.head.weight.shape) * 0.05
        )
        mx.eval(model.parameters())
        self.assertEqual(len(model.layers), 2)

        cache = model.make_cache()
        ids = mx.array([[1, 2, 3]])
        out = model.language_model(ids, cache=cache)
        mx.eval(out.logits)
        self.assertIsNotNone(out.hidden_states)
        self.assertEqual(out.hidden_states[0].shape, (1, 3, 16))

        model.language_model(mx.array([[4]]), cache=cache)
        again = model.language_model(ids, cache=model.make_cache())
        mx.eval(again.logits)
        self.assertTrue(bool(mx.allclose(out.logits, again.logits)))


class TestDeepseekV41Sanitize(unittest.TestCase):
    @staticmethod
    def _config():
        return deepseek_v41_config(hidden_size=8, o_groups=2, o_lora_rank=4)

    def test_deepseek_v41_sanitize_checkpoint_keys(self):
        from mlx_vlm.models import deepseek_v41

        model = deepseek_v41.Model(self._config())
        sanitized = model.sanitize(
            {
                "embed.weight": mx.zeros((4, 4)),
                "head.weight": mx.zeros((4, 4)),
                "layers.0.attn.wq_a.weight": mx.zeros((4, 4)),
                "model.embed_tokens.weight": mx.zeros((4, 4)),
                "lm_head.weight": mx.zeros((4, 4)),
                "vision.blocks.0.attn.wqkv.weight": mx.zeros((4, 4)),
                "aligner.w1.weight": mx.zeros((4, 4)),
                "image_start": mx.zeros((4,)),
                "mtp.0.attn.wq_a.weight": mx.zeros((4, 4)),
            }
        )
        self.assertIn("language_model.embed_tokens.weight", sanitized)
        self.assertIn("language_model.head.weight", sanitized)
        self.assertIn("language_model.layers.0.attn.wq_a.weight", sanitized)
        self.assertIn("language_model.model.embed_tokens.weight", sanitized)
        self.assertIn("language_model.lm_head.weight", sanitized)
        self.assertIn("vision.blocks.0.attn.wqkv.weight", sanitized)
        self.assertIn("aligner.w1.weight", sanitized)
        self.assertIn("image_start", sanitized)
        self.assertFalse(any(k.startswith("mtp.") for k in sanitized))

    def test_deepseek_v41_sanitize_stacks_experts(self):
        from mlx_vlm.models import deepseek_v41

        model = deepseek_v41.Model(self._config())
        weights = {}
        for e in range(4):
            for src in ("w1", "w2", "w3"):
                for suffix in ("weight", "scales", "biases"):
                    weights[f"layers.0.ffn.experts.{e}.{src}.{suffix}"] = mx.zeros(
                        (2, 2)
                    )
        weights["layers.0.ffn.shared_experts.w1.weight"] = mx.zeros((2, 2))
        weights["layers.0.attn.wo_a.weight"] = mx.zeros((16, 8))
        weights["layers.0.attn.q_norm.weight"] = mx.zeros((8,))
        out = model.sanitize(weights)
        mx.eval(out)
        self.assertEqual(
            out["language_model.layers.0.ffn.switch_mlp.gate_proj.weight"].shape,
            (4, 2, 2),
        )
        self.assertEqual(
            out["language_model.layers.0.ffn.switch_mlp.down_proj.scales"].shape,
            (4, 2, 2),
        )
        self.assertIn(
            "language_model.layers.0.ffn.shared_experts.gate_proj.weight", out
        )
        self.assertEqual(
            out["language_model.layers.0.attn.wo_a.weight"].shape, (2, 4, 16)
        )
        self.assertFalse(any(".experts.0." in k for k in out))

    def test_deepseek_v41_sanitize_dequantizes_head(self):
        from mlx_vlm.models import deepseek_v41

        model = deepseek_v41.Model(self._config())
        wq, scales, biases = mx.quantize(
            mx.random.normal((64, 128)), group_size=64, bits=4, mode="affine"
        )
        out = model.sanitize(
            {
                "head.weight": wq,
                "head.scales": scales,
                "head.biases": biases,
            }
        )
        self.assertEqual(set(out), {"language_model.head.weight"})
        self.assertEqual(out["language_model.head.weight"].dtype, mx.float32)
        self.assertEqual(out["language_model.head.weight"].shape, (64, 128))


class TestDeepseekV41VisionSplice(unittest.TestCase):
    @staticmethod
    def _image_record():
        return SimpleNamespace(
            start=1,
            patches=mx.random.normal((36, 3, 2, 2)),
            n_vit_h=6,
            n_vit_w=6,
            types=[0, 1, 1, 2, 1, 1, 2, 3],
        )

    def test_deepseek_v41_vision_splice(self):
        from mlx_vlm.models import deepseek_v41

        model = deepseek_v41.Model(deepseek_v41_config("splice"))
        mx.eval(model.parameters())
        model.image_start = mx.ones((16,))
        model.image_end = mx.ones((16,)) * 2
        model.image_newline = mx.ones((16,)) * 3

        ids = mx.array([[5, 9, 9, 9, 9, 9, 9, 9, 9, 6]])
        result = model.get_input_embeddings(ids, pixel_values=[[self._image_record()]])
        mx.eval(result.inputs_embeds)
        embeds = result.inputs_embeds
        self.assertEqual(embeds.shape, (1, 10, 16))
        self.assertTrue(
            bool(mx.all(embeds[0, 0] == model.language_model.embed_tokens(ids)[0, 0]))
        )
        self.assertTrue(bool(mx.all(embeds[0, 1] == 1)))
        self.assertTrue(bool(mx.all(embeds[0, 4] == 3)))
        self.assertTrue(bool(mx.all(embeds[0, 8] == 2)))
        self.assertTrue(
            bool(mx.all(embeds[0, 9] == model.language_model.embed_tokens(ids)[0, 9]))
        )

    def test_deepseek_v41_decode_skips_vision(self):
        from mlx_vlm.models import deepseek_v41

        model = deepseek_v41.Model(deepseek_v41_config("splice"))
        mx.eval(model.parameters())
        ids = mx.array([[5]])
        result = model.get_input_embeddings(ids, pixel_values=[[self._image_record()]])
        mx.eval(result.inputs_embeds)
        self.assertTrue(
            bool(mx.all(result.inputs_embeds == model.language_model.embed_tokens(ids)))
        )


class TestDeepseekV41Engram(unittest.TestCase):
    """Engram layers must actually run.

    They were wired up but unreachable: the language model was built without a
    tokenizer, so the hash state stayed None and the call site's guard was never
    true. Nothing failed - the weights loaded and the layers were skipped.
    """

    @staticmethod
    def _config(**overrides):
        return deepseek_v41_config("stack", **overrides)

    @staticmethod
    def _with_hash_state(model, config):
        from mlx_vlm.models.deepseek_v41 import engram as engram_mod

        token_map = [i % 7 for i in range(config.vocab_size)]
        token_map[0] = 6
        model.engram_hash = engram_mod.NgramHashState(
            config, model.layout, token_map=token_map
        )
        return model

    def _model_with_engram(self):
        from mlx_vlm.models.deepseek_v41.language import LanguageModel

        config = self._config()
        return self._with_hash_state(LanguageModel(config), config)

    def test_two_caches_on_one_model_do_not_interfere(self):
        """Generation state belongs to the cache, not the model.

        While it lived on the modules, a second cache overwrote the first one's
        buffers, so a generation depended on whatever ran beside it.
        """
        model = self._model_with_engram()
        a = mx.array([[1, 2, 3, 4, 5, 6]])
        b = mx.array([[6, 5, 4, 3, 2, 1]])

        cache_a = model.make_cache()
        model(a, cache=cache_a)
        solo = model(mx.array([[2]]), cache=cache_a)
        mx.eval(solo.logits)

        cache_a = model.make_cache()
        model(a, cache=cache_a)
        cache_b = model.make_cache()
        model(b, cache=cache_b)
        interleaved = model(mx.array([[2]]), cache=cache_a)
        mx.eval(interleaved.logits)

        diff = mx.abs(
            solo.logits.astype(mx.float32) - interleaved.logits.astype(mx.float32)
        )
        mx.eval(diff)
        self.assertLess(float(mx.max(diff).item()), 1e-4)

    def test_deepseek_v41_engram_layers_are_invoked(self):
        from mlx_vlm.models.deepseek_v41 import engram as engram_mod

        config = self._config()
        model = self._model_with_engram()
        self.assertEqual(len(model.layout.primes), len(config.engram_layer_ids))

        calls = []
        original = engram_mod.Engram.__call__

        def counting(self, x, hash_ids, token_mask=None):
            calls.append(hash_ids.shape)
            return original(self, x, hash_ids, token_mask)

        engram_mod.Engram.__call__ = counting
        try:
            out = model(mx.array([[3, 7, 11, 15, 19, 23]]), cache=model.make_cache())
            mx.eval(out.logits)
        finally:
            engram_mod.Engram.__call__ = original

        self.assertEqual(len(calls), len(config.engram_layer_ids))

    def test_deepseek_v41_engram_hash_is_built_from_checkpoint_map(self):
        from mlx_vlm.models.deepseek_v41.language import LanguageModel

        config = self._config()
        model = LanguageModel(config)
        self.assertIsNone(model.engram_hash)

        token_map = [i % 7 for i in range(config.vocab_size)]
        token_map[0] = 6
        with tempfile.TemporaryDirectory() as directory:
            with open(f"{directory}/engram_token_map.json", "w") as handle:
                json.dump(token_map, handle)
            model._engram_source = directory
            model._ensure_engram_hash()

        self.assertIsNotNone(model.engram_hash)

    def test_deepseek_v41_chunked_prefill_matches_whole_prompt(self):
        """Chunked prefill names the ids `inputs`; the engram has to still see them."""
        from mlx_vlm.models.deepseek_v41.language import LanguageModel

        config = self._config(
            num_hidden_layers=3,
            compress_ratios=[0, 2, 0],
            kv_source_layer_ids=[1],
            index_source_layer_ids=[1],
            dspark_target_layer_ids=[],
        )
        model = self._with_hash_state(LanguageModel(config), config)
        model.head.weight = mx.random.normal(model.head.weight.shape) * 0.05
        mx.eval(model.parameters())
        prompt = mx.array([[3, 7, 11, 15, 19, 23, 27, 31, 35, 39, 43, 47]])

        whole = model(prompt, cache=model.make_cache())
        mx.eval(whole.logits)

        cache = model.make_cache()
        embeds = model.embed_tokens(prompt)
        for lo, hi in ((0, 5), (5, 9), (9, 12)):
            chunked = model(
                inputs=prompt[:, lo:hi],
                inputs_embeds=embeds[:, lo:hi],
                cache=cache,
                n_to_process=hi - lo,
            )
        mx.eval(chunked.logits)

        diff = mx.abs(
            whole.logits[:, -1].astype(mx.float32)
            - chunked.logits[:, -1].astype(mx.float32)
        )
        mx.eval(diff)
        self.assertLess(float(mx.max(diff).item()), 1e-4)


class TestDeepseekV41TokenMap(unittest.TestCase):
    @staticmethod
    def _wordlevel_tokenizer():
        from tokenizers import Tokenizer
        from tokenizers.models import WordLevel
        from tokenizers.pre_tokenizers import Whitespace

        vocab = {"[UNK]": 0, "Hello": 1, "hello": 2, " THE": 3, "the": 4, "world": 5}
        tokenizer = Tokenizer(WordLevel(vocab=vocab, unk_token="[UNK]"))
        tokenizer.pre_tokenizer = Whitespace()
        return tokenizer

    def test_deepseek_v41_token_map_collapses_case(self):
        from mlx_vlm.models.deepseek_v41 import engram as engram_module

        tokenizer = self._wordlevel_tokenizer()
        lookup, size = engram_module.build_compressed_token_map(tokenizer)
        self.assertEqual(len(lookup), tokenizer.get_vocab_size())
        self.assertEqual(lookup[1], lookup[2])
        self.assertLessEqual(size, tokenizer.get_vocab_size())

    def test_deepseek_v41_token_map_unwraps_wrapper(self):
        from mlx_vlm.models.deepseek_v41 import engram as engram_module

        class Wrapper:
            def __init__(self, backend):
                self._tokenizer = backend

        raw = self._wordlevel_tokenizer()
        lookup_wrapped, _ = engram_module.build_compressed_token_map(Wrapper(raw))
        lookup_raw, _ = engram_module.build_compressed_token_map(raw)
        self.assertEqual(lookup_wrapped, lookup_raw)

    def test_deepseek_v41_hash_state_shapes(self):
        from mlx_vlm.models.deepseek_v41 import engram as engram_module
        from mlx_vlm.models.deepseek_v41.language import DeepseekV41Cache

        config = deepseek_v41_config(
            engram_layer_ids=[0],
            engram_num_embeddings=[1024],
            engram_vocab_size=64,
        )
        layout = engram_module.EngramLayout.from_config(config)
        state = engram_module.NgramHashState(
            config,
            layout,
            token_map=[i % 7 for i in range(config.vocab_size)],
        )

        cache = DeepseekV41Cache(1, [1])
        prefill = state(mx.array([[1, 2, 3, 4]]), 0, cache)
        mx.eval(prefill)
        self.assertEqual(prefill.shape, (1, 4, 1, 4))
        step = state(mx.array([[5]]), 4, cache)
        mx.eval(step)
        self.assertEqual(step.shape, (1, 1, 1, 4))
