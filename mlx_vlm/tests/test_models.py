"""JSON model contracts, checkpoint loading, sanitization, and document layout."""

from __future__ import annotations

import copy
import importlib
import inspect
import json
import logging
import math
import os
import signal
import socket
import struct
import subprocess
import sys
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

from mlx_vlm import embedding_loader
from mlx_vlm.models import deepseek_v41
from mlx_vlm.models.base import InputEmbeddingsFeatures
from mlx_vlm.models.cache import make_prompt_cache
from mlx_vlm.models.deepseek_v41 import language as deepseek_v41_language
from mlx_vlm.models.lfm2_encoder import Model as Lfm2Encoder
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

    def masked_lm(self, model, config):
        mask = mx.array([[1, 1, 1, 0]])
        first = model(mx.array([[1, 2, 3, 0]]), attention_mask=mask).logits
        second = model(mx.array([[1, 2, 3, 9]]), attention_mask=mask).logits
        assert first.shape == (1, 4, config.vocab_size)
        assert mx.allclose(first[:, :3], second[:, :3])

    def token_embeddings(self, model, config):
        output = model(mx.array([[1, 2, 3]]))
        assert output.last_hidden_state.shape == (1, 3, config.hidden_size)
        assert output.text_embeds.shape == (1, 3, config.embedding_dim)
        assert mx.allclose(
            mx.linalg.norm(output.text_embeds, axis=-1), mx.array(1.0), atol=1e-5
        )

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
            "glm4v_moe glm4v hunyuan_vl siglip2_vision_model mimovl"
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
    if kind in {"masked_lm", "token_embeddings"}:
        return (model, config), {}
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


def test_lfm2_encoder_sanitize_and_dispatch():
    case = next(case for case in DATA["cases"] if case["module"] == "lfm2_encoder")
    module = importlib.import_module("mlx_vlm.models.lfm2_encoder")
    model = module.Model(build_config(module, case["config"]))
    weights = {
        "lfm2.embed_tokens.weight": mx.zeros((32, 16)),
        "lfm2.layers.0.conv.conv.weight": mx.zeros((16, 1, 3)),
        "lm_head.weight": mx.zeros((32, 16)),
    }
    sanitized = model.sanitize(weights)
    assert "model.embed_tokens.weight" in sanitized
    assert sanitized["model.layers.0.conv.conv.weight"].shape == (16, 3, 1)
    assert "lm_head.weight" not in sanitized

    resolved, model_type = get_model_and_args(
        {
            "model_type": "lfm2",
            "architectures": ["Lfm2BidirectionalForMaskedLM"],
        }
    )
    assert resolved.Model is Lfm2Encoder
    assert model_type == "lfm2_encoder"


def test_lfm2_colbert_sanitize_and_loader(tmp_path, monkeypatch):
    case = next(case for case in DATA["cases"] if case["module"] == "lfm2_colbert")
    module = importlib.import_module("mlx_vlm.models.lfm2_colbert")
    model = module.Model(build_config(module, case["config"]))
    weights = {
        "embed_tokens.weight": mx.zeros((32, 16)),
        "layers.0.conv.conv.weight": mx.zeros((16, 1, 3)),
        "1_Dense.linear.weight": mx.zeros((8, 16)),
    }
    sanitized = model.sanitize(weights)
    assert "model.embed_tokens.weight" in sanitized
    assert sanitized["model.layers.0.conv.conv.weight"].shape == (16, 3, 1)
    assert "projection.weight" in sanitized

    dense_dir = tmp_path / "1_Dense"
    dense_dir.mkdir()
    (dense_dir / "config.json").write_text(json.dumps({"out_features": 128}))
    captured = {}

    def fake_load(model_path, **kwargs):
        captured.update(kwargs)
        return object()

    monkeypatch.setattr(embedding_loader, "load_encoder_model", fake_load)
    embedding_loader.load_embedding_model(tmp_path)
    assert captured["model_remapping"]["lfm2"] == "lfm2_colbert"
    assert captured["config_overrides"]["embedding_dim"] == 128


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


@pytest.mark.parametrize("model_type", ["deepseek_v4", "deepseek_v41"])
def test_load_model_uses_language_model_fp8_quantization_config(model_type):
    module = importlib.import_module(f"mlx_vlm.models.{model_type}")
    case = next(c for c in DATA["cases"] if c["module"] == model_type)
    language_model = module.Model(build_config(module, case["config"])).language_model

    quantization = {
        "group_size": 64,
        "bits": 8,
        "mode": "affine",
        "language_model.weight": {"group_size": 64, "bits": 8, "mode": "affine"},
    }
    with (
        patch(
            f"mlx_vlm.models.{model_type}.language.make_quantization_config",
            return_value=quantization,
        ) as make_quantization_config,
        _checkpoint_loading(
            {
                "model_type": model_type,
                "quantization_config": {"quant_method": "fp8"},
            },
            lambda config: _CheckpointModel(config, language_model=language_model),
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


def test_qwen3_omni_audio_batch_matches_repeated_input():
    from mlx_vlm.models.qwen3_omni_moe.audio import AudioModel
    from mlx_vlm.models.qwen3_omni_moe.config import AudioConfig

    config = AudioConfig(
        d_model=8,
        encoder_layers=1,
        encoder_attention_heads=2,
        encoder_ffn_dim=16,
        num_mel_bins=8,
        output_dim=8,
        downsample_hidden_size=4,
        conv_chunksize=4,
        max_source_positions=64,
    )
    model = AudioModel(config)
    sample = mx.random.normal((8, 170))
    output = model(mx.concatenate([sample, sample], axis=1), mx.array([170, 170]))

    assert output.shape == (44, 8)
    assert mx.allclose(output[:22], output[22:])


def test_qwen3_omni_rope_delta_ignores_batch_padding():
    from mlx_vlm.models.qwen3_omni_moe.language import LanguageModel

    model = LanguageModel.__new__(LanguageModel)
    model.config = SimpleNamespace(
        vision_config=SimpleNamespace(spatial_merge_size=2),
        image_token_id=10,
        video_token_id=11,
        vision_start_token_id=12,
    )
    input_ids = mx.array([[0, 0, 1, 2], [1, 2, 3, 4]])
    attention_mask = mx.array([[0, 0, 1, 1], [1, 1, 1, 1]])

    _, rope_deltas = model.get_rope_index(input_ids, attention_mask=attention_mask)

    assert rope_deltas.tolist() == [[0], [0]]


# Patch embedding layouts


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


def test_moondream2_sanitize_remaps_checkpoint_layout():
    from mlx_vlm.models.moondream2 import Model

    source = {
        "model.text.wte": "text.model.embed_tokens.weight",
        "model.text.blocks.0.attn.qkv.weight": "text.model.layers.0.attn.qkv.weight",
        "model.text.post_ln.weight": "text.model.post_ln.weight",
        "model.text.lm_head.weight": "text.lm_head.weight",
        "model.vision.patch_emb.weight": "vision.encoder.patch_emb.weight",
        "model.vision.blocks.0.ln1.weight": "vision.encoder.blocks.0.ln1.weight",
        "model.vision.proj_mlp.fc1.weight": "vision.proj_mlp.fc1.weight",
    }
    weights = {key: mx.zeros((1,)) for key in source}
    weights["model.region.coord_decoder.fc1.weight"] = mx.zeros((1,))

    assert set(Model.sanitize(None, weights)) == set(source.values())


def test_moondream3_sanitize_remaps_raw_and_preserves_converted_keys():
    from mlx_vlm.models.moondream3 import Model

    raw = {
        "model.text.blocks.0.attn.qkv.weight": mx.zeros((1,)),
        "model.vision.blocks.0.ln1.weight": mx.zeros((1,)),
    }
    converted = Model.sanitize(None, raw)
    assert set(converted) == {
        "text.model.blocks.0.attn.qkv.weight",
        "vision.encoder.blocks.0.ln1.weight",
    }
    converted["text.lm_head.weight"] = mx.zeros((1,))
    converted["vision.proj_mlp.fc1.weight"] = mx.zeros((1,))
    assert Model.sanitize(None, converted).keys() == converted.keys()


class TestQwen3_5MoeText(unittest.TestCase):
    """Decoder-only Qwen3.5 MoE checkpoints (model_type qwen3_5_moe_text)."""

    def _model(self):
        from mlx_vlm.models.qwen3_5_moe_text import Model, ModelConfig

        case = next(
            case for case in DATA["cases"] if case["module"] == "qwen3_5_moe_text"
        )
        model = Model(ModelConfig.from_dict(copy.deepcopy(case["config"])))
        model.update(
            tree_map(
                lambda p: (mx.random.randint(-8, 8, p.shape) / 4).astype(p.dtype),
                model.parameters(),
            )
        )
        return model

    def _raw_checkpoint(self, model, prefix, fused):
        """Rebuild a published checkpoint from the model's own parameters."""
        from mlx_vlm.models.qwen3_5.qwen3_5 import NORM_WEIGHT_SUFFIXES

        raw = {}
        for key, value in tree_flatten(model.parameters()):
            if ".switch_mlp." in key:
                continue
            if key.startswith("language_model.model."):
                raw_key = prefix + key[len("language_model.model.") :]
            else:
                raw_key = key.replace("language_model.lm_head", "lm_head", 1)
            if "conv1d.weight" in key:
                value = value.swapaxes(1, 2)
            if any(key.endswith(sfx) for sfx in NORM_WEIGHT_SUFFIXES):
                value = value - 1.0
            raw[raw_key] = value
        for layer_idx, layer in enumerate(model.layers):
            experts = f"{prefix}layers.{layer_idx}.mlp.experts"
            switch = layer.mlp.switch_mlp
            if fused:
                raw[f"{experts}.gate_up_proj"] = mx.concatenate(
                    [switch.gate_proj.weight, switch.up_proj.weight], axis=-2
                )
                raw[f"{experts}.down_proj"] = switch.down_proj.weight
            else:
                for name in ("gate_proj", "up_proj", "down_proj"):
                    weight = getattr(switch, name).weight
                    for e in range(weight.shape[0]):
                        raw[f"{experts}.{e}.{name}.weight"] = weight[e]
        return raw

    def test_published_layouts_sanitize_to_the_model_exactly(self):
        model = self._model()
        expected = dict(tree_flatten(model.parameters()))
        for prefix in ("model.language_model.", "model."):
            for fused in (True, False):
                with self.subTest(prefix=prefix, fused=fused):
                    raw = self._raw_checkpoint(model, prefix, fused)
                    raw[f"{prefix}layers.0.mlp.gate.input_global_scale"] = mx.ones(1)
                    sanitized = model.sanitize(raw)
                    self.assertEqual(sanitized.keys(), expected.keys())
                    for key, value in expected.items():
                        self.assertTrue(
                            mx.array_equal(sanitized[key], value).item(), key
                        )
                    model.load_weights(list(sanitized.items()), strict=True)

    def test_ragged_expert_tensors_fail_clearly(self):
        model = self._model()
        raw = self._raw_checkpoint(model, "model.", fused=True)
        raw["model.layers.0.mlp.experts.gate_up_proj"] = mx.zeros((123,))
        with self.assertRaisesRegex(ValueError, "expected \\[num_experts"):
            model.sanitize(raw)

    def test_sanitize_is_idempotent_on_converted_weights(self):
        model = self._model()
        converted = dict(tree_flatten(model.parameters()))
        again = model.sanitize(dict(converted))
        self.assertEqual(again.keys(), converted.keys())
        for key, value in converted.items():
            self.assertTrue(mx.array_equal(again[key], value).item(), key)

    def test_missing_mrope_section_is_plain_rope(self):
        from mlx_vlm.models.qwen3_5_moe_text import Model, ModelConfig

        model = self._model()
        weights = list(tree_flatten(model.parameters()))
        ids = mx.array([[3, 1, 4, 1, 5, 9, 2, 6]])

        def logits(rope_parameters):
            config = vars(model.config) | {"rope_parameters": rope_parameters}
            other = Model(ModelConfig.from_dict(config))
            other.load_weights(weights, strict=True)
            return other(ids).logits

        base = {
            "rope_type": "default",
            "rope_theta": 10000,
            "partial_rotary_factor": 1.0,
        }
        missing = logits(dict(base))
        for section in ([2, 1, 1], [1, 1, 2], [4, 0, 0]):
            with self.subTest(section=section):
                explicit = logits(dict(base, mrope_section=section))
                self.assertTrue(mx.allclose(missing, explicit, atol=1e-5).item())


# DeepSeek-V4.1 regressions beyond the shared model contracts


DEEPSEEK_V41_NATIVE_QUANTIZATION = {
    "quant_method": "fp8",
    "activation_scheme": "dynamic",
    "weight_block_size": [32, 32],
    "scale_fmt": "ue8m0",
    "expert_dtype": "fp4",
}


def _quantize_deepseek_v41_experts(model):
    def predicate(path, module):
        if hasattr(module, "to_quantized") and any(
            p in path
            for p in ("switch_mlp.", "shared_experts.", "attn.wq_b", "attn.wo_a")
        ):
            bits = 4 if "switch_mlp." in path else 8
            return dict(group_size=32, bits=bits, mode=f"mxfp{bits}")
        return False

    nn.quantize(model, class_predicate=predicate)


@pytest.mark.parametrize("nested_config", [False, True])
def test_deepseek_v41_native_checkpoint_keeps_packed_weights(tmp_path, nested_config):
    mx.random.seed(9)
    model = deepseek_v41.Model(tiny_config("deepseek_v41"))
    model.update(tree_map(lambda p: p.astype(mx.bfloat16), model.parameters()))
    model.language_model.head.weight = mx.random.normal(
        model.language_model.head.weight.shape
    )
    quantization = deepseek_v41_language.make_quantization_config(model)
    nn.quantize(model, class_predicate=lambda p, m: quantization.get(p, False))
    mx.eval(model.parameters())
    expected = dict(tree_flatten(model.parameters()))
    weights = {}
    for name, value in expected.items():
        if (
            name.endswith(".scales")
            or name.endswith(".weight")
            and name[:-6] + "scales" in expected
        ):
            suffix = "scale" if name.endswith(".scales") else "weight"
            prefix = name.rsplit(".", 1)[0]
            if suffix == "weight":
                value = value.view(mx.uint8)
            if ".switch_mlp." in prefix:
                prefix, projection = prefix.rsplit(".", 1)
                prefix = prefix.replace(".switch_mlp", ".experts")
                projection = {"gate_proj": "w1", "down_proj": "w2", "up_proj": "w3"}[
                    projection
                ]
                for i in range(value.shape[0]):
                    weights[f"{prefix}.{i}.{projection}.{suffix}"] = value[i]
            else:
                if ".attn.wo_a" in prefix:
                    value = value.flatten(0, 1)
                weights[f"{prefix}.{suffix}"] = value
        else:
            weights[name] = value
    config = model.config.to_dict()
    if nested_config:
        config["text_config"] = {
            "quantization_config": DEEPSEEK_V41_NATIVE_QUANTIZATION
        }
    else:
        config["quantization_config"] = DEEPSEEK_V41_NATIVE_QUANTIZATION
    (tmp_path / "config.json").write_text(json.dumps(config))
    mx.save_safetensors(str(tmp_path / "model.safetensors"), weights)
    loaded = load_model(tmp_path)
    actual = dict(tree_flatten(loaded.parameters()))
    for name, value in expected.items():
        assert name in actual, name
        assert mx.array_equal(value, actual[name]).item(), name
    assert loaded.layers[0].ffn.switch_mlp.down_proj.mode == "mxfp4"
    assert loaded.layers[0].ffn.shared_experts.down_proj.mode == "mxfp8"
    assert loaded.layers[0].attn.wq_a.mode == "mxfp8"
    assert loaded.layers[1].engram.wkv.mode == "mxfp8"
    assert not hasattr(loaded, "_source_quantization")
    assert not hasattr(loaded, "_preserve_source_quantization")

    # The converter must describe the already-native modules when it writes an
    # MLX checkpoint, even if the requested default for other layers is affine4.
    from mlx_vlm.convert import _preserve_existing_deepseek_v4_quantization

    _preserve_existing_deepseek_v4_quantization(config, loaded, 64, 4, "affine")
    (tmp_path / "config.json").write_text(json.dumps(config))
    mx.save_safetensors(str(tmp_path / "model.safetensors"), actual)
    reloaded = dict(tree_flatten(load_model(tmp_path).parameters()))
    assert actual.keys() == reloaded.keys()
    for name, value in actual.items():
        assert mx.array_equal(value, reloaded[name]).item(), name


@pytest.mark.parametrize("bits", [4, 8])
def test_deepseek_v41_converted_checkpoint_keeps_declared_quantization(tmp_path, bits):
    model = deepseek_v41.Model(tiny_config("deepseek_v41"))
    quantization = dict(group_size=64, bits=bits, mode="affine")
    nn.quantize(
        model,
        **quantization,
        class_predicate=lambda p, m: ".switch_mlp." in p and hasattr(m, "to_quantized"),
    )
    expected = dict(tree_flatten(model.parameters()))
    config = model.config.to_dict()
    config["quantization"] = quantization
    # Converted checkpoints can retain the source metadata; the explicit MLX
    # quantization config must win over the native checkpoint format.
    config["quantization_config"] = DEEPSEEK_V41_NATIVE_QUANTIZATION
    (tmp_path / "config.json").write_text(json.dumps(config))
    mx.save_safetensors(str(tmp_path / "model.safetensors"), expected)
    loaded = load_model(tmp_path)
    actual = dict(tree_flatten(loaded.parameters()))
    assert actual.keys() == expected.keys()
    for name, value in expected.items():
        assert mx.array_equal(value, actual[name]).item(), name
    projection = loaded.layers[0].ffn.switch_mlp.down_proj
    assert (projection.mode, projection.bits, projection.group_size) == (
        "affine",
        bits,
        64,
    )


def test_deepseek_v41_invalid_shard_preserves_parameters():
    model = deepseek_v41_language.LanguageModel(tiny_config("deepseek_v41"))
    before = dict(tree_flatten(model.parameters()))
    with pytest.raises(ValueError, match="Expert count"):
        model.shard(SimpleNamespace(size=lambda: 3, rank=lambda: 0))
    after = dict(tree_flatten(model.parameters()))
    assert all(after[k] is v for k, v in before.items())


def _deepseek_v41_distributed_worker():
    from mlx_vlm.models.deepseek_v41.engram import NgramHashState

    group = mx.distributed.init(strict=True, backend="ring")
    records = []
    for dtype in (mx.float32, mx.bfloat16):
        mx.random.seed(19)
        cfg = tiny_config("deepseek_v41")
        reference = deepseek_v41_language.LanguageModel(cfg)
        reference.head.weight = mx.random.normal(reference.head.weight.shape) * 0.05
        reference.update(tree_map(lambda p: p.astype(dtype), reference.parameters()))
        reference.head.weight = reference.head.weight.astype(mx.float32)
        _quantize_deepseek_v41_experts(reference)
        reference.engram_hash = NgramHashState(
            cfg, reference.layout, token_map=[i % 7 for i in range(cfg.vocab_size)]
        )
        mx.eval(reference.parameters())
        sharded = deepseek_v41_language.LanguageModel(copy.deepcopy(cfg))
        _quantize_deepseek_v41_experts(sharded)
        sharded.engram_hash = NgramHashState(
            cfg, sharded.layout, token_map=[i % 7 for i in range(cfg.vocab_size)]
        )
        sharded.load_weights(tree_flatten(reference.parameters()))
        sharded.shard(group)
        with pytest.raises(ValueError, match="already sharded"):
            sharded.shard(group)
        mx.eval(sharded.parameters())
        assert sharded.head.weight.size * 2 == reference.head.weight.size
        for layer, original in zip(sharded.layers, reference.layers):
            for projection in ("gate_proj", "up_proj", "down_proj"):
                assert (
                    getattr(layer.ffn.switch_mlp, projection).weight.size * 2
                    == getattr(original.ffn.switch_mlp, projection).weight.size
                )
        for batch in (1, 4):
            actual_cache, expected_cache = sharded.make_cache(), reference.make_cache()
            for step, tokens in enumerate(
                ([1, 7, 9, 3, 11, 15, 2, 6], [13], [17], [19])
            ):
                ids = mx.array(
                    [
                        [(t + row) % cfg.vocab_size for t in tokens]
                        for row in range(batch)
                    ]
                )
                expected = reference(ids, cache=expected_cache).logits
                actual = sharded(ids, cache=actual_cache).logits
                mx.eval(actual, expected)
                atol = 0.06 if dtype == mx.bfloat16 else 2e-4
                error = mx.max(mx.abs(actual - expected)).item()
                agrees = mx.distributed.all_gather(actual, group=group)
                mx.eval(agrees)
                ranks_agree = mx.array_equal(agrees[:batch], agrees[batch:]).item()
                records.append(
                    dict(
                        dtype=str(dtype),
                        batch=batch,
                        step=step,
                        max_error=error,
                        ranks_agree=ranks_agree,
                        passed=mx.allclose(
                            actual, expected, atol=atol, rtol=atol
                        ).item(),
                    )
                )
    passed = mx.distributed.all_sum(
        mx.array(int(all(r["passed"] and r["ranks_agree"] for r in records))),
        group=group,
    ).item()
    print(
        json.dumps(dict(rank=group.rank(), passed_ranks=passed, checks=records)),
        flush=True,
    )
    assert passed == 2, records


@pytest.mark.skipif(not mx.distributed.is_available("ring"), reason="ring required")
def test_deepseek_v41_two_rank_prefill_decode_and_batch():
    repo = Path(__file__).resolve().parents[2]
    while True:
        with socket.socket() as first, socket.socket() as second:
            first.bind(("127.0.0.1", 0))
            port = first.getsockname()[1]
            if port == 65535:
                continue
            try:
                second.bind(("127.0.0.1", port + 1))
            except OSError:
                continue
            break
    command = [
        sys.executable,
        "-c",
        "from mlx._distributed_utils.launch import main; main()",
        "--backend",
        "ring",
        "-n",
        "2",
        "--starting-port",
        str(port),
        "--env",
        f"PYTHONPATH={repo}",
        "--",
        sys.executable,
        "-c",
        "from mlx_vlm.tests.test_models import _deepseek_v41_distributed_worker; "
        "_deepseek_v41_distributed_worker()",
    ]
    process = subprocess.Popen(
        command,
        cwd=repo,
        env=dict(os.environ, PYTHONPATH=str(repo)),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        start_new_session=True,
    )
    try:
        output, _ = process.communicate(timeout=120)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGKILL)
        output, _ = process.communicate()
        pytest.fail(output)
    assert process.returncode == 0, output
    records = [json.loads(line) for line in output.splitlines() if line.startswith("{")]
    assert {r["rank"] for r in records if r["passed_ranks"] == 2} == {0, 1}, output


class TestDeepseekV41EndToEnd(unittest.TestCase):
    def setUp(self):
        mx.random.seed(0)

    def test_index_rotary_matches_reference_at_long_positions(self):
        from mlx_vlm.models.deepseek_v41.language import (
            _apply_index_rotary,
            _index_cos_sin,
        )

        dim, length, theta = 64, 32769, 160000
        factor, beta_fast, beta_slow, original_length = 16, 32, 1, 65536

        def correction(rotations):
            return (
                dim
                * np.log(original_length / (rotations * 2 * np.pi))
                / (2 * np.log(theta))
            )

        low = int(np.floor(correction(beta_fast)))
        high = int(np.ceil(correction(beta_slow)))
        ramp = (np.arange(dim // 2, dtype=np.float32) - low) / (high - low)
        smooth = 1 - np.clip(ramp, 0, 1)
        powers = np.power(np.float64(theta), np.arange(0, dim, 2) / dim)
        inv = 1 / powers.astype(np.float32)
        inv = inv / factor * (1 - smooth) + inv * smooth
        phase = np.arange(length, dtype=np.float32)[:, None] * inv[None, :]
        cos, sin = _index_cos_sin(
            length, dim, theta, (factor, beta_fast, beta_slow, original_length)
        )
        phase = phase.astype(np.float64)
        np.testing.assert_array_equal(np.array(cos), np.cos(phase).astype(np.float32))
        np.testing.assert_array_equal(np.array(sin), np.sin(phase).astype(np.float32))

        for batch, count in ((1, 1), (2, 17)):
            with self.subTest(batch=batch, length=count):
                x = mx.random.normal((batch, count, 4, 80)).astype(mx.bfloat16)
                c, s = cos[None, -count:, None], sin[None, -count:, None]
                data = np.array(x.astype(mx.float32))
                rotated = data[..., -dim:].copy().view(np.complex64)
                frequencies = np.array(c) + np.complex64(1j) * np.array(s)
                rotated = (rotated * frequencies).view(np.float32)
                expected = np.concatenate([data[..., :-dim], rotated], axis=-1)
                actual = _apply_index_rotary(x, c, s, dim)
                self.assertEqual(actual.dtype, x.dtype)
                self.assertTrue(
                    mx.array_equal(actual, mx.array(expected).astype(x.dtype)).item()
                )

    def test_hc_expansion_preserves_reference_rounding(self):
        from mlx_vlm.models.deepseek_v41.language import hc_expand

        for dtype in (mx.float32, mx.float16, mx.bfloat16):
            for batch, length in ((1, 1), (2, 5), (1, 256)):
                with self.subTest(dtype=dtype, batch=batch, length=length):
                    x = mx.random.normal((batch, length, 128)).astype(dtype)
                    residual = mx.random.normal((batch, length, 4, 128)).astype(dtype)
                    post = mx.random.uniform(low=0, high=2, shape=(batch, length, 4))
                    comb = mx.softmax(mx.random.normal((batch, length, 4, 4)), axis=-1)
                    xx, rr, pp, cc = [
                        np.array(a.astype(mx.float32))
                        for a in (x, residual, post, comb)
                    ]
                    # Independent CPU evaluation of the source's explicit
                    # products and reduction; matmul changes the rounding.
                    expected = pp[..., None] * xx[:, :, None, :] + np.sum(
                        cc[..., None] * rr[..., None, :], axis=2, dtype=np.float32
                    )
                    actual = hc_expand(x, residual, post, comb)
                    self.assertEqual(actual.dtype, dtype)
                    self.assertTrue(
                        mx.array_equal(actual, mx.array(expected).astype(dtype)).item()
                    )

    def test_activation_fp8_rounding_matches_reference(self):
        from mlx_vlm.models.deepseek_v41.fakequant import fake_quant_fp8_ue8m0

        # Exercise every E4M3 rounding tie and its immediate FP32 neighbors.
        # The 448 anchor fixes the block scale at one.
        levels = np.array(mx.from_fp8(mx.arange(127, dtype=mx.uint8), mx.float32))
        midpoints = (levels[:-1] + levels[1:]) / 2
        values = np.concatenate(
            [
                np.nextafter(midpoints, -np.inf),
                midpoints,
                np.nextafter(midpoints, np.inf),
            ]
        )
        ties = np.zeros((2 * len(values), 32), dtype=np.float32)
        ties[:, 0] = np.concatenate([values, -values])
        ties[:, 1] = 448
        samples = [
            mx.array(ties),
            mx.random.normal((4, 3, 512)).transpose(1, 0, 2),
            mx.random.normal((2, 64)) * 1e-8,
            mx.zeros((1, 32)),
        ]
        for dtype in (mx.float32, mx.float16, mx.bfloat16):
            for sample in samples:
                with self.subTest(dtype=dtype, shape=sample.shape):
                    x = sample.astype(dtype)
                    blocks = np.array(x.astype(mx.float32)).reshape(-1, 32)
                    amax = np.maximum(np.abs(blocks).max(-1, keepdims=True), 1e-4)
                    fraction, exponent = np.frexp(amax * np.float32(1 / 448))
                    scales = np.ldexp(np.ones_like(amax), exponent - (fraction == 0.5))
                    encoded = mx.to_fp8(mx.array(np.clip(blocks / scales, -448, 448)))
                    expected = (
                        (mx.from_fp8(encoded, mx.float32) * mx.array(scales))
                        .reshape(x.shape)
                        .astype(dtype)
                    )
                    actual = fake_quant_fp8_ue8m0(x)
                    self.assertTrue(mx.array_equal(actual, expected).item())

    def test_config_accepts_unused_prediction_layers(self):
        from mlx_vlm.models import deepseek_v41

        case = next(case for case in DATA["cases"] if case["module"] == "deepseek_v41")
        source = copy.deepcopy(case["config"])
        source["compress_ratios"] += [0, 0, 0]
        source["num_nextn_predict_layers"] = 3
        config = deepseek_v41.ModelConfig.from_dict(source)
        self.assertEqual(config.compress_ratios, case["config"]["compress_ratios"])

    def test_head_and_router_load_bf16_checkpoint_weights_as_fp32(self):
        from mlx_vlm.models import deepseek_v41

        case = next(case for case in DATA["cases"] if case["module"] == "deepseek_v41")
        model = deepseek_v41.Model(build_config(deepseek_v41, case["config"]))
        head = model.language_model.head
        gate = model.layers[0].ffn.gate
        weight = mx.random.normal(head.weight.shape).astype(mx.bfloat16)
        gate_weight = mx.random.normal(gate.weight.shape).astype(mx.bfloat16)
        x = mx.random.normal((1, 1, model.config.hidden_size)).astype(mx.bfloat16)
        gate.weight = gate_weight
        expected_indices, expected_weights = gate(x)
        mx.eval(expected_indices, expected_weights)
        sanitized = model.sanitize(
            {"head.weight": weight, "layers.0.ffn.gate.weight": gate_weight}
        )
        model.load_weights(list(sanitized.items()), strict=False)
        self.assertEqual(head.weight.dtype, mx.float32)
        self.assertEqual(gate.weight.dtype, mx.float32)
        expected = x.astype(mx.float32) @ weight.T
        self.assertTrue(mx.array_equal(head(x), expected).item())
        indices, weights = gate(x)
        self.assertTrue(mx.array_equal(indices, expected_indices).item())
        self.assertTrue(mx.array_equal(weights, expected_weights).item())

    def test_sparse_attention_preserves_precision_and_masks(self):
        from mlx_vlm.models import deepseek_v41
        from mlx_vlm.models.deepseek_v41.language import DeepseekV41Attention

        case = next(case for case in DATA["cases"] if case["module"] == "deepseek_v41")
        config = build_config(deepseek_v41, case["config"])
        attention = DeepseekV41Attention(config, 1)
        for dtype in (mx.float32, mx.bfloat16):
            for length in (1, 3):
                with self.subTest(dtype=dtype, length=length):
                    q = mx.random.normal(
                        (2, config.num_attention_heads, length, config.head_dim)
                    ).astype(dtype)
                    window = mx.random.normal((2, 7, config.head_dim)).astype(dtype)
                    pool = mx.random.normal((2, 9, config.head_dim)).astype(dtype)
                    indices = mx.broadcast_to(
                        mx.array([[[2, -1, 0, 7]]]), (2, length, 4)
                    )
                    mask = (
                        mx.arange(7)[None, None, None]
                        <= mx.arange(length)[None, None, :, None] + 3
                    )
                    attention.attn_sink = mx.random.normal(
                        (config.num_attention_heads,)
                    )
                    actual = attention._sparse_attention(q, window, pool, indices, mask)
                    expected = np.zeros(q.shape, dtype=np.float32)
                    query, local, pooled = [
                        np.array(a.astype(mx.float32)) for a in (q, window, pool)
                    ]
                    sinks = np.array(attention.attn_sink.astype(mx.float32))
                    # Select only valid rows in an independent dense FP32 oracle.
                    for batch in range(2):
                        for pos in range(length):
                            kv = np.concatenate(
                                [local[batch, : pos + 4], pooled[batch, [2, 0, 7]]]
                            )
                            scores = (query[batch, :, pos] @ kv.T) * attention.scale
                            maximum = np.maximum(scores.max(-1), sinks)[:, None]
                            probabilities = np.exp(scores - maximum)
                            denominator = probabilities.sum(-1, keepdims=True) + np.exp(
                                sinks[:, None] - maximum
                            )
                            expected[batch, :, pos] = (probabilities / denominator) @ kv
                    # BF16 output rounds the FP32 result to its nearest value.
                    tolerance = 1e-5 if dtype == mx.float32 else 2**-8
                    self.assertEqual(actual.dtype, dtype)
                    self.assertTrue(
                        mx.allclose(
                            actual.astype(mx.float32),
                            mx.array(expected),
                            atol=1e-5,
                            rtol=tolerance,
                        ).item()
                    )

    def test_engram_offloading_matches_resident_rows(self):
        import tempfile

        from mlx_vlm.models import deepseek_v41
        from mlx_vlm.models.deepseek_v41.engram import (
            OffloadedEngramEmbedding,
            QuantizedEngramEmbedding,
        )

        case = next(case for case in DATA["cases"] if case["module"] == "deepseek_v41")
        prefix = "language_model.layers.1.engram.embed."
        ids = mx.array([[0, 127, 3], [3, 1, 0]])
        for bits, mode, sharded in (
            (None, "affine", False),
            (4, "affine", False),
            (8, "affine", True),
            (4, "mxfp4", False),
            (8, "mxfp8", True),
        ):
            with (
                self.subTest(bits=bits, mode=mode, sharded=sharded),
                tempfile.TemporaryDirectory() as directory,
            ):
                path = Path(directory)
                config = copy.deepcopy(case["config"])
                model = deepseek_v41.Model(build_config(deepseek_v41, config))
                table = model.layers[1].engram.embed
                table.weight = table.weight.astype(mx.bfloat16)
                if bits is not None:
                    config["quantization"] = dict(group_size=32, bits=bits, mode=mode)
                    packed = QuantizedEngramEmbedding(128, 64, **config["quantization"])
                    packed.weight, packed.scales, *biases = mx.quantize(
                        table.weight, **config["quantization"]
                    )
                    packed.biases = biases[0] if biases else None
                    model.layers[1].engram.embed = packed
                    config["quantization"][prefix.rstrip(".")] = dict(
                        group_size=32, bits=bits, mode=mode
                    )
                expected = model.layers[1].engram.embed(ids)
                mx.eval(expected)
                self.assertEqual(expected.dtype, mx.bfloat16)
                weights = dict(tree_flatten(model.parameters()))
                if mode.startswith("mxfp"):
                    native_prefix = prefix.removeprefix("language_model.")
                    weights[native_prefix + "weight"] = weights.pop(
                        prefix + "weight"
                    ).view(mx.uint8)
                    weights[native_prefix + "scale"] = weights.pop(prefix + "scales")
                (path / "config.json").write_text(json.dumps(config))
                if sharded:
                    tables = {
                        key: weights.pop(key)
                        for key in list(weights)
                        if ".engram.embed." in key
                    }
                    mx.save_safetensors(str(path / "engram.safetensors"), tables)
                    (path / "model.safetensors.index.json").write_text(
                        json.dumps(
                            {
                                "weight_map": {
                                    **dict.fromkeys(weights, "model.safetensors"),
                                    **dict.fromkeys(tables, "engram.safetensors"),
                                }
                            }
                        )
                    )
                mx.save_safetensors(str(path / "model.safetensors"), weights)

                with patch("mlx_vlm.utils.mx.eval", wraps=mx.eval) as evaluate:
                    loaded = load_model(path)
                # Exportable tensors must be exposed after eager loading.
                self.assertNotIn(
                    prefix + "weight", dict(tree_flatten(evaluate.call_args.args[0]))
                )
                self.assertEqual(loaded.config.model_path, str(path))
                self.assertIsInstance(
                    loaded.layers[1].engram.embed, OffloadedEngramEmbedding
                )
                self.assertEqual(loaded.layers[1].engram.embed(ids).dtype, mx.bfloat16)
                self.assertTrue(
                    mx.array_equal(loaded.layers[1].engram.embed(ids), expected).item()
                )
                self.assertIn(
                    prefix + "weight", dict(tree_flatten(loaded.parameters()))
                )

                # Conversion uses lazy loading and must keep exportable table weights.
                lazy = load_model(path, lazy=True)
                self.assertEqual(lazy.config.model_path, str(path))
                self.assertIn(prefix + "weight", dict(tree_flatten(lazy.parameters())))
                self.assertTrue(
                    mx.array_equal(lazy.layers[1].engram.embed(ids), expected).item()
                )
                with patch.object(QuantizedEngramEmbedding, "_quantize_chunk_rows", 41):
                    quantized = lazy.layers[1].engram.embed.to_quantized(
                        group_size=32, bits=4
                    )
                resident = model.layers[1].engram.embed.to_quantized(
                    group_size=32, bits=4
                )
                for name in ("weight", "scales", "biases"):
                    self.assertTrue(
                        mx.array_equal(quantized[name], resident[name]).item()
                    )
                self.assertTrue(mx.array_equal(quantized(ids), resident(ids)).item())

                exported = str(path / "export.safetensors")
                lazy.save_weights(exported)
                self.assertIn(prefix + "weight", mx.load(exported))

    def test_fp8_scale_layouts_decode_exactly(self):
        from mlx_vlm.models.deepseek_v41.deepseek_v41 import _pack_source_weight

        raw = mx.full((64, 64), 56, dtype=mx.uint8)  # E4M3 encoding of 1.0.
        for rowwise in (False, True):
            with self.subTest(rowwise=rowwise):
                scale_rows = 64 if rowwise else 2
                scales = (
                    mx.arange(scale_rows * 2).reshape(scale_rows, 2) % 4 + 125
                ).astype(mx.uint8)
                packed, expanded, mode = _pack_source_weight(raw, scales)
                decoded = mx.dequantize(
                    packed, expanded, group_size=32, bits=8, mode=mode
                )
                # Keep the reference exact, independent of GPU pow rounding.
                expected = mx.array([0.25, 0.5, 1.0, 2.0])[
                    scales.astype(mx.int32) - 125
                ]
                expected = mx.repeat(expected, 32, axis=-1)
                if not rowwise:
                    expected = mx.repeat(expected, 32, axis=0)
                self.assertTrue(mx.array_equal(decoded, expected).item())

    def test_engram_chunked_requantization(self):
        from mlx_vlm.models.deepseek_v41.engram import QuantizedEngramEmbedding

        source = QuantizedEngramEmbedding(
            7, 256, group_size=32, bits=8, mode="mxfp8", scale_dtype=mx.uint8
        )
        source.weight, source.scales = mx.quantize(
            mx.random.normal((7, 256)).astype(mx.bfloat16),
            group_size=32,
            bits=8,
            mode="mxfp8",
        )
        expected = mx.quantize(
            mx.dequantize(
                source.weight, source.scales, group_size=32, bits=8, mode="mxfp8"
            ),
            group_size=64,
            bits=4,
        )
        with patch.object(QuantizedEngramEmbedding, "_quantize_chunk_rows", 3):
            converted = source.to_quantized(group_size=64, bits=4)
        for actual, reference in zip(
            (converted.weight, converted.scales, converted.biases), expected
        ):
            self.assertTrue(mx.array_equal(actual, reference).item())
        self.assertIs(converted.to_quantized(group_size=64, bits=4), converted)

    def test_indexer_uses_unrotated_latents_and_adjacent_pair_rope(self):
        from mlx_vlm.models import deepseek_v41
        from mlx_vlm.models.deepseek_v41 import language
        from mlx_vlm.models.deepseek_v41.fakequant import (
            fake_quant_fp4_e4m3,
            fake_quant_fp4_ue8m0,
            fake_quant_fp8_ue8m0,
        )

        case = next(case for case in DATA["cases"] if case["module"] == "deepseek_v41")
        config = build_config(deepseek_v41, case["config"])

        def rotate(value, positions):
            # DeepSeek's reference apply_rotary_emb views adjacent pairs as complex.
            dtype = value.dtype
            value = np.array(value.astype(mx.float32))
            rd = config.qk_rope_head_dim
            frequencies = config.compress_rope_theta ** (
                -np.arange(0, rd, 2, dtype=np.float32) / rd
            )
            angles = np.asarray(positions, dtype=np.float32)[:, None] * frequencies
            phases = np.exp(1j * angles).astype(np.complex64)
            phases = phases.reshape(1, len(positions), *((1,) * (value.ndim - 3)), -1)
            pairs = np.ascontiguousarray(value[..., -rd:]).view(np.complex64)
            rotated = np.ascontiguousarray(pairs * phases).view(np.float32)
            return mx.array(
                np.concatenate([value[..., :-rd], rotated], axis=-1)
            ).astype(dtype)

        for dtype in (mx.float32, mx.bfloat16):
            with self.subTest(dtype=dtype):
                attn = language.DeepseekV41Attention(config, 1)
                attn.update(tree_map(lambda p: p.astype(dtype), attn.parameters()))
                # Ratio-2 pooling projections remain FP32 in the reference.
                attn.compressor.wkv.weight = attn.compressor.wkv.weight.astype(
                    mx.float32
                )
                attn.compressor.wgate.weight = attn.compressor.wgate.weight.astype(
                    mx.float32
                )
                x = mx.random.normal((2, 16, config.hidden_size)).astype(dtype)
                qr = attn.q_norm(attn.wq_a(fake_quant_fp8_ue8m0(x)))
                cache = language.DeepseekV41Cache(config.num_hidden_layers)
                latent = attn.compressor(x, 0, cache)
                positions = range(0, x.shape[1], attn.compress_ratio)
                expected_keys = fake_quant_fp4_ue8m0(
                    rotate(attn.indexer.k_norm(attn.indexer.wk(latent)), positions)
                )
                q = attn.indexer.wq_b(fake_quant_fp8_ue8m0(qr)).reshape(
                    *x.shape[:2], config.index_n_heads, config.index_head_dim
                )
                expected_queries = fake_quant_fp4_ue8m0(rotate(q, range(x.shape[1])))
                expected_pool = fake_quant_fp4_e4m3(rotate(latent, positions))
                with patch.object(
                    language, "_index_scores", wraps=language._index_scores
                ) as score:
                    pool, indices = attn._compress_part(x, qr, 0, cache)
                for actual, expected in (
                    (cache.index_k, expected_keys),
                    (score.call_args.args[0], expected_queries),
                    (pool, expected_pool),
                ):
                    self.assertEqual(actual.dtype, dtype)
                    self.assertTrue(mx.allclose(actual, expected, atol=1e-6).item())

                # Match the reference's BF16 rounding after the dot product,
                # head weighting, and head reduction, including non-tied top-k.
                weights = attn.indexer.weights_proj(x) * (
                    config.index_head_dim**-0.5 * config.index_n_heads**-0.5
                )
                dots = mx.einsum(
                    "bshd,btd->bsht",
                    expected_queries.astype(mx.float32),
                    expected_keys.astype(mx.float32),
                ).astype(dtype)
                weighted = (
                    mx.maximum(dots, 0).astype(mx.float32)
                    * weights.astype(mx.float32)[..., None]
                ).astype(dtype)
                expected_scores = weighted.astype(mx.float32).sum(2).astype(dtype)
                lengths = (mx.arange(1, x.shape[1] + 1) // attn.compress_ratio)[:, None]
                expected_scores = mx.where(
                    mx.arange(expected_keys.shape[1]) < lengths,
                    expected_scores,
                    -mx.inf,
                )
                scores = language._index_scores(*score.call_args.args)
                self.assertEqual(scores.dtype, dtype)
                self.assertTrue(mx.array_equal(scores, expected_scores).item())
                selected = mx.take_along_axis(
                    expected_scores, mx.maximum(indices, 0), -1
                )
                selected = mx.where(indices < 0, -mx.inf, selected)
                best = mx.sort(expected_scores, axis=-1)[..., -indices.shape[-1] :]
                self.assertTrue(mx.array_equal(mx.sort(selected), best).item())

    def test_moe_matches_reference_activation_and_routing_precision(self):
        from mlx_vlm.models import deepseek_v41
        from mlx_vlm.models.deepseek_v41.language import DeepseekV41MoE

        case = next(case for case in DATA["cases"] if case["module"] == "deepseek_v41")
        config = build_config(deepseek_v41, case["config"])

        def project(layer, x, expert=None):
            # Use MLX's native MXFP8 codec as an independent rounding oracle.
            q, scales = mx.quantize(x, group_size=32, bits=8, mode="mxfp8")
            x = mx.dequantize(
                q, scales, group_size=32, bits=8, mode="mxfp8", dtype=x.dtype
            )
            weight = layer.weight if expert is None else layer.weight[expert]
            if hasattr(layer, "scales"):
                # Match MLX's decode/prefill GEMM rounding, keeping the expert
                # loop, activation codec and routing order independent.
                if expert is not None and x.shape[1] == 1:
                    return mx.gather_qmm(
                        x[..., None, :],
                        layer.weight,
                        layer.scales,
                        layer.biases,
                        rhs_indices=mx.full(x.shape[:-1], expert, dtype=mx.int32),
                        transpose=True,
                        group_size=layer.group_size,
                        bits=layer.bits,
                    ).squeeze(-2)
                return mx.quantized_matmul(
                    x,
                    weight,
                    layer.scales if expert is None else layer.scales[expert],
                    layer.biases if expert is None else layer.biases[expert],
                    transpose=True,
                    group_size=layer.group_size,
                    bits=layer.bits,
                )
            return x @ weight.T

        def expert(module, x, weight=None, index=None):
            gate = project(module.gate_proj, x, index).astype(mx.float32)
            up = project(module.up_proj, x, index).astype(mx.float32)
            gate = mx.minimum(gate, config.swiglu_limit)
            up = mx.clip(up, -config.swiglu_limit, config.swiglu_limit)
            activated = (gate / (1 + mx.exp(-gate))) * up
            if weight is not None:
                activated = activated * weight[..., None]
            return project(module.down_proj, activated.astype(x.dtype), index)

        for bits in (None, 4, 8):
            for length in (1, 17):  # Unsorted decode and sorted expert dispatch.
                with self.subTest(bits=bits, length=length):
                    model = DeepseekV41MoE(config)
                    model.gate.weight = mx.random.normal(model.gate.weight.shape) * 0.1
                    model.update(
                        tree_map(lambda p: p.astype(mx.bfloat16), model.parameters())
                    )
                    if bits is not None:
                        nn.quantize(
                            model,
                            group_size=32,
                            bits=bits,
                            class_predicate=lambda path, module: hasattr(
                                module, "to_quantized"
                            )
                            and path.startswith(("switch_mlp.", "shared_experts.")),
                        )
                    x = (mx.random.normal((2, length, config.hidden_size)) * 8).astype(
                        mx.bfloat16
                    )
                    indices, weights = model.gate(x)
                    expected = mx.zeros(x.shape, dtype=mx.float32)
                    for index in range(config.n_routed_experts):
                        weight = mx.where(indices == index, weights, 0).sum(-1)
                        expected = expected + expert(
                            model.switch_mlp, x, weight, index
                        ).astype(mx.float32)
                    expected = (
                        expected + expert(model.shared_experts, x).astype(mx.float32)
                    ).astype(x.dtype)
                    actual = model(x)
                    self.assertEqual(actual.dtype, x.dtype)
                    self.assertTrue(
                        mx.allclose(actual, expected, rtol=1e-5, atol=1e-5).item()
                    )

    def test_image_tokens_use_visual_routing_and_break_engram_history(self):
        """The generation path supplies token ids, including during chunked prefill.

        Match the reference's explicit image mask and masked n-gram hashes,
        even when a prefill boundary falls inside an image span.
        """
        from mlx_vlm.generate.common import _chunked_prefill_enabled
        from mlx_vlm.models import deepseek_v41
        from mlx_vlm.models.deepseek_v41.engram import NgramHashState
        from mlx_vlm.models.deepseek_v41.language import LanguageModel

        case = next(case for case in DATA["cases"] if case["module"] == "deepseek_v41")
        config = build_config(deepseek_v41, case["config"])
        model = LanguageModel(config)
        self.assertTrue(_chunked_prefill_enabled(model))
        token_map = [i % 7 for i in range(config.vocab_size)]
        token_map[0] = 6
        model.engram_hash = NgramHashState(config, model.layout, token_map=token_map)
        model.head.weight = mx.random.normal(model.head.weight.shape) * 0.05
        for layer in model.layers:
            layer.ffn.gate.bias = mx.array([10.0, 9.0, 0.0, 0.0])
            layer.ffn.gate.bias_vl = mx.array([0.0, 0.0, 10.0, 9.0])

        ids = mx.array([[3, 7, 5, 5, 5, 11, 15, 19]])
        image_mask = ids == config.image_token_id
        embeds = model.embed_tokens(ids)
        embeds = mx.where(image_mask[..., None], mx.random.normal(embeds.shape), embeds)
        reference_cache = model.make_cache()
        hashes = model.engram_hash(ids, 0, reference_cache[0], token_mask=~image_mask)
        self.assertLess(mx.max(hashes).item(), config.engram_num_embeddings[0])
        output = model(
            ids,
            inputs_embeds=embeds,
            cache=reference_cache,
            image_mask=image_mask,
            engram_hashes=hashes,
        )
        self.assertIsNone(output.hidden_states)
        expected = output.logits
        mx.eval(expected)

        for chunks in ((8,), (3, 2, 3), (1,) * 8):
            with self.subTest(chunks=chunks):
                cache = model.make_cache()
                outputs, start = [], 0
                for length in chunks:
                    stop = start + length
                    outputs.append(
                        model(
                            inputs=ids[:, start:stop],
                            inputs_embeds=embeds[:, start:stop],
                            cache=cache,
                            n_to_process=length,
                        ).logits
                    )
                    start = stop
                actual = mx.concatenate(outputs, axis=1)
                mx.eval(actual)
                self.assertTrue(
                    bool(mx.array_equal(cache[0].engram, reference_cache[0].engram))
                )
                self.assertTrue(bool(mx.all(cache[0].engram[:, 2:5] == -1)))
                self.assertTrue(bool(mx.allclose(actual, expected, atol=1e-4)))
