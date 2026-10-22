"""Checkpoint sampling defaults must agree across text generation entry points."""

import contextlib
import json
from types import SimpleNamespace as NS
from unittest.mock import MagicMock, patch

import mlx.core as mx
import pytest

from mlx_vlm.generate import ar, dispatch
from mlx_vlm.models.base import InputEmbeddingsFeatures, LanguageModelOutput
from mlx_vlm.server import ChatRequest, _build_gen_args, runtime
from mlx_vlm.tests.test_cli import _args, _text_cli
from mlx_vlm.tests.test_diffusion_models import FakeProcessor
from mlx_vlm.utils import apply_generation_config_defaults, load_config


@pytest.fixture(
    params=[
        ({}, {}, (0.0, 1.0, 0)),
        ({"temperature": None, "top_p": None, "top_k": None}, {}, (0.0, 1.0, 0)),
        ({"do_sample": True}, {}, (0.0, 1.0, 0)),
        ({"temperature": 0.7, "top_p": 0.9, "top_k": 2}, {}, (0.7, 0.9, 2)),
        (
            {"do_sample": True, "temperature": 1.0, "top_p": 0.95, "top_k": 64},
            {},
            (1.0, 0.95, 64),
        ),
        ({"top_p": 0.8}, {}, (0.0, 0.8, 0)),
        (
            {"temperature": 1.0, "top_p": 0.9, "top_k": 2},
            {"temperature": None, "top_p": None, "top_k": None},
            (1.0, 0.9, 2),
        ),
        (
            {"temperature": 1.0, "top_p": 0.9, "top_k": 2},
            {"temperature": 0.0, "top_p": 1.0, "top_k": 0},
            (0.0, 1.0, 0),
        ),
        ({"temperature": 1.0, "top_p": 0.9, "top_k": 2}, {"top_k": 0}, (1.0, 0.9, 0)),
        (
            {"do_sample": False, "temperature": 1.0, "top_p": 0.9, "top_k": 2},
            {},
            (0.0, 0.9, 2),
        ),
        ({"do_sample": False, "temperature": 1.0}, {"temperature": 0.6}, (0.6, 1.0, 0)),
    ],
    ids=[
        "missing",
        "null",
        "sample-only",
        "sampling",
        "gemma",
        "partial",
        "null-overrides",
        "explicit-greedy",
        "partial-override",
        "do-not-sample",
        "override-do-not-sample",
    ],
)
def sampling_case(request, tmp_path):
    defaults, overrides, expected = request.param
    (tmp_path / "config.json").write_text(json.dumps({"model_type": "test"}))
    (tmp_path / "generation_config.json").write_text(json.dumps(defaults))
    config = apply_generation_config_defaults(
        NS(model_type="test", eos_token_id=[]), load_config(tmp_path)
    )
    return config, overrides, dict(zip(("temperature", "top_p", "top_k"), expected))


@pytest.mark.parametrize("chat", [False, True])
def test_cli_sampling_defaults(sampling_case, chat):
    config, overrides, expected = sampling_case
    flags = ["--chat"] if chat else []
    for name, value in overrides.items():
        if value is not None:
            flags.extend(["--" + name.replace("_", "-"), str(value)])
    args = _args(*flags)
    with (
        _text_cli(args) as cli,
        patch("builtins.input", side_effect=["hello", ""]),
        patch.object(dispatch, "stream_generate", return_value=iter(())) as stream,
    ):
        cli.load.return_value = (NS(config=config), NS())
        dispatch.main()
    actual = (stream if chat else cli.generate).call_args.kwargs
    assert {key: actual[key] for key in expected} == expected


@pytest.mark.parametrize("cached_config", [False, True])
def test_server_sampling_defaults(sampling_case, monkeypatch, cached_config):
    config, overrides, expected = sampling_case
    monkeypatch.setattr(
        runtime,
        "model_cache",
        {"config": config if cached_config else NS(temperature=0.123)},
    )
    processor = None if cached_config else NS(config=config)
    request = ChatRequest(
        model="demo",
        messages=[],
        **{k: v for k, v in overrides.items() if v is not None},
    )
    actual = _build_gen_args(request, processor)
    assert {key: getattr(actual, key) for key in expected} == expected


@pytest.mark.parametrize("entrypoint", ["generate_step", "stream_generate", "generate"])
@pytest.mark.parametrize("seed", [None, 42])
def test_python_sampling_defaults(sampling_case, entrypoint, seed):
    config, overrides, expected = sampling_case
    original_config = vars(config).copy()
    model = _sampling_model(config)
    processor = FakeProcessor()
    options = dict(max_tokens=32, seed=seed, prompt_cache=[])
    with (
        patch.object(ar, "make_sampler", wraps=ar.make_sampler) as sampler,
        patch.object(
            ar, "_PositionedTargetSampler", wraps=ar._PositionedTargetSampler
        ) as positioned,
        patch.object(dispatch, "wired_limit", return_value=contextlib.nullcontext()),
    ):
        mx.random.seed(42)
        if entrypoint == "generate_step":
            tokens = [
                token
                for token, _ in ar.generate_step(
                    mx.array([[1]]), model, None, None, **options, **overrides
                )
            ]
        else:
            result = getattr(dispatch, entrypoint)(
                model, processor, "", input_ids=mx.array([[1]]), **options, **overrides
            )
            if entrypoint == "generate":
                tokens = result.token_ids
            else:
                tokens = [
                    chunk.token for chunk in result if chunk.finish_reason is None
                ]
    if seed is not None and expected["temperature"] > 0:
        positioned.assert_called_once_with(**expected, seed=seed)
        sampler.assert_not_called()
    else:
        actual = sampler.call_args.kwargs
        assert (actual["temp"], actual["top_p"], actual["top_k"]) == tuple(
            expected.values()
        )
        positioned.assert_not_called()
    assert len(tokens) == 32
    assert set(tokens) == ({1, 2} if expected["temperature"] > 0 else {2})
    assert vars(config) == original_config


def _sampling_model(config):
    model = MagicMock(config=config)
    model.language_model.return_value = LanguageModelOutput(
        logits=mx.array([[[-100.0, 0.0, 0.003] + [-100.0] * 125]])
    )
    model.get_input_embeddings.return_value = InputEmbeddingsFeatures(
        inputs_embeds=mx.zeros((1, 1, 4))
    )
    return model


@pytest.mark.parametrize("seed", [None, 42])
@pytest.mark.parametrize("do_sample", [False, True])
def test_custom_sampler_overrides_checkpoint(seed, do_sample):
    model = _sampling_model(NS(do_sample=do_sample, temperature=1.0, top_k=64))
    custom = MagicMock(
        spec=lambda logits: None,
        side_effect=lambda logits: mx.ones(logits.shape[:-1], dtype=mx.int32),
    )
    with (
        patch.object(ar, "make_sampler") as sampler,
        patch.object(ar, "_PositionedTargetSampler") as positioned,
    ):
        tokens = [
            token
            for token, _ in ar.generate_step(
                mx.array([[1]]),
                model,
                None,
                None,
                max_tokens=3,
                prompt_cache=[],
                sampler=custom,
                seed=seed,
            )
        ]
    assert tokens == [1, 1, 1]
    assert custom.called
    sampler.assert_not_called()
    positioned.assert_not_called()


def test_generate_step_without_model_config():
    model = _sampling_model(None)
    del model.config
    tokens = [
        token
        for token, _ in ar.generate_step(
            mx.array([[1]]), model, None, None, max_tokens=3, prompt_cache=[]
        )
    ]
    assert tokens == [2, 2, 2]


def test_diffusion_receives_resolved_sampling_defaults(sampling_case):
    config, overrides, expected = sampling_case
    model = NS(config=config)
    with (
        patch.object(dispatch, "is_diffusion_model", return_value=True),
        patch.object(
            dispatch, "stream_diffusion_generate_from_kwargs", return_value=iter(())
        ) as diffusion,
    ):
        list(
            dispatch.stream_generate(
                model, FakeProcessor(), "", input_ids=mx.array([[1]]), **overrides
            )
        )
    actual = diffusion.call_args.args[7]
    assert {key: actual[key] for key in expected} == expected
