"""The CLI and the library must agree on every default they both define.

`mlx_vlm.generate` (the CLI) and `generate_step` (the library) used to declare
their own copies of the same constants, so one side could be changed without
the other. These tests compare the two by introspection rather than by
restating the values, so they keep working when a default is deliberately
changed and still fail when the two sides drift apart.
"""

import inspect
import sys
from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import patch

from mlx_vlm.generate import ar, common, dispatch
from mlx_vlm.generate.ar import generate_step
from mlx_vlm.generate.dispatch import parse_arguments


def _cli_defaults():
    with patch.object(sys, "argv", ["mlx_vlm.generate"]):
        return vars(parse_arguments())


def _library_defaults():
    return {
        name: parameter.default
        for name, parameter in inspect.signature(generate_step).parameters.items()
        if parameter.default is not inspect.Parameter.empty
    }


def test_cli_and_library_agree_on_shared_defaults():
    cli, library = _cli_defaults(), _library_defaults()
    shared = sorted(set(cli) & set(library))

    # Guard against this test quietly comparing nothing: a rename on either
    # side would otherwise empty the intersection and still pass.
    assert len(shared) >= 10, f"expected many shared defaults, found {shared}"

    # A CLI default of None means "not given on the command line", and the
    # value is resolved further down (--draft-kind is inferred from the model
    # type, for instance). Any other disagreement is drift.
    mismatched = {
        name: {"cli": cli[name], "library": library[name]}
        for name in shared
        if cli[name] is not None and cli[name] != library[name]
    }
    assert not mismatched, (
        f"CLI and library defaults disagree: {mismatched}. Both sides should "
        "read the constant from mlx_vlm.generate.common."
    )


def test_shared_defaults_are_defined_once():
    """Modules that expose a default re-export `common`'s, never their own copy."""
    constants = [name for name in dir(common) if name.startswith("DEFAULT_")]
    assert constants, "no DEFAULT_* constants found in mlx_vlm.generate.common"

    for module in (ar, dispatch):
        for name in constants:
            if hasattr(module, name):
                assert getattr(module, name) is getattr(common, name), (
                    f"{module.__name__}.{name} is not the object defined in "
                    "mlx_vlm.generate.common"
                )


def test_cli_sampling_defaults_are_deferred_to_the_model_config():
    cli = _cli_defaults()

    assert cli["temperature"] is None
    assert cli["top_p"] is None
    assert cli["top_k"] is None


def test_sampling_defaults_use_model_generation_config():
    config = SimpleNamespace(
        do_sample=True,
        temperature=1.0,
        top_p=0.95,
        top_k=64,
    )

    assert common.resolve_generation_sampling_defaults(config) == (1.0, 0.95, 64)


def test_sampling_defaults_preserve_explicit_overrides():
    config = SimpleNamespace(
        do_sample=False,
        temperature=1.0,
        top_p=0.95,
        top_k=64,
    )

    assert common.resolve_generation_sampling_defaults(
        config,
        temperature=0.7,
        top_p=1.0,
        top_k=0,
    ) == (0.7, 1.0, 0)


def test_sampling_defaults_fall_back_when_config_is_incomplete():
    config = SimpleNamespace(do_sample=False, temperature=1.0, top_p=None, top_k=None)

    assert common.resolve_generation_sampling_defaults(config) == (
        0.0,
        common.DEFAULT_TOP_P,
        common.DEFAULT_TOP_K,
    )


def test_sampling_defaults_fall_back_when_config_fields_are_missing():
    assert common.resolve_generation_sampling_defaults(SimpleNamespace()) == (
        common.DEFAULT_TEMPERATURE,
        common.DEFAULT_TOP_P,
        common.DEFAULT_TOP_K,
    )


def test_stream_generate_forwards_model_sampling_defaults():
    class InputIds:
        size = 1

        def flatten(self):
            return self

        def tolist(self):
            return [1]

    tokenizer = SimpleNamespace(all_special_ids=[])
    processor = SimpleNamespace(tokenizer=tokenizer)
    model = SimpleNamespace(
        config=SimpleNamespace(
            do_sample=True,
            temperature=1.0,
            top_p=0.95,
            top_k=64,
        ),
        language_model=object(),
    )
    captured = {}
    detokenizer = SimpleNamespace(
        last_segment="",
        add_token=lambda *args, **kwargs: None,
        finalize=lambda: None,
    )

    def fake_generate_step(*args, **kwargs):
        captured.update(kwargs)
        return iter(())

    with (
        patch.object(
            dispatch,
            "_prepare_generation_inputs",
            return_value=(InputIds(), None, None, {}),
        ),
        patch.object(dispatch, "is_diffusion_model", return_value=False),
        patch.object(
            dispatch._apc, "multimodal_token_ids_from_config", return_value=set()
        ),
        patch.object(dispatch._apc, "media_safe_prefix_min", return_value=0),
        patch.object(dispatch.cache, "make_prompt_cache", return_value=[]),
        patch.object(dispatch, "wired_limit", return_value=nullcontext()),
        patch.object(dispatch, "make_streaming_detokenizer", return_value=detokenizer),
        patch.object(dispatch, "generate_step", side_effect=fake_generate_step),
    ):
        list(dispatch.stream_generate(model, processor, "hello"))

    assert captured["temperature"] == 1.0
    assert captured["top_p"] == 0.95
    assert captured["top_k"] == 64


def test_stream_generate_keeps_diffusion_sampling_defaults_compatible():
    model = SimpleNamespace(config=SimpleNamespace())
    processor = SimpleNamespace(tokenizer=SimpleNamespace(all_special_ids=[]))
    captured = []

    def fake_diffusion(*args, **kwargs):
        captured.append(dict(args[7]))
        return iter(())

    with (
        patch.object(
            dispatch,
            "_prepare_generation_inputs",
            return_value=(object(), None, None, {}),
        ),
        patch.object(dispatch, "is_diffusion_model", return_value=True),
        patch.object(
            dispatch,
            "stream_diffusion_generate_from_kwargs",
            side_effect=fake_diffusion,
        ),
    ):
        list(dispatch.stream_generate(model, processor, "hello"))
        list(
            dispatch.stream_generate(
                model,
                processor,
                "hello",
                temperature=0.0,
                top_p=0.0,
                top_k=0,
            )
        )

    assert captured[0]["temperature"] == common.DEFAULT_TEMPERATURE
    assert captured[0]["top_p"] == common.DEFAULT_TOP_P
    assert captured[0]["top_k"] == common.DEFAULT_TOP_K
    assert captured[1]["temperature"] == 0.0
    assert captured[1]["top_p"] == 0.0
    assert captured[1]["top_k"] == 0
