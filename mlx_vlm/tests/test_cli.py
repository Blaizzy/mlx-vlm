"""CLI arguments, detector display options, and shared generation defaults."""

import argparse
import ast
import inspect
import os
import subprocess
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

import pytest

from mlx_vlm.generate import ar, common, dispatch
from mlx_vlm.generate.ar import generate_step
from mlx_vlm.generate.dispatch import parse_arguments
from mlx_vlm.models.rfdetr.generate import _get_annotator, main
from mlx_vlm.models.sam3 import generate as sam3_generate

# CLI arguments

SOURCE_ROOT = Path(__file__).resolve().parents[2]


def test_package_clears_main_thread_mlx_streams_at_exit(tmp_path):
    marker = tmp_path / "streams-cleared"
    env = os.environ.copy()
    env["MLX_VLM_CLEAR_STREAMS_MARKER"] = str(marker)
    script = """
import os
from pathlib import Path

import mlx.core as mx

original_clear_streams = getattr(mx, "clear_streams", None)


def record_clear_streams():
    Path(os.environ["MLX_VLM_CLEAR_STREAMS_MARKER"]).write_text("cleared")
    if original_clear_streams is not None:
        original_clear_streams()


mx.clear_streams = record_clear_streams
import mlx_vlm
"""

    subprocess.run([sys.executable, "-c", script], check=True, env=env)

    assert marker.read_text() == "cleared"


def _load_module(path: str) -> ast.Module:
    source_path = SOURCE_ROOT / path
    return ast.parse(source_path.read_text(), filename=str(source_path))


def _find_add_argument(module: ast.Module, flag: str) -> ast.Call:
    for node in ast.walk(module):
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "add_argument"
            and node.args
            and isinstance(node.args[0], ast.Constant)
            and node.args[0].value == flag
        ):
            return node
    raise AssertionError(f"{flag} argument must be defined")


def _find_verbose_add_argument(module: ast.Module) -> ast.Call:
    return _find_add_argument(module, "--verbose")


def _keyword_map(call: ast.Call) -> dict[str, ast.expr]:
    return {kw.arg: kw.value for kw in call.keywords}


def _assert_verbose_uses_boolean_optional_action(
    path: str, *, expected_default: bool
) -> None:
    verbose_call = _find_verbose_add_argument(_load_module(path))
    keywords = _keyword_map(verbose_call)

    action = keywords["action"]
    assert isinstance(action, ast.Attribute)
    assert isinstance(action.value, ast.Name)
    assert action.value.id == "argparse"
    assert action.attr == "BooleanOptionalAction"

    default = keywords["default"]
    assert isinstance(default, ast.Constant)
    assert default.value is expected_default


def test_generate_verbose_flag_uses_boolean_optional_action():
    _assert_verbose_uses_boolean_optional_action(
        "mlx_vlm/generate/dispatch.py", expected_default=False
    )


def test_chat_verbose_flag_uses_boolean_optional_action():
    _assert_verbose_uses_boolean_optional_action(
        "mlx_vlm/chat.py", expected_default=True
    )


def test_generate_verbose_flag_semantics():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--verbose", action=argparse.BooleanOptionalAction, default=False
    )

    assert parser.parse_args([]).verbose is False
    assert parser.parse_args(["--verbose"]).verbose is True
    assert parser.parse_args(["--no-verbose"]).verbose is False


def _literal_values(node: ast.expr) -> tuple:
    if isinstance(node, (ast.Tuple, ast.List)):
        return tuple(item.value for item in node.elts if isinstance(item, ast.Constant))
    raise AssertionError("expected literal tuple or list")


def _assert_thinking_mode_flag(path: str) -> None:
    call = _find_add_argument(_load_module(path), "--thinking-mode")
    keywords = _keyword_map(call)

    assert _literal_values(keywords["choices"]) == ("enabled", "disabled", "adaptive")
    default = keywords["default"]
    assert isinstance(default, ast.Constant)
    assert default.value is None


def test_generate_thinking_mode_flag():
    _assert_thinking_mode_flag("mlx_vlm/generate/dispatch.py")


def test_chat_thinking_mode_flag():
    _assert_thinking_mode_flag("mlx_vlm/chat.py")


def _find_function_def(module: ast.Module, name: str) -> ast.FunctionDef:
    for node in ast.walk(module):
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return node
    raise AssertionError(f"{name} must be defined")


def _is_args_system(node: ast.expr) -> bool:
    return (
        isinstance(node, ast.Attribute)
        and node.attr == "system"
        and isinstance(node.value, ast.Name)
        and node.value.id == "args"
    )


def test_generate_one_shot_applies_system_prompt():
    main = _find_function_def(_load_module("mlx_vlm/generate/dispatch.py"), "main")

    def _assigns_prompt(block: ast.If) -> bool:
        return any(
            isinstance(stmt, ast.Assign)
            and any(
                isinstance(target, ast.Name) and target.id == "prompt"
                for target in stmt.targets
            )
            for stmt in ast.walk(block)
        )

    assert any(
        isinstance(node, ast.If)
        and _is_args_system(node.test)
        and _assigns_prompt(node)
        for node in ast.walk(main)
    ), "one-shot generate must prepend args.system to the prompt"


# SAM3 thresholds


class TestSam3ThresholdDefaults(unittest.TestCase):
    def _dispatch(self, argv):
        recorded = {}

        def recorder(name):
            def _fn(**kwargs):
                recorded[name] = kwargs

            return _fn

        with (
            patch.object(sam3_generate, "run_image", recorder("run_image")),
            patch.object(sam3_generate, "track_video", recorder("track_video")),
            patch.object(
                sam3_generate, "track_video_realtime", recorder("track_video_realtime")
            ),
            patch.object(sys, "argv", ["generate.py"] + argv),
        ):
            sam3_generate.main()

        self.assertEqual(len(recorded), 1, f"expected one dispatch, got {recorded}")
        return next(iter(recorded.values()))

    def test_image_task_uses_its_documented_default(self):
        kwargs = self._dispatch(
            ["--task", "segment", "--image", "img.png", "--prompt", "a dog"]
        )
        self.assertEqual(kwargs["threshold"], 0.3)

    def test_track_task_uses_its_documented_default(self):
        kwargs = self._dispatch(
            ["--task", "track", "--video", "clip.mp4", "--prompt", "a car"]
        )
        self.assertEqual(kwargs["threshold"], 0.15)

    def test_realtime_task_default_is_unchanged(self):
        kwargs = self._dispatch(["--task", "realtime", "--prompt", "a car"])
        self.assertEqual(kwargs["threshold"], 0.5)


# RF-DETR box display


def _chain_names(annotator):
    return [type(a).__name__ for a in getattr(annotator, "annotators", [annotator])]


def test_segment_chain_drops_boxes_and_labels_when_show_boxes_is_off():
    assert _chain_names(_get_annotator(None, "segment", show_boxes=True)) == [
        "MaskAnnotator",
        "BoxAnnotator",
        "LabelAnnotator",
    ]
    assert _chain_names(_get_annotator(None, "segment", show_boxes=False)) == [
        "MaskAnnotator"
    ]


def test_detect_keeps_boxes_because_nothing_else_is_drawn():
    # A detect chain has nothing but boxes and labels, so turning them off
    # would render an empty overlay.
    for show_boxes in (True, False):
        assert _chain_names(_get_annotator(None, "detect", show_boxes=show_boxes)) == [
            "BoxAnnotator",
            "LabelAnnotator",
        ]


def _capture_parser():
    """Return the ArgumentParser that main() builds."""

    class _Stop(Exception):
        pass

    captured = {}

    def _fake_parse(self, *args, **kwargs):
        captured["parser"] = self
        raise _Stop()

    with patch.object(argparse.ArgumentParser, "parse_args", _fake_parse):
        try:
            main()
        except _Stop:
            pass
    return captured["parser"]


def test_cli_can_turn_boxes_off():
    parser = _capture_parser()

    args = parser.parse_args(["--model", "m", "--image", "i.jpg"])
    assert args.show_boxes is True

    args = parser.parse_args(["--model", "m", "--image", "i.jpg", "--no-show-boxes"])
    assert args.show_boxes is False


# CLI and library default parity


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


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
