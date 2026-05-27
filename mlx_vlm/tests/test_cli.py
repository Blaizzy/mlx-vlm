import argparse
import ast
from pathlib import Path

SOURCE_ROOT = Path(__file__).resolve().parents[2]


def _load_module(path: str) -> ast.Module:
    source_path = SOURCE_ROOT / path
    return ast.parse(source_path.read_text(), filename=str(source_path))


def _find_verbose_add_argument(module: ast.Module) -> ast.Call:
    for node in ast.walk(module):
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "add_argument"
            and node.args
            and isinstance(node.args[0], ast.Constant)
            and node.args[0].value == "--verbose"
        ):
            return node
    raise AssertionError("--verbose argument must be defined")


def _keyword_map(call: ast.Call) -> dict[str, ast.expr]:
    return {kw.arg: kw.value for kw in call.keywords}


def _assert_verbose_uses_boolean_optional_action(path: str) -> None:
    verbose_call = _find_verbose_add_argument(_load_module(path))
    keywords = _keyword_map(verbose_call)

    action = keywords["action"]
    assert isinstance(action, ast.Attribute)
    assert isinstance(action.value, ast.Name)
    assert action.value.id == "argparse"
    assert action.attr == "BooleanOptionalAction"

    default = keywords["default"]
    assert isinstance(default, ast.Constant)
    assert default.value is True


def test_generate_verbose_flag_uses_boolean_optional_action():
    _assert_verbose_uses_boolean_optional_action("mlx_vlm/generate/dispatch.py")


def test_chat_verbose_flag_uses_boolean_optional_action():
    _assert_verbose_uses_boolean_optional_action("mlx_vlm/chat.py")


def test_video_generate_verbose_flag_uses_boolean_optional_action():
    _assert_verbose_uses_boolean_optional_action("mlx_vlm/video_generate.py")


def test_verbose_flag_semantics():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--verbose",
        action=argparse.BooleanOptionalAction,
        default=True,
    )

    assert parser.parse_args([]).verbose is True
    assert parser.parse_args(["--verbose"]).verbose is True
    assert parser.parse_args(["--no-verbose"]).verbose is False
