import argparse
import ast
from pathlib import Path

SOURCE_ROOT = Path(__file__).resolve().parents[2]


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
