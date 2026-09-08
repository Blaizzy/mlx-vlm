import argparse
import ast
import os
import subprocess
import sys
from pathlib import Path

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
        "--verbose",
        action=argparse.BooleanOptionalAction,
        default=False,
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


# --- Server-aware generate (--base-url) ----------------------------------------

import importlib.util
import io


def _load_remote():
    """Load generate/remote.py in isolation (stdlib-only, no mlx import)."""
    path = SOURCE_ROOT / "mlx_vlm" / "generate" / "remote.py"
    spec = importlib.util.spec_from_file_location("mlx_vlm_remote_under_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _remote_args(**overrides):
    base = dict(
        model="mlx-community/Model",
        prompt="hello",
        system=None,
        image=None,
        base_url=None,
        api_key=None,
        verbose=False,
        temperature=0.0,
        max_tokens=256,
        seed=None,
        repetition_penalty=None,
        frequency_penalty=None,
        presence_penalty=None,
        adapter_path=None,
        draft_model=None,
        quantize_activations=False,
    )
    base.update(overrides)
    return argparse.Namespace(**base)


class _FakeResponse:
    def __init__(self, lines):
        self._lines = lines

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def __iter__(self):
        return iter(self._lines)


def test_generate_defines_base_url_and_api_key():
    module = _load_module("mlx_vlm/generate/dispatch.py")
    _find_add_argument(module, "--base-url")
    _find_add_argument(module, "--api-key")


def test_generate_main_routes_to_server_before_load():
    module = _load_module("mlx_vlm/generate/dispatch.py")
    main = _find_function_def(module, "main")
    calls = [
        node.func.id
        for node in ast.walk(main)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
    ]
    assert "run_on_server" in calls and "load" in calls
    assert calls.index("run_on_server") < calls.index("load")


def test_resolve_base_url_precedence(monkeypatch):
    remote = _load_remote()
    monkeypatch.delenv("MLX_VLM_BASE_URL", raising=False)
    assert remote.resolve_base_url(_remote_args()) is None
    monkeypatch.setenv("MLX_VLM_BASE_URL", "http://env:1/")
    assert remote.resolve_base_url(_remote_args()) == "http://env:1/"
    assert (
        remote.resolve_base_url(_remote_args(base_url="http://flag:2"))
        == "http://flag:2"
    )


def test_run_on_server_returns_false_without_base_url(monkeypatch):
    remote = _load_remote()
    monkeypatch.delenv("MLX_VLM_BASE_URL", raising=False)

    def _fail(*a, **k):
        raise AssertionError("must not touch the network without a base URL")

    monkeypatch.setattr(remote.urllib.request, "urlopen", _fail)
    assert remote.run_on_server(_remote_args()) is False


def test_build_messages_passes_images_through_unencoded(tmp_path):
    remote = _load_remote()
    img = tmp_path / "pic.png"
    img.write_bytes(b"not-a-real-png")
    msgs = remote.build_messages(_remote_args(prompt="what?", image=[str(img)]))
    content = msgs[0]["content"]
    assert content[0] == {"type": "text", "text": "what?"}
    url = content[1]["image_url"]["url"]
    assert url == str(img)  # passed through, NOT base64-encoded
    assert not url.startswith("data:")


def test_build_messages_system_and_nargs_prompt():
    remote = _load_remote()
    msgs = remote.build_messages(
        _remote_args(system="be terse", prompt=["hi", "there"])
    )
    assert msgs[0] == {"role": "system", "content": "be terse"}
    assert msgs[1] == {"role": "user", "content": "hi there"}


def test_sampling_params_default_is_spec_only():
    remote = _load_remote()
    # A default request carries only OpenAI-spec fields — no mlx-vlm extras.
    params = remote.sampling_params(_remote_args(temperature=0.0, max_tokens=256))
    assert params == {"temperature": 0.0, "max_tokens": 256}
    assert "enable_thinking" not in params
    assert "repetition_context_size" not in params


def test_sampling_params_adds_extras_only_when_engaged():
    remote = _load_remote()
    params = remote.sampling_params(
        _remote_args(
            temperature=0.7,
            repetition_penalty=1.1,
            repetition_context_size=20,
            enable_thinking=True,
            thinking_budget=64,
            thinking_start_token="<t>",
            thinking_end_token="</t>",
        )
    )
    assert params["temperature"] == 0.7
    assert params["repetition_penalty"] == 1.1
    assert params["repetition_context_size"] == 20  # only because its penalty is set
    assert params["enable_thinking"] is True
    assert params["thinking_budget"] == 64
    assert params["thinking_start_token"] == "<t>"


def test_sampling_params_context_size_dropped_without_penalty():
    remote = _load_remote()
    # A context size with no matching penalty is a meaningless extra — omit it.
    params = remote.sampling_params(_remote_args(repetition_context_size=20))
    assert "repetition_context_size" not in params


def test_stream_chat_parses_sse(monkeypatch):
    remote = _load_remote()
    lines = [
        b'data: {"choices":[{"delta":{"role":"assistant"}}]}\n',
        b'data: {"choices":[{"delta":{"content":"Hel"}}]}\n',
        b'data: {"choices":[{"delta":{"content":"lo"}}]}\n',
        b"data: {bad}\n",
        b"data: [DONE]\n",
        b'data: {"choices":[{"delta":{"content":"after"}}]}\n',
    ]
    monkeypatch.setattr(
        remote.urllib.request, "urlopen", lambda *a, **k: _FakeResponse(lines)
    )
    assert list(remote.stream_chat("http://x", "m", [], {})) == ["Hel", "lo"]


def test_run_on_server_streams_and_returns_true(monkeypatch, capsys):
    remote = _load_remote()
    lines = [b'data: {"choices":[{"delta":{"content":"Hi!"}}]}\n', b"data: [DONE]\n"]
    monkeypatch.setattr(
        remote.urllib.request, "urlopen", lambda *a, **k: _FakeResponse(lines)
    )
    assert remote.run_on_server(_remote_args(base_url="http://x")) is True
    assert "Hi!" in capsys.readouterr().out


def test_run_on_server_conflicting_flag_errors(monkeypatch):
    remote = _load_remote()

    def _fail(*a, **k):
        raise AssertionError("must not touch the network on a conflict")

    monkeypatch.setattr(remote.urllib.request, "urlopen", _fail)
    try:
        remote.run_on_server(_remote_args(base_url="http://x", adapter_path="/a"))
        raise AssertionError("expected SystemExit")
    except SystemExit as exc:
        assert "adapter-path" in str(exc)


def test_run_on_server_unreachable_errors(monkeypatch):
    remote = _load_remote()

    def _refused(*a, **k):
        raise remote.urllib.error.URLError("connection refused")

    monkeypatch.setattr(remote.urllib.request, "urlopen", _refused)
    try:
        remote.run_on_server(_remote_args(base_url="http://x"))
        raise AssertionError("expected SystemExit")
    except SystemExit as exc:
        assert "unreachable" in str(exc)


def test_run_on_server_http_error_surfaced(monkeypatch):
    remote = _load_remote()

    def _boom(*a, **k):
        raise remote.urllib.error.HTTPError(
            "http://x", 500, "err", {}, io.BytesIO(b"kaboom")
        )

    monkeypatch.setattr(remote.urllib.request, "urlopen", _boom)
    try:
        remote.run_on_server(_remote_args(base_url="http://x"))
        raise AssertionError("expected SystemExit")
    except SystemExit as exc:
        assert "HTTP 500" in str(exc)
