import argparse
import ast
import importlib.util
import io
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

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


def _load_remote():
    path = SOURCE_ROOT / "mlx_vlm/generate/remote.py"
    spec = importlib.util.spec_from_file_location("remote_for_tests", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _remote_args(**overrides):
    values = {
        "adapter_path": None,
        "api_key": None,
        "audio": None,
        "base_url": None,
        "chat": False,
        "enable_thinking": False,
        "eos_tokens": None,
        "force_download": False,
        "frequency_context_size": 20,
        "frequency_penalty": None,
        "gen_kwargs": {},
        "image": None,
        "kv_bits": None,
        "kv_key_bits": None,
        "kv_key_scheme": None,
        "kv_value_bits": None,
        "kv_value_scheme": None,
        "local": False,
        "max_kv_size": None,
        "max_tokens": 32,
        "model": "demo/model",
        "output_modality": "text",
        "prefill_step_size": 2048,
        "presence_context_size": 20,
        "presence_penalty": None,
        "processor_kwargs": {},
        "prompt": ["hello", "there"],
        "quantize_activations": False,
        "repetition_context_size": 20,
        "repetition_penalty": None,
        "resize_shape": None,
        "revision": "main",
        "seed": None,
        "server": False,
        "system": None,
        "temperature": 0.0,
        "thinking_budget": None,
        "thinking_end_token": "</think>",
        "thinking_mode": None,
        "thinking_start_token": "<think>",
        "trust_remote_code": False,
        "verbose": False,
        "video": None,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def test_generate_server_flags_are_mutually_exclusive_and_routed_before_load():
    source = (SOURCE_ROOT / "mlx_vlm/generate/dispatch.py").read_text()
    assert "server_mode = parser.add_mutually_exclusive_group()" in source
    assert 'server_mode.add_argument(\n        "--server"' in source
    assert 'server_mode.add_argument(\n        "--local"' in source
    assert source.index("if run_on_server(args):") < source.index(
        "model, processor = load("
    )


def test_chat_exposes_the_same_server_controls():
    source = (SOURCE_ROOT / "mlx_vlm/chat.py").read_text()
    assert '"--base-url"' in source
    assert '"--api-key"' in source
    assert "server_mode = parser.add_mutually_exclusive_group()" in source
    assert "remote_args=args if use_server else None" in source
    assert "for chunk in stream_chat(" in source


def test_remote_eligibility_accepts_plain_generate_and_rejects_local_features():
    remote = _load_remote()

    assert remote.eligible_for_server(_remote_args())
    assert not remote.eligible_for_server(_remote_args(local=True))
    assert not remote.eligible_for_server(_remote_args(audio=["speech.wav"]))
    assert not remote.eligible_for_server(_remote_args(chat=True))
    assert not remote.eligible_for_server(_remote_args(kv_bits=4))
    assert not remote.eligible_for_server(_remote_args(revision="abc123"))
    assert not remote.eligible_for_server(_remote_args(prefill_step_size=512))
    assert not remote.eligible_for_server(_remote_args(prompt=[{"role": "user"}]))


def test_remote_build_messages_joins_prompt_and_preserves_image_urls(tmp_path):
    remote = _load_remote()
    image = tmp_path / "example.png"
    image.write_bytes(b"png")

    messages = remote.build_messages(
        _remote_args(
            system="be concise",
            image=[str(image), "https://example.test/remote.png"],
        )
    )

    assert messages[0] == {"role": "system", "content": "be concise"}
    content = messages[1]["content"]
    assert content[0] == {"type": "text", "text": "hello there"}
    assert content[1]["image_url"]["url"].startswith("data:image/png;base64,")
    assert content[2]["image_url"]["url"] == "https://example.test/remote.png"


def test_remote_sampling_params_preserve_supported_cli_values():
    remote = _load_remote()
    params = remote.sampling_params(
        _remote_args(
            enable_thinking=True,
            frequency_penalty=0.2,
            presence_penalty=0.3,
            repetition_penalty=1.1,
            resize_shape=[512, 768],
            seed=7,
            thinking_budget=64,
        )
    )

    assert params == {
        "enable_thinking": True,
        "frequency_context_size": 20,
        "frequency_penalty": 0.2,
        "max_tokens": 32,
        "presence_context_size": 20,
        "presence_penalty": 0.3,
        "repetition_context_size": 20,
        "repetition_penalty": 1.1,
        "resize_shape": [512, 768],
        "seed": 7,
        "temperature": 0.0,
        "thinking_budget": 64,
        "thinking_end_token": "</think>",
        "thinking_start_token": "<think>",
    }


def test_remote_stream_parses_sse_and_posts_expected_request(monkeypatch):
    remote = _load_remote()
    seen = {}

    class Response:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def __iter__(self):
            return iter(
                [
                    b"event: message" + bytes([10]),
                    b'data: {"choices":[{"delta":{"content":"hello"}}]}' + bytes([10]),
                    b'data: {"choices":[{"delta":{"content":" world"}}]}' + bytes([10]),
                    b"data: [DONE]" + bytes([10]),
                ]
            )

    def urlopen(request):
        seen["url"] = request.full_url
        seen["body"] = json.loads(request.data)
        return Response()

    monkeypatch.setattr(remote.urllib.request, "urlopen", urlopen)
    assert list(
        remote.stream_chat(
            "http://127.0.0.1:8080",
            "demo/model",
            [{"role": "user", "content": "hello"}],
            {"max_tokens": 32},
        )
    ) == ["hello", " world"]
    assert seen["url"] == "http://127.0.0.1:8080/v1/chat/completions"
    assert seen["body"]["stream"] is True
    assert seen["body"]["model"] == "demo/model"


def test_remote_server_probe_returns_false_for_missing_endpoint(monkeypatch):
    remote = _load_remote()

    def missing(*args, **kwargs):
        raise remote.urllib.error.HTTPError(
            "http://127.0.0.1:8080/v1/models", 404, "missing", {}, io.BytesIO()
        )

    monkeypatch.setattr(remote.urllib.request, "urlopen", missing)
    assert not remote.server_available("http://127.0.0.1:8080")


def test_remote_server_probe_surfaces_auth_errors(monkeypatch):
    remote = _load_remote()

    def unauthorized(*args, **kwargs):
        raise remote.urllib.error.HTTPError(
            "http://127.0.0.1:8080/v1/models",
            401,
            "unauthorized",
            {},
            io.BytesIO(b'{"detail":"bad key"}'),
        )

    monkeypatch.setattr(remote.urllib.request, "urlopen", unauthorized)
    with pytest.raises(remote.RemoteServerError, match="HTTP 401"):
        remote.server_available("http://127.0.0.1:8080")


def test_remote_run_streams_when_available(monkeypatch, capsys):
    remote = _load_remote()
    seen = {}
    monkeypatch.setattr(remote, "server_available", lambda *args: True)

    def stream(base_url, model, messages, params, args):
        seen.update(base_url=base_url, model=model, messages=messages, params=params)
        yield "answer"

    monkeypatch.setattr(remote, "stream_chat", stream)
    assert remote.run_on_server(_remote_args())
    assert capsys.readouterr().out == "answer" + chr(10)
    assert seen["messages"] == [{"role": "user", "content": "hello there"}]
    assert seen["params"]["enable_thinking"] is False


def test_remote_run_forced_rejects_unsupported_features():
    remote = _load_remote()
    with pytest.raises(remote.RemoteServerError, match="--server cannot"):
        remote.run_on_server(_remote_args(server=True, chat=True))


def test_remote_run_forced_errors_when_server_is_unavailable(monkeypatch):
    remote = _load_remote()
    monkeypatch.setattr(remote, "server_available", lambda *args: False)
    with pytest.raises(remote.RemoteServerError, match="No mlx-vlm server"):
        remote.run_on_server(_remote_args(server=True))


def test_remote_run_falls_back_only_before_any_output(monkeypatch):
    remote = _load_remote()
    monkeypatch.setattr(remote, "server_available", lambda *args: True)

    def disconnected(*args):
        raise remote.urllib.error.URLError("connection reset")
        yield

    monkeypatch.setattr(remote, "stream_chat", disconnected)
    assert not remote.run_on_server(_remote_args())
