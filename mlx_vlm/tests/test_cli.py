import argparse
import ast
import io
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from mlx_vlm.decision import comparison_report, encode_record, format_response
from mlx_vlm.decision import main as decision_main
from mlx_vlm.tests.test_processors import _decision_request, _DecisionTokenizer

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


# Decision CLI: parsing, output streams, and comparison reports.


@pytest.fixture
def decision_cli_stub(monkeypatch):
    from mlx_vlm.decision_scheduler import DecisionJob

    calls = []

    def load_model(path):
        print("loader diagnostic")
        return SimpleNamespace(config=SimpleNamespace())

    class Scheduler:
        def __init__(self, loader, **kwargs):
            self.loader = loader
            self.loaded = set()

        def submit(self, body):
            calls.append(body)
            if body["model"] not in self.loaded:
                self.loader(body["model"])
                self.loaded.add(body["model"])
            record = encode_record(_DecisionTokenizer(), body)
            probs = [[0.8, 0.2], [0.25, 0.75], [0.1, 0.2, 0.7]]
            if body["model"] == "reference":
                probs = [p[::-1] for p in probs]
            job = DecisionJob(body)
            job.future.set_result(
                {
                    "response": format_response(body, record, probs),
                    "probabilities": probs,
                }
            )
            return job

        def stop_and_join(self):
            pass

    monkeypatch.setattr("mlx_vlm.utils.get_model_path", lambda name: name)
    monkeypatch.setattr("mlx_vlm.utils.load_model", load_model)
    monkeypatch.setattr(
        "mlx_vlm.utils.load_processor",
        lambda *a, **k: SimpleNamespace(tokenizer=_DecisionTokenizer()),
    )
    monkeypatch.setattr("mlx_vlm.decision_scheduler.DecisionScheduler", Scheduler)
    return calls


@pytest.mark.parametrize("kind", ["file", "stdin", "inline", "jsonl"])
def test_decision_cli_input_modes_and_json_only_stdout(
    decision_cli_stub, monkeypatch, tmp_path, capsys, kind
):
    body = _decision_request()
    path = tmp_path / "requests.json"
    if kind == "file":
        path.write_text(json.dumps(body))
        argv = ["--request", str(path)]
    elif kind == "stdin":
        monkeypatch.setattr("sys.stdin", io.StringIO(json.dumps(body)))
        argv = ["--request", "-"]
    elif kind == "inline":
        questions = tmp_path / "questions.json"
        questions.write_text(json.dumps(body["questions"]))
        argv = [
            "--model",
            body["model"],
            "--state",
            body["state"],
            "--questions",
            "@" + str(questions),
        ]
    else:
        monkeypatch.setattr(
            "sys.stdin",
            io.StringIO(
                "\n".join(json.dumps({**body, "state": str(i)}) for i in range(5))
            ),
        )
        argv = ["--request", "-", "--jsonl", "--batch-size", "2"]
    decision_main(argv)
    captured = capsys.readouterr()
    assert "loader diagnostic" in captured.err
    outputs = (
        [json.loads(line) for line in captured.out.splitlines()]
        if kind == "jsonl"
        else [json.loads(captured.out)]
    )
    assert len(outputs) == len(decision_cli_stub) == (5 if kind == "jsonl" else 1)
    assert all(output["model"] == body["model"] for output in outputs)
    assert [call["state"] for call in decision_cli_stub] == (
        [str(i) for i in range(5)] if kind == "jsonl" else [body["state"]]
    )


def test_decision_comparison_uses_unrounded_probabilities_and_gates_changes():
    report = comparison_report([[[0.50001, 0.49999]]], [[[0.49999, 0.50001]]])
    assert report["argmax_flips"] == 1
    assert 0 < report["max_probability_drift"] < 0.0001
    assert not report["passed"]
    assert comparison_report([[[0.5, 0.5]]], [[[0.5, 0.5]]])["passed"]
    assert not comparison_report(
        [[[0.7, 0.3]]], [[[0.6, 0.4]]], max_probability_drift=0.05
    )["passed"]


def test_decision_cli_comparison_report_and_exit_code(
    decision_cli_stub, monkeypatch, tmp_path
):
    monkeypatch.setattr("sys.stdin", io.StringIO(json.dumps(_decision_request())))
    report = tmp_path / "drift.json"
    with pytest.raises(SystemExit) as exc:
        decision_main(
            [
                "--request",
                "-",
                "--compare-model",
                "reference",
                "--report",
                str(report),
                "--max-probability-drift",
                "0",
            ]
        )
    assert exc.value.code == 1
    result = json.loads(report.read_text())
    assert not result["passed"]
    assert result["argmax_flips"] == 3
    assert [body["model"] for body in decision_cli_stub] == [
        "decision-test",
        "reference",
    ]
