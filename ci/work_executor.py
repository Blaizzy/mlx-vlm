from __future__ import annotations

import argparse
import json
import os
import subprocess
from pathlib import Path
from typing import Any, Mapping, Sequence

from ci.components.base import ExecutionContext
from ci.components.registry import phase_commands
from ci.execution_security import ExecutionSecurityError, verify_execution

FORWARDED_ENVIRONMENT = frozenset(
    {
        "CI_CHECKPOINT_PATH",
        "CI_REQUIRE_SANDBOX",
        "CI_JOB_PYTHON",
        "HF_ASSETS_CACHE",
        "HF_HOME",
        "HF_HUB_CACHE",
        "HF_HUB_OFFLINE",
        "HOME",
        "LANG",
        "LC_ALL",
        "PATH",
        "TMPDIR",
        "TRANSFORMERS_OFFLINE",
        "UV_CACHE_DIR",
        "UV_OFFLINE",
        "UV_PYTHON_INSTALL_DIR",
        "UV_PROJECT_ENVIRONMENT",
    }
)


def _run(command: list[str], findings: Path) -> tuple[int, dict[str, Any]]:
    environment = {
        key: value for key, value in os.environ.items() if key in FORWARDED_ENVIRONMENT
    }
    environment["CI_JOB_FINDINGS"] = str(findings)
    environment["PYTHONPATH"] = str(Path(__file__).resolve().parent.parent)
    completed = subprocess.run(command, env=environment)
    if not findings.is_file():
        return completed.returncode or 1, {
            "verdict": "test_failure",
            "error": "phase produced no findings",
        }
    value = json.loads(findings.read_text())
    if not isinstance(value, Mapping):
        return completed.returncode or 1, {
            "verdict": "test_failure",
            "error": "phase findings must be an object",
        }
    return completed.returncode, dict(value)


def _phase(findings: Mapping[str, Any], returncode: int) -> dict[str, Any]:
    verdict = str(findings.get("verdict", "test_failure"))
    outcome = (
        verdict if verdict in {"passed", "improved", "regressed"} else "test_failure"
    )
    if returncode and outcome != "test_failure":
        outcome = "test_failure"
    return {"outcome": outcome, "findings": dict(findings)}


def run(
    args: argparse.Namespace, *, validate_execution: bool = True
) -> tuple[int, dict[str, Any]]:
    output = Path(os.environ.get("CI_JOB_FINDINGS", "findings.json"))
    job = json.loads(args.job.read_text())
    configured = job.get("phases", [])
    if not isinstance(configured, list) or not configured:
        raise ValueError("work item has no phases")
    context = ExecutionContext(
        job_path=args.job,
        control=args.control,
        base=args.base,
        head=args.head,
        image=args.image or args.control / "ci" / "assets" / "cat.jpg",
        max_tokens=args.max_tokens,
    )
    commands = phase_commands(context)
    if validate_execution:
        try:
            verify_execution(
                job,
                control=args.control,
                base=args.base,
                head=args.head,
                commands=commands,
            )
        except (ExecutionSecurityError, OSError, subprocess.SubprocessError) as error:
            result = {
                "verdict": "test_failure",
                "security": {
                    "passed": False,
                    "reason": f"{type(error).__name__}: {error}",
                },
                "phases": {},
            }
            output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
            return 2, result
    phases: dict[str, Any] = {}
    for index, name in enumerate(configured):
        if name not in commands:
            raise ValueError(f"unsupported work phase: {name}")
        code, findings = _run(
            commands[name], output.with_name(f"{output.stem}-{name}.json")
        )
        phases[name] = _phase(findings, code)
        if phases[name]["outcome"] == "test_failure":
            for pending in configured[index + 1 :]:
                phases[pending] = {
                    "outcome": "skipped",
                    "findings": {"reason": f"{name}_failed"},
                }
            break
    outcomes = {phase["outcome"] for phase in phases.values()}
    if "test_failure" in outcomes:
        verdict = "test_failure"
    elif "regressed" in outcomes:
        verdict = "regressed"
    elif "improved" in outcomes:
        verdict = "improved"
    else:
        verdict = "passed"
    result = {"verdict": verdict, "phases": phases}
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    return (2 if verdict == "test_failure" else 0), result


def main(argv: Sequence[str] | None = None) -> int:
    directory = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser()
    parser.add_argument("--job", type=Path, required=True)
    parser.add_argument("--control", type=Path, default=directory.parent)
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--head", type=Path, required=True)
    parser.add_argument("--image", type=Path)
    parser.add_argument("--max-tokens", type=int, default=16)
    args = parser.parse_args(argv)
    return run(args)[0]


if __name__ == "__main__":
    raise SystemExit(main())
