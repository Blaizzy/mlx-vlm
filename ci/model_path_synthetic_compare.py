from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Mapping, Sequence

from ci.probe_process import run_project_probe


def run_probe(
    project: Path,
    probe: Path,
    job: Path,
    profiles: Path,
    output: Path,
) -> Mapping[str, Any]:
    run_project_probe(
        project,
        probe,
        [
            "--job",
            str(job),
            "--profiles",
            str(profiles),
            "--output",
            str(output),
        ],
    )
    value = json.loads(output.read_text())
    if not isinstance(value, Mapping):
        raise RuntimeError("synthetic probe output must be an object")
    return value


def compare(base: Mapping[str, Any], head: Mapping[str, Any]) -> dict[str, Any]:
    base_output = tuple(float(value) for value in base.get("output", []))
    head_output = tuple(float(value) for value in head.get("output", []))
    same_shape = base.get("output_shape") == head.get("output_shape")
    same_parameters = base.get("parameter_signature") == head.get("parameter_signature")
    finite = bool(base.get("finite")) and bool(head.get("finite"))
    comparable = (
        same_shape and bool(base_output) and len(base_output) == len(head_output)
    )
    max_abs_diff = (
        max(
            abs(base_value - head_value)
            for base_value, head_value in zip(base_output, head_output)
        )
        if comparable
        else None
    )
    match = bool(
        same_shape
        and same_parameters
        and finite
        and max_abs_diff is not None
        and max_abs_diff <= 1e-5
    )
    return {
        "verdict": "passed" if match else "test_failure",
        "correctness": {
            "match": match,
            "same_shape": same_shape,
            "same_parameters": same_parameters,
            "finite": finite,
            "max_abs_diff": max_abs_diff,
            "base_output_hash": base.get("output_hash"),
            "head_output_hash": head.get("output_hash"),
        },
        "base": {key: value for key, value in base.items() if key != "output"},
        "head": {key: value for key, value in head.items() if key != "output"},
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--job", type=Path, required=True)
    parser.add_argument("--profiles", type=Path, required=True)
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--head", type=Path, required=True)
    parser.add_argument("--probe", type=Path, required=True)
    args = parser.parse_args(argv)
    findings_path = Path(os.environ.get("CI_JOB_FINDINGS", "findings.json"))
    base_output = findings_path.with_suffix(".base.json")
    head_output = findings_path.with_suffix(".head.json")
    try:
        base = run_probe(args.base, args.probe, args.job, args.profiles, base_output)
        head = run_probe(args.head, args.probe, args.job, args.profiles, head_output)
        result = compare(base, head)
    except Exception as error:
        result = {
            "verdict": "test_failure",
            "error": f"{type(error).__name__}: {error}",
        }
    findings_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    return 2 if result["verdict"] == "test_failure" else 0


if __name__ == "__main__":
    raise SystemExit(main())
