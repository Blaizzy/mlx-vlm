from __future__ import annotations

import argparse
import json
import re
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from ci.bot import BotOutput
from ci.worker_result import finalize


class ReportError(ValueError):
    pass


COMMIT_PATTERN = re.compile(r"[0-9a-f]{40}")
FILE_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}\.json")
REPOSITORY_PATTERN = re.compile(
    r"[A-Za-z0-9][A-Za-z0-9._-]*/[A-Za-z0-9][A-Za-z0-9._-]*"
)


def build_report(
    exported: Mapping[str, Any],
    results_directory: Path,
    *,
    run_url: str,
    attempt_id: str,
    head_sha: str,
) -> dict[str, Any]:
    if (
        exported.get("schema_version") != 1
        or exported.get("attempt_id") != attempt_id
        or exported.get("head_sha") != head_sha
        or COMMIT_PATTERN.fullmatch(head_sha) is None
    ):
        raise ReportError("repository control identity does not match the attempt")
    control = exported.get("control")
    device_jobs = exported.get("device_jobs")
    repository = exported.get("repository")
    base_sha = exported.get("base_sha")
    contract_sha = exported.get("contract_sha")
    if (
        not isinstance(control, Mapping)
        or not isinstance(device_jobs, list)
        or len(device_jobs) > 512
        or not isinstance(repository, str)
        or REPOSITORY_PATTERN.fullmatch(repository) is None
        or not isinstance(base_sha, str)
        or COMMIT_PATTERN.fullmatch(base_sha) is None
        or not isinstance(contract_sha, str)
        or COMMIT_PATTERN.fullmatch(contract_sha) is None
    ):
        raise ReportError("repository control payload is incomplete")
    if any(
        control.get(field) != expected
        for field, expected in (
            ("repository", repository),
            ("target_sha", base_sha),
            ("head_sha", head_sha),
            ("contract_sha", contract_sha),
        )
    ):
        raise ReportError("repository control payload identity is inconsistent")

    results = []
    identifiers = set()
    filenames = set()
    for item in device_jobs:
        if not isinstance(item, Mapping) or not isinstance(
            item.get("manifest"), Mapping
        ):
            raise ReportError("repository device job is invalid")
        filename = item.get("file")
        if (
            not isinstance(filename, str)
            or FILE_PATTERN.fullmatch(filename) is None
            or item.get("id") != item["manifest"].get("id")
            or item.get("id") in identifiers
            or filename in filenames
            or item["manifest"].get("repository") != repository
            or item["manifest"].get("base_sha") != base_sha
            or item["manifest"].get("head_sha") != head_sha
            or item["manifest"].get("contract_sha") != contract_sha
        ):
            raise ReportError("repository device job has no result filename")
        identifiers.add(item["id"])
        filenames.add(filename)
        try:
            raw = _runner_result(results_directory, filename)
        except (OSError, ValueError, ReportError):
            raw = {}
        results.append(finalize(item["manifest"], raw))

    record = dict(control)
    outcomes = {str(result.get("outcome", "")) for result in results}
    fallback = str(control.get("outcome", "infrastructure_failure"))
    outcome = next(
        (
            value
            for value in (
                "test_failure",
                "regressed",
                "no_eligible_runner",
                "infrastructure_failure",
                "cancelled",
                "improved",
                "passed",
            )
            if value in outcomes
        ),
        fallback,
    )
    record.update(
        {
            "kind": "ci_execution",
            "outcome": outcome,
            "run_url": run_url,
            "attempt_id": attempt_id,
            "head_sha": head_sha,
            "results": results,
        }
    )
    return record


def render_report(record: Mapping[str, Any]) -> str:
    return BotOutput(record).render()


def _runner_result(directory: Path, filename: str) -> dict[str, Any] | None:
    if directory.is_symlink() or not directory.is_dir():
        raise ReportError("runner result directory is invalid")
    matches = tuple(directory.rglob(f"{filename}.result.json"))
    if not matches:
        return None
    if (
        len(matches) != 1
        or matches[0].is_symlink()
        or matches[0].stat().st_size > 1_000_000
    ):
        raise ReportError("runner result artifact is ambiguous or invalid")
    value = json.loads(matches[0].read_text())
    if not isinstance(value, Mapping):
        raise ReportError("runner result artifact must contain an object")
    return dict(value)


def _load(path: Path) -> dict[str, Any]:
    if path.is_symlink() or path.stat().st_size > 5_000_000:
        raise ReportError("repository control artifact is invalid")
    value = json.loads(path.read_text())
    if not isinstance(value, Mapping):
        raise ReportError("repository control artifact must contain an object")
    return dict(value)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--control", type=Path, required=True)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--run-url", required=True)
    parser.add_argument("--attempt-id", required=True)
    parser.add_argument("--head-sha", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    record = build_report(
        _load(args.control),
        args.results,
        run_url=args.run_url,
        attempt_id=args.attempt_id,
        head_sha=args.head_sha,
    )
    args.output.write_text(render_report(record))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
