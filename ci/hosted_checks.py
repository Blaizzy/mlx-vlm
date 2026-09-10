from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path
from typing import Any, Mapping, Sequence

from ci.bot import BotOutput
from ci.docs_check import DocsCheckError, compare_docs


class HostedCheckError(ValueError):
    pass


def run_hosted_checks(
    control: Mapping[str, Any], repository: Path, base: str, head: str
) -> list[dict[str, Any]]:
    """Execute trusted handlers for every GitHub-hosted check in a plan."""

    checks = control.get("checks")
    if not isinstance(checks, list):
        raise HostedCheckError("control record has no hosted checks list")
    results: list[dict[str, Any]] = []
    for check in checks:
        if not isinstance(check, Mapping):
            raise HostedCheckError("hosted check must be an object")
        if check.get("execution_target") != "github_hosted":
            continue
        if check.get("work_type") != "Docs":
            results.append(_infrastructure_failure(check, "unsupported work type"))
            continue
        try:
            results.append(
                compare_docs(
                    repository,
                    base,
                    head,
                    [str(path) for path in check.get("changed_paths", [])],
                )
            )
        except (DocsCheckError, OSError, subprocess.CalledProcessError) as error:
            results.append(_infrastructure_failure(check, str(error)))
    return results


def resolved_record(
    control: Mapping[str, Any], results: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    """Attach hosted results without changing device work or approval gates."""

    record = dict(control)
    record["results"] = [dict(result) for result in results]
    outcomes = {str(result.get("outcome", "")) for result in results}
    for outcome in ("test_failure", "infrastructure_failure", "cancelled"):
        if outcome in outcomes:
            record["outcome"] = outcome
            record["hosted_outcome"] = outcome
            break
    else:
        record["hosted_outcome"] = "passed" if results else "skipped"
    return record


def _infrastructure_failure(check: Mapping[str, Any], message: str) -> dict[str, Any]:
    return {
        "component": str(check.get("component", "hosted_check")),
        "check_id": str(check.get("id", "unknown")),
        "outcome": "infrastructure_failure",
        "changed_paths": list(check.get("changed_paths", [])),
        "findings": {"new_errors": [message[:500]]},
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--control", type=Path, required=True)
    parser.add_argument("--repository-path", type=Path, required=True)
    parser.add_argument("--base", required=True)
    parser.add_argument("--head", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--markdown", type=Path, required=True)
    parser.add_argument("--github-output", type=Path)
    args = parser.parse_args(argv)

    control = json.loads(args.control.read_text())
    if not isinstance(control, Mapping):
        raise HostedCheckError("control record must be an object")
    results = run_hosted_checks(control, args.repository_path, args.base, args.head)
    record = resolved_record(control, results)
    args.output.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
    args.markdown.write_text(BotOutput(record).render())
    if args.github_output:
        with args.github_output.open("a") as stream:
            stream.write(f"outcome={record['hosted_outcome']}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
