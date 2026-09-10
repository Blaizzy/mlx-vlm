from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

RESULT_FIELDS = frozenset(
    {
        "schema_version",
        "kind",
        "job_id",
        "component",
        "subject",
        "device",
        "decision",
        "outcome",
        "reason",
        "repository",
        "observed",
        "cache",
        "exit_code",
        "model",
        "checkpoint_failure",
        "findings",
        "findings_error",
    }
)
CHECKPOINT_FAILURE_OUTCOMES = {
    "checkpoint_not_found": "test_failure",
    "access_denied": "test_failure",
    "disk_full": "declined",
    "network_transient": "infrastructure_failure",
    "checkpoint_policy_failed": "test_failure",
    "checkpoint_internal_error": "test_failure",
}


def _bounded_json(value: Any, depth: int = 0) -> bool:
    if depth > 8:
        return False
    if value is None or isinstance(value, (bool, int)):
        return True
    if isinstance(value, float):
        return math.isfinite(value)
    if isinstance(value, str):
        return len(value) <= 10_000
    if isinstance(value, Mapping):
        return len(value) <= 128 and all(
            isinstance(key, str) and len(key) <= 128 and _bounded_json(item, depth + 1)
            for key, item in value.items()
        )
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return len(value) <= 1_024 and all(
            _bounded_json(item, depth + 1) for item in value
        )
    return False


def _identity_matches(job: Mapping[str, Any], result: Mapping[str, Any]) -> bool:
    for field in ("component", "subject", "model"):
        if field in result and result[field] != job.get(field):
            return False
    return True


def _checkpoint_failure_is_valid(result: Mapping[str, Any]) -> bool:
    value = result.get("checkpoint_failure")
    if value is None:
        return True
    if not isinstance(value, Mapping):
        return False
    required = {"phase", "operation", "code", "retryable", "attempts"}
    allowed = required | {"diagnostic", "role"}
    if not required.issubset(value) or set(value) - allowed:
        return False
    code = value.get("code")
    attempts = value.get("attempts")
    return (
        code in CHECKPOINT_FAILURE_OUTCOMES
        and value.get("phase") == "hf_checkpoint"
        and value.get("operation") == "download"
        and isinstance(value.get("retryable"), bool)
        and value["retryable"] == (code == "network_transient")
        and isinstance(attempts, int)
        and not isinstance(attempts, bool)
        and 1 <= attempts <= 100
        and result.get("reason") == code
        and result.get("outcome") == CHECKPOINT_FAILURE_OUTCOMES[code]
    )


def finalize(
    job: Mapping[str, Any],
    result: Mapping[str, Any] | None,
    infrastructure_error: str | None = None,
) -> dict[str, Any]:
    if infrastructure_error:
        return _infrastructure_failure(job, infrastructure_error)

    if result is None:
        return _infrastructure_failure(job, "runner produced no result")

    if (
        bool(set(result) - RESULT_FIELDS)
        or not _bounded_json(result)
        or result.get("schema_version") != 1
        or result.get("kind") != "device_job_result"
        or result.get("job_id") != job.get("id")
        or result.get("repository") != job.get("repository")
        or not _identity_matches(job, result)
        or not _checkpoint_failure_is_valid(result)
        or not isinstance(result.get("device"), str)
        or not 1 <= len(result["device"]) <= 128
        or result.get("decision") not in {"accepted", "declined"}
        or result.get("outcome")
        not in {
            "accepted",
            "passed",
            "test_failure",
            "regressed",
            "infrastructure_failure",
            "declined",
        }
    ):
        return _infrastructure_failure(job, "runner result failed validation")
    decision = result["decision"]
    reason = result.get("reason")
    decline_reasons = {
        "declined_busy",
        "declined_memory",
        "declined_disk",
        "declined_thermal",
        "disk_full",
        "unhealthy",
    }
    accepted_reasons = {
        None,
        "access_denied",
        "checkpoint_internal_error",
        "checkpoint_not_found",
        "checkpoint_policy_failed",
        "correctness_regression",
        "network_transient",
    }
    if (decision == "accepted" and reason not in accepted_reasons) or (
        decision == "declined" and reason not in decline_reasons
    ):
        return _infrastructure_failure(job, "runner result failed validation")
    if (decision == "accepted") != (result["outcome"] != "declined"):
        return _infrastructure_failure(job, "runner result failed validation")
    if "findings" in result and not isinstance(result["findings"], Mapping):
        return _infrastructure_failure(job, "runner result failed validation")
    expected_reasons = {
        "accepted": {None},
        "passed": {None},
        "regressed": {"correctness_regression"},
        "infrastructure_failure": {"network_transient"},
    }
    if (
        result["outcome"] in expected_reasons
        and reason not in expected_reasons[result["outcome"]]
    ):
        return _infrastructure_failure(job, "runner result failed validation")

    output = dict(result)
    output.update(
        {
            "component": str(job.get("component", "runner")),
            "model": job.get("model"),
            "profile": job.get("profile"),
            "job_id": str(job.get("id", "")),
        }
    )
    findings = output.get("findings")
    if isinstance(findings, Mapping):
        phases = findings.get("phases")
        if isinstance(phases, Mapping):
            output["phases"] = dict(phases)
        metrics = findings.get("metrics")
        if isinstance(metrics, Mapping):
            output["metrics"] = dict(metrics)
        verdict = findings.get("verdict")
        if output.get("decision") == "accepted" and verdict in {
            "improved",
            "regressed",
            "test_failure",
        }:
            output["outcome"] = verdict
    if decision == "declined":
        output["outcome"] = "no_eligible_runner"
    return output


def _infrastructure_failure(job: Mapping[str, Any], message: str) -> dict[str, Any]:
    return {
        "component": str(job.get("component", "runner")),
        "model": job.get("model"),
        "profile": job.get("profile"),
        "job_id": str(job.get("id", "")),
        "outcome": "infrastructure_failure",
        "findings": {"error": message},
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--job", type=Path, required=True)
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--infrastructure-error", type=Path)
    args = parser.parse_args(argv)

    job = json.loads(args.job.read_text())
    raw_result = json.loads(args.result.read_text()) if args.result.is_file() else None
    infrastructure_error = None
    if args.infrastructure_error and args.infrastructure_error.is_file():
        error_lines = args.infrastructure_error.read_text().strip().splitlines()
        if error_lines:
            infrastructure_error = f"lease heartbeat failed: {error_lines[-1][:500]}"
    result = finalize(job, raw_result, infrastructure_error)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
