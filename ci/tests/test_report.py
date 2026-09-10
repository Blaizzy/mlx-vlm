import json

import pytest

from ci.report import ReportError, build_report, render_report

HEAD_SHA = "b" * 40
BASE_SHA = "a" * 40
CONTRACT_SHA = "c" * 40


def manifest():
    return {
        "id": "model_path:example",
        "component": "model_path",
        "model": "example",
        "profile": "dense",
        "repository": "Example/project",
        "base_sha": BASE_SHA,
        "head_sha": HEAD_SHA,
        "contract_sha": CONTRACT_SHA,
        "changed_paths": ["mlx_vlm/models/example/model.py"],
        "phases": ["synthetic"],
    }


def exported():
    job = manifest()
    return {
        "schema_version": 1,
        "attempt_id": "attempt:1",
        "repository": "Example/project",
        "base_sha": BASE_SHA,
        "head_sha": HEAD_SHA,
        "contract_sha": CONTRACT_SHA,
        "device_jobs": [{"id": job["id"], "file": "000.json", "manifest": job}],
        "control": {
            "schema_version": 1,
            "kind": "approved_job_plan",
            "repository": "Example/project",
            "target_sha": BASE_SHA,
            "head_sha": HEAD_SHA,
            "contract_sha": CONTRACT_SHA,
            "outcome": "ready",
            "jobs": [job],
            "gates": [],
            "checks": [],
            "errors": [],
        },
    }


def runner_result(**updates):
    value = {
        "schema_version": 1,
        "kind": "device_job_result",
        "job_id": manifest()["id"],
        "repository": "Example/project",
        "decision": "accepted",
        "outcome": "passed",
        "device": "runner-one",
        "findings": {"verdict": "passed"},
    }
    value.update(updates)
    return value


def test_report_renders_validated_runner_result(tmp_path):
    (tmp_path / "000.json.result.json").write_text(json.dumps(runner_result()))

    record = build_report(
        exported(),
        tmp_path,
        run_url="https://example.test/run/1",
        attempt_id="attempt:1",
        head_sha=HEAD_SHA,
    )

    assert record["outcome"] == "passed"
    assert record["results"][0]["device"] == "runner-one"
    rendered = render_report(record)
    assert "Attempt: `attempt:1`" in rendered
    assert "runner-one" in rendered


def test_missing_result_is_terminal_infrastructure_failure(tmp_path):
    record = build_report(
        exported(),
        tmp_path,
        run_url="run",
        attempt_id="attempt:1",
        head_sha=HEAD_SHA,
    )

    assert record["outcome"] == "infrastructure_failure"
    assert record["results"][0]["findings"] == {"error": "runner produced no result"}


def test_report_rejects_attempt_identity_mismatch(tmp_path):
    with pytest.raises(ReportError, match="identity"):
        build_report(
            exported(),
            tmp_path,
            run_url="run",
            attempt_id="different",
            head_sha=HEAD_SHA,
        )


def test_report_rejects_result_filename_patterns(tmp_path):
    control = exported()
    control["device_jobs"][0]["file"] = "*.json"

    with pytest.raises(ReportError, match="filename"):
        build_report(
            control,
            tmp_path,
            run_url="run",
            attempt_id="attempt:1",
            head_sha=HEAD_SHA,
        )


def test_report_rejects_inconsistent_control_identity(tmp_path):
    control = exported()
    control["control"]["repository"] = "Different/project"

    with pytest.raises(ReportError, match="inconsistent"):
        build_report(
            control,
            tmp_path,
            run_url="run",
            attempt_id="attempt:1",
            head_sha=HEAD_SHA,
        )


def test_report_rejects_duplicate_device_jobs(tmp_path):
    control = exported()
    control["device_jobs"].append(dict(control["device_jobs"][0]))

    with pytest.raises(ReportError, match="filename"):
        build_report(
            control,
            tmp_path,
            run_url="run",
            attempt_id="attempt:1",
            head_sha=HEAD_SHA,
        )


@pytest.mark.parametrize("contents", ("not json", "[]"))
def test_malformed_result_is_terminal_infrastructure_failure(tmp_path, contents):
    (tmp_path / "000.json.result.json").write_text(contents)

    record = build_report(
        exported(),
        tmp_path,
        run_url="run",
        attempt_id="attempt:1",
        head_sha=HEAD_SHA,
    )

    assert record["outcome"] == "infrastructure_failure"
    assert record["results"][0]["findings"] == {
        "error": "runner result failed validation"
    }
