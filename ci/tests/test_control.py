import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from ci.control import (
    ControlError,
    ExecutionOutcome,
    PlanningOutcome,
    configuration_digest,
    control_record,
    execution_outcome,
    export_repository_plan,
    main,
    planning_outcome,
    release_control,
    render_status,
)

BASE_SHA = "a" * 40
HEAD_SHA = "b" * 40


def plan(*, jobs=None, gates=None, checks=None, blocked=None, head_sha=HEAD_SHA):
    return {
        "schema_version": 1,
        "base_sha": BASE_SHA,
        "target_sha": BASE_SHA,
        "head_sha": head_sha,
        "rules": [],
        "components": [],
        "jobs": jobs or [],
        "gates": gates or [],
        "checks": checks or [],
        "blocked": blocked or [],
    }


def pending_work():
    return {
        "id": "model_path:example",
        "work_type": "ModelPath",
        "component": "model_path",
        "subject": "example",
        "model": "example",
        "phases": ["synthetic", "hf_checkpoint"],
        "required_memory_gib": 8,
        "required_disk_gib": 4,
        "synthetic": {"adapter": "example", "profile": "dense_vlm"},
        "hf_checkpoint": {
            "repo": "example/model",
            "revision": "c" * 40,
            "expected_model_type": "example",
            "weight": {"bytes": 1024},
        },
        "scenarios": ["vlm_animal"],
    }


def test_released_job_accepts_bound_repository_identity():
    record = control_record(plan(jobs=[pending_work()]), "example/repository", 8)
    released = release_control(record, HEAD_SHA)
    job = dict(released["jobs"][0])
    job["repository"] = "example/repository"
    job.pop("manifest_digest")

    from ci.execution_security import canonical_digest, validate_job

    validate_job(job, require_digest=False)
    job["manifest_digest"] = canonical_digest(job)
    validate_job(job)


def approval_gate():
    configuration = {
        "synthetic": {"adapter": "example", "profile": "dense_vlm"},
        "hf_checkpoint": {
            "repo": "example/model",
            "revision": "c" * 40,
            "expected_model_type": "example",
            "weight": {"bytes": 1024},
        },
        "scenarios": ["vlm_animal"],
    }
    return {
        "id": f"new_model_path:example:{HEAD_SHA}",
        "type": "maintainer_approval",
        "status": "awaiting_maintainer_approval",
        "component": "new_model_path",
        "model": "example",
        "head_sha": HEAD_SHA,
        "configuration_digest": configuration_digest(configuration),
        "changed_paths": ["mlx_vlm/models/example/model.py"],
        "requested_phases": ["synthetic", "hf_checkpoint"],
        "configuration": configuration,
        "pending_work": pending_work(),
    }


def test_planning_outcome_precedence():
    assert planning_outcome(plan()) is PlanningOutcome.READY
    assert (
        planning_outcome(plan(gates=[approval_gate()]))
        is PlanningOutcome.AWAITING_APPROVAL
    )
    assert (
        planning_outcome(
            plan(gates=[approval_gate()], blocked=[{"reason": "missing_config"}])
        )
        is PlanningOutcome.BLOCKED
    )


def test_control_record_normalizes_blockers():
    record = control_record(
        plan(
            blocked=[
                {
                    "component": "new_model_path",
                    "model": "example",
                    "reason": "model_manifest_not_updated",
                    "changed_paths": ["mlx_vlm/models/example/model.py"],
                }
            ]
        ),
        "example/repository",
        8,
    )

    assert record["outcome"] == "blocked"
    assert record["errors"] == [
        {
            "code": "model_manifest_not_updated",
            "category": "configuration",
            "component": "new_model_path",
            "subject": "example",
            "retryable": False,
            "user_actionable": True,
            "details": {"changed_paths": ["mlx_vlm/models/example/model.py"]},
        }
    ]


def test_control_record_preserves_hosted_checks():
    check = {
        "id": "docs",
        "work_type": "Docs",
        "component": "docs_change",
        "execution_target": "github_hosted",
        "changed_paths": ["README.md"],
    }

    record = control_record(plan(checks=[check]), "example/repository", 8)

    assert record["checks"] == [check]


def test_control_record_rejects_duplicate_hosted_check_ids():
    check = {
        "id": "docs",
        "work_type": "Docs",
        "component": "docs_change",
        "execution_target": "github_hosted",
        "changed_paths": ["README.md"],
    }

    with pytest.raises(ControlError, match="hosted check ids must be unique"):
        control_record(plan(checks=[check, check]), "example/repository", 8)


def test_release_turns_approved_gate_into_jobs():
    contract_sha = "c" * 40
    record = control_record(
        plan(gates=[approval_gate()]),
        "example/repository",
        8,
        contract_sha=contract_sha,
    )

    released = release_control(record, HEAD_SHA, approve_gates=True)

    assert released["kind"] == "approved_job_plan"
    assert released["outcome"] == "ready"
    assert len(released["jobs"]) == 1
    assert released["jobs"][0]["id"] == pending_work()["id"]
    assert released["jobs"][0]["repository"] == "example/repository"
    assert released["jobs"][0]["base_sha"] == BASE_SHA
    assert released["jobs"][0]["head_sha"] == HEAD_SHA
    assert released["jobs"][0]["contract_sha"] == contract_sha
    assert released["jobs"][0]["manifest_digest"].startswith("sha256:")
    assert released["gates"][0]["status"] == "approved"
    assert released["approval"] == {
        "mechanism": "github_environment",
        "head_sha": HEAD_SHA,
        "gate_ids": [f"new_model_path:example:{HEAD_SHA}"],
    }
    assert released["job_plan_digest"].startswith("sha256:")


def test_release_rejects_stale_head():
    record = control_record(plan(gates=[approval_gate()]), "example/repository", 8)

    with pytest.raises(ControlError, match="head changed"):
        release_control(record, "different")


def test_release_requires_explicit_gate_approval():
    record = control_record(plan(gates=[approval_gate()]), "example/repository", 8)

    with pytest.raises(ControlError, match="approval gates"):
        release_control(record, HEAD_SHA)


def test_release_rejects_tampered_configuration():
    gate = approval_gate()
    gate["configuration"]["hf_checkpoint"]["repo"] = "attacker/model"
    record = control_record(plan(gates=[gate]), "example/repository", 8)

    with pytest.raises(ControlError, match="digest does not match"):
        release_control(record, HEAD_SHA, approve_gates=True)


def test_release_rejects_duplicate_job_ids():
    gate = approval_gate()
    duplicate_gate = approval_gate()
    duplicate_gate["id"] = "new_model_path:example:abc123:duplicate"
    record = control_record(plan(gates=[gate, duplicate_gate]), "example/repository", 8)

    with pytest.raises(ControlError, match="job ids must be unique"):
        release_control(record, HEAD_SHA, approve_gates=True)


def test_release_rejects_job_outside_gate_scope():
    gate = approval_gate()
    gate["pending_work"]["model"] = "different"
    record = control_record(plan(gates=[gate]), "example/repository", 8)

    with pytest.raises(ControlError, match="exceeds its scope"):
        release_control(record, HEAD_SHA, approve_gates=True)


def test_status_renderer_is_centralized_and_suppresses_mentions():
    gate = approval_gate()
    gate["model"] = "@reviewer|model"
    record = control_record(
        plan(gates=[gate]), "example/repository", 8, "https://example.com/run"
    )

    rendered = render_status(record)

    assert rendered.startswith("<!-- mlx-vlm:ci:plan -->")
    assert "Awaiting maintainer approval" in rendered
    assert "@\u200breviewer\\|model" in rendered
    assert "No Apple Silicon job starts" in rendered


@pytest.mark.parametrize(
    ("exit_code", "cancelled", "infrastructure_failure", "expected"),
    [
        (0, False, False, ExecutionOutcome.PASSED),
        (1, False, False, ExecutionOutcome.TEST_FAILURE),
        (None, False, False, ExecutionOutcome.INFRASTRUCTURE_FAILURE),
        (1, False, True, ExecutionOutcome.INFRASTRUCTURE_FAILURE),
        (1, True, False, ExecutionOutcome.CANCELLED),
    ],
)
def test_execution_outcomes(exit_code, cancelled, infrastructure_failure, expected):
    assert (
        execution_outcome(
            exit_code,
            cancelled=cancelled,
            infrastructure_failure=infrastructure_failure,
        )
        is expected
    )


def test_release_cli_writes_runner_manifest(tmp_path):
    record = control_record(plan(gates=[approval_gate()]), "example/repository", 8)
    source = tmp_path / "control.json"
    output = tmp_path / "runner-plan.json"
    markdown = tmp_path / "summary.md"
    source.write_text(json.dumps(record))

    assert (
        main(
            [
                "release",
                "--control",
                str(source),
                "--current-head",
                HEAD_SHA,
                "--approve-gates",
                "--output",
                str(output),
                "--markdown",
                str(markdown),
            ]
        )
        == 0
    )
    assert json.loads(output.read_text())["kind"] == "approved_job_plan"
    assert "Ready for runner dispatch" in markdown.read_text()


def test_plan_cli_uses_immutable_repository_head(tmp_path):
    repository_root = Path(__file__).parents[2]
    output = tmp_path / "control.json"
    markdown = tmp_path / "summary.md"

    assert (
        main(
            [
                "plan",
                "--base",
                "HEAD",
                "--head",
                "HEAD",
                "--repository",
                "example/repository",
                "--contract-sha",
                BASE_SHA,
                "--repository-path",
                str(repository_root),
                "--pr",
                "8",
                "--rules-config",
                str(repository_root / "ci/change-rules.yaml"),
                "--protected-config",
                str(repository_root / "ci/protected_paths.yaml"),
                "--output",
                str(output),
                "--markdown",
                str(markdown),
            ]
        )
        == 0
    )
    record = json.loads(output.read_text())
    assert record["outcome"] == "ready"
    assert record["jobs"] == []
    assert record["head_sha"]


def test_plan_cli_accepts_legacy_contributor_config_arguments(tmp_path):
    repository_root = Path(__file__).parents[2]
    output = tmp_path / "control.json"
    markdown = tmp_path / "summary.md"

    assert (
        main(
            [
                "plan",
                "--base",
                "HEAD",
                "--head",
                "HEAD",
                "--repository",
                "example/repository",
                "--contract-sha",
                BASE_SHA,
                "--repository-path",
                str(repository_root),
                "--pr",
                "8",
                "--rules-config",
                str(repository_root / "ci/change-rules.yaml"),
                "--model-config",
                str(repository_root / "ci/model_path.yaml"),
                "--scenario-config",
                str(repository_root / "ci/model-path-scenario.yaml"),
                "--protected-config",
                str(repository_root / "ci/protected_paths.yaml"),
                "--output",
                str(output),
                "--markdown",
                str(markdown),
            ]
        )
        == 0
    )
    assert json.loads(output.read_text())["outcome"] == "ready"


def test_shared_export_preserves_repository_policy_as_opaque_jobs(
    monkeypatch, tmp_path
):
    repository_root = Path(__file__).parents[2]
    planned = plan(jobs=[pending_work()])
    diff = SimpleNamespace(changed_files=(), context=lambda: object())
    delegator = SimpleNamespace(plan_context=lambda context: planned)
    monkeypatch.setattr("ci.control.import_checkout", lambda *args: None)
    monkeypatch.setattr("ci.control.materialize", lambda *args: ())
    monkeypatch.setattr("ci.control.diff_from_git", lambda *args: diff)
    monkeypatch.setattr("ci.control.create_delegator", lambda *args: delegator)
    output = tmp_path / "control.json"
    jobs = tmp_path / "jobs"

    exported = export_repository_plan(
        repository_path=repository_root,
        base_checkout=tmp_path / "base",
        head_checkout=tmp_path / "head",
        base_sha=BASE_SHA,
        head_sha=HEAD_SHA,
        contract_sha="c" * 40,
        repository="Example/project",
        pr_number=8,
        attempt_id="attempt:8",
        run_url="https://example.test/run/8",
        output=output,
        jobs=jobs,
    )

    assert exported["terminal_state"] == "planned"
    assert exported["control"]["kind"] == "approved_job_plan"
    assert exported["device_jobs"][0]["file"] == "000.json"
    manifest = json.loads((jobs / "000.json").read_text())
    assert manifest["repository"] == "Example/project"
    assert manifest["head_sha"] == HEAD_SHA
    assert exported["device_jobs"][0]["manifest"] == manifest
    assert json.loads(output.read_text()) == exported


def test_shared_export_turns_invalid_configuration_into_blocked_plan(
    monkeypatch, tmp_path
):
    repository_root = Path(__file__).parents[2]
    monkeypatch.setattr("ci.control.import_checkout", lambda *args: None)
    monkeypatch.setattr(
        "ci.control.materialize",
        lambda *args: (_ for _ in ()).throw(ValueError("invalid contributor config")),
    )

    exported = export_repository_plan(
        repository_path=repository_root,
        base_checkout=tmp_path / "base",
        head_checkout=tmp_path / "head",
        base_sha=BASE_SHA,
        head_sha=HEAD_SHA,
        contract_sha="c" * 40,
        repository="Example/project",
        pr_number=8,
        attempt_id="attempt:blocked",
        run_url="https://example.test/run/blocked",
        output=tmp_path / "control.json",
        jobs=tmp_path / "jobs",
    )

    assert exported["terminal_state"] == "blocked"
    assert exported["base_sha"] == BASE_SHA
    assert exported["device_jobs"] == []
    assert exported["control"]["errors"][0]["code"] == "invalid_ci_configuration"


def test_shared_export_never_emits_device_work_for_blocked_plan(monkeypatch, tmp_path):
    repository_root = Path(__file__).parents[2]
    planned = plan(jobs=[pending_work()])
    diff = SimpleNamespace(changed_files=("ci/control.py",), context=lambda: object())
    delegator = SimpleNamespace(plan_context=lambda context: planned)
    monkeypatch.setattr("ci.control.import_checkout", lambda *args: None)
    monkeypatch.setattr("ci.control.materialize", lambda *args: ())
    monkeypatch.setattr("ci.control.diff_from_git", lambda *args: diff)
    monkeypatch.setattr("ci.control.create_delegator", lambda *args: delegator)
    monkeypatch.setattr("ci.control._refused_paths", lambda *args: ["ci/control.py"])

    exported = export_repository_plan(
        repository_path=repository_root,
        base_checkout=tmp_path / "base",
        head_checkout=tmp_path / "head",
        base_sha=BASE_SHA,
        head_sha=HEAD_SHA,
        contract_sha="c" * 40,
        repository="Example/project",
        pr_number=8,
        attempt_id="attempt:blocked-work",
        run_url="https://example.test/run/blocked-work",
        output=tmp_path / "control.json",
        jobs=tmp_path / "jobs",
    )

    assert exported["terminal_state"] == "blocked"
    assert exported["device_jobs"] == []
    assert list((tmp_path / "jobs").iterdir()) == []
