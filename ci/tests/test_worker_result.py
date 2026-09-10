import pytest

from ci.worker_result import finalize


def job():
    return {
        "id": "model_path:qwen2_vl:hf_checkpoint",
        "component": "model_path",
        "model": "qwen2_vl",
        "subject": "qwen2_vl",
        "mode": "hf_checkpoint",
        "repository": "Example/project",
    }


def runner_result(**updates):
    result = {
        "schema_version": 1,
        "kind": "device_job_result",
        "job_id": job()["id"],
        "component": "model_path",
        "model": "qwen2_vl",
        "subject": "qwen2_vl",
        "repository": "Example/project",
        "device": "runner-one",
        "decision": "accepted",
        "outcome": "passed",
    }
    result.update(updates)
    return result


def test_missing_runner_result_is_infrastructure_failure():
    work = job()
    work["profile"] = "dense"
    result = finalize(work, None)
    assert result["outcome"] == "infrastructure_failure"
    assert result["profile"] == "dense"
    assert result["job_id"] == "model_path:qwen2_vl:hf_checkpoint"


def test_findings_verdict_and_metrics_are_promoted():
    result = finalize(
        job(),
        runner_result(
            findings={
                "verdict": "regressed",
                "metrics": {"ttft_ms": {"base": 10, "head": 20}},
            },
        ),
    )

    assert result["outcome"] == "regressed"
    assert "ttft_ms" in result["metrics"]
    assert result["component"] == "model_path"
    assert result["job_id"] == "model_path:qwen2_vl:hf_checkpoint"


def test_lease_failure_overrides_a_passing_workload():
    result = finalize(
        job(),
        runner_result(findings={"verdict": "passed"}),
        "lease heartbeat failed: gh is unavailable",
    )

    assert result["outcome"] == "infrastructure_failure"
    assert result["findings"] == {"error": "lease heartbeat failed: gh is unavailable"}
    assert result["job_id"] == "model_path:qwen2_vl:hf_checkpoint"


def test_result_identity_cannot_override_planned_work():
    result = finalize(
        job(),
        runner_result(
            component="security",
            model="different",
        ),
    )

    assert result["outcome"] == "infrastructure_failure"
    assert result["component"] == "model_path"
    assert result["model"] == "qwen2_vl"


@pytest.mark.parametrize(
    "updates",
    (
        {"unexpected": "field"},
        {"device": ""},
        {"device": "x" * 129},
    ),
)
def test_unbounded_or_ambiguous_result_is_infrastructure_failure(updates):
    result = finalize(job(), runner_result(**updates))

    assert result["outcome"] == "infrastructure_failure"
    assert result["findings"] == {"error": "runner result failed validation"}


def test_deeply_nested_result_is_infrastructure_failure():
    nested = True
    for _ in range(10):
        nested = {"value": nested}

    result = finalize(job(), runner_result(findings=nested))

    assert result["outcome"] == "infrastructure_failure"
    assert result["findings"] == {"error": "runner result failed validation"}


def test_decline_becomes_no_eligible_runner():
    result = finalize(
        job(),
        runner_result(
            decision="declined",
            outcome="declined",
            reason="declined_memory",
        ),
    )

    assert result["outcome"] == "no_eligible_runner"


def test_typed_runner_failures_preserve_their_outcome():
    regressed = finalize(
        job(),
        runner_result(
            outcome="regressed",
            reason="correctness_regression",
            findings={"verdict": "regressed"},
        ),
    )
    transient = finalize(
        job(),
        runner_result(
            outcome="infrastructure_failure",
            reason="network_transient",
        ),
    )
    disk_full = finalize(
        job(),
        runner_result(
            decision="declined",
            outcome="declined",
            reason="disk_full",
        ),
    )

    assert regressed["outcome"] == "regressed"
    assert transient["outcome"] == "infrastructure_failure"
    assert disk_full["outcome"] == "no_eligible_runner"


def test_checkpoint_failure_contract_rejects_misclassification():
    raw = runner_result(
        outcome="test_failure",
        reason="checkpoint_not_found",
    )
    raw["checkpoint_failure"] = {
        "phase": "hf_checkpoint",
        "operation": "download",
        "code": "network_transient",
        "retryable": True,
        "attempts": 3,
    }

    result = finalize(job(), raw)

    assert result["outcome"] == "infrastructure_failure"
    assert result["findings"] == {"error": "runner result failed validation"}


def test_unknown_runner_outcome_is_infrastructure_failure():
    result = finalize(job(), runner_result(outcome="unexpected"))

    assert result["outcome"] == "infrastructure_failure"
    assert result["findings"] == {"error": "runner result failed validation"}


@pytest.mark.parametrize(
    "updates",
    (
        {"decision": "accepted", "outcome": "declined", "reason": None},
        {
            "decision": "declined",
            "outcome": "passed",
            "reason": "declined_busy",
        },
        {"findings": "untrusted"},
        {"repository": "Different/project"},
        {"outcome": "regressed", "reason": None},
    ),
)
def test_inconsistent_result_is_infrastructure_failure(updates):
    raw = runner_result()
    raw.update(updates)

    result = finalize(job(), raw)

    assert result["outcome"] == "infrastructure_failure"
    assert result["findings"] == {"error": "runner result failed validation"}
