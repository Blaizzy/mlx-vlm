from ci.hosted_checks import resolved_record


def control():
    return {
        "schema_version": 1,
        "kind": "ci_control",
        "head_sha": "abc123",
        "outcome": "ready",
        "components": ["docs_change"],
        "checks": [
            {
                "id": "docs",
                "work_type": "Docs",
                "component": "docs_change",
                "execution_target": "github_hosted",
                "changed_paths": ["README.md"],
            }
        ],
        "jobs": [],
        "gates": [],
        "errors": [],
    }


def test_resolved_docs_only_record_passes():
    record = resolved_record(
        control(),
        [
            {
                "component": "docs_change",
                "check_id": "docs",
                "outcome": "passed",
            }
        ],
    )

    assert record["outcome"] == "ready"
    assert record["hosted_outcome"] == "passed"
    assert record["jobs"] == []


def test_resolved_record_preserves_device_jobs_and_reports_docs_failure():
    value = control()
    value["jobs"] = [{"id": "model_path:qwen2_vl"}]

    record = resolved_record(
        value,
        [
            {
                "component": "docs_change",
                "check_id": "docs",
                "outcome": "test_failure",
            }
        ],
    )

    assert record["outcome"] == "test_failure"
    assert record["hosted_outcome"] == "test_failure"
    assert record["jobs"] == [{"id": "model_path:qwen2_vl"}]
