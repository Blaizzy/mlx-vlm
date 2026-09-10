import json

import pytest

from ci.execution_security import (
    ExecutionSecurityError,
    canonical_digest,
    seal_job,
    validate_job,
    verify_execution,
)

SHA = "a" * 40
REPOSITORY = "Example/project"


def job():
    return {
        "id": "model_path:qwen2_vl",
        "work_type": "ModelPath",
        "component": "model_path",
        "subject": "qwen2_vl",
        "model": "qwen2_vl",
        "changed_paths": ["mlx_vlm/models/qwen2_vl/vision.py"],
        "phases": ["synthetic"],
        "required_memory_gib": 8,
        "required_disk_gib": 4,
        "synthetic": {"adapter": "qwen2_vl", "profile": "dense_vlm"},
    }


def test_sealed_job_rejects_manifest_mutation():
    sealed = seal_job(
        job(),
        repository=REPOSITORY,
        base_sha=SHA,
        head_sha="b" * 40,
        contract_sha=SHA,
    )
    validate_job(sealed)
    sealed["required_memory_gib"] = 1

    with pytest.raises(ExecutionSecurityError, match="digest"):
        validate_job(sealed)


def test_job_rejects_unknown_fields_and_phases():
    value = job()
    value["command"] = "curl attacker"

    with pytest.raises(ExecutionSecurityError, match="unregistered fields"):
        seal_job(
            value,
            repository=REPOSITORY,
            base_sha=SHA,
            head_sha="b" * 40,
            contract_sha=SHA,
        )

    value = job()
    value["phases"] = ["attacker_phase"]
    with pytest.raises(ExecutionSecurityError, match="unregistered phases"):
        seal_job(
            value,
            repository=REPOSITORY,
            base_sha=SHA,
            head_sha="b" * 40,
            contract_sha=SHA,
        )


def test_execution_rejects_wrong_sha(monkeypatch, tmp_path):
    control = tmp_path / "control"
    base = tmp_path / "base"
    head = tmp_path / "head"
    for path in (control / "ci", base, head):
        path.mkdir(parents=True)
    probe = control / "ci" / "probe.py"
    probe.write_text("pass\n")
    sealed = seal_job(
        job(),
        repository=REPOSITORY,
        base_sha=SHA,
        head_sha="b" * 40,
        contract_sha=SHA,
    )

    def fake_git(repository, *arguments, raw=False):
        if arguments[0] == "rev-parse":
            return "c" * 40 if repository == head else SHA
        if arguments[0] == "status":
            return ""
        return probe.read_bytes() if raw else ""

    monkeypatch.setattr("ci.execution_security._git", fake_git)

    with pytest.raises(ExecutionSecurityError, match="head checkout"):
        verify_execution(
            sealed,
            control=control,
            base=base,
            head=head,
            commands={"synthetic": ["python", str(probe)]},
        )


def test_execution_rejects_substituted_probe(monkeypatch, tmp_path):
    control = tmp_path / "control"
    base = tmp_path / "base"
    head = tmp_path / "head"
    for path in (control / "ci", base, head):
        path.mkdir(parents=True)
    probe = control / "ci" / "probe.py"
    probe.write_text("malicious\n")
    sealed = seal_job(
        job(),
        repository=REPOSITORY,
        base_sha=SHA,
        head_sha="b" * 40,
        contract_sha=SHA,
    )

    def fake_git(repository, *arguments, raw=False):
        if arguments[0] == "rev-parse":
            return "b" * 40 if repository == head else SHA
        if arguments[0] == "status":
            return ""
        return b"trusted\n" if raw else ""

    monkeypatch.setattr("ci.execution_security._git", fake_git)

    with pytest.raises(ExecutionSecurityError, match="differs from control"):
        verify_execution(
            sealed,
            control=control,
            base=base,
            head=head,
            commands={"synthetic": ["python", str(probe)]},
        )


def test_canonical_digest_is_stable():
    left = {"b": 2, "a": 1}
    right = json.loads(json.dumps(left, sort_keys=True))
    assert canonical_digest(left) == canonical_digest(right)
