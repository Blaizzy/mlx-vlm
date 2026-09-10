import re
from pathlib import Path

ROOT = Path(__file__).parents[2]
APP_ACTION_SHA = "bcd2ba49218906704ab6c1aa796996da409d3eb1"


def test_benchmark_dispatches_only_authorized_identity():
    workflow = (ROOT / ".github/workflows/bench.yml").read_text()

    assert "issue_comment:" in workflow
    assert "permissions: {}" in workflow
    assert f"uses: actions/create-github-app-token@{APP_ACTION_SHA}" in workflow
    assert "repos/Marvis-Labs/mlx-ci/dispatches" in workflow
    assert "github.event.comment.body == '/ci run'" in workflow
    assert "collaborators/$COMMENTER/permission" in workflow
    assert "client_payload" in workflow
    assert "author_association" not in workflow
    assert "self-hosted" not in workflow
    assert "secrets: inherit" not in workflow
    assert "actions/checkout" not in workflow
    assert "github.event.pull_request.head" not in workflow


def test_pull_request_plan_runs_only_trusted_base_adapter():
    workflow = (ROOT / ".github/workflows/ci-control.yml").read_text()

    assert "pull_request:" in workflow
    assert "permissions: {}" in workflow
    assert "ref: ${{ github.event.pull_request.base.sha }}" in workflow
    assert "python -m ci.repository_adapter plan" in workflow
    assert "python -m ci.repository_adapter hosted-checks" in workflow
    assert "PYTHONPATH=head" not in workflow
    assert "pull_request_target:" not in workflow
    assert "secrets: inherit" not in workflow


def test_repository_keeps_only_repository_owned_interfaces():
    assert (ROOT / "ci/control.py").is_file()
    assert (ROOT / "ci/repository_adapter.py").is_file()
    assert (ROOT / "ci/work_executor.py").is_file()
    assert (ROOT / "ci/report.py").is_file()
    assert (ROOT / "ci/hosted-requirements.txt").is_file()
    for name in (
        "attempt_lease.py",
        "device_inventory.py",
        "device_lease.py",
        "runner_selection.py",
        "scheduler.py",
        "workflow_report.py",
    ):
        assert not (ROOT / "ci" / name).exists()


def test_workflows_pin_actions_and_minimize_default_permissions():
    workflows = ROOT / ".github/workflows"

    for name in ("bench.yml", "ci-control.yml", "tests.yml"):
        path = workflows / name
        source = path.read_text()
        assert "permissions: {}" in source
        assert re.search(r"^\s*pull_request_target:", source, re.MULTILINE) is None
        assert re.search(r"^\s*workflow_run:", source, re.MULTILINE) is None
        for line in source.splitlines():
            if re.search(r"^\s*-?\s*uses:", line):
                assert re.search(r"uses:\s+[^@\s]+@[0-9a-f]{40}(?:\s|$)", line)
