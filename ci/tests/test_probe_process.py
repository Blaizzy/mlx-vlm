import subprocess

from ci.probe_process import run_project_probe


def test_runner_uses_prebuilt_trusted_python_without_uv_sync(monkeypatch, tmp_path):
    observed = {}

    def invoke(command, env):
        observed["command"] = command
        observed["environment"] = env
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr("ci.probe_process.subprocess.run", invoke)
    monkeypatch.setenv("CI_JOB_PYTHON", "/trusted/venv/bin/python")
    project = tmp_path / "head"
    probe = tmp_path / "control" / "ci" / "probe.py"

    run_project_probe(project, probe, ["--output", "result.json"])

    assert observed["command"] == [
        "/trusted/venv/bin/python",
        str(probe),
        "--output",
        "result.json",
    ]
    assert observed["environment"]["PYTHONPATH"] == str(project)


def test_local_fallback_is_frozen_and_offline(monkeypatch, tmp_path):
    observed = {}

    def invoke(command, env):
        observed["command"] = command
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr("ci.probe_process.subprocess.run", invoke)
    monkeypatch.delenv("CI_JOB_PYTHON", raising=False)

    run_project_probe(tmp_path / "head", tmp_path / "probe.py", [])

    assert observed["command"][:4] == ["uv", "run", "--frozen", "--offline"]
