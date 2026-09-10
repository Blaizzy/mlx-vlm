import subprocess

from ci.component_config import materialize


def test_materialize_reads_only_registered_contributor_configuration(
    monkeypatch, tmp_path
):
    calls = []

    def fake_run(command, **kwargs):
        calls.append((command, kwargs))
        if command[1:3] == ["cat-file", "-s"]:
            return subprocess.CompletedProcess(command, 0, stdout="18\n")
        return subprocess.CompletedProcess(command, 0, stdout=b"schema_version: 1\n")

    monkeypatch.setattr("ci.component_config.subprocess.run", fake_run)

    written = materialize(tmp_path, "head", tmp_path / "output")

    assert [path.name for path in written] == [
        "model-path-scenario.yaml",
        "model_path.yaml",
    ]
    assert [call[0][-1] for call in calls if call[0][1] == "show"] == [
        "head:ci/model-path-scenario.yaml",
        "head:ci/model_path.yaml",
    ]
    assert all(path.read_text() == "schema_version: 1\n" for path in written)


def test_materialize_rejects_oversized_contributor_configuration(monkeypatch, tmp_path):
    def fake_run(command, **kwargs):
        return subprocess.CompletedProcess(command, 0, stdout="1048577\n")

    monkeypatch.setattr("ci.component_config.subprocess.run", fake_run)

    try:
        materialize(tmp_path, "head", tmp_path / "output")
    except ValueError as error:
        assert "too large" in str(error)
    else:
        raise AssertionError("oversized contributor configuration was accepted")
