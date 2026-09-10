from pathlib import Path

import pytest

from ci.components import registry
from ci.components.base import ExecutionContext


def test_component_registrations_are_unique_and_self_contained():
    names = [registration.name for registration in registry.REGISTRATIONS]

    assert len(names) == len(set(names))
    assert registry.supported_work() == frozenset(
        {
            ("ModelPath", "model_path"),
        }
    )
    assert registry.supported_phases() == frozenset(
        {
            "synthetic",
            "hf_checkpoint",
        }
    )
    assert registry.contributor_config_paths() == (
        "model-path-scenario.yaml",
        "model_path.yaml",
    )


def test_removing_a_registration_removes_its_work_and_phases(monkeypatch):
    retained = tuple(
        registration
        for registration in registry.REGISTRATIONS
        if registration.name != "model_path"
    )
    monkeypatch.setattr(registry, "REGISTRATIONS", retained)

    assert ("ModelPath", "model_path") not in registry.supported_work()
    assert "synthetic" not in registry.supported_phases()
    assert all(
        "model_path" not in output.component_names for output in registry.outputs()
    )
    context = ExecutionContext(
        Path("job.json"),
        Path("control"),
        Path("base"),
        Path("head"),
        Path("image.jpg"),
        16,
    )
    assert "synthetic" not in registry.phase_commands(context)


def test_removing_docs_registration_removes_its_planner_and_output(monkeypatch):
    retained = tuple(
        registration
        for registration in registry.REGISTRATIONS
        if registration.name != "docs_change"
    )
    monkeypatch.setattr(registry, "REGISTRATIONS", retained)

    assert all(
        planner.name != "docs_change"
        for planner in registry.planners(Path("ci"), Path("."))
    )
    assert all(
        "docs_change" not in output.component_names for output in registry.outputs()
    )


def test_registered_phases_build_commands_without_executor_switches():
    context = ExecutionContext(
        Path("job.json"),
        Path("control"),
        Path("base"),
        Path("head"),
        Path("image.jpg"),
        16,
    )

    commands = registry.phase_commands(context)

    assert set(commands) == registry.supported_phases()
    assert commands["synthetic"][1].endswith("ci/model_path_synthetic_compare.py")
    assert "--scenarios" in commands["hf_checkpoint"]
    source = (Path(__file__).parents[1] / "work_executor.py").read_text()
    assert all(phase not in source for phase in commands)


def test_workflow_calls_only_generic_component_entry_points():
    workflow = Path(__file__).parents[2] / ".github" / "workflows" / "bench.yml"
    source = workflow.read_text()

    assert "repos/Marvis-Labs/mlx-ci/dispatches" in source
    assert (Path(__file__).parents[1] / "control.py").is_file()
    assert (Path(__file__).parents[1] / "work_executor.py").is_file()
    assert (Path(__file__).parents[1] / "report.py").is_file()
    assert "model_path_work.py" not in source
    assert "model_path_compare.py" not in source
    assert "kv_cache_contract_compare.py" not in source


def test_gate_validation_is_owned_by_the_registered_component(monkeypatch):
    gate = {
        "component": "new_model_path",
        "model": "new_model",
        "requested_phases": ["synthetic"],
        "pending_work": {
            "work_type": "ModelPath",
            "component": "model_path",
            "model": "new_model",
            "phases": ["synthetic"],
        },
    }
    registry.validate_gate(gate)
    retained = tuple(
        registration
        for registration in registry.REGISTRATIONS
        if registration.name != "model_path"
    )
    monkeypatch.setattr(registry, "REGISTRATIONS", retained)

    with pytest.raises(ValueError, match="no unique gate validator"):
        registry.validate_gate(gate)
