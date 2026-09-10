from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

from ci.components.base import ComponentContext, ExecutionContext
from ci.components.docs import REGISTRATION as DOCS
from ci.components.model_path import REGISTRATION as MODEL_PATH

REGISTRATIONS = (DOCS, MODEL_PATH)


def planners(
    config_directory: Path,
    repository: Path,
    contributor_config_directory: Path | None = None,
) -> tuple[Any, ...]:
    context = ComponentContext(
        config_directory,
        repository,
        contributor_config_directory,
    )
    return tuple(
        planner
        for registration in REGISTRATIONS
        for planner in registration.planner_factory(context)
    )


def outputs() -> tuple[Any, ...]:
    return tuple(registration.output_factory() for registration in REGISTRATIONS)


def supported_work() -> frozenset[tuple[str, str]]:
    return frozenset(
        value for registration in REGISTRATIONS for value in registration.work
    )


def supported_phases() -> frozenset[str]:
    return frozenset(
        phase.name for registration in REGISTRATIONS for phase in registration.phases
    )


def supported_job_fields() -> frozenset[str]:
    return frozenset(
        field for registration in REGISTRATIONS for field in registration.job_fields
    )


def phase_commands(context: ExecutionContext) -> dict[str, list[str]]:
    commands: dict[str, list[str]] = {}
    for registration in REGISTRATIONS:
        for phase in registration.phases:
            if phase.name in commands:
                raise ValueError(f"duplicate phase registration: {phase.name}")
            commands[phase.name] = phase.command(context)
    return commands


def contributor_config_paths() -> tuple[str, ...]:
    return tuple(
        sorted(
            {
                path
                for registration in REGISTRATIONS
                for path in registration.contributor_configs
            }
        )
    )


def validate_gate(gate: Mapping[str, Any]) -> None:
    component = str(gate.get("component", ""))
    registrations = [
        registration
        for registration in REGISTRATIONS
        if component in registration.components
        and registration.gate_validator is not None
    ]
    if len(registrations) != 1:
        raise ValueError(f"no unique gate validator for component: {component}")
    validator = registrations[0].gate_validator
    if validator is None:
        raise ValueError(f"component does not support approval gates: {component}")
    validator(gate)
