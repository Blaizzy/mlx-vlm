from __future__ import annotations

import math
import sys
from typing import Any, Mapping

from ci.components.base import (
    ComponentContext,
    ComponentRegistration,
    ExecutionContext,
    PhaseRegistration,
)

SYNTHETIC_ADAPTERS = frozenset(
    {
        "bert",
        "deepseek_vl_v2",
        "granite_vision",
        "internvl_chat",
        "qwen2_5_vl",
        "qwen2_vl",
    }
)


def resource_requirements(
    checkpoint: Mapping[str, Any] | None,
) -> tuple[int, int]:
    if checkpoint is None:
        return 8, 2
    weight = checkpoint.get("weight")
    if not isinstance(weight, Mapping):
        raise ValueError("checkpoint has no weight metadata")
    weight_bytes = weight.get("bytes")
    if not isinstance(weight_bytes, int) or weight_bytes <= 0:
        raise ValueError("checkpoint weight bytes must be positive")
    weights_gib = weight_bytes / 2**30
    return (
        max(8, math.ceil(weights_gib * 1.5 + 4)),
        max(4, math.ceil(weights_gib * 1.25 + 2)),
    )


def model_path_service(context: ComponentContext):
    from ci.delegator import ModelPath

    return context.service(
        "model_path",
        lambda: ModelPath(
            context.config("model_path.yaml", contributor=True),
            context.config("model-path-scenario.yaml", contributor=True),
            supported_synthetic_adapters=SYNTHETIC_ADAPTERS,
        ),
    )


def _planners(context: ComponentContext) -> tuple[Any, ...]:
    from ci.delegator import NewModelPath

    model_path = model_path_service(context)
    return NewModelPath(model_path), model_path


def _output():
    from ci.bot import ModelPathOutput

    return ModelPathOutput()


def _synthetic(context: ExecutionContext) -> list[str]:
    directory = context.config_directory
    return [
        sys.executable,
        str(directory / "model_path_synthetic_compare.py"),
        "--job",
        str(context.job_path),
        "--profiles",
        str(directory / "model_path.yaml"),
        "--base",
        str(context.base),
        "--head",
        str(context.head),
        "--probe",
        str(directory / "model_path_synthetic_probe.py"),
    ]


def _hf_checkpoint(context: ExecutionContext) -> list[str]:
    directory = context.config_directory
    return [
        sys.executable,
        str(directory / "model_path_compare.py"),
        "--job",
        str(context.job_path),
        "--scenarios",
        str(directory / "model-path-scenario.yaml"),
        "--base",
        str(context.base),
        "--head",
        str(context.head),
        "--probe",
        str(directory / "model_path_probe.py"),
        "--image",
        str(context.image),
        "--max-tokens",
        str(context.max_tokens),
    ]


def _validate_gate(gate: Mapping[str, Any]) -> None:
    pending_work = gate.get("pending_work")
    requested = gate.get("requested_phases")
    if not isinstance(pending_work, Mapping):
        raise ValueError("approval gate has no pending work")
    if not isinstance(requested, list) or not requested:
        raise ValueError("approval gate has no requested phases")
    if (
        pending_work.get("work_type") != "ModelPath"
        or pending_work.get("component") != "model_path"
        or pending_work.get("model") != gate.get("model")
        or pending_work.get("phases") != requested
    ):
        raise ValueError("approval gate pending work exceeds its scope")


REGISTRATION = ComponentRegistration(
    name="model_path",
    components=frozenset({"model_path", "new_model_path"}),
    planner_factory=_planners,
    output_factory=_output,
    work=frozenset({("ModelPath", "model_path")}),
    phases=(
        PhaseRegistration("synthetic", _synthetic),
        PhaseRegistration("hf_checkpoint", _hf_checkpoint),
    ),
    contributor_configs=frozenset({"model_path.yaml", "model-path-scenario.yaml"}),
    gate_validator=_validate_gate,
    job_fields=frozenset(
        {
            "synthetic",
            "hf_checkpoint",
            "scenarios",
            "unavailable_phases",
        }
    ),
)
