from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from huggingface_hub import snapshot_download

from .weights import SOURCE_REPO_ID, SOURCE_REVISION

Partition = Literal["fl2va", "ref2va"]
Workflow = Literal["t2va", "fl2va", "ref2va"]
MiniMaxH3Partition = Partition
MiniMaxH3Workflow = Workflow

_WORKFLOW_PARTITIONS: dict[Workflow, Partition] = {
    "t2va": "fl2va",
    "fl2va": "fl2va",
    "ref2va": "ref2va",
}
_SHARED_COMPONENTS = (
    "text_encoder",
    "tokenizer",
    "processor",
    "vae",
    "audio_vae",
    "scheduler",
    "audio_scheduler",
)
_ROOT_PATTERNS = (
    "LICENSE",
    "LICENSE.*",
    "README.md",
    "model_index.json",
    "modular_model_index.json",
)


@dataclass(frozen=True, slots=True)
class MiniMaxH3DownloadPlan:
    repo_id: str
    revision: str | None
    workflow: Workflow
    partition: Partition
    components: tuple[str, ...]
    patterns: tuple[str, ...]


def partition_for_workflow(workflow: Workflow) -> Partition:
    try:
        return _WORKFLOW_PARTITIONS[workflow]
    except KeyError as exc:
        raise ValueError(
            f"workflow must be 't2va', 'fl2va', or 'ref2va', got {workflow!r}"
        ) from exc


def _resolve_workflow(
    workflow: Workflow | None,
    partition: Partition | None,
) -> Workflow:
    if workflow is None:
        if partition is None:
            raise ValueError(
                "workflow is required; choose 't2va', 'fl2va', or 'ref2va'"
            )
        workflow = partition
    selected_partition = partition_for_workflow(workflow)
    if partition is not None and partition != selected_partition:
        raise ValueError(
            f"workflow {workflow!r} uses the {selected_partition!r} partition, "
            f"not {partition!r}"
        )
    return workflow


def _resolve_revision(repo_id: str, revision: str | None) -> str | None:
    if revision is None and repo_id == SOURCE_REPO_ID:
        return SOURCE_REVISION
    return revision


def download_plan(
    workflow: Workflow | None = None,
    *,
    partition: Partition | None = None,
    repo_id: str = SOURCE_REPO_ID,
    revision: str | None = None,
) -> MiniMaxH3DownloadPlan:
    """Plan the Diffusers components needed by exactly one H3 workflow.

    ``t2va`` and ``fl2va`` share the ``transformer`` component. ``ref2va``
    uses ``transformer_ref``. Shared components live once at the repository
    root, so the allowlist never enters the duplicated legacy ``FL2VA`` or
    ``Ref2VA`` trees.
    """
    workflow = _resolve_workflow(workflow, partition)
    partition = partition_for_workflow(workflow)
    transformer = "transformer" if partition == "fl2va" else "transformer_ref"
    components = _SHARED_COMPONENTS + (transformer,)
    patterns = _ROOT_PATTERNS + tuple(f"{component}/**" for component in components)
    return MiniMaxH3DownloadPlan(
        repo_id=repo_id,
        revision=_resolve_revision(repo_id, revision),
        workflow=workflow,
        partition=partition,
        components=components,
        patterns=patterns,
    )


def download_model(
    *,
    workflow: Workflow | None = None,
    partition: Partition | None = None,
    repo_id: str = SOURCE_REPO_ID,
    revision: str | None = None,
    local_dir: str | Path | None = None,
    token: str | None = None,
    force_download: bool = False,
    max_workers: int = 16,
) -> Path:
    """Download one Diffusers H3 workflow and its shared components."""
    plan = download_plan(
        workflow,
        partition=partition,
        repo_id=repo_id,
        revision=revision,
    )
    kwargs = {
        "repo_id": plan.repo_id,
        "revision": plan.revision,
        "allow_patterns": list(plan.patterns),
        "token": token or os.environ.get("HF_TOKEN") or None,
        "force_download": force_download,
        "max_workers": max_workers,
    }
    if local_dir is not None:
        target = Path(local_dir).expanduser()
        target.mkdir(parents=True, exist_ok=True)
        kwargs["local_dir"] = str(target)
    path = Path(snapshot_download(**kwargs))
    return validate_official_layout(path, partition=plan.partition)


def resolve_model_path(
    path_or_repo: str | Path,
    *,
    workflow: Workflow | None = None,
    partition: Partition | None = None,
    revision: str | None = None,
    local_dir: str | Path | None = None,
    token: str | None = None,
    force_download: bool = False,
    max_workers: int = 16,
) -> Path:
    """Return a local path, selectively downloading a Hub repo if needed."""
    path = Path(path_or_repo).expanduser()
    if path.exists():
        return path
    return download_model(
        workflow=workflow,
        partition=partition,
        repo_id=str(path_or_repo),
        revision=revision,
        local_dir=local_dir,
        token=token,
        force_download=force_download,
        max_workers=max_workers,
    )


def validate_official_layout(
    model_path: str | Path,
    *,
    partition: Partition,
) -> Path:
    root = Path(model_path).expanduser()
    if not root.is_dir():
        raise FileNotFoundError(f"model directory does not exist: {root}")
    transformer = "transformer" if partition == "fl2va" else "transformer_ref"
    required = (
        f"{transformer}/config.json",
        f"{transformer}/*.safetensors",
        "vae/config.json",
        "vae/*.safetensors",
        "audio_vae/config.json",
        "audio_vae/*.safetensors",
        "text_encoder/config.json",
        "text_encoder/*.safetensors",
        "tokenizer/tokenizer.json",
    )
    missing = [pattern for pattern in required if not list(root.glob(pattern))]
    if missing:
        formatted = "\n".join(f"  - {pattern}" for pattern in missing)
        raise FileNotFoundError(
            f"MiniMax-H3 {partition} snapshot is incomplete:\n{formatted}"
        )
    return root


__all__ = [
    "MiniMaxH3DownloadPlan",
    "MiniMaxH3Partition",
    "MiniMaxH3Workflow",
    "download_model",
    "download_plan",
    "partition_for_workflow",
    "resolve_model_path",
    "validate_official_layout",
]
