from __future__ import annotations

import hashlib
import json
import re
import subprocess
from pathlib import Path
from typing import Any, Mapping, Sequence

from ci.components.registry import (
    supported_job_fields,
    supported_phases,
    supported_work,
)

COMMIT_PATTERN = re.compile(r"[0-9a-f]{40}")
REPOSITORY_PATTERN = re.compile(
    r"[A-Za-z0-9][A-Za-z0-9._-]*/[A-Za-z0-9][A-Za-z0-9._-]*"
)
COMMON_JOB_FIELDS = frozenset(
    {
        "id",
        "work_type",
        "component",
        "subject",
        "model",
        "profile",
        "changed_paths",
        "origins",
        "phases",
        "required_memory_gib",
        "required_disk_gib",
        "repository",
        "base_sha",
        "head_sha",
        "contract_sha",
        "manifest_digest",
    }
)


class ExecutionSecurityError(ValueError):
    pass


def canonical_digest(value: Mapping[str, Any]) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return "sha256:" + hashlib.sha256(payload).hexdigest()


def seal_job(
    job: Mapping[str, Any],
    *,
    repository: str,
    base_sha: str,
    head_sha: str,
    contract_sha: str,
) -> dict[str, Any]:
    sealed = dict(job)
    sealed.update(
        {
            "repository": repository,
            "base_sha": base_sha,
            "head_sha": head_sha,
            "contract_sha": contract_sha,
        }
    )
    sealed.pop("manifest_digest", None)
    validate_job(sealed, require_digest=False)
    sealed["manifest_digest"] = canonical_digest(sealed)
    return sealed


def validate_job(job: Mapping[str, Any], *, require_digest: bool = True) -> None:
    allowed = COMMON_JOB_FIELDS | supported_job_fields()
    unexpected = sorted(set(job) - allowed)
    if unexpected:
        raise ExecutionSecurityError(
            "work manifest contains unregistered fields: " + ", ".join(unexpected)
        )
    work = (job.get("work_type"), job.get("component"))
    if work not in supported_work():
        raise ExecutionSecurityError(f"unregistered work item: {work!r}")
    for field in ("id", "subject"):
        if not isinstance(job.get(field), str) or not job[field]:
            raise ExecutionSecurityError(f"work manifest requires {field}")
    repository = job.get("repository")
    if (
        not isinstance(repository, str)
        or REPOSITORY_PATTERN.fullmatch(repository) is None
    ):
        raise ExecutionSecurityError("work manifest requires repository owner/name")
    phases = job.get("phases")
    if not isinstance(phases, list) or not phases:
        raise ExecutionSecurityError("work manifest requires phases")
    if len(phases) != len(set(phases)) or any(
        not isinstance(phase, str) or phase not in supported_phases()
        for phase in phases
    ):
        raise ExecutionSecurityError("work manifest contains unregistered phases")
    for field in ("required_memory_gib", "required_disk_gib"):
        if not isinstance(job.get(field), int) or job[field] <= 0:
            raise ExecutionSecurityError(f"work manifest requires positive {field}")
    for field in ("base_sha", "head_sha", "contract_sha"):
        value = job.get(field)
        if not isinstance(value, str) or COMMIT_PATTERN.fullmatch(value) is None:
            raise ExecutionSecurityError(f"work manifest requires immutable {field}")
    repository = job.get("repository")
    if (
        not isinstance(repository, str)
        or REPOSITORY_PATTERN.fullmatch(repository) is None
    ):
        raise ExecutionSecurityError("work manifest requires repository identity")
    if require_digest:
        supplied = job.get("manifest_digest")
        unsigned = dict(job)
        unsigned.pop("manifest_digest", None)
        if supplied != canonical_digest(unsigned):
            raise ExecutionSecurityError("work manifest digest does not match")


def verify_execution(
    job: Mapping[str, Any],
    *,
    control: Path,
    base: Path,
    head: Path,
    commands: Mapping[str, Sequence[str]],
) -> None:
    validate_job(job)
    _require_checkout(control, str(job["contract_sha"]), "control")
    _require_checkout(base, str(job["base_sha"]), "base")
    _require_checkout(head, str(job["head_sha"]), "head")
    for phase in job["phases"]:
        command = commands.get(phase)
        if command is None or len(command) < 2:
            raise ExecutionSecurityError(f"phase has no trusted command: {phase}")
        _require_tracked_control_file(control, Path(command[1]))


def _require_checkout(repository: Path, expected: str, role: str) -> None:
    resolved = repository.resolve(strict=True)
    if repository.is_symlink() or not resolved.is_dir():
        raise ExecutionSecurityError(f"{role} checkout is not a real directory")
    actual = _git(repository, "rev-parse", "--verify", "HEAD^{commit}")
    if actual != expected:
        raise ExecutionSecurityError(
            f"{role} checkout is {actual or 'unknown'}, expected {expected}"
        )
    status = _git(
        repository,
        "status",
        "--porcelain=v1",
        "--untracked-files=all",
    )
    if status:
        raise ExecutionSecurityError(f"{role} checkout is not clean")


def _require_tracked_control_file(control: Path, path: Path) -> None:
    try:
        resolved = path.resolve(strict=True)
        relative = resolved.relative_to(control.resolve(strict=True))
    except (FileNotFoundError, ValueError) as error:
        raise ExecutionSecurityError(
            "phase entry point is outside trusted control"
        ) from error
    if path.is_symlink() or relative.parts[:1] != ("ci",):
        raise ExecutionSecurityError("phase entry point is not trusted CI code")
    expected = _git(control, "show", f"HEAD:{relative.as_posix()}", raw=True)
    if not isinstance(expected, bytes) or resolved.read_bytes() != expected:
        raise ExecutionSecurityError("phase entry point differs from control commit")


def _git(repository: Path, *arguments: str, raw: bool = False) -> str | bytes:
    completed = subprocess.run(
        ["git", *arguments],
        cwd=repository,
        check=True,
        capture_output=True,
        text=not raw,
    )
    return completed.stdout if raw else completed.stdout.strip()
