from __future__ import annotations

import os
import subprocess
from pathlib import Path
from typing import Mapping, Sequence


def run_project_probe(
    project: Path,
    probe: Path,
    arguments: Sequence[str],
    *,
    python_path: Sequence[Path] = (),
    environment: Mapping[str, str] | None = None,
    allowed_returncodes: frozenset[int] = frozenset({0}),
) -> None:
    job_python = os.environ.get("CI_JOB_PYTHON")
    if job_python:
        command = [job_python, str(probe), *arguments]
    else:
        command = [
            "uv",
            "run",
            "--frozen",
            "--offline",
            "--project",
            str(project),
            "--python",
            "3.10",
            "python",
            str(probe),
            *arguments,
        ]
    child_environment = dict(os.environ)
    paths = [*(str(path) for path in python_path), str(project)]
    child_environment["PYTHONPATH"] = os.pathsep.join(paths)
    if environment:
        child_environment.update(environment)
    completed = subprocess.run(command, env=child_environment)
    if completed.returncode not in allowed_returncodes:
        raise subprocess.CalledProcessError(completed.returncode, command)
