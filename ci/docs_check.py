from __future__ import annotations

import posixpath
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Mapping, Sequence
from urllib.parse import unquote, urlsplit

import yaml

MAX_DOCUMENT_BYTES = 2_000_000
INLINE_LINK = re.compile(r"!?\[[^\]]*\]\(\s*(<[^>]+>|[^\s)]+)")
REFERENCE_LINK = re.compile(r"^\s*\[[^\]]+\]:\s*(<[^>]+>|\S+)", re.MULTILINE)
INLINE_CODE = re.compile(r"`[^`]*`")


class DocsCheckError(ValueError):
    pass


@dataclass(frozen=True, order=True)
class Diagnostic:
    path: str
    code: str
    detail: str

    def render(self) -> str:
        return f"{self.path}: {self.code}: {self.detail}"


@dataclass(frozen=True)
class RepositorySnapshot:
    files: frozenset[str]
    documents: Mapping[str, str]


def compare_docs(
    repository: Path,
    base: str,
    head: str,
    changed_paths: Sequence[str],
) -> dict[str, Any]:
    """Compare documentation diagnostics between immutable Git trees."""

    base_snapshot = snapshot_from_git(repository, base)
    head_snapshot = snapshot_from_git(repository, head)
    base_diagnostics = audit_snapshot(base_snapshot)
    head_diagnostics = audit_snapshot(head_snapshot)
    new_diagnostics = sorted(head_diagnostics - base_diagnostics)
    return {
        "component": "docs_change",
        "check_id": "docs",
        "outcome": "test_failure" if new_diagnostics else "passed",
        "changed_paths": sorted(set(changed_paths)),
        "findings": {
            "base_diagnostic_count": len(base_diagnostics),
            "head_diagnostic_count": len(head_diagnostics),
            "new_errors": [item.render() for item in new_diagnostics],
        },
    }


def snapshot_from_git(repository: Path, ref: str) -> RepositorySnapshot:
    """Read documentation as inert blobs from an immutable Git commit."""

    commit = _git(repository, "rev-parse", "--verify", f"{ref}^{{commit}}", text=True)
    names = _git(
        repository,
        "ls-tree",
        "-r",
        "--name-only",
        "-z",
        commit.strip(),
        binary=True,
    )
    files = frozenset(
        value.decode("utf-8", "surrogateescape")
        for value in names.rstrip(b"\0").split(b"\0")
        if value
    )
    documents: dict[str, str] = {}
    for path in sorted(item for item in files if _is_document(item)):
        content = _git(repository, "show", f"{commit.strip()}:{path}", binary=True)
        if len(content) > MAX_DOCUMENT_BYTES:
            raise DocsCheckError(f"{path} exceeds the documentation size limit")
        try:
            documents[path] = content.decode("utf-8")
        except UnicodeDecodeError as error:
            raise DocsCheckError(f"{path} is not valid UTF-8") from error
    return RepositorySnapshot(files=files, documents=documents)


def audit_snapshot(snapshot: RepositorySnapshot) -> frozenset[Diagnostic]:
    """Return deterministic local-link and MkDocs navigation diagnostics."""

    diagnostics: set[Diagnostic] = set()
    for path, content in snapshot.documents.items():
        if path.lower().endswith(".md"):
            diagnostics.update(_link_diagnostics(path, content, snapshot.files))
    diagnostics.update(_mkdocs_diagnostics(snapshot))
    return frozenset(diagnostics)


def _link_diagnostics(
    path: str, content: str, files: frozenset[str]
) -> set[Diagnostic]:
    diagnostics: set[Diagnostic] = set()
    searchable = _without_code_blocks(content)
    targets = [match.group(1) for match in INLINE_LINK.finditer(searchable)]
    targets.extend(match.group(1) for match in REFERENCE_LINK.finditer(searchable))
    for raw_target in targets:
        target = raw_target.strip("<>")
        parsed = urlsplit(target)
        if (
            not parsed.path
            or parsed.scheme
            or parsed.netloc
            or target.startswith(("#", "//", "/"))
        ):
            continue
        decoded = unquote(parsed.path)
        resolved = posixpath.normpath(posixpath.join(posixpath.dirname(path), decoded))
        if resolved == ".." or resolved.startswith("../"):
            diagnostics.add(Diagnostic(path, "link_outside_repository", target))
            continue
        if not _target_exists(resolved, files):
            diagnostics.add(Diagnostic(path, "missing_local_target", target))
    return diagnostics


def _mkdocs_diagnostics(snapshot: RepositorySnapshot) -> set[Diagnostic]:
    content = snapshot.documents.get("mkdocs.yml")
    if content is None:
        return set()
    try:
        config = yaml.safe_load(content)
    except yaml.YAMLError as error:
        return {Diagnostic("mkdocs.yml", "invalid_yaml", str(error).splitlines()[0])}
    if not isinstance(config, Mapping):
        return {Diagnostic("mkdocs.yml", "invalid_configuration", "expected mapping")}
    docs_dir = config.get("docs_dir", "docs")
    if not isinstance(docs_dir, str) or not docs_dir:
        return {Diagnostic("mkdocs.yml", "invalid_docs_dir", str(docs_dir))}
    diagnostics: set[Diagnostic] = set()
    for target in _nav_targets(config.get("nav", [])):
        if urlsplit(target).scheme:
            continue
        candidate = PurePosixPath(docs_dir, target).as_posix()
        if candidate not in snapshot.files:
            diagnostics.add(Diagnostic("mkdocs.yml", "missing_nav_target", target))
    return diagnostics


def _nav_targets(value: Any) -> list[str]:
    if isinstance(value, str):
        return [value]
    if isinstance(value, list):
        return [target for item in value for target in _nav_targets(item)]
    if isinstance(value, Mapping):
        return [target for item in value.values() for target in _nav_targets(item)]
    return []


def _target_exists(path: str, files: frozenset[str]) -> bool:
    candidates = {path, f"{path}.md", f"{path}/README.md", f"{path}/index.md"}
    if path.endswith(".html"):
        candidates.add(f"{path[:-5]}.md")
    return any(candidate in files for candidate in candidates) or any(
        item.startswith(f"{path.rstrip('/')}/") for item in files
    )


def _without_code_blocks(content: str) -> str:
    retained: list[str] = []
    fence: str | None = None
    for line in content.splitlines():
        stripped = line.lstrip()
        marker = (
            "```"
            if stripped.startswith("```")
            else "~~~" if stripped.startswith("~~~") else None
        )
        if marker:
            fence = None if fence == marker else marker if fence is None else fence
            continue
        if fence is None:
            retained.append(INLINE_CODE.sub("", line))
    return "\n".join(retained)


def _is_document(path: str) -> bool:
    return path == "mkdocs.yml" or path.lower().endswith(".md")


def _git(
    repository: Path,
    *arguments: str,
    text: bool = False,
    binary: bool = False,
) -> str | bytes:
    result = subprocess.run(
        ["git", *arguments],
        cwd=repository,
        check=True,
        capture_output=True,
        text=text,
    )
    if binary:
        return result.stdout
    return result.stdout if text else result.stdout.decode()
