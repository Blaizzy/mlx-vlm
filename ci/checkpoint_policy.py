from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Mapping

REPOSITORY_PATTERN = re.compile(
    r"[A-Za-z0-9][A-Za-z0-9._-]*/[A-Za-z0-9][A-Za-z0-9._-]*"
)
REVISION_PATTERN = re.compile(r"[0-9a-f]{40}")
SAFE_SUFFIXES = frozenset(
    {
        ".json",
        ".model",
        ".safetensors",
        ".tiktoken",
        ".txt",
    }
)
INERT_SUFFIXES = frozenset({".gitattributes", ".jpeg", ".jpg", ".md", ".png"})
INERT_NAMES = frozenset({".gitattributes", "LICENSE", "NOTICE"})
UNSAFE_SUFFIXES = frozenset(
    {".bin", ".ckpt", ".dill", ".joblib", ".pickle", ".pkl", ".pt", ".pth", ".py"}
)


class CheckpointPolicyError(ValueError):
    pass


def validate_checkpoint(checkpoint: Mapping[str, Any]) -> None:
    allowed = {"repo", "revision", "expected_model_type", "weight"}
    unexpected = sorted(set(checkpoint) - allowed)
    if unexpected:
        raise CheckpointPolicyError(
            "checkpoint contains unsupported fields: " + ", ".join(unexpected)
        )
    repo = checkpoint.get("repo")
    revision = checkpoint.get("revision")
    if not isinstance(repo, str) or REPOSITORY_PATTERN.fullmatch(repo) is None:
        raise CheckpointPolicyError("checkpoint repo must be an owner/name slug")
    if not isinstance(revision, str) or REVISION_PATTERN.fullmatch(revision) is None:
        raise CheckpointPolicyError("checkpoint revision must be a full commit SHA")
    model_type = checkpoint.get("expected_model_type")
    if not isinstance(model_type, str) or not model_type:
        raise CheckpointPolicyError("checkpoint expected_model_type is required")
    weight = checkpoint.get("weight")
    if not isinstance(weight, Mapping) or not isinstance(weight.get("bytes"), int):
        raise CheckpointPolicyError("checkpoint weight bytes are required")
    if weight["bytes"] <= 0:
        raise CheckpointPolicyError("checkpoint weight bytes must be positive")
    if weight.get("format", "safetensors") != "safetensors":
        raise CheckpointPolicyError("checkpoint weight format must be safetensors")
    if "files" in weight and (
        not isinstance(weight["files"], int) or weight["files"] <= 0
    ):
        raise CheckpointPolicyError("checkpoint weight file count must be positive")


def validate_snapshot(snapshot: Path, checkpoint: Mapping[str, Any]) -> dict[str, Any]:
    validate_checkpoint(checkpoint)
    root = snapshot.resolve(strict=True)
    if snapshot.is_symlink() or not root.is_dir():
        raise CheckpointPolicyError("checkpoint snapshot must be a real directory")
    entries = sorted(root.rglob("*"))
    for path in entries:
        if path.is_symlink():
            raise CheckpointPolicyError(f"checkpoint contains a symlink: {path.name}")
    files = [path for path in entries if path.is_file()]
    if not files or len(files) > 10_000:
        raise CheckpointPolicyError("checkpoint snapshot has an invalid file count")
    total = 0
    safe_tensors = 0
    config_seen = False
    for path in files:
        try:
            relative = path.resolve(strict=True).relative_to(root)
        except ValueError as error:
            raise CheckpointPolicyError("checkpoint file escapes snapshot") from error
        if any(part in {"", ".", ".."} for part in relative.parts):
            raise CheckpointPolicyError("checkpoint contains an invalid path")
        suffix = path.suffix.lower()
        if suffix in UNSAFE_SUFFIXES:
            raise CheckpointPolicyError(f"unsafe checkpoint file: {relative}")
        if (
            path.name not in INERT_NAMES
            and suffix not in SAFE_SUFFIXES | INERT_SUFFIXES
        ):
            raise CheckpointPolicyError(f"unsupported checkpoint file: {relative}")
        size = path.stat().st_size
        if size > int(checkpoint["weight"]["bytes"]) + 512 * 2**20:
            raise CheckpointPolicyError(
                f"checkpoint file exceeds size policy: {relative}"
            )
        total += size
        safe_tensors += int(suffix == ".safetensors")
        if relative.as_posix() == "config.json":
            config_seen = True
            config = json.loads(path.read_text())
            _validate_config(config)
            if config.get("model_type") != checkpoint["expected_model_type"]:
                raise CheckpointPolicyError(
                    "checkpoint model_type does not match manifest"
                )
    limit = int(checkpoint["weight"]["bytes"] * 1.25) + 512 * 2**20
    if total > limit:
        raise CheckpointPolicyError("checkpoint snapshot exceeds declared size")
    if not config_seen or safe_tensors == 0:
        raise CheckpointPolicyError("checkpoint requires config.json and safetensors")
    expected_files = checkpoint["weight"].get("files")
    if isinstance(expected_files, int) and safe_tensors != expected_files:
        raise CheckpointPolicyError("checkpoint safetensor file count does not match")
    return {"files": len(files), "bytes": total, "safetensors": safe_tensors}


def _validate_config(value: Any) -> None:
    if not isinstance(value, Mapping):
        raise CheckpointPolicyError("checkpoint config must be an object")
    stack = [value]
    while stack:
        current = stack.pop()
        for key, nested in current.items():
            if key in {"auto_map", "model_file"}:
                raise CheckpointPolicyError(
                    f"checkpoint config requires remote code: {key}"
                )
            if key == "trust_remote_code" and nested is True:
                raise CheckpointPolicyError("checkpoint config enables remote code")
            if isinstance(nested, Mapping):
                stack.append(nested)
            elif isinstance(nested, list):
                stack.extend(item for item in nested if isinstance(item, Mapping))
