from __future__ import annotations

import hashlib
import json
import os
import re
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Callable, Iterable, Mapping, Optional

from huggingface_hub import scan_cache_dir
from huggingface_hub.errors import CacheNotFound

MODEL_PATHS_ENV = "MLX_VLM_MODEL_PATHS"
MODEL_ALIASES_ENV = "MLX_VLM_MODEL_ALIASES"

_DISCOVERY_DEPTH = 2
_MODEL_CONFIG_FILES = {"config.json", "model_index.json"}
_MODEL_WEIGHT_INDEX = "model.safetensors.index.json"
_HF_REQUIRED_FILES = {"config.json", "tokenizer_config.json"}


class ModelRegistryError(ValueError):
    pass


class InvalidModelRegistryConfiguration(ModelRegistryError):
    pass


class AmbiguousModelIdentifier(ModelRegistryError):
    pass


class LocalModelNotFound(ModelRegistryError):
    pass


@dataclass(frozen=True)
class ModelEntry:
    id: str
    path: Path
    created: int


@dataclass(frozen=True)
class ModelResolution:
    id: str
    load_target: str
    path: Optional[Path]


@dataclass(frozen=True)
class _Catalog:
    entries: tuple[ModelEntry, ...]
    by_path: Mapping[Path, ModelEntry]


def parse_model_paths(value: Optional[str]) -> list[Path]:
    if not value:
        return []

    paths = []
    seen = set()
    for raw_path in value.split(os.pathsep):
        raw_path = raw_path.strip()
        if not raw_path:
            continue
        path = _canonical_path(raw_path)
        if path not in seen:
            seen.add(path)
            paths.append(path)
    return paths


def parse_model_aliases(value: Optional[str]) -> dict[str, Path]:
    if not value:
        return {}
    try:
        decoded = json.loads(value)
    except json.JSONDecodeError as error:
        raise InvalidModelRegistryConfiguration(
            f"{MODEL_ALIASES_ENV} must be a JSON object"
        ) from error
    if not isinstance(decoded, dict):
        raise InvalidModelRegistryConfiguration(
            f"{MODEL_ALIASES_ENV} must be a JSON object"
        )

    aliases = {}
    for model_id, raw_path in decoded.items():
        if not isinstance(model_id, str) or not isinstance(raw_path, str):
            raise InvalidModelRegistryConfiguration(
                f"{MODEL_ALIASES_ENV} keys and values must be strings"
            )
        model_id = _validate_model_id(model_id)
        if not raw_path.strip():
            raise InvalidModelRegistryConfiguration(
                f"Model alias {model_id!r} has an empty path"
            )
        path = _canonical_path(raw_path.strip())
        existing = aliases.get(model_id)
        if existing is not None and existing != path:
            raise InvalidModelRegistryConfiguration(
                f"Model alias {model_id!r} is configured more than once"
            )
        aliases[model_id] = path
    return aliases


def parse_model_alias(value: str) -> tuple[str, Path]:
    model_id, separator, raw_path = value.partition("=")
    if not separator or not raw_path.strip():
        raise InvalidModelRegistryConfiguration("Model aliases must use ID=PATH")
    return _validate_model_id(model_id), _canonical_path(raw_path.strip())


def encode_model_aliases(aliases: Mapping[str, Path | str]) -> str:
    return json.dumps(
        {model_id: str(path) for model_id, path in aliases.items()},
        sort_keys=True,
        separators=(",", ":"),
    )


def private_model_id(path: Path | str) -> str:
    canonical = _canonical_path(path)
    name = re.sub(r"[^A-Za-z0-9._-]+", "-", canonical.name).strip(".-")
    digest = hashlib.sha256(str(canonical).encode()).hexdigest()[:12]
    return f"local/{name or 'model'}-{digest}"


def public_model_id(reference: str) -> str:
    reference = str(reference)
    expanded = Path(reference).expanduser()
    if (
        expanded.is_absolute()
        or expanded.exists()
        or reference.startswith(("~/", "./", "../"))
    ):
        return private_model_id(expanded)
    return reference


class ModelRegistry:
    def __init__(
        self,
        search_paths: Iterable[Path | str] = (),
        aliases: Optional[Mapping[str, Path | str]] = None,
        *,
        include_hf_cache: bool = True,
        cache_scanner: Callable = scan_cache_dir,
    ):
        self.search_paths = _unique_paths(search_paths)
        self.aliases = {}
        for model_id, path in (aliases or {}).items():
            model_id = _validate_model_id(model_id)
            path = _canonical_path(path)
            existing = self.aliases.get(model_id)
            if existing is not None and existing != path:
                raise InvalidModelRegistryConfiguration(
                    f"Model alias {model_id!r} is configured more than once"
                )
            self.aliases[model_id] = path
        self.include_hf_cache = include_hf_cache
        self.cache_scanner = cache_scanner

    @classmethod
    def from_environment(
        cls,
        *,
        include_hf_cache: bool = True,
        cache_scanner: Callable = scan_cache_dir,
    ) -> "ModelRegistry":
        return cls(
            search_paths=parse_model_paths(os.environ.get(MODEL_PATHS_ENV)),
            aliases=parse_model_aliases(os.environ.get(MODEL_ALIASES_ENV)),
            include_hf_cache=include_hf_cache,
            cache_scanner=cache_scanner,
        )

    def entries(self) -> list[ModelEntry]:
        return list(self._catalog().entries)

    def resolve(self, reference: str) -> ModelResolution:
        reference = reference.strip()
        if not reference:
            raise ModelRegistryError("Model identifier cannot be empty")

        local_candidate = Path(reference).expanduser()
        if local_candidate.exists():
            if not local_candidate.is_dir():
                raise LocalModelNotFound(
                    f"Local model path is not a directory: {reference}"
                )
            path = _canonical_path(local_candidate)
            catalog = self._catalog()
            entry = catalog.by_path.get(path)
            return ModelResolution(
                id=entry.id if entry else private_model_id(path),
                load_target=str(path),
                path=path,
            )

        alias_path = self.aliases.get(reference)
        if alias_path is not None:
            self._validate_alias_path(reference, alias_path)
            return ModelResolution(
                id=reference,
                load_target=str(alias_path),
                path=alias_path,
            )

        search_entry = self._resolve_search_path_id(reference)
        if search_entry is not None:
            return ModelResolution(
                id=search_entry.id,
                load_target=str(search_entry.path),
                path=search_entry.path,
            )

        if local_candidate.is_absolute() or reference.startswith(("~/", "./", "../")):
            raise LocalModelNotFound(f"Local model path does not exist: {reference}")

        return ModelResolution(id=reference, load_target=reference, path=None)

    def _resolve_search_path_id(self, reference: str) -> Optional[ModelEntry]:
        if not reference.startswith("local/"):
            return None
        _validate_model_id(reference)
        relative_parts = PurePosixPath(reference).parts[1:]
        if not relative_parts or len(relative_parts) > _DISCOVERY_DEPTH:
            return None

        matches = {}
        for root in self.search_paths:
            candidates = [root / Path(*relative_parts)]
            if len(relative_parts) == 1 and relative_parts[0] == root.name:
                candidates.insert(0, root)
            for candidate in candidates:
                if not is_model_directory(candidate):
                    continue
                path = _canonical_path(candidate)
                if path in self.aliases.values():
                    continue
                matches[path] = ModelEntry(
                    id=reference,
                    path=path,
                    created=_modified_at(path),
                )

        if len(matches) > 1:
            raise AmbiguousModelIdentifier(
                f"Model identifier {reference!r} is ambiguous across configured "
                "model roots; configure an explicit alias"
            )
        entry = next(iter(matches.values()), None)
        if entry is not None and self.include_hf_cache:
            for cached_entry in self._hugging_face_entries():
                if cached_entry.path == entry.path:
                    return cached_entry
        return entry

    @staticmethod
    def _validate_alias_path(model_id: str, path: Path) -> None:
        if not path.is_dir():
            raise InvalidModelRegistryConfiguration(
                f"Model alias {model_id!r} points to a missing directory"
            )
        if not is_model_directory(path):
            raise InvalidModelRegistryConfiguration(
                f"Model alias {model_id!r} does not point to a valid model"
            )

    def _catalog(self) -> _Catalog:
        candidates = []
        if self.include_hf_cache:
            candidates.extend(self._hugging_face_entries())
        candidates.extend(self._search_path_entries())

        alias_by_path = {}
        for model_id, path in self.aliases.items():
            self._validate_alias_path(model_id, path)
            existing_alias = alias_by_path.get(path)
            if existing_alias and existing_alias != model_id:
                raise InvalidModelRegistryConfiguration(
                    f"Model aliases {existing_alias!r} and {model_id!r} refer to "
                    "the same model"
                )
            alias_by_path[path] = model_id

        candidates_by_id = {}
        candidates_by_path = {}
        for entry in candidates:
            candidates_by_path.setdefault(entry.path, []).append(entry)
            if entry.path in alias_by_path:
                continue
            existing = candidates_by_id.get(entry.id)
            if existing and existing.path != entry.path:
                raise AmbiguousModelIdentifier(
                    f"Model identifier {entry.id!r} is ambiguous across configured "
                    "model roots; configure an explicit alias"
                )
            candidates_by_id[entry.id] = entry

        for model_id, path in self.aliases.items():
            existing = candidates_by_id.get(model_id)
            if existing and existing.path != path:
                raise AmbiguousModelIdentifier(
                    f"Model alias {model_id!r} conflicts with a discovered model ID"
                )

        canonical_by_path = {}
        for path, path_candidates in candidates_by_path.items():
            alias = alias_by_path.get(path)
            if alias:
                canonical_by_path[path] = ModelEntry(
                    id=alias,
                    path=path,
                    created=_modified_at(path),
                )
            else:
                canonical_by_path[path] = path_candidates[0]

        for path, alias in alias_by_path.items():
            canonical_by_path.setdefault(
                path,
                ModelEntry(
                    id=alias,
                    path=path,
                    created=_modified_at(path),
                ),
            )

        entries = tuple(sorted(canonical_by_path.values(), key=lambda entry: entry.id))
        return _Catalog(entries=entries, by_path=canonical_by_path)

    def _search_path_entries(self) -> list[ModelEntry]:
        entries = []
        seen_paths = set()
        for root in self.search_paths:
            if not root.is_dir():
                continue
            for path, relative_path in _discover_models(root):
                if path in seen_paths:
                    continue
                seen_paths.add(path)
                relative_id = (
                    root.name
                    if relative_path == Path(".")
                    else relative_path.as_posix()
                )
                entries.append(
                    ModelEntry(
                        id=_validate_model_id(f"local/{relative_id}"),
                        path=path,
                        created=_modified_at(path),
                    )
                )
        return entries

    def _hugging_face_entries(self) -> list[ModelEntry]:
        try:
            cache_info = self.cache_scanner()
        except CacheNotFound:
            return []

        entries = []
        for repo in cache_info.repos:
            if not _is_loadable_hugging_face_repo(repo):
                continue
            revision = repo.refs["main"]
            snapshot_path = getattr(revision, "snapshot_path", None)
            if snapshot_path is None:
                continue
            path = _canonical_path(snapshot_path)
            entries.append(
                ModelEntry(
                    id=_validate_model_id(repo.repo_id),
                    path=path,
                    created=int(repo.last_modified),
                )
            )
        return entries


def get_model_registry(
    *, include_hf_cache: bool = True, cache_scanner: Callable = scan_cache_dir
) -> ModelRegistry:
    return ModelRegistry.from_environment(
        include_hf_cache=include_hf_cache,
        cache_scanner=cache_scanner,
    )


def is_model_directory(path: Path | str) -> bool:
    path = _canonical_path(path)
    if not path.is_dir():
        return False
    if not any((path / name).is_file() for name in _MODEL_CONFIG_FILES):
        return False
    return _contains_weights(path)


def _contains_weights(root: Path) -> bool:
    queue = [(root, 0)]
    seen = set()
    while queue:
        current, depth = queue.pop(0)
        try:
            canonical = current.resolve()
        except (OSError, RuntimeError):
            continue
        if canonical in seen:
            continue
        seen.add(canonical)
        try:
            children = list(current.iterdir())
        except OSError:
            continue
        if any(
            child.is_file()
            and (child.name == _MODEL_WEIGHT_INDEX or child.suffix == ".safetensors")
            for child in children
        ):
            return True
        if depth < _DISCOVERY_DEPTH:
            queue.extend(
                (child, depth + 1)
                for child in children
                if child.is_dir() and not child.name.startswith(".")
            )
    return False


def _discover_models(root: Path):
    queue = [(root, Path("."), 0)]
    seen = set()
    while queue:
        visible_path, relative_path, depth = queue.pop(0)
        try:
            canonical = visible_path.resolve()
        except (OSError, RuntimeError):
            continue
        if canonical in seen:
            continue
        seen.add(canonical)
        if is_model_directory(canonical):
            yield canonical, relative_path
            continue
        if depth >= _DISCOVERY_DEPTH:
            continue
        try:
            children = sorted(visible_path.iterdir(), key=lambda path: path.name)
        except OSError:
            continue
        queue.extend(
            (child, relative_path / child.name, depth + 1)
            for child in children
            if child.is_dir() and not child.name.startswith(".")
        )


def _is_loadable_hugging_face_repo(repo) -> bool:
    if repo.repo_type != "model" or "main" not in repo.refs:
        return False
    file_names = {file.file_path.name for file in repo.refs["main"].files}
    has_weights = _MODEL_WEIGHT_INDEX in file_names or any(
        name.endswith(".safetensors") for name in file_names
    )
    return _HF_REQUIRED_FILES.issubset(file_names) and has_weights


def _validate_model_id(value: str) -> str:
    model_id = value.strip()
    if not model_id:
        raise InvalidModelRegistryConfiguration("Model identifier cannot be empty")
    path = PurePosixPath(model_id)
    if (
        "\\" in model_id
        or path.is_absolute()
        or path.as_posix() != model_id
        or any(part in ("", ".", "..") for part in path.parts)
    ):
        raise InvalidModelRegistryConfiguration(
            "Model identifier must be a relative, normalized name"
        )
    return model_id


def _unique_paths(paths: Iterable[Path | str]) -> tuple[Path, ...]:
    result = []
    seen = set()
    for value in paths:
        path = _canonical_path(value)
        if path not in seen:
            seen.add(path)
            result.append(path)
    return tuple(result)


def _canonical_path(path: Path | str) -> Path:
    expanded = Path(path).expanduser()
    try:
        return expanded.resolve()
    except (OSError, RuntimeError):
        return expanded.absolute()


def _modified_at(path: Path) -> int:
    try:
        return int(path.stat().st_mtime)
    except OSError:
        return 0
