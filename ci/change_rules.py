from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from string import Formatter
from typing import Any, Iterable, Mapping

import yaml


class _UniqueKeyLoader(yaml.SafeLoader):
    pass


def _unique_mapping(loader, node, deep=False):
    loader.flatten_mapping(node)
    mapping = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node, deep=deep)
        try:
            duplicate = key in mapping
        except TypeError as error:
            raise ValueError("YAML mapping keys must be scalar") from error
        if duplicate:
            raise ValueError(f"duplicate YAML key: {key}")
        mapping[key] = loader.construct_object(value_node, deep=deep)
    return mapping


_UniqueKeyLoader.add_constructor(
    yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG,
    _unique_mapping,
)


def load_yaml_mapping(path: Path, max_bytes: int = 2_000_000) -> dict[str, Any]:
    if path.stat().st_size > max_bytes:
        raise ValueError(f"{path} exceeds the {max_bytes}-byte configuration limit")
    data = yaml.load(path.read_text(), Loader=_UniqueKeyLoader)
    if not isinstance(data, dict):
        raise ValueError(f"{path} must contain a YAML mapping")
    return data


def normalize_path(path: str) -> str | None:
    normalized = PurePosixPath(path.replace("\\", "/"))
    if normalized.is_absolute() or ".." in normalized.parts:
        return None
    return normalized.as_posix()


class PathPattern:
    """Match repository paths with wildcards and named segment captures."""

    def __init__(self, pattern: str):
        normalized = normalize_path(pattern)
        if not normalized or normalized == ".":
            raise ValueError(f"invalid path pattern: {pattern!r}")
        self.pattern = normalized
        self.regex, self.capture_names = self._compile(normalized)

    @staticmethod
    def _compile(pattern: str) -> tuple[re.Pattern[str], frozenset[str]]:
        expression: list[str] = ["^"]
        captures: set[str] = set()
        index = 0
        while index < len(pattern):
            if pattern.startswith("**", index):
                expression.append(".*")
                index += 2
                continue
            character = pattern[index]
            if character == "*":
                expression.append("[^/]*")
                index += 1
                continue
            if character == "?":
                expression.append("[^/]")
                index += 1
                continue
            if character == "{":
                closing = pattern.find("}", index + 1)
                if closing == -1:
                    raise ValueError(f"unclosed capture in path pattern: {pattern}")
                name = pattern[index + 1 : closing]
                if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", name):
                    raise ValueError(f"invalid capture name in path pattern: {pattern}")
                if name in captures:
                    raise ValueError(f"duplicate capture {name!r} in path pattern")
                captures.add(name)
                expression.append(f"(?P<{name}>[^/]+)")
                index = closing + 1
                continue
            if character == "}":
                raise ValueError(f"unopened capture in path pattern: {pattern}")
            expression.append(re.escape(character))
            index += 1
        expression.append("$")
        return re.compile("".join(expression)), frozenset(captures)

    def match(self, path: str) -> dict[str, str] | None:
        normalized = normalize_path(path)
        if normalized is None:
            return None
        match = self.regex.fullmatch(normalized)
        return match.groupdict() if match else None


@dataclass(frozen=True)
class ChangeContext:
    changed_files: tuple[str, ...]
    base_files: frozenset[str] = frozenset()
    head_files: frozenset[str] = frozenset()
    base_sha: str | None = None
    head_sha: str | None = None
    target_sha: str | None = None
    tree_state_known: bool = False

    @classmethod
    def create(
        cls,
        changed_files: Iterable[str],
        base_files: Iterable[str] = (),
        head_files: Iterable[str] = (),
        head_sha: str | None = None,
        base_sha: str | None = None,
        target_sha: str | None = None,
        tree_state_known: bool = False,
    ) -> ChangeContext:
        return cls(
            changed_files=tuple(sorted(set(_valid_paths(changed_files)))),
            base_files=frozenset(_tree_paths(base_files)),
            head_files=frozenset(_tree_paths(head_files)),
            head_sha=head_sha,
            base_sha=base_sha,
            target_sha=target_sha,
            tree_state_known=tree_state_known,
        )

    def base_contains(self, path: str) -> bool:
        return path in self.base_files

    def head_contains(self, path: str) -> bool:
        return path in self.head_files


@dataclass(frozen=True)
class ChangeMatch:
    rule: str
    component: str
    path: str
    captures: Mapping[str, str]


@dataclass(frozen=True)
class ChangeRule:
    name: str
    component: str
    include: tuple[PathPattern, ...]
    exclude: tuple[PathPattern, ...]
    base_path_absent: str | None
    base_path_present: str | None
    head_path_absent: str | None
    head_path_present: str | None
    supersedes: tuple[str, ...]

    @property
    def capture_names(self) -> frozenset[str]:
        return self.include[0].capture_names

    def match(self, path: str, context: ChangeContext) -> ChangeMatch | None:
        if any(pattern.match(path) is not None for pattern in self.exclude):
            return None
        captures = next(
            (
                result
                for pattern in self.include
                if (result := pattern.match(path)) is not None
            ),
            None,
        )
        if captures is None or not self._conditions_match(captures, context):
            return None
        return ChangeMatch(self.name, self.component, path, captures)

    def _conditions_match(
        self, captures: Mapping[str, str], context: ChangeContext
    ) -> bool:
        conditions = (
            (self.base_path_absent, context.base_contains, False),
            (self.base_path_present, context.base_contains, True),
            (self.head_path_absent, context.head_contains, False),
            (self.head_path_present, context.head_contains, True),
        )
        for template, lookup, expected in conditions:
            if template is None:
                continue
            if not context.tree_state_known:
                return False
            path = template.format_map(captures)
            if lookup(path) is not expected:
                return False
        return True


class ChangeDetector:
    """Apply declarative change rules to a normalized PR change set."""

    def __init__(self, rules: Iterable[ChangeRule]):
        self.rules = tuple(rules)

    @classmethod
    def from_yaml(cls, path: Path) -> ChangeDetector:
        data = load_yaml_mapping(path)
        if data.get("schema_version") != 1:
            raise ValueError(f"{path}: unsupported schema_version")
        configured = data.get("rules")
        if not isinstance(configured, dict) or not configured:
            raise ValueError(f"{path}: rules must be a non-empty mapping")
        rules = tuple(
            _parse_rule(name, value, path) for name, value in configured.items()
        )
        _validate_supersedes(rules, path)
        return cls(rules)

    def detect(self, context: ChangeContext) -> tuple[ChangeMatch, ...]:
        matches = [
            match
            for path in context.changed_files
            for rule in self.rules
            if (match := rule.match(path, context)) is not None
        ]
        return tuple(self._without_superseded(matches))

    def _without_superseded(self, matches: list[ChangeMatch]) -> list[ChangeMatch]:
        rules = {rule.name: rule for rule in self.rules}
        suppressed: set[tuple[str, tuple[tuple[str, str], ...]]] = set()
        for match in matches:
            rule = rules[match.rule]
            for target_name in rule.supersedes:
                target = rules[target_name]
                shared = sorted(rule.capture_names & target.capture_names)
                scope = tuple((name, match.captures[name]) for name in shared)
                suppressed.add((target_name, scope))

        retained: list[ChangeMatch] = []
        for match in matches:
            if any(
                target_name == match.rule
                and all(match.captures.get(name) == value for name, value in scope)
                for target_name, scope in suppressed
            ):
                continue
            retained.append(match)
        return retained


def _parse_rule(name: Any, value: Any, source: Path) -> ChangeRule:
    if not isinstance(name, str) or not name:
        raise ValueError(f"{source}: rule names must be non-empty strings")
    if not isinstance(value, dict):
        raise ValueError(f"{source}: rule {name} must be a mapping")
    allowed = {
        "component",
        "include",
        "exclude",
        "base_path_absent",
        "base_path_present",
        "head_path_absent",
        "head_path_present",
        "supersedes",
    }
    unknown = set(value) - allowed
    if unknown:
        raise ValueError(f"{source}: rule {name} has unknown fields: {sorted(unknown)}")
    component = value.get("component")
    if not isinstance(component, str) or not component:
        raise ValueError(f"{source}: rule {name} requires a component")
    include = _patterns(value.get("include"), source, name, "include", required=True)
    exclude = _patterns(value.get("exclude", []), source, name, "exclude")
    captures = include[0].capture_names
    if any(pattern.capture_names != captures for pattern in include):
        raise ValueError(f"{source}: rule {name} include captures must match")
    conditions = {
        field: _condition_template(value.get(field), captures, source, name, field)
        for field in (
            "base_path_absent",
            "base_path_present",
            "head_path_absent",
            "head_path_present",
        )
    }
    if conditions["base_path_absent"] and conditions["base_path_present"]:
        raise ValueError(f"{source}: rule {name} has conflicting base conditions")
    if conditions["head_path_absent"] and conditions["head_path_present"]:
        raise ValueError(f"{source}: rule {name} has conflicting head conditions")
    supersedes = value.get("supersedes", [])
    if not isinstance(supersedes, list) or any(
        not isinstance(item, str) or not item for item in supersedes
    ):
        raise ValueError(f"{source}: rule {name} supersedes must be a string list")
    return ChangeRule(
        name=name,
        component=component,
        include=include,
        exclude=exclude,
        supersedes=tuple(dict.fromkeys(supersedes)),
        **conditions,
    )


def _patterns(
    value: Any,
    source: Path,
    rule: str,
    field: str,
    required: bool = False,
) -> tuple[PathPattern, ...]:
    if not isinstance(value, list) or any(not isinstance(item, str) for item in value):
        raise ValueError(f"{source}: rule {rule} {field} must be a string list")
    if required and not value:
        raise ValueError(f"{source}: rule {rule} {field} cannot be empty")
    return tuple(PathPattern(item) for item in value)


def _condition_template(
    value: Any,
    captures: frozenset[str],
    source: Path,
    rule: str,
    field: str,
) -> str | None:
    if value is None:
        return None
    normalized = normalize_path(value) if isinstance(value, str) else None
    if not normalized or normalized == ".":
        raise ValueError(f"{source}: rule {rule} {field} must be a relative path")
    referenced = {
        name for _, name, _, _ in Formatter().parse(normalized) if name is not None
    }
    if not referenced <= captures:
        raise ValueError(f"{source}: rule {rule} {field} has unknown captures")
    return normalized


def _validate_supersedes(rules: tuple[ChangeRule, ...], source: Path) -> None:
    by_name = {rule.name: rule for rule in rules}
    for rule in rules:
        for target_name in rule.supersedes:
            if target_name == rule.name:
                raise ValueError(f"{source}: rule {rule.name} cannot supersede itself")
            target = by_name.get(target_name)
            if target is None:
                raise ValueError(f"{source}: rule {rule.name} supersedes unknown rule")
            if not rule.capture_names & target.capture_names:
                raise ValueError(
                    f"{source}: rule {rule.name} and {target_name} need a shared capture"
                )


def _valid_paths(paths: Iterable[str]) -> Iterable[str]:
    for path in paths:
        normalized = normalize_path(path)
        if normalized and normalized != ".":
            yield normalized


def _tree_paths(files: Iterable[str]) -> Iterable[str]:
    for file in _valid_paths(files):
        path = PurePosixPath(file)
        yield path.as_posix()
        for parent in path.parents:
            if parent.as_posix() != ".":
                yield parent.as_posix()
