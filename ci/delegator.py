from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Protocol, Sequence

from ci.change_rules import (
    ChangeContext,
    ChangeDetector,
    ChangeMatch,
    load_yaml_mapping,
)


class ChangeComponent(Protocol):
    name: str

    def plan(
        self, matches: Sequence[ChangeMatch], context: ChangeContext
    ) -> dict[str, Any]: ...


class ModelPath:
    """Build CI jobs for configured existing-model changes."""

    name = "model_path"

    def __init__(
        self,
        model_config: Path,
        scenario_config: Path,
        supported_synthetic_adapters: frozenset[str] | None = None,
    ):
        self.model_data = self._load_yaml(model_config)
        self.scenario_data = self._load_yaml(scenario_config)
        self.models = self._mapping(self.model_data, "models", model_config)
        self.profiles = self._mapping(
            self.model_data, "synthetic_profiles", model_config
        )
        self.scenarios = self._mapping(self.scenario_data, "scenarios", scenario_config)
        self.defaults_by_capability = self._mapping(
            self.scenario_data, "defaults_by_capability", scenario_config
        )
        self.supported_synthetic_adapters = supported_synthetic_adapters
        self._require_schema_version(self.model_data, model_config)
        self._require_schema_version(self.scenario_data, scenario_config)

    def plan(
        self, matches: Sequence[ChangeMatch], context: ChangeContext
    ) -> dict[str, Any]:
        changed_models, invalid = self._changed_models(matches)
        jobs: list[dict[str, Any]] = []
        blocked = list(invalid)

        if {"ci/model_path.yaml", "ci/model-path-scenario.yaml"} & set(
            context.changed_files
        ):
            blocked.extend(
                self._blocker(
                    model_name,
                    paths,
                    None,
                    "existing_model_ci_configuration_changed",
                )
                for model_name, paths in sorted(changed_models.items())
            )
            return {
                "component": self.name,
                "jobs": [],
                "gates": [],
                "blocked": blocked,
            }

        for model_name, paths in sorted(changed_models.items()):
            model = self.models.get(model_name)
            if not isinstance(model, dict):
                blocked.append(
                    self._blocker(model_name, paths, None, "missing_model_config")
                )
                continue

            configuration, configuration_blockers = self.configuration(
                model_name, paths
            )
            blocked.extend(configuration_blockers)
            if configuration is not None:
                jobs.append(self.work_item(model_name, paths, configuration))

        return {
            "component": self.name,
            "jobs": jobs,
            "gates": [],
            "blocked": blocked,
        }

    def work_item(
        self,
        model_name: str,
        paths: list[str],
        configuration: Mapping[str, Any],
        *,
        component: str | None = None,
    ) -> dict[str, Any]:
        from ci.components.model_path import resource_requirements

        checkpoint = dict(configuration["hf_checkpoint"])
        required_memory, required_disk = resource_requirements(checkpoint)
        return {
            "id": f"model_path:{model_name}",
            "work_type": "ModelPath",
            "component": component or self.name,
            "subject": model_name,
            "model": model_name,
            "changed_paths": paths,
            "phases": ["synthetic", "hf_checkpoint"],
            "scenarios": list(configuration["scenarios"]),
            "synthetic": dict(configuration["synthetic"]),
            "hf_checkpoint": checkpoint,
            "required_memory_gib": required_memory,
            "required_disk_gib": required_disk,
        }

    def configuration(
        self, model_name: str, paths: list[str]
    ) -> tuple[dict[str, Any] | None, list[dict[str, Any]]]:
        model = self.models.get(model_name)
        if not isinstance(model, dict):
            return None, [
                self._blocker(model_name, paths, None, "missing_model_config")
            ]

        scenario_ids, scenario_error = self._scenario_ids(model)
        synthetic = model.get("synthetic")
        checkpoint = model.get("hf_checkpoint")
        errors = (
            ("synthetic", self._synthetic_error(synthetic) or scenario_error),
            (
                "hf_checkpoint",
                self._checkpoint_error(checkpoint) or scenario_error,
            ),
        )
        blockers = [
            self._blocker(model_name, paths, mode, reason)
            for mode, reason in errors
            if reason
        ]
        if blockers:
            return None, blockers
        return {
            "synthetic": {
                "adapter": synthetic["adapter"],
                "profile": synthetic["profile"],
            },
            "hf_checkpoint": {
                "repo": checkpoint["repo"],
                "revision": checkpoint["revision"],
                "expected_model_type": checkpoint["expected_model_type"],
                "weight": checkpoint["weight"],
            },
            "scenarios": scenario_ids,
        }, []

    @staticmethod
    def _load_yaml(path: Path) -> dict[str, Any]:
        return load_yaml_mapping(path)

    @staticmethod
    def _mapping(data: dict[str, Any], key: str, source: Path) -> dict[str, Any]:
        value = data.get(key)
        if not isinstance(value, dict):
            raise ValueError(f"{source}: {key} must be a mapping")
        return value

    @staticmethod
    def _require_schema_version(data: dict[str, Any], source: Path) -> None:
        if data.get("schema_version") != 1:
            raise ValueError(f"{source}: unsupported schema_version")

    @staticmethod
    def _changed_models(
        matches: Sequence[ChangeMatch],
    ) -> tuple[dict[str, list[str]], list[dict[str, Any]]]:
        paths_by_model: dict[str, set[str]] = defaultdict(set)
        blocked: list[dict[str, Any]] = []
        for match in matches:
            model_name = match.captures.get("model")
            if not model_name:
                blocked.append(
                    {
                        "component": match.component,
                        "rule": match.rule,
                        "changed_paths": [match.path],
                        "reason": "missing_model_capture",
                    }
                )
                continue
            paths_by_model[model_name].add(match.path)
        return (
            {model: sorted(paths) for model, paths in paths_by_model.items()},
            blocked,
        )

    def _scenario_ids(self, model: dict[str, Any]) -> tuple[list[str], str | None]:
        configured = model.get("scenarios")
        if configured is None:
            capabilities = model.get("capabilities", [])
            if not isinstance(capabilities, list):
                return [], "invalid_capabilities"
            configured = [
                self.defaults_by_capability.get(capability)
                for capability in capabilities
            ]
        if not isinstance(configured, list):
            return [], "invalid_scenarios"
        if not configured:
            return [], "missing_scenarios"
        if any(not isinstance(item, str) for item in configured):
            return [], "invalid_scenarios"

        scenario_ids = list(dict.fromkeys(configured))
        if any(item not in self.scenarios for item in scenario_ids):
            return [], "unknown_scenario"
        return scenario_ids, None

    def _synthetic_error(self, synthetic: Any) -> str | None:
        if not isinstance(synthetic, dict) or synthetic.get("status") != "configured":
            return "not_configured"
        if not isinstance(synthetic.get("adapter"), str) or not synthetic["adapter"]:
            return "invalid_synthetic_adapter"
        if (
            self.supported_synthetic_adapters is not None
            and synthetic["adapter"] not in self.supported_synthetic_adapters
        ):
            return "unsupported_synthetic_adapter"
        profile = synthetic.get("profile")
        if not isinstance(profile, str) or profile not in self.profiles:
            return "invalid_synthetic_profile"
        return None

    @staticmethod
    def _checkpoint_error(checkpoint: Any) -> str | None:
        if not isinstance(checkpoint, dict) or checkpoint.get("status") != "configured":
            return "not_configured"
        for key in ("repo", "revision", "expected_model_type"):
            if not isinstance(checkpoint.get(key), str) or not checkpoint[key]:
                return f"invalid_checkpoint_{key}"
        weight = checkpoint.get("weight")
        if not isinstance(weight, dict):
            return "invalid_checkpoint_weight"
        if not isinstance(weight.get("bytes"), int) or weight["bytes"] <= 0:
            return "invalid_checkpoint_weight_bytes"
        from ci.checkpoint_policy import CheckpointPolicyError, validate_checkpoint

        try:
            validate_checkpoint(
                {
                    "repo": checkpoint["repo"],
                    "revision": checkpoint["revision"],
                    "expected_model_type": checkpoint["expected_model_type"],
                    "weight": checkpoint["weight"],
                }
            )
        except CheckpointPolicyError:
            return "unsafe_checkpoint_configuration"
        return None

    @staticmethod
    def _blocker(
        model: str, paths: list[str], mode: str | None, reason: str
    ) -> dict[str, Any]:
        return {
            "component": ModelPath.name,
            "model": model,
            "mode": mode,
            "changed_paths": paths,
            "reason": reason,
        }


class NewModelPath:
    """Require approval before executing contributor-supplied new-model tests."""

    name = "new_model_path"

    def __init__(self, model_path: ModelPath):
        self.model_path = model_path

    def plan(
        self, matches: Sequence[ChangeMatch], context: ChangeContext
    ) -> dict[str, Any]:
        changed_models, invalid = self.model_path._changed_models(matches)
        gates: list[dict[str, Any]] = []
        blocked = list(invalid)

        for model_name, paths in sorted(changed_models.items()):
            if "ci/model_path.yaml" not in context.changed_files:
                blocked.append(
                    {
                        "component": self.name,
                        "model": model_name,
                        "mode": None,
                        "changed_paths": paths,
                        "reason": "model_manifest_not_updated",
                    }
                )
                continue
            configuration, configuration_blockers = self.model_path.configuration(
                model_name, paths
            )
            blocked.extend(
                {**blocker, "component": self.name}
                for blocker in configuration_blockers
            )
            if configuration is None:
                continue
            if not context.head_sha:
                blocked.append(
                    {
                        "component": self.name,
                        "model": model_name,
                        "mode": None,
                        "changed_paths": paths,
                        "reason": "missing_head_sha",
                    }
                )
                continue
            configuration_digest = (
                "sha256:"
                + hashlib.sha256(
                    json.dumps(
                        configuration, sort_keys=True, separators=(",", ":")
                    ).encode()
                ).hexdigest()
            )
            gates.append(
                {
                    "id": f"new_model_path:{model_name}:{context.head_sha}",
                    "type": "maintainer_approval",
                    "status": "awaiting_maintainer_approval",
                    "component": self.name,
                    "model": model_name,
                    "head_sha": context.head_sha,
                    "configuration_digest": configuration_digest,
                    "changed_paths": paths,
                    "requested_phases": ["synthetic", "hf_checkpoint"],
                    "configuration": configuration,
                    "pending_work": self.model_path.work_item(
                        model_name,
                        paths,
                        configuration,
                    ),
                }
            )

        return {
            "component": self.name,
            "jobs": [],
            "gates": gates,
            "blocked": blocked,
        }


class Delegator:
    """Detect change ownership and merge independent component plans."""

    def __init__(self, detector: ChangeDetector, components: Sequence[ChangeComponent]):
        self.detector = detector
        self.components = tuple(components)
        names = [component.name for component in self.components]
        if len(names) != len(set(names)):
            raise ValueError("component names must be unique")

    def plan(
        self,
        changed_files: Iterable[str],
        *,
        base_files: Iterable[str] = (),
        head_files: Iterable[str] = (),
        head_sha: str | None = None,
        base_sha: str | None = None,
        target_sha: str | None = None,
        tree_state_known: bool = False,
    ) -> dict[str, Any]:
        context = ChangeContext.create(
            changed_files,
            base_files,
            head_files,
            head_sha,
            base_sha,
            target_sha,
            tree_state_known,
        )
        return self.plan_context(context)

    def plan_context(self, context: ChangeContext) -> dict[str, Any]:
        matches = self.detector.detect(context)
        matches_by_component: dict[str, list[ChangeMatch]] = defaultdict(list)
        for match in matches:
            matches_by_component[match.component].append(match)

        plans: list[dict[str, Any]] = []
        for component in self.components:
            component_matches = matches_by_component.pop(component.name, [])
            if component_matches:
                plans.append(component.plan(component_matches, context))

        unregistered = [
            {
                "component": component,
                "rule": component_matches[0].rule,
                "changed_paths": sorted(match.path for match in component_matches),
                "reason": "unregistered_component",
            }
            for component, component_matches in sorted(matches_by_component.items())
        ]
        return {
            "schema_version": 1,
            "base_sha": context.base_sha,
            "target_sha": context.target_sha,
            "head_sha": context.head_sha,
            "rules": list(dict.fromkeys(match.rule for match in matches)),
            "components": [plan["component"] for plan in plans],
            "jobs": [job for plan in plans for job in plan["jobs"]],
            "gates": [gate for plan in plans for gate in plan["gates"]],
            "checks": [check for plan in plans for check in plan.get("checks", [])],
            "blocked": [item for plan in plans for item in plan["blocked"]]
            + unregistered,
        }


@dataclass(frozen=True)
class GitDiff:
    base_sha: str
    target_sha: str
    head_sha: str
    changed_files: tuple[str, ...]
    base_files: tuple[str, ...]
    head_files: tuple[str, ...]

    def context(self) -> ChangeContext:
        return ChangeContext.create(
            self.changed_files,
            self.base_files,
            self.head_files,
            head_sha=self.head_sha,
            base_sha=self.base_sha,
            target_sha=self.target_sha,
            tree_state_known=True,
        )


def _resolve_commit(ref: str, cwd: Path | None) -> str:
    result = subprocess.run(
        ["git", "rev-parse", "--verify", "--end-of-options", f"{ref}^{{commit}}"],
        cwd=cwd,
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def _merge_base(base: str, head: str, cwd: Path | None) -> str:
    result = subprocess.run(
        ["git", "merge-base", base, head],
        cwd=cwd,
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def _parse_name_status(output: bytes) -> tuple[str, ...]:
    fields = output.rstrip(b"\0").split(b"\0") if output else []
    changed: list[str] = []
    index = 0
    while index < len(fields):
        status = fields[index].decode("ascii")
        index += 1
        path_count = 2 if status.startswith(("R", "C")) else 1
        if index + path_count > len(fields):
            raise ValueError("malformed git diff output")
        changed.extend(os.fsdecode(path) for path in fields[index : index + path_count])
        index += path_count
    return tuple(changed)


def _tree_files(commit: str, cwd: Path | None) -> tuple[str, ...]:
    result = subprocess.run(
        ["git", "ls-tree", "-r", "--name-only", "-z", commit, "--"],
        cwd=cwd,
        check=True,
        capture_output=True,
    )
    return tuple(
        os.fsdecode(path) for path in result.stdout.rstrip(b"\0").split(b"\0") if path
    )


def diff_from_git(base: str, head: str, cwd: Path | None = None) -> GitDiff:
    """Load changed paths and immutable base/head tree snapshots."""

    base_commit = _resolve_commit(base, cwd)
    head_commit = _resolve_commit(head, cwd)
    base_tree = _merge_base(base_commit, head_commit, cwd)
    result = subprocess.run(
        [
            "git",
            "diff",
            "--name-status",
            "-z",
            "--find-renames",
            "--diff-filter=ACDMRTUXB",
            f"{base_tree}..{head_commit}",
            "--",
        ],
        cwd=cwd,
        check=True,
        capture_output=True,
    )
    return GitDiff(
        base_sha=base_tree,
        target_sha=base_commit,
        head_sha=head_commit,
        changed_files=_parse_name_status(result.stdout),
        base_files=_tree_files(base_tree, cwd),
        head_files=_tree_files(head_commit, cwd),
    )


def changed_files_from_git(
    base: str, head: str, cwd: Path | None = None
) -> tuple[str, ...]:
    """Return changed paths between the merge base and head."""

    return diff_from_git(base, head, cwd).changed_files


def create_delegator(
    rules_config: Path,
    contributor_config_directory: Path | None = None,
    repository: Path | None = None,
) -> Delegator:
    """Create a delegator from trusted rules and registered component plugins."""

    from ci.components.registry import planners

    config_directory = rules_config.parent
    components = planners(
        config_directory,
        repository or config_directory.parent,
        contributor_config_directory,
    )
    return Delegator(ChangeDetector.from_yaml(rules_config), components)


def default_delegator(config_directory: Path | None = None) -> Delegator:
    """Create the delegator using repository rules and model manifests."""

    config_directory = config_directory or Path(__file__).resolve().parent
    return create_delegator(
        config_directory / "change-rules.yaml",
        repository=config_directory.parent,
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base")
    parser.add_argument("--head")
    parser.add_argument("--changed-file", action="append", default=[])
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)

    if args.changed_file and (args.base or args.head):
        parser.error("use --changed-file or --base/--head, not both")
    if bool(args.base) != bool(args.head):
        parser.error("--base and --head must be provided together")
    if not args.changed_file and not args.base:
        parser.error("provide --changed-file or --base/--head")

    delegator = default_delegator()
    plan = (
        delegator.plan(args.changed_file)
        if args.changed_file
        else delegator.plan_context(diff_from_git(args.base, args.head).context())
    )
    output = json.dumps(plan, indent=2) + "\n"
    if args.output:
        args.output.write_text(output)
    else:
        print(output, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
