"""Inspect local model candidates without importing checkpoint code or weights."""

import json
from functools import lru_cache
from pathlib import Path
from typing import Iterable

from jinja2 import Environment, TemplateSyntaxError, nodes

MODEL_PATHS_ENV = "MLX_VLM_MODEL_PATHS"


def read_json_object(path: Path) -> dict:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return value if isinstance(value, dict) else {}


@lru_cache(maxsize=128)
def _declared_reasoning_efforts(template: str) -> list[str] | None:
    """Read explicit effort lists and off branches without rendering templates."""
    try:
        tree = Environment(extensions=["jinja2.ext.loopcontrols"]).parse(template)
    except TemplateSyntaxError:
        return None

    aliases = {"reasoning_effort", "reasoning_strength"}

    def is_effort(expression):
        while isinstance(expression, nodes.Filter) and expression.name in {
            "default",
            "d",
            "lower",
        }:
            expression = expression.node
        return isinstance(expression, nodes.Name) and expression.name in aliases

    allowlists = []
    has_none = False
    for statement in tree.find_all((nodes.Assign, nodes.If, nodes.CondExpr)):
        # Qwen validates a defaulted alias; follow assignments before their use.
        if isinstance(statement, nodes.Assign):
            if isinstance(statement.target, nodes.Name) and is_effort(statement.node):
                aliases.add(statement.target.name)
            continue

        test = statement.test
        # North uses a conditional expression rather than an effort allowlist.
        if not any(
            isinstance(call.node, nodes.Name) and call.node.name == "raise_exception"
            for call in statement.find_all(nodes.Call)
        ):
            for comparison in (test, *test.find_all(nodes.Compare)):
                if (
                    isinstance(comparison, nodes.Compare)
                    and is_effort(comparison.expr)
                    and len(comparison.ops) == 1
                    and comparison.ops[0].op == "eq"
                    and isinstance(comparison.ops[0].expr, nodes.Const)
                    and comparison.ops[0].expr.value == "none"
                ):
                    has_none = True
        if not (
            isinstance(statement, nodes.If)
            and isinstance(test, nodes.Compare)
            and is_effort(test.expr)
            and len(test.ops) == 1
            and test.ops[0].op == "notin"
            and isinstance(test.ops[0].expr, (nodes.List, nodes.Tuple))
        ):
            continue
        # A comparison used for formatting is not a declaration of valid values.
        if not any(
            isinstance(call.node, nodes.Name) and call.node.name == "raise_exception"
            for body in statement.body
            for call in body.find_all(nodes.Call)
        ):
            continue
        items = test.ops[0].expr.items
        if items and all(
            isinstance(item, nodes.Const) and isinstance(item.value, str) and item.value
            for item in items
        ):
            allowlists.append(list(dict.fromkeys(item.value for item in items)))

    # Conflicting conditional allowlists need request context to resolve.
    if allowlists and all(set(values) == set(allowlists[0]) for values in allowlists):
        return allowlists[0]
    return ["none"] if has_none and not allowlists else None


def model_reasoning_efforts(directory: Path | None = None, processor=None):
    """Inspect the active/default template, preferring the loaded processor."""
    template = None
    for owner in (processor, getattr(processor, "tokenizer", None)):
        template = getattr(owner, "chat_template", None)
        if template is not None:
            break
    if template is None and directory is not None:
        try:
            template = (directory / "chat_template.jinja").read_text(encoding="utf-8")
        except FileNotFoundError:
            for filename in ("processor_config.json", "tokenizer_config.json"):
                template = read_json_object(directory / filename).get("chat_template")
                if template is not None:
                    break
        except (OSError, UnicodeError):
            return None
    if isinstance(template, list):
        template = {
            item.get("name"): item.get("template")
            for item in template
            if isinstance(item, dict)
            and item.get("name") == "default"
            and "template" in item
        }
    if isinstance(template, dict):
        template = template.get("default")
    return _declared_reasoning_efforts(template) if isinstance(template, str) else None


def _weight_status(directory: Path) -> bool | None:
    """Return weight validity, or None when no weight files or indexes exist."""

    def present(path):
        return path.is_file() and path.stat().st_size > 0

    shards = set()
    for index in directory.glob("*.safetensors.index.json"):
        weight_map = read_json_object(index).get("weight_map")
        if not isinstance(weight_map, dict) or not weight_map:
            return False
        for filename in weight_map.values():
            if not isinstance(filename, str):
                return False
            shards.add(Path(filename))
    if shards:
        return all(
            not shard.is_absolute()
            and ".." not in shard.parts
            and shard.suffix == ".safetensors"
            and present(directory / shard)
            for shard in shards
        )
    weights = list(directory.glob("*.safetensors"))
    if not weights:
        return None
    return any(
        present(path)
        for path in weights
        if path.name not in {"adapter_model.safetensors", "consolidated.safetensors"}
    )


def is_model_directory(directory: Path) -> bool:
    """Check metadata and weight availability, not architecture compatibility."""
    try:
        pipeline = read_json_object(directory / "model_index.json")
        if isinstance(pipeline.get("_class_name"), str):
            components = [
                valid
                for child in directory.iterdir()
                if child.is_dir() and (valid := _weight_status(child)) is not None
            ]
            return bool(components) and all(components)
        config = read_json_object(directory / "config.json")
        model_type = config.get("model_type") or config.get("speculators_model_type")
        architectures = config.get("architectures")
        has_metadata = (isinstance(model_type, str) and bool(model_type.strip())) or (
            isinstance(architectures, list)
            and bool(architectures)
            and all(isinstance(a, str) and a.strip() for a in architectures)
        )
        return has_metadata and _weight_status(directory) is True
    except (OSError, RuntimeError):
        return False


def discover_models(cache_info, paths: Iterable[str] = ()) -> list[dict]:
    """Discover HF snapshots and custom model folders or their immediate children.

    Prefer main, then the newest usable revision. Deduplicate by resolved path,
    also used as the ID for non-main snapshots and custom models.
    """
    models = {}

    def add(path, *, model_id=None, created=None):
        try:
            path = Path(path).expanduser().resolve()
            if path in models or not is_model_directory(path):
                return False
            models[path] = {
                "id": str(path if model_id is None else model_id),
                "path": path,
                "created": int(path.stat().st_mtime if created is None else created),
            }
            return True
        except (OSError, RuntimeError):
            return False

    for repo in sorted(cache_info.repos, key=lambda r: r.repo_id):
        if repo.repo_type != "model":
            continue
        main = repo.refs.get("main")
        for revision in sorted(
            repo.revisions,
            key=lambda r: (r != main, -r.last_modified, str(r.snapshot_path)),
        ):
            if add(
                revision.snapshot_path,
                model_id=repo.repo_id if revision == main else None,
                created=revision.last_modified,
            ):
                break

    for value in paths:
        try:
            root = Path(value).expanduser()
            # A model directory must not have its components listed separately.
            if (root / "config.json").exists() or (root / "model_index.json").exists():
                candidates = [root]
            else:
                candidates = sorted(p for p in root.iterdir() if p.is_dir())
        except (OSError, RuntimeError):
            continue
        for path in candidates:
            add(path)
    return sorted(models.values(), key=lambda model: model["id"].lower())
