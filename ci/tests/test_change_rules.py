from pathlib import Path

import pytest
import yaml

from ci.change_rules import ChangeContext, ChangeDetector, PathPattern
from ci.delegator import Delegator


def write_rules(tmp_path: Path, rules: dict) -> Path:
    path = tmp_path / "change-rules.yaml"
    path.write_text(yaml.safe_dump({"schema_version": 1, "rules": rules}))
    return path


def test_named_capture_and_recursive_wildcard():
    pattern = PathPattern("mlx_vlm/models/{model}/**")

    assert pattern.match("mlx_vlm/models/qwen/model.py") == {"model": "qwen"}
    assert pattern.match("mlx_vlm/models/attention.py") is None


def test_rule_can_be_added_without_detector_code_change(tmp_path):
    rules = write_rules(
        tmp_path,
        {
            "docs": {
                "component": "lightweight",
                "include": ["docs/**", "*.md"],
                "exclude": ["docs/generated/**"],
            }
        },
    )
    detector = ChangeDetector.from_yaml(rules)

    matches = detector.detect(
        ChangeContext.create(
            ["docs/guide/setup.md", "docs/generated/index.md", "README.md"]
        )
    )

    assert [(match.rule, match.component, match.path) for match in matches] == [
        ("docs", "lightweight", "README.md"),
        ("docs", "lightweight", "docs/guide/setup.md"),
    ]


def test_base_and_head_conditions_classify_new_directory(tmp_path):
    rules = write_rules(
        tmp_path,
        {
            "new_model": {
                "component": "new_model",
                "include": ["mlx_vlm/models/{model}/**"],
                "base_path_absent": "mlx_vlm/models/{model}",
                "head_path_present": "mlx_vlm/models/{model}",
            }
        },
    )
    detector = ChangeDetector.from_yaml(rules)
    context = ChangeContext.create(
        ["mlx_vlm/models/new/model.py"],
        base_files=["mlx_vlm/models/old/model.py"],
        head_files=["mlx_vlm/models/new/model.py"],
        tree_state_known=True,
    )

    matches = detector.detect(context)

    assert matches[0].captures == {"model": "new"}


def test_new_model_rule_supersedes_existing_model_for_same_capture(tmp_path):
    rules = write_rules(
        tmp_path,
        {
            "new_model": {
                "component": "new_model",
                "include": ["mlx_vlm/models/{model}/**"],
                "base_path_absent": "mlx_vlm/models/{model}",
                "head_path_present": "mlx_vlm/models/{model}",
                "supersedes": ["model"],
            },
            "model": {
                "component": "model",
                "include": ["mlx_vlm/models/{model}/**"],
            },
        },
    )
    detector = ChangeDetector.from_yaml(rules)
    context = ChangeContext.create(
        [
            "mlx_vlm/models/new/config.py",
            "mlx_vlm/models/old/config.py",
        ],
        base_files=["mlx_vlm/models/old/model.py"],
        head_files=[
            "mlx_vlm/models/new/model.py",
            "mlx_vlm/models/old/model.py",
        ],
        tree_state_known=True,
    )

    matches = detector.detect(context)

    assert [(match.rule, match.captures["model"]) for match in matches] == [
        ("new_model", "new"),
        ("model", "old"),
    ]


def test_unknown_rule_fields_fail_validation(tmp_path):
    rules = write_rules(
        tmp_path,
        {
            "docs": {
                "component": "lightweight",
                "include": ["docs/**"],
                "typo": True,
            }
        },
    )

    with pytest.raises(ValueError, match="unknown fields"):
        ChangeDetector.from_yaml(rules)


def test_duplicate_yaml_keys_fail_validation(tmp_path):
    rules = tmp_path / "change-rules.yaml"
    rules.write_text(
        "schema_version: 1\nrules:\n  docs:\n    component: first\n"
        "    component: second\n    include: ['*.md']\n"
    )

    with pytest.raises(ValueError, match="duplicate YAML key: component"):
        ChangeDetector.from_yaml(rules)


def test_matched_unregistered_component_is_a_blocker(tmp_path):
    rules = write_rules(
        tmp_path,
        {
            "docs": {
                "component": "lightweight",
                "include": ["docs/**"],
            }
        },
    )

    plan = Delegator(ChangeDetector.from_yaml(rules), []).plan(["docs/guide.md"])

    assert plan["jobs"] == []
    assert plan["blocked"] == [
        {
            "component": "lightweight",
            "rule": "docs",
            "changed_paths": ["docs/guide.md"],
            "reason": "unregistered_component",
        }
    ]
