import json
from pathlib import Path

import yaml

from ci.change_rules import ChangeDetector
from ci.checkpoint_policy import validate_checkpoint
from ci.delegator import (
    Delegator,
    ModelPath,
    NewModelPath,
    _parse_name_status,
    create_delegator,
    main,
)
from ci.docs_change import DocsChange


def test_temporary_model_manifests_do_not_disable_other_components(tmp_path):
    config_directory = Path(__file__).parents[1]
    model_config = tmp_path / "model_path.yaml"
    scenario_config = tmp_path / "model-path-scenario.yaml"
    model_config.write_text((config_directory / "model_path.yaml").read_text())
    scenario_config.write_text(
        (config_directory / "model-path-scenario.yaml").read_text()
    )

    delegator = create_delegator(
        config_directory / "change-rules.yaml",
        tmp_path,
        config_directory.parent,
    )

    assert [component.name for component in delegator.components] == [
        "docs_change",
        "new_model_path",
        "model_path",
    ]


def write_configs(tmp_path: Path) -> tuple[Path, Path, Path]:
    model_config = {
        "schema_version": 1,
        "synthetic_profiles": {"dense_vlm": {}},
        "models": {
            "ready": {
                "capabilities": ["vision_language"],
                "synthetic": {
                    "status": "configured",
                    "adapter": "ready",
                    "profile": "dense_vlm",
                },
                "hf_checkpoint": {
                    "status": "configured",
                    "repo": "example/ready",
                    "revision": "a" * 40,
                    "expected_model_type": "ready",
                    "weight": {"bytes": 1024, "gib": 0.01},
                },
            },
            "second": {
                "capabilities": ["vision_language"],
                "synthetic": {
                    "status": "configured",
                    "adapter": "second",
                    "profile": "dense_vlm",
                },
                "hf_checkpoint": {
                    "status": "configured",
                    "repo": "example/second",
                    "revision": "b" * 40,
                    "expected_model_type": "second",
                    "weight": {"bytes": 2048, "gib": 0.01},
                },
            },
            "todo": {
                "synthetic": {"status": "TODO"},
                "hf_checkpoint": {"status": "TODO"},
            },
        },
    }
    scenario_config = {
        "schema_version": 1,
        "defaults_by_capability": {"vision_language": "vlm_animal"},
        "scenarios": {"vlm_animal": {}},
    }
    rules_config = {
        "schema_version": 1,
        "rules": {
            "docs_change": {
                "component": "docs_change",
                "include": ["*.md", "**/*.md", "mkdocs.yml", "docs/**"],
                "exclude": [],
            },
            "new_model_path": {
                "component": "new_model_path",
                "include": ["mlx_vlm/models/{model}/**"],
                "exclude": [
                    "mlx_vlm/models/{model}/*.md",
                    "mlx_vlm/models/{model}/**/*.md",
                ],
                "base_path_absent": "mlx_vlm/models/{model}",
                "head_path_present": "mlx_vlm/models/{model}",
                "supersedes": ["model_path"],
            },
            "model_path": {
                "component": "model_path",
                "include": ["mlx_vlm/models/{model}/**"],
                "exclude": [
                    "mlx_vlm/models/{model}/*.md",
                    "mlx_vlm/models/{model}/**/*.md",
                ],
            },
        },
    }
    model_path = tmp_path / "model_path.yaml"
    scenario_path = tmp_path / "model-path-scenario.yaml"
    rules_path = tmp_path / "change-rules.yaml"
    model_path.write_text(yaml.safe_dump(model_config))
    scenario_path.write_text(yaml.safe_dump(scenario_config))
    rules_path.write_text(yaml.safe_dump(rules_config))
    return model_path, scenario_path, rules_path


def make_delegator(tmp_path: Path) -> Delegator:
    model_config, scenario_config, rules_config = write_configs(tmp_path)
    model_path = ModelPath(model_config, scenario_config)
    return Delegator(
        ChangeDetector.from_yaml(rules_config),
        [DocsChange(), NewModelPath(model_path), model_path],
    )


def test_configured_model_emits_one_model_path_work_item(tmp_path):
    plan = make_delegator(tmp_path).plan(
        [
            "mlx_vlm/models/ready/vision.py",
            "mlx_vlm/models/ready/config.py",
            "mlx_vlm/models/ready/config.py",
        ]
    )

    assert plan["rules"] == ["model_path"]
    assert plan["components"] == ["model_path"]
    assert len(plan["jobs"]) == 1
    work = plan["jobs"][0]
    assert work["work_type"] == "ModelPath"
    assert work["phases"] == ["synthetic", "hf_checkpoint"]
    assert work["changed_paths"] == [
        "mlx_vlm/models/ready/config.py",
        "mlx_vlm/models/ready/vision.py",
    ]
    assert work["scenarios"] == ["vlm_animal"]
    assert work["hf_checkpoint"]["weight"]["bytes"] == 1024
    assert plan["gates"] == []
    assert plan["blocked"] == []


def test_existing_model_manifest_change_cannot_redefine_its_own_job(tmp_path):
    plan = make_delegator(tmp_path).plan(
        ["mlx_vlm/models/ready/vision.py", "ci/model_path.yaml"]
    )

    assert plan["jobs"] == []
    assert plan["blocked"][0]["reason"] == "existing_model_ci_configuration_changed"


def test_multiple_models_emit_independent_jobs(tmp_path):
    plan = make_delegator(tmp_path).plan(
        [
            "mlx_vlm/models/ready/model.py",
            "mlx_vlm/models/second/model.py",
        ]
    )

    assert [job["model"] for job in plan["jobs"]] == ["ready", "second"]
    assert all(job["work_type"] == "ModelPath" for job in plan["jobs"])


def test_new_model_emits_approval_gate_and_no_jobs(tmp_path):
    plan = make_delegator(tmp_path).plan(
        [
            "ci/model_path.yaml",
            "mlx_vlm/models/ready/__init__.py",
            "mlx_vlm/models/ready/model.py",
        ],
        base_files=["mlx_vlm/models/existing/model.py"],
        head_files=["mlx_vlm/models/ready/model.py"],
        head_sha="abc123",
        tree_state_known=True,
    )

    assert plan["rules"] == ["new_model_path"]
    assert plan["components"] == ["new_model_path"]
    assert plan["jobs"] == []
    assert plan["blocked"] == []
    assert len(plan["gates"]) == 1
    gate = plan["gates"][0]
    assert gate["status"] == "awaiting_maintainer_approval"
    assert gate["head_sha"] == "abc123"
    assert gate["configuration_digest"].startswith("sha256:")
    assert gate["requested_phases"] == ["synthetic", "hf_checkpoint"]
    assert gate["pending_work"]["work_type"] == "ModelPath"
    assert gate["pending_work"]["phases"] == ["synthetic", "hf_checkpoint"]
    assert gate["configuration"]["hf_checkpoint"]["repo"] == "example/ready"


def test_new_model_without_manifest_entry_is_blocked(tmp_path):
    plan = make_delegator(tmp_path).plan(
        ["ci/model_path.yaml", "mlx_vlm/models/missing/model.py"],
        base_files=["mlx_vlm/models/existing/model.py"],
        head_files=["mlx_vlm/models/missing/model.py"],
        head_sha="abc123",
        tree_state_known=True,
    )

    assert plan["jobs"] == []
    assert plan["gates"] == []
    assert plan["blocked"][0]["component"] == "new_model_path"
    assert plan["blocked"][0]["reason"] == "missing_model_config"


def test_new_model_requires_manifest_change_in_same_pr(tmp_path):
    plan = make_delegator(tmp_path).plan(
        ["mlx_vlm/models/ready/model.py"],
        base_files=["mlx_vlm/models/existing/model.py"],
        head_files=["mlx_vlm/models/ready/model.py"],
        head_sha="abc123",
        tree_state_known=True,
    )

    assert plan["jobs"] == []
    assert plan["gates"] == []
    assert plan["blocked"][0]["reason"] == "model_manifest_not_updated"


def test_todo_model_is_blocked_for_both_modes(tmp_path):
    plan = make_delegator(tmp_path).plan(["mlx_vlm/models/todo/model.py"])

    assert plan["jobs"] == []
    assert [(item["mode"], item["reason"]) for item in plan["blocked"]] == [
        ("synthetic", "not_configured"),
        ("hf_checkpoint", "not_configured"),
    ]


def test_invalid_scenario_config_blocks_configured_modes(tmp_path):
    delegator = make_delegator(tmp_path)
    model_path = next(
        component
        for component in delegator.components
        if component.name == "model_path"
    )
    model_path.models["ready"]["scenarios"] = [{"invalid": True}]

    plan = delegator.plan(["mlx_vlm/models/ready/model.py"])

    assert plan["jobs"] == []
    assert [(item["mode"], item["reason"]) for item in plan["blocked"]] == [
        ("synthetic", "invalid_scenarios"),
        ("hf_checkpoint", "invalid_scenarios"),
    ]


def test_shared_model_component_and_unrelated_code_are_ignored(tmp_path):
    plan = make_delegator(tmp_path).plan(
        ["mlx_vlm/models/attention.py", "mlx_vlm/server/app.py"]
    )

    assert plan == {
        "schema_version": 1,
        "base_sha": None,
        "target_sha": None,
        "head_sha": None,
        "rules": [],
        "components": [],
        "jobs": [],
        "gates": [],
        "checks": [],
        "blocked": [],
    }


def test_documentation_change_emits_one_hosted_check(tmp_path):
    plan = make_delegator(tmp_path).plan(["README.md", "docs/guide.md"])

    assert plan["rules"] == ["docs_change"]
    assert plan["components"] == ["docs_change"]
    assert plan["jobs"] == []
    assert plan["checks"] == [
        {
            "id": "docs",
            "work_type": "Docs",
            "component": "docs_change",
            "execution_target": "github_hosted",
            "changed_paths": ["README.md", "docs/guide.md"],
        }
    ]


def test_model_documentation_does_not_schedule_model_execution(tmp_path):
    plan = make_delegator(tmp_path).plan(["mlx_vlm/models/ready/README.md"])

    assert plan["rules"] == ["docs_change"]
    assert plan["components"] == ["docs_change"]
    assert plan["jobs"] == []
    assert plan["checks"][0]["changed_paths"] == ["mlx_vlm/models/ready/README.md"]


def test_documentation_and_model_code_route_independently(tmp_path):
    plan = make_delegator(tmp_path).plan(
        ["mlx_vlm/models/ready/README.md", "mlx_vlm/models/ready/model.py"]
    )

    assert plan["rules"] == ["docs_change", "model_path"]
    assert plan["components"] == ["docs_change", "model_path"]
    assert plan["checks"][0]["changed_paths"] == ["mlx_vlm/models/ready/README.md"]
    assert [item["id"] for item in plan["jobs"]] == ["model_path:ready"]


def test_rename_includes_old_and_new_paths():
    output = (
        b"R100\0mlx_vlm/models/old/model.py\0"
        b"mlx_vlm/models/new/model.py\0M\0README.md\0"
    )

    assert _parse_name_status(output) == (
        "mlx_vlm/models/old/model.py",
        "mlx_vlm/models/new/model.py",
        "README.md",
    )


def test_cli_writes_json_plan(tmp_path, monkeypatch):
    output = tmp_path / "plan.json"
    delegator = make_delegator(tmp_path)
    monkeypatch.setattr("ci.delegator.default_delegator", lambda: delegator)

    assert (
        main(
            [
                "--changed-file",
                "mlx_vlm/models/ready/model.py",
                "--output",
                str(output),
            ]
        )
        == 0
    )
    plan = json.loads(output.read_text())
    assert len(plan["jobs"]) == 1
    assert plan["gates"] == []
    assert plan["blocked"] == []


def test_repository_configured_models_are_routable():
    config_directory = Path(__file__).parents[1]
    from ci.components.model_path import SYNTHETIC_ADAPTERS

    model_path = ModelPath(
        config_directory / "model_path.yaml",
        config_directory / "model-path-scenario.yaml",
        supported_synthetic_adapters=SYNTHETIC_ADAPTERS,
    )
    configured = [
        name
        for name, model in model_path.models.items()
        if model.get("synthetic", {}).get("status") == "configured"
        and model.get("hf_checkpoint", {}).get("status") == "configured"
    ]
    detector = ChangeDetector.from_yaml(config_directory / "change-rules.yaml")
    delegator = Delegator(
        ChangeDetector(
            rule for rule in detector.rules if rule.component != "docs_change"
        ),
        [NewModelPath(model_path), model_path],
    )

    plan = delegator.plan([f"mlx_vlm/models/{name}/config.py" for name in configured])

    assert {job["model"] for job in plan["jobs"]} == {
        "bert",
        "deepseek_vl_v2",
        "granite_vision",
        "internvl_chat",
        "qwen2_5_vl",
        "qwen2_vl",
    }
    assert plan["gates"] == []
    assert plan["blocked"] == []


def test_unsupported_synthetic_adapter_is_blocked(tmp_path):
    model_config, scenario_config, rules_config = write_configs(tmp_path)
    model_path = ModelPath(
        model_config,
        scenario_config,
        supported_synthetic_adapters=frozenset({"ready"}),
    )
    delegator = Delegator(
        ChangeDetector.from_yaml(rules_config),
        [DocsChange(), NewModelPath(model_path), model_path],
    )

    plan = delegator.plan(["mlx_vlm/models/second/model.py"])

    assert plan["jobs"] == []
    assert plan["blocked"][0]["reason"] == "unsupported_synthetic_adapter"


def test_model_catalog_covers_every_repository_model_directory():
    config_directory = Path(__file__).parents[1]
    model_path = ModelPath(
        config_directory / "model_path.yaml",
        config_directory / "model-path-scenario.yaml",
    )
    repository_models = {
        path.name
        for path in (config_directory.parent / "mlx_vlm" / "models").iterdir()
        if path.is_dir() and (path / "__init__.py").is_file()
    }

    assert set(model_path.models) == repository_models


def test_recently_synced_models_have_pinned_candidate_configurations():
    config_directory = Path(__file__).parents[1]
    model_path = ModelPath(
        config_directory / "model_path.yaml",
        config_directory / "model-path-scenario.yaml",
    )
    models = {
        "dinov2",
        "hy_v4",
        "k2_horizon",
        "llava_onevision",
        "longcat_flash_sparse",
        "moge3",
        "openai_privacy_filter",
        "pp_doclayout_v3",
        "spark2_5",
        "video_depth_anything",
        "yolo11",
        "z1t",
    }

    for name in models:
        configuration = model_path.models[name]
        assert configuration["synthetic"]["status"] == "TODO"
        checkpoint = configuration["hf_checkpoint"]
        assert checkpoint["status"] == "TODO"
        validate_checkpoint(
            {
                key: checkpoint[key]
                for key in ("repo", "revision", "expected_model_type", "weight")
            }
        )
