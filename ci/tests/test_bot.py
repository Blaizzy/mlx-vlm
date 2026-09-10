import pytest

from ci.bot import BotOutput, BotOutputError


def job(model):
    return {
        "id": f"model_path:{model}",
        "work_type": "ModelPath",
        "component": "model_path",
        "model": model,
        "phases": ["synthetic", "hf_checkpoint"],
        "changed_paths": [f"mlx_vlm/models/{model}/model.py"],
        "synthetic": {"adapter": model, "profile": "dense_vlm"},
        "hf_checkpoint": {
            "repo": f"mlx-community/{model}-4bit",
            "revision": "abcdef1234567890",
        },
    }


def record(*, jobs=None, gates=None, checks=None, errors=None, results=None):
    return {
        "schema_version": 1,
        "kind": "ci_control",
        "head_sha": "abc123",
        "outcome": "ready",
        "run_url": "https://example.com/run",
        "jobs": jobs or [],
        "gates": gates or [],
        "checks": checks or [],
        "errors": errors or [],
        "results": results or [],
    }


def test_each_model_gets_an_independent_section():
    rendered = BotOutput(
        record(jobs=[job("qwen2_vl"), job("gemma3"), job("pixtral")])
    ).render()

    assert rendered.count("· ModelPath · Awaiting /ci run") == 3
    assert rendered.count("| Synthetic | Planned |") == 3
    assert rendered.count("| HF checkpoint | Planned |") == 3


def test_comment_identifies_commit_without_a_heading():
    rendered = BotOutput(record(jobs=[job("qwen2_vl")])).render()

    assert rendered.startswith("<!-- mlx-vlm:ci:plan -->")
    assert "Commit: `abc123`" in rendered
    assert not rendered.startswith("## ")


def test_execution_comment_is_immutable_per_attempt():
    value = record(jobs=[job("qwen2_vl")])
    value.update({"kind": "ci_execution", "attempt_id": "42"})

    rendered = BotOutput(value).render()

    assert rendered.startswith("<!-- mlx-vlm:ci:attempt:42 -->")
    assert "Attempt: `42`" in rendered


def test_success_collapses_and_failure_stays_open():
    passed = record(
        jobs=[job("qwen2_vl")],
        results=[
            {
                "component": "model_path",
                "model": "qwen2_vl",
                "outcome": "passed",
            }
        ],
    )
    failed = record(
        jobs=[job("qwen2_vl")],
        results=[
            {
                "component": "model_path",
                "model": "qwen2_vl",
                "outcome": "test_failure",
            }
        ],
    )

    assert "<details>\n<summary>✅" in BotOutput(passed).render()
    assert "<details open>\n<summary>❌" in BotOutput(failed).render()


def test_correctness_failure_makes_performance_advisory():
    result = {
        "component": "model_path",
        "model": "qwen2_vl",
        "outcome": "test_failure",
        "device": "mini-1",
        "cache": {"reused": True},
        "phases": {
            "synthetic": {
                "outcome": "passed",
                "findings": {"correctness": {"match": True}},
            },
            "hf_checkpoint": {
                "outcome": "test_failure",
                "findings": {
                    "correctness": {"match": False},
                    "metrics": {
                        "decode_tps": {
                            "base": 10,
                            "head": 12,
                            "change_pct": 20,
                            "verdict": "improved",
                            "unit": "tok/s",
                        }
                    },
                },
            },
        },
    }

    rendered = BotOutput(record(jobs=[job("qwen2_vl")], results=[result])).render()

    assert (
        "| Decode throughput | 10 tok/s | 12 tok/s | +20.00% | advisory |" in rendered
    )
    assert (
        "Checkpoint correctness failed; performance numbers are advisory." in rendered
    )
    assert "Runner: mini-1 · Cache: reused." in rendered


def test_embedding_contract_failure_makes_performance_advisory():
    result = {
        "component": "model_path",
        "model": "bert",
        "outcome": "test_failure",
        "phases": {
            "hf_checkpoint": {
                "outcome": "test_failure",
                "findings": {
                    "correctness": {"match": True, "contracts_pass": False},
                    "metrics": {
                        "embedding_tps": {
                            "base": 100,
                            "head": 110,
                            "change_pct": 10,
                            "verdict": "improved",
                            "unit": "tok/s",
                        }
                    },
                },
            }
        },
    }

    rendered = BotOutput(record(jobs=[job("bert")], results=[result])).render()

    assert (
        "| Embedding throughput | 100 tok/s | 110 tok/s | +10.00% | advisory |"
        in rendered
    )
    assert "Correctness: failed · Performance: advisory" in rendered


def test_new_model_waits_for_maintainer_approval():
    pending = job("new_family")
    gate = {
        "component": "new_model_path",
        "model": "new_family",
        "status": "awaiting_maintainer_approval",
        "changed_paths": ["mlx_vlm/models/new_family/model.py"],
        "pending_work": pending,
    }

    rendered = BotOutput(record(gates=[gate])).render()

    assert "· ModelPath · Awaiting maintainer approval" in rendered
    assert rendered.count("| Synthetic | Awaiting approval |") == 1
    assert rendered.count("| HF checkpoint | Awaiting approval |") == 1


def test_model_names_cannot_create_mentions_or_break_tables():
    rendered = BotOutput(record(jobs=[job("@reviewer|model")])).render()

    assert "@\u200breviewer\\|model" in rendered
    assert "@reviewer|model" not in rendered


@pytest.mark.parametrize(
    ("failure", "message"),
    [
        ("checkpoint_not_found", "checkpoint or revision was not found"),
        ("access_denied", "requires access"),
        ("disk_full", "enough disk space"),
        ("network_transient", "failed temporarily"),
    ],
)
def test_checkpoint_failures_have_contributor_facing_messages(failure, message):
    result = {
        "component": "model_path",
        "model": "qwen2_vl",
        "outcome": "test_failure",
        "checkpoint_failure": {
            "phase": "hf_checkpoint",
            "operation": "download",
            "code": failure,
            "retryable": failure == "network_transient",
            "attempts": 1,
        },
    }

    assert (
        message in BotOutput(record(jobs=[job("qwen2_vl")], results=[result])).render()
    )


def test_docs_change_renders_result():
    check = {
        "id": "docs",
        "component": "docs_change",
        "execution_target": "github_hosted",
        "changed_paths": ["README.md"],
    }
    value = record(
        checks=[check],
        results=[
            {
                "component": "docs_change",
                "outcome": "passed",
                "changed_paths": ["README.md"],
                "findings": {"new_errors": []},
            }
        ],
    )
    value["components"] = ["docs_change"]

    rendered = BotOutput(value).render()

    assert "<strong>Documentation</strong> · DocsChange · Passed" in rendered


def test_unknown_component_is_rejected():
    with pytest.raises(BotOutputError, match="component_path"):
        BotOutput(
            record(jobs=[{"component": "component_path", "id": "unknown"}])
        ).render()
