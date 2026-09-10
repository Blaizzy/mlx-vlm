import sys
from types import ModuleType, SimpleNamespace

import mlx.core as mx
import numpy as np
import pytest

from ci.model_path_probe import (
    aggregate,
    cached_checkpoint,
    checkpoint,
    embed,
    embedding_scenario,
    formatted_prompt,
    prepare_processor,
    summarize,
)


def test_embedding_scenario_is_selected_from_model_path_work():
    scenario = {"embedding": {"input": {"texts": ["related", "similar", "unrelated"]}}}

    assert (
        embedding_scenario({"scenarios": ["embedding"]}, scenario)
        == scenario["embedding"]
    )
    assert embedding_scenario({"scenarios": ["vlm_animal"]}, scenario) is None


def test_embedding_probe_checks_semantics_and_normalization():
    class Tokenizer:
        model_max_length = 32

        def __call__(self, texts, **kwargs):
            assert len(texts) == 3
            return {
                "input_ids": np.array([[1, 2], [3, 4], [5, 6]]),
                "attention_mask": np.ones((3, 2)),
            }

    class Model:
        def __call__(self, input_ids, attention_mask):
            return SimpleNamespace(
                text_embeds=mx.array([[1.0, 0.0], [0.995, 0.099875], [0.0, 1.0]])
            )

    findings = embed(
        Model(),
        SimpleNamespace(tokenizer=Tokenizer()),
        ["related", "similar", "unrelated"],
    )

    assert findings["output_shape"] == [3, 2]
    assert all(findings["correctness_checks"].values())
    assert findings["similarities"]["related"] > findings["similarities"]["unrelated"]


def test_checkpoint_requires_pinned_repo():
    job = {"hf_checkpoint": {"repo": "org/model", "revision": "abc123"}}
    assert checkpoint(job) == ("org/model", "abc123")


@pytest.mark.parametrize(
    "value",
    [None, {}, {"repo": "org/model"}, {"revision": "abc123"}],
)
def test_checkpoint_rejects_incomplete_metadata(value):
    with pytest.raises(ValueError):
        checkpoint({"hf_checkpoint": value})


def test_summarize_reports_e2e_metrics(monkeypatch):
    monkeypatch.setattr("ci.model_path_probe.time.perf_counter", lambda: 3.0)
    result = SimpleNamespace(
        text="a cat",
        token=42,
        prompt_tokens=12,
        generation_tokens=2,
        prompt_tps=100.0,
        generation_tps=20.0,
        peak_memory=18.5,
        finish_reason="length",
    )
    findings = summarize([result], 1.0, 250.0)
    assert findings["generated_text"] == "a cat"
    assert findings["prefill_tps"] == 100.0
    assert findings["decode_tps"] == 20.0
    assert findings["ttft_ms"] == 250.0
    assert findings["wall_ms"] == 2000.0
    assert findings["peak_memory_gib"] == 18.5


def test_summarize_accepts_empty_decoded_text_with_a_generation_result(monkeypatch):
    monkeypatch.setattr("ci.model_path_probe.time.perf_counter", lambda: 3.0)
    result = SimpleNamespace(
        text="",
        token=2,
        prompt_tokens=12,
        generation_tokens=1,
        prompt_tps=100.0,
        generation_tps=20.0,
        peak_memory=18.5,
        finish_reason="stop",
    )

    findings = summarize([result], 1.0, 250.0)

    assert findings["generated_text"] == ""
    assert findings["output_hash"]
    assert findings["generation_tokens"] == 1


def test_aggregate_uses_median_and_retains_runs():
    first = {
        "output_hash": "same",
        "prompt_tokens": 10,
        "generation_tokens": 2,
        "prefill_tps": 100,
        "decode_tps": 20,
        "ttft_ms": 500,
        "wall_ms": 900,
        "peak_memory_gib": 4,
    }
    second = dict(first, prefill_tps=120, ttft_ms=400)
    third = dict(first, prefill_tps=110, ttft_ms=450)

    findings = aggregate([first, second, third])

    assert findings["prefill_tps"] == 110
    assert findings["ttft_ms"] == 450
    assert len(findings["runs"]) == 3


def test_cached_checkpoint_uses_pinned_local_snapshot(tmp_path, monkeypatch):
    snapshot = tmp_path / "models--org--model" / "snapshots" / "abc123"
    snapshot.mkdir(parents=True)
    monkeypatch.setenv("HF_HUB_CACHE", str(tmp_path))

    assert cached_checkpoint("org/model", "abc123") == snapshot
    assert cached_checkpoint("org/model", "different") is None


def test_prepare_processor_fills_missing_vision_metadata():
    processor = SimpleNamespace(
        patch_size=None,
        vision_feature_select_strategy=None,
        num_additional_image_tokens=0,
    )

    prepare_processor(
        processor,
        {
            "vision_config": {
                "patch_size": 14,
                "model_type": "clip_vision_model",
            },
            "vision_feature_select_strategy": "default",
        },
    )

    assert processor.patch_size == 14
    assert processor.vision_feature_select_strategy == "default"
    assert processor.num_additional_image_tokens == 1


def test_formatted_prompt_flattens_media_for_text_tokenizer(monkeypatch):
    tokenizer = SimpleNamespace()
    processor = SimpleNamespace(tokenizer=tokenizer, image_token="<image>")
    calls = []

    def apply(processor, config, prompt, **kwargs):
        if kwargs.get("return_messages"):
            return [
                {
                    "role": "user",
                    "content": [
                        {"type": "image"},
                        {"type": "text", "text": prompt},
                    ],
                }
            ]
        raise TypeError('can only concatenate str (not "list") to str')

    def render(messages, **kwargs):
        calls.append((messages, kwargs))
        return "rendered"

    tokenizer.apply_chat_template = render
    prompt_utils = ModuleType("mlx_vlm.prompt_utils")
    prompt_utils.apply_chat_template = apply
    monkeypatch.setitem(sys.modules, "mlx_vlm.prompt_utils", prompt_utils)

    assert (
        formatted_prompt(processor, {"model_type": "llava_next"}, "cat") == "rendered"
    )
    assert calls[0][0] == [{"role": "user", "content": "<image>\ncat"}]
