from ci.model_path_compare import compare, merge_measurements, metric_verdict


def measurements(**overrides):
    value = {
        "prefill_tps": 100.0,
        "decode_tps": 20.0,
        "ttft_ms": 500.0,
        "wall_ms": 1000.0,
        "peak_memory_gib": 4.0,
        "prompt_tokens": 32,
        "generation_tokens": 16,
        "output_hash": "same",
    }
    value.update(overrides)
    return value


def test_metric_direction_accounts_for_throughput_and_latency():
    assert metric_verdict("prefill_tps", 8) == "improved"
    assert metric_verdict("decode_tps", -8) == "regressed"
    assert metric_verdict("ttft_ms", -8) == "improved"
    assert metric_verdict("peak_memory_gib", 8) == "regressed"
    assert metric_verdict("wall_ms", 2) == "noise"


def test_comparison_reports_improvement_with_matching_output():
    result = compare(
        measurements(),
        measurements(prefill_tps=110.0, ttft_ms=450.0),
    )

    assert result["verdict"] == "improved"
    assert result["correctness"]["match"] is True
    assert result["metrics"]["prefill_tps"]["change_pct"] == 10.0


def test_output_mismatch_overrides_performance():
    result = compare(measurements(), measurements(output_hash="different"))

    assert result["verdict"] == "test_failure"
    assert result["correctness"]["match"] is False


def test_embedding_contract_failure_overrides_matching_output():
    base = measurements(
        correctness_checks={
            "finite_embeddings": True,
            "normalized_embeddings": True,
            "related_pair_similarity_above_unrelated": True,
        }
    )
    head = measurements(
        correctness_checks={
            "finite_embeddings": True,
            "normalized_embeddings": True,
            "related_pair_similarity_above_unrelated": False,
        }
    )

    result = compare(base, head)

    assert result["verdict"] == "test_failure"
    assert result["correctness"]["match"] is True
    assert result["correctness"]["contracts_pass"] is False


def test_decode_throughput_is_unavailable_for_short_generations():
    result = compare(
        measurements(generation_tokens=1),
        measurements(generation_tokens=1, decode_tps=200.0),
    )

    assert result["verdict"] == "passed"
    assert "decode_tps" not in result["metrics"]
    assert result["unavailable_metrics"]["decode_tps"] == {
        "reason": "requires_at_least_8_generation_tokens",
        "base_generation_tokens": 1,
        "head_generation_tokens": 1,
    }


def test_merge_measurements_counterbalances_order_drift():
    first = measurements(prefill_tps=120.0, ttft_ms=400.0)
    first["runs"] = [first]
    second = measurements(prefill_tps=100.0, ttft_ms=500.0)
    second["runs"] = [second]

    merged = merge_measurements(first, second)

    assert merged["prefill_tps"] == 110.0
    assert merged["ttft_ms"] == 450.0
    assert len(merged["runs"]) == 2
