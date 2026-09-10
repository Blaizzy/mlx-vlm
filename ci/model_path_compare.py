from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Mapping, Sequence

from ci.model_path_probe import aggregate
from ci.probe_process import run_project_probe

POSITIVE_METRICS = {
    "prefill_tps": "tok/s",
    "decode_tps": "tok/s",
    "embedding_tps": "tok/s",
}
NEGATIVE_METRICS = {
    "ttft_ms": "ms",
    "embedding_latency_ms": "ms",
    "wall_ms": "ms",
    "peak_memory_gib": "GiB",
}


def run_probe(
    project: Path,
    probe: Path,
    job: Path,
    scenarios: Path,
    image: Path,
    max_tokens: int,
    output: Path,
) -> Mapping[str, Any]:
    arguments = [
        "--job",
        str(job),
        "--scenarios",
        str(scenarios),
        "--image",
        str(image),
        "--max-tokens",
        str(max_tokens),
        "--warmup",
        "1",
        "--iterations",
        "3",
        "--output",
        str(output),
    ]
    run_project_probe(project, probe, arguments)
    value = json.loads(output.read_text())
    if not isinstance(value, Mapping):
        raise RuntimeError("model path probe output must be an object")
    return value


def metric_verdict(name: str, change_pct: float, threshold: float = 5.0) -> str:
    if abs(change_pct) < threshold:
        return "noise"
    if name in POSITIVE_METRICS:
        return "improved" if change_pct > 0 else "regressed"
    return "improved" if change_pct < 0 else "regressed"


def compare(base: Mapping[str, Any], head: Mapping[str, Any]) -> dict[str, Any]:
    metrics: dict[str, dict[str, Any]] = {}
    unavailable_metrics: dict[str, dict[str, Any]] = {}
    for name, unit in (POSITIVE_METRICS | NEGATIVE_METRICS).items():
        if name not in base or name not in head:
            continue
        if (
            name == "decode_tps"
            and min(
                int(base.get("generation_tokens", 0)),
                int(head.get("generation_tokens", 0)),
            )
            < 8
        ):
            unavailable_metrics[name] = {
                "reason": "requires_at_least_8_generation_tokens",
                "base_generation_tokens": int(base.get("generation_tokens", 0)),
                "head_generation_tokens": int(head.get("generation_tokens", 0)),
            }
            continue
        base_value = float(base[name])
        head_value = float(head[name])
        change_pct = ((head_value - base_value) / base_value * 100) if base_value else 0
        metrics[name] = {
            "base": round(base_value, 4),
            "head": round(head_value, 4),
            "change_pct": round(change_pct, 2),
            "verdict": metric_verdict(name, change_pct),
            "unit": unit,
        }

    hashes_match = base.get("output_hash") == head.get("output_hash")
    checks = [base.get("correctness_checks"), head.get("correctness_checks")]
    has_contract = any(isinstance(item, Mapping) for item in checks)
    contracts_pass = not has_contract or all(
        isinstance(item, Mapping) and all(bool(value) for value in item.values())
        for item in checks
    )
    metric_verdicts = {item["verdict"] for item in metrics.values()}
    if not hashes_match or not contracts_pass:
        verdict = "test_failure"
    elif "regressed" in metric_verdicts:
        verdict = "regressed"
    elif "improved" in metric_verdicts:
        verdict = "improved"
    else:
        verdict = "passed"
    return {
        "verdict": verdict,
        "correctness": {
            "base_output_hash": base.get("output_hash"),
            "head_output_hash": head.get("output_hash"),
            "match": hashes_match,
            "contracts_pass": contracts_pass,
        },
        "metrics": metrics,
        "unavailable_metrics": unavailable_metrics,
        "base": dict(base),
        "head": dict(head),
    }


def merge_measurements(*measurements: Mapping[str, Any]) -> dict[str, Any]:
    runs = [
        run
        for measurement in measurements
        for run in measurement.get("runs", [measurement])
    ]
    return aggregate(runs)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--job", type=Path, required=True)
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--head", type=Path, required=True)
    parser.add_argument("--probe", type=Path, required=True)
    parser.add_argument("--scenarios", type=Path, required=True)
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--max-tokens", type=int, default=32)
    args = parser.parse_args(argv)

    findings_path = Path(os.environ.get("CI_JOB_FINDINGS", "findings.json"))
    base_output = findings_path.with_suffix(".base.json")
    head_output = findings_path.with_suffix(".head.json")
    try:
        base = run_probe(
            args.base,
            args.probe,
            args.job,
            args.scenarios,
            args.image,
            args.max_tokens,
            base_output,
        )
        head = run_probe(
            args.head,
            args.probe,
            args.job,
            args.scenarios,
            args.image,
            args.max_tokens,
            head_output,
        )
        result = compare(base, head)
        if result["verdict"] == "regressed":
            confirmation_head = run_probe(
                args.head,
                args.probe,
                args.job,
                args.scenarios,
                args.image,
                args.max_tokens,
                findings_path.with_suffix(".confirmation-head.json"),
            )
            confirmation_base = run_probe(
                args.base,
                args.probe,
                args.job,
                args.scenarios,
                args.image,
                args.max_tokens,
                findings_path.with_suffix(".confirmation-base.json"),
            )
            result = compare(
                merge_measurements(base, confirmation_base),
                merge_measurements(head, confirmation_head),
            )
            result["performance_confirmation"] = "counterbalanced"
    except Exception as error:
        result = {
            "verdict": "test_failure",
            "error": f"{type(error).__name__}: {error}",
        }
    findings_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, sort_keys=True))
    return 2 if result["verdict"] == "test_failure" else 0


if __name__ == "__main__":
    raise SystemExit(main())
