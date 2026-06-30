#!/usr/bin/env python3
"""Microbenchmark continuous-batching decode latency under concurrent prefill.

The benchmark intentionally isolates BatchGenerator scheduling behavior. It uses
small fake decode and prompt batches with deterministic sleep-based costs so the
before/after comparison reflects how long decode responses are held by prompt
prefill work inside a single scheduler iteration.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


def _percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    rank = (len(ordered) - 1) * (pct / 100.0)
    lo = math.floor(rank)
    hi = math.ceil(rank)
    if lo == hi:
        return ordered[lo]
    return ordered[lo] + (ordered[hi] - ordered[lo]) * (rank - lo)


def _parse_ints(raw: str) -> list[int]:
    values = [int(part.strip()) for part in raw.split(",") if part.strip()]
    if not values:
        raise argparse.ArgumentTypeError("expected at least one integer")
    return values


def _repo_commit(repo: Path) -> str:
    try:
        return (
            subprocess.check_output(
                ["git", "-C", str(repo), "rev-parse", "--short", "HEAD"],
                text=True,
                stderr=subprocess.DEVNULL,
            )
            .strip()
            .splitlines()[0]
        )
    except Exception:
        return "unknown"


@dataclass
class ScenarioResult:
    label: str
    commit: str
    repo: str
    version_has_token_budget: bool
    mode: str
    decode_rows: int
    iterations: int
    warmup: int
    max_num_batched_tokens: int
    prefill_step_size: int
    remaining_prompt_tokens: int
    prefill_token_ms: float
    decode_token_ms: float
    median_ms: float
    mean_ms: float
    p95_ms: float
    min_ms: float
    max_ms: float
    mean_prefill_tokens: float
    median_prefill_tokens: float
    median_decode_tps: float
    raw_latency_ms: list[float]
    raw_prefill_tokens: list[int]

    def to_dict(self) -> dict:
        return self.__dict__.copy()


def _run_one_scenario(
    ar,
    *,
    mode: str,
    decode_rows: int,
    iterations: int,
    warmup: int,
    max_num_batched_tokens: int,
    prefill_step_size: int,
    remaining_prompt_tokens: int,
    prefill_token_ms: float,
    decode_token_ms: float,
) -> tuple[list[float], list[int]]:
    class FakeGenerationBatch:
        def __init__(self, rows: int):
            self.rows = rows
            self.prompt_cache = []

        def __len__(self):
            return self.rows

        def next(self):
            if decode_token_ms > 0:
                time.sleep(self.rows * decode_token_ms / 1000.0)
            return [
                ar.GenerationBatch.Response(i, 100 + i, 0.0, None)
                for i in range(self.rows)
            ]

    class FakePromptBatch:
        def __init__(self):
            self.remaining = remaining_prompt_tokens
            self.last_processed = 0
            self.recorded_prompt_time = 0.0

        def needs_processing(self, max_tokens=None):
            if max_tokens is None:
                return self.remaining > prefill_step_size
            return self.remaining > max(0, int(max_tokens))

        def prompt_step(self, max_tokens=None):
            if max_tokens is None:
                n = min(prefill_step_size, self.remaining - 1)
            else:
                n = min(prefill_step_size, max(0, int(max_tokens)), self.remaining - 1)
            n = max(0, n)
            if n > 0 and prefill_token_ms > 0:
                time.sleep(n * prefill_token_ms / 1000.0)
            self.remaining -= n
            self.last_processed = n
            return n

        def record_prompt_time(self, elapsed_s):
            self.recorded_prompt_time += elapsed_s

    latencies: list[float] = []
    prefill_tokens: list[int] = []
    total_runs = warmup + iterations
    for run_idx in range(total_runs):
        prompt_batch = None if mode == "decode_only" else FakePromptBatch()
        bg = object.__new__(ar.BatchGenerator)
        bg._wire_stack = None
        bg._generation_batch = FakeGenerationBatch(decode_rows)
        bg._prompt_batch = prompt_batch
        bg._unprocessed_sequences = []
        bg._gen_tokens_counter = 0
        bg._steps_counter = 0
        bg._cache_eval_interval = 0
        bg._prompt_time_counter = 0.0
        bg._prompt_tokens_counter = 0
        bg.completion_batch_size = max_num_batched_tokens + decode_rows + 1
        bg.max_num_batched_tokens = max_num_batched_tokens

        start = time.perf_counter()
        _, generation_responses = ar.BatchGenerator._next(bg)
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        if len(generation_responses) != decode_rows:
            raise RuntimeError(
                f"expected {decode_rows} decode responses, got {len(generation_responses)}"
            )
        processed = 0 if prompt_batch is None else prompt_batch.last_processed
        bg.close()

        if run_idx >= warmup:
            latencies.append(elapsed_ms)
            prefill_tokens.append(processed)

    return latencies, prefill_tokens


def run_benchmark(args) -> None:
    repo = Path(args.repo).resolve()
    sys.path.insert(0, str(repo))
    from mlx_vlm.generate import ar

    commit = _repo_commit(repo)
    version_has_token_budget = hasattr(ar.BatchGenerator, "_token_budget_after_decode")
    results: list[ScenarioResult] = []
    for mode in args.modes:
        for decode_rows in args.decode_rows:
            latencies, prefill_tokens = _run_one_scenario(
                ar,
                mode=mode,
                decode_rows=decode_rows,
                iterations=args.iterations,
                warmup=args.warmup,
                max_num_batched_tokens=args.max_num_batched_tokens,
                prefill_step_size=args.prefill_step_size,
                remaining_prompt_tokens=args.remaining_prompt_tokens,
                prefill_token_ms=args.prefill_token_ms,
                decode_token_ms=args.decode_token_ms,
            )
            median_ms = statistics.median(latencies)
            results.append(
                ScenarioResult(
                    label=args.label,
                    commit=commit,
                    repo=str(repo),
                    version_has_token_budget=version_has_token_budget,
                    mode=mode,
                    decode_rows=decode_rows,
                    iterations=args.iterations,
                    warmup=args.warmup,
                    max_num_batched_tokens=args.max_num_batched_tokens,
                    prefill_step_size=args.prefill_step_size,
                    remaining_prompt_tokens=args.remaining_prompt_tokens,
                    prefill_token_ms=args.prefill_token_ms,
                    decode_token_ms=args.decode_token_ms,
                    median_ms=median_ms,
                    mean_ms=statistics.mean(latencies),
                    p95_ms=_percentile(latencies, 95),
                    min_ms=min(latencies),
                    max_ms=max(latencies),
                    mean_prefill_tokens=statistics.mean(prefill_tokens),
                    median_prefill_tokens=statistics.median(prefill_tokens),
                    median_decode_tps=decode_rows / (median_ms / 1000.0),
                    raw_latency_ms=latencies,
                    raw_prefill_tokens=prefill_tokens,
                )
            )

    payload = {
        "label": args.label,
        "commit": commit,
        "repo": str(repo),
        "version_has_token_budget": version_has_token_budget,
        "python": sys.version,
        "results": [result.to_dict() for result in results],
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _load_results(paths: Iterable[str]) -> list[dict]:
    rows: list[dict] = []
    for path in paths:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        rows.extend(payload["results"])
    return rows


def write_csv(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "label",
        "commit",
        "version_has_token_budget",
        "mode",
        "decode_rows",
        "median_ms",
        "mean_ms",
        "p95_ms",
        "min_ms",
        "max_ms",
        "mean_prefill_tokens",
        "median_prefill_tokens",
        "median_decode_tps",
        "max_num_batched_tokens",
        "prefill_step_size",
        "remaining_prompt_tokens",
        "prefill_token_ms",
        "decode_token_ms",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field) for field in fields})


def _svg_text(
    x, y, text, *, size=12, anchor="middle", weight="normal", color="#1f2937"
):
    escaped = str(text).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    return (
        f'<text x="{x:.1f}" y="{y:.1f}" font-family="Arial, sans-serif" '
        f'font-size="{size}" font-weight="{weight}" text-anchor="{anchor}" '
        f'fill="{color}">{escaped}</text>'
    )


def _nice_max(value: float) -> float:
    if value <= 0:
        return 1.0
    exp = math.floor(math.log10(value))
    frac = value / (10**exp)
    if frac <= 1:
        nice = 1
    elif frac <= 2:
        nice = 2
    elif frac <= 5:
        nice = 5
    else:
        nice = 10
    return nice * (10**exp)


def _draw_bar_panel(
    *,
    rows: list[dict],
    metric: str,
    title: str,
    y_label: str,
    x: int,
    y: int,
    width: int,
    height: int,
) -> list[str]:
    labels = []
    for row in rows:
        label = row["label"]
        if label not in labels:
            labels.append(label)
    decode_rows = sorted({int(row["decode_rows"]) for row in rows})
    by_key = {
        (row["label"], int(row["decode_rows"])): float(row[metric]) for row in rows
    }
    max_value = max(by_key.values()) if by_key else 1.0
    y_max = _nice_max(max_value * 1.15)
    colors = ["#2563eb", "#dc2626", "#059669", "#7c3aed"]
    left = x + 70
    top = y + 32
    plot_w = width - 95
    plot_h = height - 85
    bottom = top + plot_h
    group_w = plot_w / max(1, len(decode_rows))
    bar_w = min(38, group_w / (len(labels) + 1.4))
    parts = [
        f'<rect x="{x}" y="{y}" width="{width}" height="{height}" fill="#ffffff" stroke="#d1d5db"/>',
        _svg_text(x + width / 2, y + 22, title, size=15, weight="bold"),
        f'<line x1="{left}" y1="{top}" x2="{left}" y2="{bottom}" stroke="#374151"/>',
        f'<line x1="{left}" y1="{bottom}" x2="{left + plot_w}" y2="{bottom}" stroke="#374151"/>',
    ]
    for tick in range(5):
        value = y_max * tick / 4
        ty = bottom - (value / y_max) * plot_h
        parts.append(
            f'<line x1="{left - 4}" y1="{ty:.1f}" x2="{left + plot_w}" y2="{ty:.1f}" stroke="#e5e7eb"/>'
        )
        parts.append(_svg_text(left - 8, ty + 4, f"{value:.0f}", size=10, anchor="end"))
    for group_idx, decode in enumerate(decode_rows):
        group_center = left + group_w * (group_idx + 0.5)
        parts.append(_svg_text(group_center, bottom + 18, str(decode), size=11))
        for label_idx, label in enumerate(labels):
            value = by_key.get((label, decode), 0.0)
            bx = group_center - (len(labels) * bar_w) / 2 + label_idx * bar_w
            bh = (value / y_max) * plot_h
            by = bottom - bh
            parts.append(
                f'<rect x="{bx:.1f}" y="{by:.1f}" width="{bar_w - 4:.1f}" '
                f'height="{bh:.1f}" fill="{colors[label_idx % len(colors)]}"/>'
            )
            parts.append(
                _svg_text(bx + (bar_w - 4) / 2, by - 4, f"{value:.1f}", size=9)
            )
    parts.append(
        _svg_text(left + plot_w / 2, y + height - 12, "active decode rows", size=11)
    )
    parts.append(_svg_text(x + 15, top + plot_h / 2, y_label, size=11, anchor="middle"))
    legend_x = left + plot_w - 150
    legend_y = top + 4
    for idx, label in enumerate(labels):
        ly = legend_y + idx * 18
        parts.append(
            f'<rect x="{legend_x}" y="{ly}" width="11" height="11" fill="{colors[idx % len(colors)]}"/>'
        )
        parts.append(_svg_text(legend_x + 18, ly + 10, label, size=11, anchor="start"))
    return parts


def write_svg(rows: list[dict], path: Path) -> None:
    mixed = [row for row in rows if row["mode"] == "decode_with_prefill"]
    decode_only = [row for row in rows if row["mode"] == "decode_only"]
    width = 1040
    height = 720
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#f8fafc"/>',
        _svg_text(
            width / 2,
            28,
            "Continuous Batching Scheduler: Before vs After",
            size=20,
            weight="bold",
        ),
        _svg_text(
            width / 2,
            50,
            "Synthetic one-step benchmark: decode responses emitted after concurrent prefill work",
            size=12,
            color="#475569",
        ),
    ]
    parts.extend(
        _draw_bar_panel(
            rows=mixed,
            metric="median_ms",
            title="Mixed Decode + Prefill: Median _next() Latency",
            y_label="ms",
            x=24,
            y=76,
            width=992,
            height=285,
        )
    )
    parts.extend(
        _draw_bar_panel(
            rows=mixed,
            metric="mean_prefill_tokens",
            title="Prefill Tokens Scheduled In Same Iteration",
            y_label="tokens",
            x=24,
            y=386,
            width=992,
            height=285,
        )
    )
    if decode_only:
        before_after = sorted(
            decode_only, key=lambda row: (int(row["decode_rows"]), row["label"])
        )
        summary = ", ".join(
            f'{row["label"]} {int(row["decode_rows"])} rows: {float(row["median_ms"]):.2f} ms'
            for row in before_after[:4]
        )
        parts.append(
            _svg_text(width / 2, 700, f"Decode-only sanity sample: {summary}", size=11)
        )
    parts.append("</svg>")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(parts), encoding="utf-8")


def plot_results(args) -> None:
    rows = _load_results(args.inputs)
    if args.csv_out:
        write_csv(rows, Path(args.csv_out))
    if args.svg_out:
        write_svg(rows, Path(args.svg_out))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    run = subparsers.add_parser("run")
    run.add_argument("--repo", required=True)
    run.add_argument("--label", required=True)
    run.add_argument("--out", required=True)
    run.add_argument("--iterations", type=int, default=30)
    run.add_argument("--warmup", type=int, default=5)
    run.add_argument("--decode-rows", type=_parse_ints, default=[1, 8, 32, 64])
    run.add_argument(
        "--modes", nargs="+", default=["decode_only", "decode_with_prefill"]
    )
    run.add_argument("--max-num-batched-tokens", type=int, default=64)
    run.add_argument("--prefill-step-size", type=int, default=2048)
    run.add_argument("--remaining-prompt-tokens", type=int, default=8192)
    run.add_argument("--prefill-token-ms", type=float, default=0.02)
    run.add_argument("--decode-token-ms", type=float, default=0.005)
    run.set_defaults(func=run_benchmark)

    plot = subparsers.add_parser("plot")
    plot.add_argument("--inputs", nargs="+", required=True)
    plot.add_argument("--csv-out")
    plot.add_argument("--svg-out")
    plot.set_defaults(func=plot_results)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    args.func(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
