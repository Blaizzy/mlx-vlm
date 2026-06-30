#!/usr/bin/env python3
"""MiniMax M3 continuous-batching scheduler benchmark.

This uses the real MiniMax M3 model path and times a scheduler tick when decode
rows are already active and a new long prompt is waiting for prefill.
"""

from __future__ import annotations

import argparse
import csv
import inspect
import json
import math
import statistics
import subprocess
import sys
import time
from pathlib import Path


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


def _percentile(values: list[float], pct: float) -> float:
    ordered = sorted(values)
    if not ordered:
        return 0.0
    if len(ordered) == 1:
        return ordered[0]
    rank = (len(ordered) - 1) * pct / 100.0
    lo = math.floor(rank)
    hi = math.ceil(rank)
    if lo == hi:
        return ordered[lo]
    return ordered[lo] + (ordered[hi] - ordered[lo]) * (rank - lo)


def _sync(mx):
    mx.synchronize()


def _clear_mlx(mx):
    mx.clear_cache()
    if hasattr(mx, "reset_peak_memory"):
        mx.reset_peak_memory()


def _prepare_rows(model, processor, prompts, ar, prompt_utils, utils):
    formatted = [
        prompt_utils.apply_chat_template(processor, model.config, prompt, num_images=0)
        for prompt in prompts
    ]
    add_special_tokens = (
        getattr(processor, "chat_template", None) is None
        if model.config.model_type in ["gemma3", "gemma3n", "gemma4", "gemma4_unified"]
        else True
    )
    inputs = utils.prepare_inputs(
        processor,
        images=None,
        audio=None,
        prompts=formatted,
        image_token_index=getattr(model.config, "image_token_index", None),
        add_special_tokens=add_special_tokens,
        pad_to_uniform_size=False,
    )
    input_ids = inputs["input_ids"]
    mask = inputs.get("attention_mask")
    data_kwargs = {
        k: v
        for k, v in inputs.items()
        if k not in ["input_ids", "pixel_values", "attention_mask"]
    }
    embedding_output = model.get_input_embeddings(
        input_ids,
        inputs.get("pixel_values"),
        mask=mask,
        **data_kwargs,
    )
    gen_kwargs = {**data_kwargs, **embedding_output.to_dict()}
    return input_ids.tolist(), ar._split_prompt_kwargs_per_row(gen_kwargs, len(prompts))


def _make_generator(ar, model, processor, decode_rows, args):
    kwargs = {
        "max_tokens": args.max_generated_tokens,
        "completion_batch_size": decode_rows + 1,
        "prefill_batch_size": args.prefill_batch_size or max(1, decode_rows),
        "prefill_step_size": args.prefill_step_size,
        "compute_logprobs": False,
    }
    sig = inspect.signature(ar.BatchGenerator)
    if "max_num_batched_tokens" in sig.parameters:
        kwargs["max_num_batched_tokens"] = args.max_num_batched_tokens
    return ar.BatchGenerator(model.language_model, processor, **kwargs)


def _prime_active_decode(
    ar,
    mx,
    model,
    processor,
    active_input_ids,
    active_prompt_kwargs,
    decode_rows,
    args,
):
    gen = _make_generator(ar, model, processor, decode_rows, args)
    uids = gen.insert(
        active_input_ids,
        max_tokens=[args.max_generated_tokens] * decode_rows,
        prompt_kwargs=active_prompt_kwargs,
    )
    deadline = time.perf_counter() + args.setup_timeout_s
    while len(gen._generation_batch) < decode_rows:
        if time.perf_counter() > deadline:
            gen.close()
            raise TimeoutError("timed out while priming active decode batch")
        gen.next()
        _sync(mx)
    gen.prefill_batch_size = 1
    _sync(mx)
    return gen, set(uids)


def _drop_pending_prompt(gen, active_uids, mx):
    if getattr(gen, "_prompt_batch", None) is not None:
        gen._prompt_batch.uids = []
        gen._prompt_batch.prompt_cache = []
        gen._prompt_batch = None
    gen._unprocessed_sequences = [
        item for item in gen._unprocessed_sequences if item[0] in active_uids
    ]
    if len(gen._generation_batch) > 0:
        keep = [
            i for i, uid in enumerate(gen._generation_batch.uids) if uid in active_uids
        ]
        if len(keep) < len(gen._generation_batch.uids):
            gen._generation_batch.filter(keep)
    mx.clear_cache()


def _remaining_prompt_tokens(prompt_batch) -> int:
    remaining = getattr(prompt_batch, "remaining_prompt_tokens", None)
    if callable(remaining):
        return int(remaining())
    inputs_embeds = getattr(prompt_batch, "_inputs_embeds", None)
    if inputs_embeds is not None:
        return int(inputs_embeds.shape[1])
    input_ids = getattr(prompt_batch, "_input_ids", None)
    if input_ids is not None:
        return int(input_ids.shape[1])
    return 0


def _run_scenario(
    *,
    ar,
    mx,
    model,
    processor,
    args,
    mode: str,
    decode_rows: int,
    active_input_ids,
    active_prompt_kwargs,
    pending_input_ids,
    pending_prompt_kwargs,
):
    gen, active_uids = _prime_active_decode(
        ar,
        mx,
        model,
        processor,
        active_input_ids[:decode_rows],
        active_prompt_kwargs[:decode_rows],
        decode_rows,
        args,
    )
    latencies = []
    prefill_tokens = []
    decode_counts = []
    total_runs = args.warmup + args.iterations
    pending_len = len(pending_input_ids[0])

    for idx in range(total_runs):
        pending_uid = None
        if mode == "decode_with_prefill":
            pending_uid = gen.insert(
                pending_input_ids,
                max_tokens=[1],
                prompt_kwargs=pending_prompt_kwargs,
            )[0]

        _sync(mx)
        start = time.perf_counter()
        _, generation_responses = gen.next()
        _sync(mx)
        elapsed_ms = (time.perf_counter() - start) * 1000.0

        processed = 0
        if mode == "decode_with_prefill":
            prompt_batch = getattr(gen, "_prompt_batch", None)
            if prompt_batch is not None and pending_uid in prompt_batch.uids:
                processed = pending_len - _remaining_prompt_tokens(prompt_batch)
            else:
                processed = pending_len

        if idx >= args.warmup:
            latencies.append(elapsed_ms)
            prefill_tokens.append(processed)
            decode_counts.append(len(generation_responses))

        if mode == "decode_with_prefill":
            _drop_pending_prompt(gen, active_uids, mx)

        if len(gen._generation_batch) != decode_rows:
            gen.close()
            raise RuntimeError(
                f"active decode batch changed size: expected {decode_rows}, "
                f"got {len(gen._generation_batch)}"
            )

    gen.close()
    return {
        "mode": mode,
        "decode_rows": decode_rows,
        "latency_ms": latencies,
        "prefill_tokens": prefill_tokens,
        "decode_counts": decode_counts,
        "median_ms": statistics.median(latencies),
        "mean_ms": statistics.mean(latencies),
        "p95_ms": _percentile(latencies, 95),
        "min_ms": min(latencies),
        "max_ms": max(latencies),
        "median_prefill_tokens": statistics.median(prefill_tokens),
        "mean_prefill_tokens": statistics.mean(prefill_tokens),
        "median_decode_responses": statistics.median(decode_counts),
    }


def _pending_remaining(gen, pending_uid: int | None, pending_len: int) -> int:
    if pending_uid is None:
        return 0
    for uid, *_ in getattr(gen, "_unprocessed_sequences", []):
        if uid == pending_uid:
            return pending_len
    prompt_batch = getattr(gen, "_prompt_batch", None)
    if prompt_batch is not None and pending_uid in getattr(prompt_batch, "uids", []):
        return _remaining_prompt_tokens(prompt_batch)
    generation_batch = getattr(gen, "_generation_batch", None)
    if generation_batch is not None and pending_uid in getattr(
        generation_batch, "uids", []
    ):
        return 0
    return 0


def _run_stream_scenario(
    *,
    ar,
    mx,
    model,
    processor,
    args,
    mode: str,
    decode_rows: int,
    active_input_ids,
    active_prompt_kwargs,
    pending_input_ids,
    pending_prompt_kwargs,
):
    gen, active_uids = _prime_active_decode(
        ar,
        mx,
        model,
        processor,
        active_input_ids[:decode_rows],
        active_prompt_kwargs[:decode_rows],
        decode_rows,
        args,
    )
    active_counts = {uid: 0 for uid in active_uids}
    pending_uid = None
    pending_len = len(pending_input_ids[0])
    if mode == "decode_with_prefill":
        pending_uid = gen.insert(
            pending_input_ids,
            max_tokens=[1],
            prompt_kwargs=pending_prompt_kwargs,
        )[0]

    latencies = []
    prefill_tokens = []
    decode_counts = []
    start_total = time.perf_counter()
    while min(active_counts.values()) < args.max_generated_tokens:
        remaining_before = _pending_remaining(gen, pending_uid, pending_len)
        _sync(mx)
        start = time.perf_counter()
        _, generation_responses = gen.next()
        _sync(mx)
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        remaining_after = _pending_remaining(gen, pending_uid, pending_len)

        active_responses = 0
        for response in generation_responses:
            if response.uid in active_counts:
                active_counts[response.uid] += 1
                active_responses += 1
        latencies.append(elapsed_ms)
        prefill_tokens.append(max(0, remaining_before - remaining_after))
        decode_counts.append(active_responses)

        if not gen.has_work and min(active_counts.values()) < args.max_generated_tokens:
            gen.close()
            raise RuntimeError("generator stopped before active rows reached target")

    total_ms = (time.perf_counter() - start_total) * 1000.0
    gen.close()
    total_active_tokens = sum(active_counts.values())
    nonzero_prefill = [tokens for tokens in prefill_tokens if tokens > 0]
    return {
        "mode": mode,
        "decode_rows": decode_rows,
        "target_tokens_per_row": args.max_generated_tokens,
        "active_tokens": total_active_tokens,
        "total_ms": total_ms,
        "active_decode_tps": total_active_tokens / (total_ms / 1000.0),
        "tick_count": len(latencies),
        "first_tick_ms": latencies[0],
        "median_tick_ms": statistics.median(latencies),
        "mean_tick_ms": statistics.mean(latencies),
        "p95_tick_ms": _percentile(latencies, 95),
        "max_tick_ms": max(latencies),
        "prefill_ticks": len(nonzero_prefill),
        "total_prefill_tokens": sum(prefill_tokens),
        "median_prefill_tokens_per_prefill_tick": (
            statistics.median(nonzero_prefill) if nonzero_prefill else 0
        ),
        "latency_ms": latencies,
        "prefill_tokens": prefill_tokens,
        "decode_counts": decode_counts,
        "active_counts": active_counts,
    }


def run(args) -> None:
    repo = Path(args.repo).resolve()
    sys.path.insert(0, str(repo))

    import mlx.core as mx

    from mlx_vlm import load, prompt_utils, utils
    from mlx_vlm.generate import ar

    _clear_mlx(mx)
    commit = _repo_commit(repo)
    load_start = time.perf_counter()
    model, processor = load(args.model_path, lazy=args.lazy)
    _sync(mx)
    load_s = time.perf_counter() - load_start

    active_prompt = args.active_prompt
    pending_prompt = (args.pending_prompt + " ") * args.pending_repeat
    max_decode_rows = max(args.decode_rows)

    active_input_ids, active_prompt_kwargs = _prepare_rows(
        model,
        processor,
        [active_prompt] * max_decode_rows,
        ar,
        prompt_utils,
        utils,
    )
    pending_input_ids, pending_prompt_kwargs = _prepare_rows(
        model,
        processor,
        [pending_prompt],
        ar,
        prompt_utils,
        utils,
    )
    _sync(mx)

    results = []
    for mode in args.modes:
        for decode_rows in args.decode_rows:
            scenario = _run_scenario(
                ar=ar,
                mx=mx,
                model=model,
                processor=processor,
                args=args,
                mode=mode,
                decode_rows=decode_rows,
                active_input_ids=active_input_ids,
                active_prompt_kwargs=active_prompt_kwargs,
                pending_input_ids=pending_input_ids,
                pending_prompt_kwargs=pending_prompt_kwargs,
            )
            scenario.update(
                {
                    "label": args.label,
                    "commit": commit,
                    "repo": str(repo),
                    "model_path": args.model_path,
                    "version_has_token_budget": hasattr(
                        ar.BatchGenerator, "_token_budget_after_decode"
                    ),
                    "max_num_batched_tokens": args.max_num_batched_tokens,
                    "prefill_step_size": args.prefill_step_size,
                    "active_prompt_tokens": len(active_input_ids[0]),
                    "pending_prompt_tokens": len(pending_input_ids[0]),
                    "iterations": args.iterations,
                    "warmup": args.warmup,
                    "load_s": load_s,
                    "peak_memory_gb": mx.get_peak_memory() / 1e9,
                }
            )
            results.append(scenario)
            print(
                f"{args.label} {mode} rows={decode_rows}: "
                f"median={scenario['median_ms']:.2f} ms "
                f"prefill={scenario['median_prefill_tokens']:.0f}"
            )

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(
            {
                "label": args.label,
                "commit": commit,
                "repo": str(repo),
                "model_path": args.model_path,
                "load_s": load_s,
                "results": results,
            },
            indent=2,
        ),
        encoding="utf-8",
    )


def run_stream(args) -> None:
    repo = Path(args.repo).resolve()
    sys.path.insert(0, str(repo))

    import mlx.core as mx

    from mlx_vlm import load, prompt_utils, utils
    from mlx_vlm.generate import ar

    _clear_mlx(mx)
    commit = _repo_commit(repo)
    load_start = time.perf_counter()
    model, processor = load(args.model_path, lazy=args.lazy)
    _sync(mx)
    load_s = time.perf_counter() - load_start

    active_prompt = args.active_prompt
    pending_prompt = (args.pending_prompt + " ") * args.pending_repeat
    max_decode_rows = max(args.decode_rows)

    active_input_ids, active_prompt_kwargs = _prepare_rows(
        model,
        processor,
        [active_prompt] * max_decode_rows,
        ar,
        prompt_utils,
        utils,
    )
    pending_input_ids, pending_prompt_kwargs = _prepare_rows(
        model,
        processor,
        [pending_prompt],
        ar,
        prompt_utils,
        utils,
    )
    _sync(mx)

    results = []
    for mode in args.modes:
        for decode_rows in args.decode_rows:
            scenario = _run_stream_scenario(
                ar=ar,
                mx=mx,
                model=model,
                processor=processor,
                args=args,
                mode=mode,
                decode_rows=decode_rows,
                active_input_ids=active_input_ids,
                active_prompt_kwargs=active_prompt_kwargs,
                pending_input_ids=pending_input_ids,
                pending_prompt_kwargs=pending_prompt_kwargs,
            )
            scenario.update(
                {
                    "label": args.label,
                    "commit": commit,
                    "repo": str(repo),
                    "model_path": args.model_path,
                    "version_has_token_budget": hasattr(
                        ar.BatchGenerator, "_token_budget_after_decode"
                    ),
                    "max_num_batched_tokens": args.max_num_batched_tokens,
                    "prefill_step_size": args.prefill_step_size,
                    "active_prompt_tokens": len(active_input_ids[0]),
                    "pending_prompt_tokens": len(pending_input_ids[0]),
                    "load_s": load_s,
                    "peak_memory_gb": mx.get_peak_memory() / 1e9,
                }
            )
            results.append(scenario)
            print(
                f"{args.label} {mode} rows={decode_rows}: "
                f"tokens={scenario['active_tokens']} "
                f"total={scenario['total_ms']:.2f} ms "
                f"tps={scenario['active_decode_tps']:.2f} "
                f"max_tick={scenario['max_tick_ms']:.2f} ms"
            )

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(
            {
                "label": args.label,
                "commit": commit,
                "repo": str(repo),
                "model_path": args.model_path,
                "load_s": load_s,
                "results": results,
            },
            indent=2,
        ),
        encoding="utf-8",
    )


def _load_rows(paths):
    rows = []
    for path in paths:
        rows.extend(json.loads(Path(path).read_text(encoding="utf-8"))["results"])
    return rows


def write_csv(rows, path: Path) -> None:
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
        "median_prefill_tokens",
        "mean_prefill_tokens",
        "median_decode_responses",
        "max_num_batched_tokens",
        "prefill_step_size",
        "active_prompt_tokens",
        "pending_prompt_tokens",
        "iterations",
        "warmup",
        "load_s",
        "peak_memory_gb",
        "model_path",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field) for field in fields})


def _text(x, y, value, *, size=12, anchor="middle", weight="normal", color="#1f2937"):
    escaped = str(value).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    return (
        f'<text x="{x:.1f}" y="{y:.1f}" font-family="Arial, sans-serif" '
        f'font-size="{size}" font-weight="{weight}" text-anchor="{anchor}" '
        f'fill="{color}">{escaped}</text>'
    )


def _nice_max(value):
    if value <= 0:
        return 1
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


def _panel(rows, metric, title, y_label, x, y, width, height):
    labels = []
    for row in rows:
        if row["label"] not in labels:
            labels.append(row["label"])
    decode_rows = sorted({int(row["decode_rows"]) for row in rows})
    values = {
        (row["label"], int(row["decode_rows"])): float(row[metric]) for row in rows
    }
    y_max = _nice_max(max(values.values()) * 1.15 if values else 1)
    colors = ["#2563eb", "#dc2626", "#059669", "#7c3aed", "#ea580c"]
    left = x + 72
    top = y + 34
    plot_w = width - 100
    plot_h = height - 88
    bottom = top + plot_h
    group_w = plot_w / max(1, len(decode_rows))
    bar_w = min(42, group_w / (len(labels) + 1.3))
    parts = [
        f'<rect x="{x}" y="{y}" width="{width}" height="{height}" fill="#fff" stroke="#d1d5db"/>',
        _text(x + width / 2, y + 22, title, size=15, weight="bold"),
        f'<line x1="{left}" y1="{top}" x2="{left}" y2="{bottom}" stroke="#374151"/>',
        f'<line x1="{left}" y1="{bottom}" x2="{left + plot_w}" y2="{bottom}" stroke="#374151"/>',
    ]
    for tick in range(5):
        value = y_max * tick / 4
        ty = bottom - (value / y_max) * plot_h
        parts.append(
            f'<line x1="{left - 4}" y1="{ty:.1f}" x2="{left + plot_w}" y2="{ty:.1f}" stroke="#e5e7eb"/>'
        )
        parts.append(_text(left - 8, ty + 4, f"{value:.0f}", size=10, anchor="end"))
    for group_idx, decode in enumerate(decode_rows):
        group_center = left + group_w * (group_idx + 0.5)
        parts.append(_text(group_center, bottom + 18, decode, size=11))
        for label_idx, label in enumerate(labels):
            value = values.get((label, decode), 0.0)
            bx = group_center - (len(labels) * bar_w) / 2 + label_idx * bar_w
            bh = (value / y_max) * plot_h
            by = bottom - bh
            parts.append(
                f'<rect x="{bx:.1f}" y="{by:.1f}" width="{bar_w - 4:.1f}" '
                f'height="{bh:.1f}" fill="{colors[label_idx % len(colors)]}"/>'
            )
            parts.append(_text(bx + (bar_w - 4) / 2, by - 4, f"{value:.1f}", size=9))
    parts.append(
        _text(left + plot_w / 2, y + height - 12, "active decode rows", size=11)
    )
    parts.append(_text(x + 15, top + plot_h / 2, y_label, size=11))
    legend_x = left + plot_w - 170
    legend_y = top + 5
    for idx, label in enumerate(labels):
        ly = legend_y + idx * 18
        parts.append(
            f'<rect x="{legend_x}" y="{ly}" width="11" height="11" fill="{colors[idx % len(colors)]}"/>'
        )
        parts.append(_text(legend_x + 18, ly + 10, label, size=11, anchor="start"))
    return parts


def write_svg(rows, path: Path) -> None:
    mixed = [row for row in rows if row["mode"] == "decode_with_prefill"]
    width = 1040
    height = 690
    prompt_tokens = mixed[0].get("pending_prompt_tokens", "") if mixed else ""
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#f8fafc"/>',
        _text(
            width / 2,
            28,
            "MiniMax M3 Continuous Batching: Before vs After",
            size=20,
            weight="bold",
        ),
        _text(
            width / 2,
            50,
            f"Real M3 model, pending prompt {prompt_tokens} tokens, one measured scheduler tick",
            size=12,
            color="#475569",
        ),
    ]
    parts.extend(
        _panel(
            mixed,
            "median_ms",
            "Mixed Decode + Long Prefill: Median _next() Latency",
            "ms",
            24,
            76,
            992,
            275,
        )
    )
    parts.extend(
        _panel(
            mixed,
            "median_prefill_tokens",
            "Long-Prompt Prefill Tokens Scheduled During The Decode Tick",
            "tokens",
            24,
            382,
            992,
            275,
        )
    )
    parts.append("</svg>")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(parts), encoding="utf-8")


def plot(args) -> None:
    rows = _load_rows(args.inputs)
    if args.csv_out:
        write_csv(rows, Path(args.csv_out))
    if args.svg_out:
        write_svg(rows, Path(args.svg_out))


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    run_parser = subparsers.add_parser("run")
    run_parser.add_argument("--repo", required=True)
    run_parser.add_argument("--label", required=True)
    run_parser.add_argument("--model-path", default="~/MiniMax-M3-4bit")
    run_parser.add_argument("--out", required=True)
    run_parser.add_argument("--decode-rows", type=_parse_ints, default=[1])
    run_parser.add_argument(
        "--modes", nargs="+", default=["decode_only", "decode_with_prefill"]
    )
    run_parser.add_argument("--iterations", type=int, default=3)
    run_parser.add_argument("--warmup", type=int, default=1)
    run_parser.add_argument("--prefill-step-size", type=int, default=2048)
    run_parser.add_argument("--prefill-batch-size", type=int)
    run_parser.add_argument("--max-num-batched-tokens", type=int, default=64)
    run_parser.add_argument("--max-generated-tokens", type=int, default=64)
    run_parser.add_argument("--setup-timeout-s", type=float, default=900)
    run_parser.add_argument("--lazy", action="store_true")
    run_parser.add_argument(
        "--active-prompt",
        default="Write one short sentence about MiniMax sparse attention.",
    )
    run_parser.add_argument(
        "--pending-prompt",
        default=(
            "MiniMax M3 continuous batching benchmark: explain sparse attention, "
            "prefill scheduling, decode fairness, and latency tradeoffs."
        ),
    )
    run_parser.add_argument("--pending-repeat", type=int, default=700)
    run_parser.set_defaults(func=run)

    stream_parser = subparsers.add_parser("run-stream")
    stream_parser.add_argument("--repo", required=True)
    stream_parser.add_argument("--label", required=True)
    stream_parser.add_argument("--model-path", default="~/MiniMax-M3-4bit")
    stream_parser.add_argument("--out", required=True)
    stream_parser.add_argument("--decode-rows", type=_parse_ints, default=[1])
    stream_parser.add_argument("--modes", nargs="+", default=["decode_with_prefill"])
    stream_parser.add_argument("--prefill-step-size", type=int, default=2048)
    stream_parser.add_argument("--prefill-batch-size", type=int)
    stream_parser.add_argument("--max-num-batched-tokens", type=int, default=64)
    stream_parser.add_argument("--max-generated-tokens", type=int, default=128)
    stream_parser.add_argument("--setup-timeout-s", type=float, default=900)
    stream_parser.add_argument("--lazy", action="store_true")
    stream_parser.add_argument(
        "--active-prompt",
        default="Write one short sentence about MiniMax sparse attention.",
    )
    stream_parser.add_argument(
        "--pending-prompt",
        default=(
            "MiniMax M3 continuous batching benchmark: explain sparse attention, "
            "prefill scheduling, decode fairness, and latency tradeoffs."
        ),
    )
    stream_parser.add_argument("--pending-repeat", type=int, default=100)
    stream_parser.set_defaults(func=run_stream)

    plot_parser = subparsers.add_parser("plot")
    plot_parser.add_argument("--inputs", nargs="+", required=True)
    plot_parser.add_argument("--csv-out")
    plot_parser.add_argument("--svg-out")
    plot_parser.set_defaults(func=plot)
    return parser


def main(argv=None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if hasattr(args, "model_path"):
        args.model_path = str(Path(args.model_path).expanduser())
    args.func(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
