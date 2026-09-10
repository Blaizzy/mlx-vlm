"""Benchmark native MTP against AR with fixed, reproducible target sampling.

The optional stage profiler evaluates phase boundaries and changes execution
scheduling. Profiled timings are diagnostic; normal runs measure throughput.
"""

import argparse
import json
import time
from collections import defaultdict
from pathlib import Path

import mlx.core as mx

from mlx_vlm.generate.ar import _make_cache, _PositionedTargetSampler
from mlx_vlm.generate.common import wired_limit
from mlx_vlm.speculative.cache_state import SpeculativePrefill
from mlx_vlm.speculative.drafters import load_drafter, validate_drafter_compatibility
from mlx_vlm.speculative.mtp import mtp_rounds
from mlx_vlm.speculative.stats import speculative_stats_snapshot
from mlx_vlm.utils import load


class PhaseProfiler:
    def __init__(self):
        self.seconds = defaultdict(float)
        self.calls = defaultdict(int)
        self.previous = None

    def __call__(self, phase, arrays):
        mx.eval(arrays)
        mx.synchronize()
        now = time.perf_counter()
        if phase != "start":
            self.seconds[phase] += now - self.previous
            self.calls[phase] += 1
        self.previous = now

    def report(self):
        return {
            phase: {
                "seconds": seconds,
                "calls": self.calls[phase],
                "mean_ms": seconds * 1000 / self.calls[phase],
            }
            for phase, seconds in self.seconds.items()
        }


def run(
    target,
    draft,
    tokens,
    padding,
    max_tokens,
    block_size,
    temperature=0,
    profile=False,
    prefill_step_size=512,
):
    sampler = _PositionedTargetSampler(temperature=temperature, top_p=0.9, seed=123)
    rows = list(range(tokens.shape[0]))

    def sample(logits, position):
        if temperature == 0:
            return mx.argmax(logits, axis=-1)
        logprobs = logits - mx.logsumexp(logits, axis=-1, keepdims=True)
        return sampler.sample_target(
            logprobs, row_ids=rows, positions=[position] * len(rows)
        )

    with wired_limit([target, draft]):
        for name in ("_position_ids", "_rope_deltas"):
            if hasattr(target, name):
                setattr(target, name, None)
        cache = _make_cache(target, padding)
        profiler = PhaseProfiler() if profile else None
        mx.reset_peak_memory()
        start = time.perf_counter()
        prefill_state = SpeculativePrefill("mtp", draft, tokens)
        if draft is not None:
            prefill_state.start(target, cache, draft)
        for offset in range(0, tokens.shape[1] - 1, prefill_step_size):
            chunk = tokens[
                :, offset : min(offset + prefill_step_size, tokens.shape[1] - 1)
            ]
            mask = (
                mx.arange(offset, offset + chunk.shape[1])[None]
                >= mx.array(padding)[:, None]
            )
            output = target(
                chunk, cache=cache, return_hidden=draft is not None, attention_mask=mask
            )
            prefill_state.append(output)
            mx.eval([c.state for c in cache])
        output = target(tokens[:, -1:], cache=cache, return_hidden=draft is not None)
        first = sample(output.logits[:, -1], 0)
        prefill_state.finish(output, first)
        mx.eval(first)
        if draft is not None:
            mx.eval(prefill_state.state.seed.token, prefill_state.state.seed.hidden)
        prefill = time.perf_counter() - start
        result = [[value] for value in first.tolist()]
        start = time.perf_counter()
        if draft is None:
            token = first
            for position in range(1, max_tokens):
                output = target(token[:, None], cache=cache)
                token = sample(output.logits[:, -1], position)
                for row, value in zip(result, token.tolist()):
                    row.append(value)
        else:
            for values, _ in mtp_rounds(
                target,
                draft,
                cache,
                output.hidden_states[-1],
                prompt_tokens=tokens,
                first_bonus=first,
                max_tokens=max_tokens,
                sampler=sampler,
                row_ids=rows,
                greedy_sampling=temperature == 0,
                draft_block_size=block_size,
                phase_observer=profiler,
                state=prefill_state.state,
            ):
                for row, value in zip(result, values):
                    if value is not None:
                        row.append(value)
        mx.synchronize()
        elapsed = time.perf_counter() - start
        report = {
            "tokens": result,
            "prefill_seconds": prefill,
            "decode_seconds": elapsed,
            "tokens_per_second": sum(len(row) - 1 for row in result) / elapsed,
            "peak_memory_gb": mx.get_peak_memory() / 1e9,
        }
        if profiler:
            report["phases"] = profiler.report()
        return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--draft-model", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--batch-size", type=int, nargs="+", default=[1])
    parser.add_argument("--num-prompts", type=int, default=2)
    parser.add_argument("--context-tokens", type=int, nargs="+", default=[0])
    parser.add_argument("--temperatures", type=float, nargs="+", default=[0])
    parser.add_argument("--block-sizes", type=int, nargs="+", default=[2, 3])
    parser.add_argument("--profile", action="store_true")
    parser.add_argument("--prefill-step-size", type=int, default=512)
    args = parser.parse_args()
    if (
        args.max_tokens < 2
        or any(n < 1 for n in args.batch_size)
        or args.prefill_step_size < 1
    ):
        parser.error(
            "Use at least two generated tokens and positive batch/chunk sizes."
        )
    print("Loading target and MTP weights", flush=True)
    model, processor = load(args.model)
    draft, kind = load_drafter(args.draft_model)
    validate_drafter_compatibility(model, draft, kind)
    target = model.language_model
    tokenizer = getattr(processor, "tokenizer", processor)
    prompts = [
        "Explain why the sky is blue in three sentences.",
        "Write a Python function that returns the Fibonacci sequence.",
        "List the numbers from one to twenty, separated by commas.",
        "What is the difference between a list and a tuple in Python?",
    ][: args.num_prompts]
    report = {
        "model": args.model,
        "draft_model": args.draft_model,
        "settings": vars(args),
        "mlx_version": mx.__version__,
        "runs": [],
    }
    print(
        f"Device: {mx.default_device()}; weights: {mx.get_active_memory() / 1e9:.1f} GB",
        flush=True,
    )
    for context in args.context_tokens:
        encoded = []
        for prompt in prompts:
            reference = "Reference notes: A library stores books. Its index maps titles to shelves.\n"
            content = (
                reference * (context // 15)
                + "\nAnswer the following request:\n"
                + prompt
                if context
                else prompt
            )
            values = tokenizer.apply_chat_template(
                [{"role": "user", "content": content}],
                add_generation_prompt=True,
                tokenize=True,
                reasoning_effort="low",
            )
            encoded.append(values["input_ids"] if hasattr(values, "keys") else values)
        for batch_size in args.batch_size:
            for offset in range(0, len(prompts), batch_size):
                group = encoded[offset : offset + batch_size]
                width = max(map(len, group))
                padding = [width - len(row) for row in group]
                tokens = mx.array([[0] * pad + row for row, pad in zip(group, padding)])
                for temperature in args.temperatures:
                    baseline = run(
                        target,
                        None,
                        tokens,
                        padding,
                        args.max_tokens,
                        2,
                        temperature,
                        prefill_step_size=args.prefill_step_size,
                    )
                    print(
                        f"AR B={len(group)} context={width} T={temperature}: {baseline['tokens_per_second']:.2f} tok/s",
                        flush=True,
                    )
                    for block_size in args.block_sizes:
                        before = speculative_stats_snapshot(draft)
                        actual = run(
                            target,
                            draft,
                            tokens,
                            padding,
                            args.max_tokens,
                            block_size,
                            temperature,
                            args.profile,
                            args.prefill_step_size,
                        )
                        stats = [
                            a - b
                            for a, b in zip(speculative_stats_snapshot(draft), before)
                        ]
                        parity = actual["tokens"] == baseline["tokens"]
                        report["runs"].append(
                            {
                                "prompts": prompts[offset : offset + batch_size],
                                "prompt_lengths": [len(row) for row in group],
                                "temperature": temperature,
                                "block_size": block_size,
                                "parity": parity,
                                "baseline": baseline,
                                "mtp": actual,
                                "stats": stats,
                                "speedup": baseline["decode_seconds"]
                                / actual["decode_seconds"],
                                "text": [
                                    tokenizer.decode(row) for row in actual["tokens"]
                                ],
                            }
                        )
                        Path(args.output).write_text(json.dumps(report, indent=2))
                        print(
                            f"MTP block {block_size}: {actual['tokens_per_second']:.2f} tok/s; parity={parity}; stats={stats}",
                            flush=True,
                        )
    if not all(run["parity"] for run in report["runs"]):
        raise SystemExit("Token differences found; inspect the saved report.")


if __name__ == "__main__":
    main()
