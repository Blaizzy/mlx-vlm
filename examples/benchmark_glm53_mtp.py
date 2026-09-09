"""Compare GLM-5.3-Flash FP8 MTP with ordinary decoding, including token parity.

python examples/benchmark_glm53_mtp.py --model /path/to/GLM-5.3-Flash-FP8 \
    --draft-model /path/to/GLM-5.3-Flash-MTP-FP8 --output results.json
"""

import argparse
import json
import time
from pathlib import Path

import mlx.core as mx

from mlx_vlm.generate.ar import _make_cache
from mlx_vlm.generate.common import wired_limit
from mlx_vlm.speculative.drafters import load_drafter, validate_drafter_compatibility
from mlx_vlm.speculative.mtp import mtp_rounds
from mlx_vlm.speculative.stats import speculative_stats_snapshot
from mlx_vlm.utils import load


def run(target, draft, tokens, padding, max_tokens, block_size):
    with wired_limit([target, draft]):
        return _run(target, draft, tokens, padding, max_tokens, block_size)


def _run(target, draft, tokens, padding, max_tokens, block_size):
    cache = _make_cache(target, padding)
    start = time.perf_counter()
    output = target(tokens, cache=cache, return_hidden=True, attention_mask=tokens != 0)
    first = mx.argmax(output.logits[:, -1], axis=-1)
    mx.eval(first)
    prefill = time.perf_counter() - start
    result = [[value] for value in first.tolist()]
    start = time.perf_counter()
    if draft is None:
        token = first
        for _ in range(max_tokens - 1):
            output = target(token[:, None], cache=cache)
            token = mx.argmax(output.logits[:, -1], axis=-1)
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
            sampler=None,
            greedy_sampling=True,
            draft_block_size=block_size,
        ):
            for row, value in zip(result, values):
                if value is not None:
                    row.append(value)
    mx.synchronize()
    elapsed = time.perf_counter() - start
    return {
        "tokens": result,
        "prefill_seconds": prefill,
        "decode_seconds": elapsed,
        "tokens_per_second": sum(len(row) - 1 for row in result) / elapsed,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--draft-model", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--max-tokens", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-prompts", type=int, default=4)
    parser.add_argument("--block-sizes", type=int, nargs="+", default=[2, 3, 5])
    args = parser.parse_args()
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
    ]
    report = {"model": args.model, "draft_model": args.draft_model, "runs": []}
    prompts = prompts[: args.num_prompts]
    print(
        f"Device: {mx.default_device()}; weights: {mx.get_active_memory() / 1e9:.1f} GB",
        flush=True,
    )
    for start in range(0, len(prompts), args.batch_size):
        group = prompts[start : start + args.batch_size]
        encoded = [
            tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}],
                add_generation_prompt=True,
                tokenize=True,
                reasoning_effort="low",
            )
            for prompt in group
        ]
        encoded = [row["input_ids"] if hasattr(row, "keys") else row for row in encoded]
        width = max(map(len, encoded))
        padding = [width - len(row) for row in encoded]
        tokens = mx.array([[0] * pad + row for row, pad in zip(encoded, padding)])
        baseline = run(target, None, tokens, padding, args.max_tokens, 2)
        print(f"AR: {baseline['tokens_per_second']:.2f} tok/s", flush=True)
        for block_size in args.block_sizes:
            before = speculative_stats_snapshot(draft)
            actual = run(target, draft, tokens, padding, args.max_tokens, block_size)
            stats = tuple(
                a - b for a, b in zip(speculative_stats_snapshot(draft), before)
            )
            parity = actual["tokens"] == baseline["tokens"]
            report["runs"].append(
                {
                    "prompts": group,
                    "block_size": block_size,
                    "parity": parity,
                    "baseline": baseline,
                    "mtp": actual,
                    "stats": stats,
                    "speedup": baseline["decode_seconds"] / actual["decode_seconds"],
                    "text": [tokenizer.decode(row) for row in actual["tokens"]],
                }
            )
            Path(args.output).write_text(json.dumps(report, indent=2))
            print(
                f"MTP block {block_size}: {actual['tokens_per_second']:.2f} tok/s; parity={parity}; stats={stats}",
                flush=True,
            )
            if not parity:
                raise RuntimeError(
                    "MTP tokens differ from ordinary decode; see the report."
                )
    report["peak_memory_gb"] = mx.get_peak_memory() / 1e9
    Path(args.output).write_text(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
