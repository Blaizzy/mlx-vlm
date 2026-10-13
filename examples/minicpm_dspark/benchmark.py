"""Reproduce MiniCPM/DSpark generation on the opt-in SIMD verifier.

Run with PYTHONPATH=. from the repository root. No synchronization is added
inside decoding; complete-round timing uses the existing host completion point.
"""

import argparse
import importlib
import json
import statistics
import time
from importlib.metadata import version
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
from adapter import install_padded_head
from mlx.utils import tree_flatten

from mlx_vlm import stream_generate
from mlx_vlm.models.qwen3_5.speculative_verifier import Qwen3_5BatchInvariantForward
from mlx_vlm.speculative.drafters.dspark.config import DSparkConfig
from mlx_vlm.speculative.drafters.dspark.dspark import DSparkDraftModel
from mlx_vlm.utils import load

CASES = {
    "text": "Write a detailed practical guide to growing vegetables in a small garden. Include 20 numbered sections, each with at least 100 words. Cover planning, soil, seeds, planting, watering, compost, pests, harvesting, storage, and seasonal tasks. Continue until all 20 sections are complete.",
    "code": "Implement a complete Python library for a task scheduler. Include priority queues, dependencies, retries with backoff, cancellation, persistence, and detailed unit tests. Explain each component and provide the complete source code. Continue until every component is implemented.",
    "image": "Write a long illustrated-story-style narrative inspired by this image. Include 20 numbered chapters with at least 100 words each. Describe the setting, characters, their daily lives, and a gentle adventure. Continue until all 20 chapters are complete.",
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--draft", required=True)
    parser.add_argument("--mode", choices=["native", "simd"], default="simd")
    parser.add_argument("--draft-bits", type=int, choices=[4, 8])
    parser.add_argument("--draft-head-bits", type=int, choices=[4, 8])
    parser.add_argument("--widths", nargs="+", default=["serial", "4", "6"])
    parser.add_argument("--cases", nargs="+", choices=list(CASES), default=list(CASES))
    parser.add_argument("--image", type=Path, default=Path("examples/images/cats.jpg"))
    parser.add_argument("--length", type=int, default=1024)
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument(
        "--output-dir", type=Path, default=Path("benchmark_results/minicpm_dspark")
    )
    args = parser.parse_args()
    if args.length < 1 or args.runs < 1:
        parser.error("--length and --runs must be positive")
    if any(
        w not in ("serial", "adaptive") and (not w.isdigit() or int(w) < 2)
        for w in args.widths
    ):
        parser.error("widths must be serial, adaptive, or integers >= 2")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    model, processor = load(args.model)
    lm = model.language_model
    if args.mode == "simd":
        install_padded_head(lm)
    else:
        standard = Qwen3_5BatchInvariantForward()
        type(lm)._get_speculative_verifier = lambda self: standard

    before = dict(tree_flatten(lm.parameters()))
    checkpoint = Path(args.draft)
    config = json.loads((checkpoint / "config.json").read_text())
    # Older local DSpark exports used a flat projector field.
    if "dflash_config" not in config and config.get("projector_type") == "dspark":
        config["dflash_config"] = {"projector_type": "dspark"}
    draft = DSparkDraftModel(DSparkConfig.from_dict(config))
    weights = {}
    for shard in sorted(checkpoint.glob("*.safetensors")):
        weights.update(mx.load(str(shard)))
    draft.load_weights(list(draft.sanitize(weights).items()))
    del weights
    if args.draft_bits:
        # Quantize owned weights before binding shared target modules.
        nn.quantize(draft, group_size=64, bits=args.draft_bits)
    mx.eval(draft.parameters())
    draft.bind(model)
    proposal_head = None
    if args.draft_head_bits:
        proposal_head = nn.QuantizedLinear.from_linear(
            lm.lm_head, group_size=64, bits=args.draft_head_bits
        )
        mx.eval(proposal_head.parameters())
        original_bind = draft.bind

        def bind_with_proposal_head(target):
            result = original_bind(target)
            draft.lm_head = proposal_head
            return result

        draft.bind = bind_with_proposal_head
        draft.bind(model)
    after = dict(tree_flatten(lm.parameters()))
    assert before.keys() == after.keys() and all(before[k] is after[k] for k in before)
    metadata = dict(
        arguments=vars(args),
        mlx=version("mlx"),
        device=mx.device_info(),
        target_weights_unchanged=True,
    )
    (args.output_dir / "config.json").write_text(
        json.dumps(metadata, indent=2, default=str)
    )

    rounds = importlib.import_module("mlx_vlm.speculative.dflash")
    original_choose, original_walk = (
        rounds._dflash_next_block_size,
        rounds._speculative_walk,
    )
    records, pending, policy = [], None, None

    def finish(boundary=None):
        nonlocal pending
        if pending is not None:
            pending["seconds"] = (
                time.perf_counter() if boundary is None else boundary
            ) - pending.pop("start")
            records.append(pending)
            pending = None

    def choose(drafter, ceiling, remaining, initial):
        nonlocal pending
        boundary = time.perf_counter()
        finish(boundary)
        width = (
            min(int(policy), ceiling, remaining)
            if policy.isdigit()
            else original_choose(drafter, ceiling, remaining, initial)
        )
        pending = dict(start=boundary, width=width, emitted=0, accepted=0)
        return width

    def walk(*args, **kwargs):
        accepted, tokens = original_walk(*args, **kwargs)
        pending.update(emitted=len(tokens), accepted=int(accepted))
        return accepted, tokens

    rounds._dflash_next_block_size, rounds._speculative_walk = choose, walk
    results = []
    try:
        for case in args.cases:
            content = ([{"type": "image"}] if case == "image" else []) + [
                {"type": "text", "text": CASES[case]}
            ]
            prompt = processor.apply_chat_template(
                [{"role": "user", "content": content}],
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )

            def run(selected_policy, length):
                nonlocal policy, records, pending
                policy, records, pending = selected_policy, [], None
                draft.prefer_requested_block_size = policy.isdigit()
                draft.config.runtime_block_size = int(policy) if policy.isdigit() else 8
                processor.tokenizer.stopping_criteria.reset(model.config.eos_token_id)
                mx.synchronize()
                mx.reset_peak_memory()
                start = time.perf_counter()
                options = (
                    {}
                    if policy == "serial"
                    else dict(draft_model=draft, draft_kind="dflash")
                )
                for output in stream_generate(
                    model,
                    processor,
                    prompt,
                    image=[str(args.image)] if case == "image" else None,
                    max_tokens=length,
                    temperature=0,
                    **options,
                ):
                    pass
                mx.synchronize()
                seconds = time.perf_counter() - start
                finish()
                if proposal_head is not None:
                    assert draft.lm_head is proposal_head
                # Remove EOS/trailing proposals that were verified but not emitted.
                excess = max(
                    0,
                    sum(r["emitted"] for r in records)
                    - max(0, len(output.token_ids) - 1),
                )
                for record in reversed(records):
                    removed = min(excess, record["emitted"])
                    record["emitted"] -= removed
                    excess -= removed
                emitted = sum(r["emitted"] for r in records)
                round_seconds = sum(r["seconds"] for r in records)
                return dict(
                    case=case,
                    policy=policy,
                    budget=length,
                    tokens=output.token_ids,
                    seconds=seconds,
                    end_to_end_tps=len(output.token_ids) / seconds,
                    decode_tps=output.generation_tps,
                    prefill_tps=output.prompt_tps,
                    prompt_tokens=output.prompt_tokens,
                    rounds=len(records),
                    accepted=sum(r["accepted"] for r in records),
                    round_emitted=emitted,
                    round_seconds=round_seconds,
                    round_tps=emitted / round_seconds if round_seconds else 0,
                    peak_gb=mx.get_peak_memory() / 1e9,
                    round_records=records,
                )

            for selected in args.widths:
                run(selected, 48)
            for repeat in range(args.runs):
                order = args.widths if repeat % 2 == 0 else list(reversed(args.widths))
                for selected in order:
                    result = run(selected, args.length)
                    result["repeat"] = repeat
                    results.append(result)
                    (args.output_dir / "raw.json").write_text(json.dumps(results))
                    print(
                        json.dumps(
                            {
                                k: v
                                for k, v in result.items()
                                if k not in ("tokens", "round_records")
                            }
                        ),
                        flush=True,
                    )
    finally:
        rounds._dflash_next_block_size, rounds._speculative_walk = (
            original_choose,
            original_walk,
        )
    summary = []
    for case in args.cases:
        reference = next(
            (
                r["tokens"]
                for r in results
                if r["case"] == case and r["policy"] == "serial"
            ),
            None,
        )
        for selected in args.widths:
            samples = [
                r for r in results if r["case"] == case and r["policy"] == selected
            ]
            summary.append(
                dict(
                    case=case,
                    policy=selected,
                    tokens=len(samples[0]["tokens"]),
                    deterministic=all(
                        r["tokens"] == samples[0]["tokens"] for r in samples
                    ),
                    matches_serial=(
                        all(r["tokens"] == reference for r in samples)
                        if reference is not None
                        else None
                    ),
                    **{
                        key: statistics.median(r[key] for r in samples)
                        for key in (
                            "seconds",
                            "end_to_end_tps",
                            "decode_tps",
                            "prefill_tps",
                            "rounds",
                        )
                    },
                )
            )
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    assert all(r["deterministic"] and r["matches_serial"] is not False for r in summary)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
