#!/usr/bin/env python3
"""Run NIAH (Needle-in-a-Haystack) evaluation via mlx_vlm.

Usage:
    python mlx_vlm/tests/run_niah.py \
        --model mlx-community/Qwen3.5-27B-4bit \
        --dataset single               # or "multi" or "both"
        --context-lengths 2k 8k 24k    # optional filter, default all
        --kv-bits 3.5                   # optional, triggers TurboQuant
        --kv-quant-scheme turboquant    # optional
        --max-tokens 128               # answer generation budget
        --output results.json           # where to save results
"""

import argparse
import gc
import json
import time
from pathlib import Path

import mlx.core as mx

from mlx_vlm.generate import generate, GenerationResult, maybe_quantize_kv_cache
from mlx_vlm.models import cache as cache_mod
from mlx_vlm.prompt_utils import apply_chat_template
from mlx_vlm.turboquant import TurboQuantKVCache, _state_nbytes
from mlx_vlm.utils import load

NIAH_DIR = Path(__file__).parent / "niah"

ALL_CONTEXT_LABELS = ["2k", "8k", "24k", "32k", "64k", "128k", "256k"]


def score_answer(expected: str, generated: str) -> bool:
    """Check if the expected answer appears in the generated text."""
    return expected.lower() in generated.lower()


def strip_thinking(text: str) -> str:
    """Remove <think>...</think> blocks from model output."""
    import re
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()


def estimate_kv_size_gb(model, n_tokens: int, kv_bits: float | None = None) -> float:
    """Estimate KV cache size from model config."""
    config = model.config
    # Extract dimensions from config
    text_config = getattr(config, "text_config", config)
    n_layers = getattr(text_config, "num_hidden_layers", 28)
    n_kv_heads = getattr(text_config, "num_key_value_heads",
                         getattr(text_config, "num_attention_heads", 4))
    head_dim = getattr(text_config, "head_dim", None)
    if head_dim is None:
        hidden = getattr(text_config, "hidden_size", 2048)
        n_heads = getattr(text_config, "num_attention_heads", 16)
        head_dim = hidden // n_heads

    if kv_bits is not None:
        # TurboQuant: approximate bits per value stored
        # For 3.5-bit SplitCodec with float16 norms:
        #   Keys (ProdState): norms(16b) + mse_indices(~bits-1 packed) + residual_norms(16b) + signs(1b packed)
        #   Values (MSEState): norms(16b) + indices(~bits packed)
        # Rough: ~(bits + 0.5) bits per dim for keys, ~(bits + 0.25) for values
        effective_bits_per_dim = kv_bits + 0.5  # average key+value
        # Plus norms overhead: ~16 bits per token per head (shared across dims)
        norms_bits_per_token = 16 * 3  # norms + residual_norms + value_norms
        bytes_per_token_per_layer = (
            n_kv_heads * (head_dim * effective_bits_per_dim + norms_bits_per_token) / 8 * 2  # K+V
        )
    else:
        # FP16: 2 bytes per element, K+V
        bytes_per_token_per_layer = n_kv_heads * head_dim * 2 * 2  # 2 bytes * 2 (K+V)

    total_bytes = bytes_per_token_per_layer * n_layers * n_tokens
    return total_bytes / (1024 ** 3)


def run_single(
    model, processor, config, tests, gen_kwargs, context_filter,
    enable_thinking=False,
) -> list[dict]:
    results = []
    for test in tests:
        label = test["context_length_label"]
        if context_filter and label not in context_filter:
            continue

        depth = test["depth"]
        print(f"\n{'='*60}")
        print(f"[Single Needle] context={label}, depth={depth}")
        print(f"{'='*60}")

        prompt = apply_chat_template(
            processor, config, test["prompt"],
            enable_thinking=enable_thinking,
        )

        t0 = time.perf_counter()
        result = generate(model, processor, prompt, verbose=True, **gen_kwargs)
        elapsed = time.perf_counter() - t0
        active_mem_gb = round(mx.get_active_memory() / 1e9, 2)

        raw_text = result.text
        clean_text = strip_thinking(raw_text)
        correct = score_answer(test["expected_answer"], clean_text)
        kv_gb = estimate_kv_size_gb(model, result.prompt_tokens, gen_kwargs.get("kv_bits"))

        entry = {
            "dataset": "single_needle",
            "context_length": label,
            "context_tokens": test["context_length_tokens"],
            "depth": depth,
            "needle": test["needle"],
            "question": test["question"],
            "expected_answer": test["expected_answer"],
            "generated_answer": clean_text,
            "correct": correct,
            "prompt_tokens": result.prompt_tokens,
            "generation_tokens": result.generation_tokens,
            "prompt_tps": round(result.prompt_tps, 1),
            "generation_tps": round(result.generation_tps, 1),
            "active_memory_gb": active_mem_gb,
            "peak_memory_gb": round(result.peak_memory, 2),
            "kv_size_gb": round(kv_gb, 2),
            "elapsed_seconds": round(elapsed, 1),
        }
        results.append(entry)

        status = "PASS" if correct else "FAIL"
        print(f"  Answer: {clean_text[:200]}")
        print(f"  Expected: {test['expected_answer']}")
        print(f"  [{status}]  prefill={result.prompt_tps:.1f} tok/s  gen={result.generation_tps:.1f} tok/s  active={active_mem_gb:.1f}GB  kv={kv_gb:.2f}GB  time={elapsed:.1f}s")

        mx.clear_cache()
        gc.collect()

    return results


def run_multi(
    model, processor, config, tests, gen_kwargs, context_filter,
    enable_thinking=False,
) -> list[dict]:
    results = []
    for test in tests:
        label = test["context_length_label"]
        if context_filter and label not in context_filter:
            continue

        n_needles = test["num_needles"]
        print(f"\n{'='*60}")
        print(f"[Multi Needle] context={label}, needles={n_needles}")
        print(f"{'='*60}")

        prompt = apply_chat_template(
            processor, config, test["prompt"],
            enable_thinking=enable_thinking,
        )

        t0 = time.perf_counter()
        result = generate(model, processor, prompt, verbose=True, **gen_kwargs)
        elapsed = time.perf_counter() - t0

        raw_text = result.text
        clean_text = strip_thinking(raw_text)
        active_mem_gb = round(mx.get_active_memory() / 1e9, 2)
        kv_gb = estimate_kv_size_gb(model, result.prompt_tokens, gen_kwargs.get("kv_bits"))

        needle_results = []
        for needle in test["needles"]:
            found = score_answer(needle["answer"], clean_text)
            needle_results.append({
                "id": needle["id"],
                "question": needle["question"],
                "expected_answer": needle["answer"],
                "found": found,
            })

        n_found = sum(1 for nr in needle_results if nr["found"])

        entry = {
            "dataset": "multi_needle",
            "context_length": label,
            "context_tokens": test["context_length_tokens"],
            "num_needles": n_needles,
            "needles_found": n_found,
            "needle_details": needle_results,
            "generated_answer": clean_text,
            "prompt_tokens": result.prompt_tokens,
            "generation_tokens": result.generation_tokens,
            "prompt_tps": round(result.prompt_tps, 1),
            "generation_tps": round(result.generation_tps, 1),
            "active_memory_gb": active_mem_gb,
            "peak_memory_gb": round(result.peak_memory, 2),
            "kv_size_gb": round(kv_gb, 2),
            "elapsed_seconds": round(elapsed, 1),
        }
        results.append(entry)

        print(f"  Answer: {clean_text[:300]}")
        print(f"  Needles found: {n_found}/{n_needles}")
        for nr in needle_results:
            s = "PASS" if nr["found"] else "FAIL"
            print(f"    [{s}] {nr['expected_answer']}")
        print(f"  prefill={result.prompt_tps:.1f} tok/s  gen={result.generation_tps:.1f} tok/s  active={active_mem_gb:.1f}GB  kv={kv_gb:.2f}GB  time={elapsed:.1f}s")

        mx.clear_cache()
        gc.collect()

    return results


def print_summary(results: list[dict]):
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")

    single = [r for r in results if r["dataset"] == "single_needle"]
    multi = [r for r in results if r["dataset"] == "multi_needle"]

    if single:
        print("\nSingle Needle:")
        print(f"  {'Context':<8} {'Depth':<6} {'Status':<6} {'Prefill':<12} {'Gen':<10} {'KV':<8} {'Active':<8} {'Time':<8}")
        print(f"  {'-'*66}")
        for r in single:
            s = "PASS" if r["correct"] else "FAIL"
            print(
                f"  {r['context_length']:<8} {r['depth']:<6} {s:<6} "
                f"{r['prompt_tps']:<12.1f} {r['generation_tps']:<10.1f} "
                f"{r['kv_size_gb']:<8.2f} "
                f"{r['active_memory_gb']:<8.1f} {r['elapsed_seconds']:<8.1f}"
            )
        total = len(single)
        passed = sum(1 for r in single if r["correct"])
        print(f"\n  Accuracy: {passed}/{total} ({100*passed/total:.1f}%)")

    if multi:
        print("\nMulti Needle:")
        print(f"  {'Context':<8} {'Found':<10} {'Prefill':<12} {'Gen':<10} {'KV':<8} {'Active':<8} {'Time':<8}")
        print(f"  {'-'*64}")
        for r in multi:
            print(
                f"  {r['context_length']:<8} {r['needles_found']}/{r['num_needles']:<8} "
                f"{r['prompt_tps']:<12.1f} {r['generation_tps']:<10.1f} "
                f"{r['kv_size_gb']:<8.2f} "
                f"{r['active_memory_gb']:<8.1f} {r['elapsed_seconds']:<8.1f}"
            )
        total_n = sum(r["num_needles"] for r in multi)
        found_n = sum(r["needles_found"] for r in multi)
        print(f"\n  Accuracy: {found_n}/{total_n} ({100*found_n/total_n:.1f}%)")


def main():
    parser = argparse.ArgumentParser(description="Run NIAH evaluation")
    parser.add_argument("--model", required=True)
    parser.add_argument("--dataset", choices=["single", "multi", "both"], default="both")
    parser.add_argument("--context-lengths", nargs="*", default=None,
                        help="e.g. 2k 8k 32k")
    parser.add_argument("--kv-bits", type=float, default=None)
    parser.add_argument("--kv-quant-scheme", type=str, default="uniform")
    parser.add_argument("--kv-group-size", type=int, default=64)
    parser.add_argument("--quantized-kv-start", type=int, default=5000)
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument("--prefill-step-size", type=int, default=2048)
    parser.add_argument("--enable-thinking", action="store_true",
                        help="Enable model thinking (default: disabled for efficiency)")
    parser.add_argument("--output", type=str, default="niah_results.json")
    args = parser.parse_args()

    context_filter = set(args.context_lengths) if args.context_lengths else None

    print(f"Model:       {args.model}")
    print(f"Dataset:     {args.dataset}")
    print(f"Contexts:    {args.context_lengths or 'all'}")
    print(f"KV bits:     {args.kv_bits or 'none (full precision)'}")
    print(f"KV scheme:   {args.kv_quant_scheme}")
    print()

    # Load model once
    print("Loading model...")
    model, processor = load(args.model)
    config = model.config
    print("Model loaded.\n")

    gen_kwargs = {
        "max_tokens": args.max_tokens,
        "temperature": 0.0,
        "kv_bits": args.kv_bits,
        "kv_group_size": args.kv_group_size,
        "kv_quant_scheme": args.kv_quant_scheme,
        "quantized_kv_start": args.quantized_kv_start,
        "prefill_step_size": args.prefill_step_size,
    }

    all_results = []

    if args.dataset in ("single", "both"):
        with open(NIAH_DIR / "niah_single_needle.json") as f:
            single_tests = json.load(f)
        all_results.extend(
            run_single(model, processor, config, single_tests, gen_kwargs,
                       context_filter, enable_thinking=args.enable_thinking)
        )

    if args.dataset in ("multi", "both"):
        with open(NIAH_DIR / "niah_multi_needle.json") as f:
            multi_tests = json.load(f)
        all_results.extend(
            run_multi(model, processor, config, multi_tests, gen_kwargs,
                      context_filter, enable_thinking=args.enable_thinking)
        )

    print_summary(all_results)

    output_data = {
        "model": args.model,
        "kv_bits": args.kv_bits,
        "kv_quant_scheme": args.kv_quant_scheme,
        "results": all_results,
    }
    with open(args.output, "w") as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)
    print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
