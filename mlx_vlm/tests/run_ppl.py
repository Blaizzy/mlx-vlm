#!/usr/bin/env python3
"""Perplexity evaluation on WikiText-2 for mlx_vlm models.

Uses the same KV cache, chunked prefill, and maybe_quantize_kv_cache
functions from mlx_vlm.generate — the identical code path as generation.

Usage:
    # Baseline
    python mlx_vlm/tests/run_ppl.py \
        --model mlx-community/Qwen3.5-4B-4bit

    # TurboQuant
    python mlx_vlm/tests/run_ppl.py \
        --model mlx-community/Qwen3.5-4B-4bit \
        --kv-bits 3.5 --kv-quant-scheme turboquant
"""

import argparse
import functools
import json
import math
import time

import mlx.core as mx
from datasets import load_dataset
from tqdm import tqdm

from mlx_vlm.generate import maybe_quantize_kv_cache
from mlx_vlm.models import cache
from mlx_vlm.utils import load


def _prefill_and_quantize(
    model,
    input_ids: mx.array,
    prompt_cache: list,
    quantize_cache_fn,
    prefill_step_size: int,
):
    """Chunked prefill + KV cache quantization — mirrors generate_step exactly.

    See generate.py generate_step lines 486-504.
    """
    total = input_ids.shape[1]
    if total > prefill_step_size:
        pos = 0
        while pos < total:
            chunk_end = min(pos + prefill_step_size, total)
            model.language_model(input_ids[:, pos:chunk_end], cache=prompt_cache)
            quantize_cache_fn(prompt_cache)
            mx.eval([c.state for c in prompt_cache])
            pos = chunk_end
            mx.clear_cache()
    else:
        model.language_model(input_ids, cache=prompt_cache)
        quantize_cache_fn(prompt_cache)
        mx.eval([c.state for c in prompt_cache])


def evaluate_ppl(
    model,
    tokenizer,
    dataset_text: str,
    stride: int = 512,
    max_length: int = 2048,
    kv_bits: float | None = None,
    kv_group_size: int = 64,
    kv_quant_scheme: str = "uniform",
    quantized_kv_start: int = 0,
    prefill_step_size: int = 2048,
) -> dict:
    """Compute perplexity with optional KV cache quantization.

    For each sliding window [begin, end):
      context = tokens[begin : begin + context_len]   (overlap from prior window)
      scoring = tokens[begin + context_len : end]     (new stride tokens)

    Steps per window:
      1. make_prompt_cache — same as generate_step line 427
      2. Prefill context into KV cache with chunked prefill + quantization
         — same as generate_step lines 486-504
      3. Single forward pass of scoring tokens through the (quantized) cache
      4. Collect NLL from the logits (teacher-forced)

    For baseline (kv_bits=None), the quantize step is a no-op.
    """
    encodings = tokenizer.encode(dataset_text)
    total_tokens = len(encodings)

    n_total_chunks = 0
    for begin in range(0, total_tokens, stride):
        n_total_chunks += 1
        if min(begin + max_length, total_tokens) >= total_tokens:
            break

    kv_label = f" kv={kv_bits}-bit {kv_quant_scheme}" if kv_bits else ""
    print(f"  Tokens: {total_tokens:,} | stride={stride} | "
          f"max_length={max_length}{kv_label} | chunks={n_total_chunks}")

    # Same partial as generate_step line 398-404
    quantize_cache_fn = functools.partial(
        maybe_quantize_kv_cache,
        quantized_kv_start=quantized_kv_start,
        kv_group_size=kv_group_size,
        kv_bits=kv_bits,
        kv_quant_scheme=kv_quant_scheme,
    )

    nlls = []
    n_scored = 0
    t0 = time.perf_counter()

    desc = f"PPL ({kv_bits}-bit {kv_quant_scheme})" if kv_bits else "PPL (baseline)"
    pbar = tqdm(total=n_total_chunks, desc=desc, unit="chunk",
                bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} "
                           "[{elapsed}<{remaining}, {rate_fmt}] ppl={postfix}")
    pbar.set_postfix_str("...")

    prev_end = 0
    for begin in range(0, total_tokens, stride):
        end = min(begin + max_length, total_tokens)
        target_len = end - prev_end if begin > 0 else end - begin
        context_len = (end - begin) - target_len

        # Step 1: make_prompt_cache — same as generate_step line 427
        prompt_cache = cache.make_prompt_cache(model.language_model)

        if context_len > 0:
            # Step 2: prefill context with chunked prefill + quantization
            context_ids = mx.array(encodings[begin : begin + context_len])[None, :]
            _prefill_and_quantize(
                model, context_ids, prompt_cache,
                quantize_cache_fn, prefill_step_size,
            )

            # Step 3: forward scoring tokens through (quantized) cache
            score_ids = mx.array(encodings[begin + context_len : end])[None, :]
            outputs = model.language_model(score_ids, cache=prompt_cache)
            logits = outputs.logits

            shift_logits = logits[:, :-1, :]
            shift_labels = score_ids[:, 1:]
        else:
            # First window — no context, same as baseline
            input_ids = mx.array(encodings[begin:end])[None, :]
            outputs = model.language_model(input_ids, cache=prompt_cache)
            logits = outputs.logits

            shift_logits = logits[:, :-1, :]
            shift_labels = input_ids[:, 1:]

        # Step 4: compute NLL
        log_probs = shift_logits - mx.logsumexp(shift_logits, axis=-1, keepdims=True)
        token_log_probs = mx.take_along_axis(
            log_probs, shift_labels[:, :, None], axis=-1
        ).squeeze(-1)
        nll = -mx.sum(token_log_probs).item()
        count = shift_labels.size

        nlls.append(nll)
        n_scored += count

        del outputs, logits, prompt_cache
        mx.clear_cache()

        if n_scored > 0:
            pbar.set_postfix_str(f"{math.exp(sum(nlls) / n_scored):.4f}")
        pbar.update(1)

        prev_end = end
        if end >= total_tokens:
            break

    pbar.close()
    total_time = time.perf_counter() - t0
    ppl = math.exp(sum(nlls) / n_scored) if n_scored > 0 else float("inf")

    return {
        "perplexity": round(ppl, 4),
        "total_nll": round(sum(nlls), 4),
        "n_tokens_scored": n_scored,
        "n_chunks": n_total_chunks,
        "elapsed_seconds": round(total_time, 1),
        "tokens_per_second": round(n_scored / total_time, 1),
        "peak_memory_gb": round(mx.get_peak_memory() / 1e9, 2),
    }


def main():
    parser = argparse.ArgumentParser(description="Perplexity evaluation on WikiText-2")
    parser.add_argument("--model", required=True)
    parser.add_argument("--max-length", type=int, default=2048,
                        help="Context window size")
    parser.add_argument("--stride", type=int, default=512,
                        help="Sliding window stride")
    parser.add_argument("--kv-bits", type=float, default=None,
                        help="KV cache quantization bits (enables TurboQuant)")
    parser.add_argument("--kv-quant-scheme", type=str, default="turboquant",
                        choices=("uniform", "turboquant"))
    parser.add_argument("--kv-group-size", type=int, default=64)
    parser.add_argument("--quantized-kv-start", type=int, default=0)
    parser.add_argument("--prefill-step-size", type=int, default=2048)
    parser.add_argument("--output", type=str, default="ppl_results.json")
    args = parser.parse_args()

    print(f"Model:      {args.model}")
    print(f"Max length: {args.max_length}")
    print(f"Stride:     {args.stride}")
    if args.kv_bits:
        print(f"KV bits:    {args.kv_bits}")
        print(f"KV scheme:  {args.kv_quant_scheme}")
    else:
        print(f"Mode:       baseline (no KV quantization)")
    print()

    print("Loading WikiText-2 test set...")
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    text = "\n\n".join([t for t in dataset["text"] if t.strip()])
    print(f"Dataset: {len(text):,} chars\n")

    print("Loading model...")
    model, processor = load(args.model)
    tokenizer = processor.tokenizer if hasattr(processor, "tokenizer") else processor
    print("Model loaded.\n")

    result = evaluate_ppl(
        model, tokenizer, text,
        stride=args.stride,
        max_length=args.max_length,
        kv_bits=args.kv_bits,
        kv_group_size=args.kv_group_size,
        kv_quant_scheme=args.kv_quant_scheme,
        quantized_kv_start=args.quantized_kv_start,
        prefill_step_size=args.prefill_step_size,
    )

    kv_label = f" (KV {args.kv_bits}-bit {args.kv_quant_scheme})" if args.kv_bits else ""
    print(f"\n{'='*50}")
    print(f"RESULTS — {args.model}{kv_label}")
    print(f"{'='*50}")
    print(f"  Perplexity:     {result['perplexity']:.4f}")
    print(f"  Tokens scored:  {result['n_tokens_scored']:,}")
    print(f"  Chunks:         {result['n_chunks']}")
    print(f"  Time:           {result['elapsed_seconds']:.1f}s")
    print(f"  Throughput:     {result['tokens_per_second']:.1f} tok/s")
    print(f"  Peak memory:    {result['peak_memory_gb']:.2f} GB")

    output_data = {
        "model": args.model,
        "dataset": "wikitext-2-raw-v1",
        "max_length": args.max_length,
        "stride": args.stride,
        "kv_bits": args.kv_bits,
        "kv_quant_scheme": args.kv_quant_scheme if args.kv_bits else None,
        **result,
    }
    with open(args.output, "w") as f:
        json.dump(output_data, f, indent=2)
    print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
