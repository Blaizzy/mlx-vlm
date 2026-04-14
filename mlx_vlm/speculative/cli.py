"""Speculative-decoding CLI:

    python -m mlx_vlm.speculative.cli \
        --target Qwen/Qwen3.5-4B \
        --drafter z-lab/Qwen3.5-4B-DFlash \
        --prompt "Explain speculative decoding" \
        --max-new-tokens 200

Reports per-round acceptance length and total speedup estimate vs. a
baseline autoregressive run of the same prompt/token budget.
"""

import argparse
import time

import mlx.core as mx

from ..models.qwen3_5_dflash import load_dflash_drafter
from ..utils import load
from .dflash_loop import dflash_generate


def _apply_chat(tokenizer, prompt: str) -> str:
    if hasattr(tokenizer, "apply_chat_template"):
        try:
            return tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}],
                tokenize=False,
                add_generation_prompt=True,
            )
        except Exception:  # noqa: BLE001
            pass
    return prompt


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--target", required=True, help="target VLM path or HF id")
    p.add_argument("--drafter", required=True, help="DFlash drafter path or HF id")
    p.add_argument("--prompt", required=True)
    p.add_argument("--max-new-tokens", type=int, default=200)
    p.add_argument("--compare-ar", action="store_true", help="Also run plain AR for wall-time comparison")
    args = p.parse_args()

    print(f"Loading target model: {args.target}")
    target, processor = load(args.target)

    print(f"Loading DFlash drafter: {args.drafter}")
    drafter = load_dflash_drafter(args.drafter)

    tokenizer = (
        processor.tokenizer if hasattr(processor, "tokenizer") else processor
    )
    text = _apply_chat(tokenizer, args.prompt)
    input_ids = mx.array([tokenizer.encode(text)], dtype=mx.int32)
    print(f"Prompt tokens: {input_ids.shape[1]}")
    eos = tokenizer.eos_token_id

    def _run(gen_fn, label):
        print(f"\n=== {label} ===")
        produced: list = []
        accept_lens: list = []
        round_accepts = 0
        t0 = time.perf_counter()
        for tok, accept_idx in gen_fn:
            produced.append(tok)
            if accept_idx == 0:
                if round_accepts or produced:
                    accept_lens.append(round_accepts)
                round_accepts = 0
            else:
                round_accepts = max(round_accepts, accept_idx)
        elapsed = time.perf_counter() - t0
        text_out = tokenizer.decode(produced)
        print(text_out)
        tok_per_s = len(produced) / elapsed if elapsed > 0 else 0.0
        mean_accept = (
            sum(accept_lens) / len(accept_lens) if accept_lens else 0.0
        )
        print(
            f"\n[{label}] {len(produced)} tokens in {elapsed:.2f}s "
            f"({tok_per_s:.1f} tok/s), mean accepted drafted/round = {mean_accept:.2f}"
        )
        return elapsed, len(produced)

    _run(
        dflash_generate(
            target,
            drafter,
            input_ids,
            max_new_tokens=args.max_new_tokens,
            eos_token_id=eos,
        ),
        "DFlash",
    )

    if args.compare_ar:
        from ..generate import generate_step

        print("\n=== plain AR (baseline) ===")
        t0 = time.perf_counter()
        tokens = []
        for tok, _ in generate_step(
            input_ids,
            target,
            pixel_values=None,
            mask=None,
            max_tokens=args.max_new_tokens,
            temperature=0.0,
        ):
            if isinstance(tok, mx.array):
                tok = int(tok.item())
            tokens.append(tok)
            if len(tokens) >= args.max_new_tokens:
                break
            if eos is not None and tok == eos:
                break
        elapsed = time.perf_counter() - t0
        print(tokenizer.decode(tokens))
        print(
            f"\n[AR] {len(tokens)} tokens in {elapsed:.2f}s "
            f"({len(tokens)/elapsed:.1f} tok/s)"
        )


if __name__ == "__main__":
    main()
