import argparse

import mlx.core as mx
import mlx.nn as nn

from ...drafters import load_drafter


class _DummyEmbed(nn.Module):
    def __init__(self, vocab_size: int, hidden_size: int):
        super().__init__()
        self.weight = mx.random.normal((vocab_size, hidden_size)) * 0.02

    def __call__(self, ids: mx.array) -> mx.array:
        return self.weight[ids]

    def as_linear(self, x: mx.array) -> mx.array:
        return x @ self.weight.T


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--drafter", default="z-lab/Qwen3.5-4B-DFlash")
    args = p.parse_args()

    model, _ = load_drafter(args.drafter)
    cfg = model.config

    dummy = _DummyEmbed(cfg.vocab_size, cfg.hidden_size)
    model.embed_tokens = dummy
    model.lm_head = dummy.as_linear

    B = 1
    block = cfg.block_size
    T = 16

    ids = mx.zeros((B, block), dtype=mx.int32)
    target_hidden = (
        mx.random.normal((B, T, len(cfg.target_layer_ids) * cfg.hidden_size)) * 0.02
    )
    cache = model.make_cache()

    logits = model(ids, target_hidden, cache)
    mx.eval(logits)
    print(
        f"forward OK: logits shape={tuple(logits.shape)} "
        f"mean={float(logits.mean()):.5f} std={float(logits.std()):.5f}"
    )


if __name__ == "__main__":
    main()
