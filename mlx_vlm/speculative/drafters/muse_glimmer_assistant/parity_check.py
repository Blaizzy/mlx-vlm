"""Structural smoke test for the Muse-Glimmer DFlash drafter.

Exercises the drafter's forward + ``draft_block`` on random weights (no HF repo /
network needed), asserting the block-diffusion contract the DFlash engine relies
on: ``draft_block`` returns ``block_size - 1`` proposal tokens per row.

Run: python -m mlx_vlm.speculative.drafters.muse_glimmer_assistant.parity_check
Optionally point at a real drafter: ``--drafter meta-models/Muse-Glimmer-30B-assistant``.
"""

import argparse

import mlx.core as mx
import mlx.nn as nn

from .config import MuseGlimmerAssistantConfig
from .muse_glimmer_assistant import MuseGlimmerAssistantDraftModel


class _DummyEmbed(nn.Module):
    def __init__(self, vocab_size: int, hidden_size: int):
        super().__init__()
        self.weight = mx.random.normal((vocab_size, hidden_size)) * 0.02

    def __call__(self, ids: mx.array) -> mx.array:
        return self.weight[ids]

    def as_linear(self, x: mx.array) -> mx.array:
        return x @ self.weight.T


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--drafter", default=None, help="Optional HF drafter repo to load")
    args = p.parse_args()

    if args.drafter:
        from ...drafters import load_drafter

        model, _ = load_drafter(args.drafter)
        cfg = model.config
        vocab = cfg.mask_token_id + 1
    else:
        cfg = MuseGlimmerAssistantConfig()
        model = MuseGlimmerAssistantDraftModel(cfg)
        vocab = cfg.mask_token_id + 1

    dummy = _DummyEmbed(vocab, cfg.hidden_size)
    model.embed_tokens = dummy
    model.lm_head = dummy.as_linear

    block = cfg.block_size
    aux_dim = len(cfg.target_layer_ids) * cfg.hidden_size

    # Full-block forward: [B, block] tokens + [B, block] aux hidden -> logits.
    ids = mx.zeros((1, block), dtype=mx.int32)
    target_hidden = mx.random.normal((1, block, aux_dim)) * 0.02
    cache = model.make_cache()
    logits = model(ids, target_hidden, cache)
    mx.eval(logits)
    assert logits.shape == (1, block, vocab), logits.shape

    # draft_block: anchor + (block-1) masks -> (block-1) proposal tokens.
    cache = model.make_cache()
    proposals = model.draft_block(
        7, target_hidden, cache, block, lambda lg: mx.argmax(lg, axis=-1)
    )
    mx.eval(proposals)
    assert proposals.shape == (1, block - 1), proposals.shape

    print(
        f"OK: forward logits {tuple(logits.shape)}, "
        f"draft_block proposals {tuple(proposals.shape)} "
        f"(block_size={block}, target_layer_ids={cfg.target_layer_ids})"
    )


if __name__ == "__main__":
    main()
