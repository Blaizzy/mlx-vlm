import argparse

import mlx.core as mx
import mlx.nn as nn

from ...drafters import load_drafter


class _FakeTargetEmbed(nn.Module):
    def __init__(self, vocab_size: int, backbone_hidden_size: int):
        super().__init__()
        self.weight = mx.random.normal((vocab_size, backbone_hidden_size)) * 0.02

    def __call__(self, ids: mx.array) -> mx.array:
        return self.weight[ids]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--drafter", default="gg-hf-am/gemma-4-26B-A4B-it-assistant")
    args = p.parse_args()

    model, _ = load_drafter(args.drafter, kind="mtp")
    cfg = model.config
    text_cfg = cfg.text_config

    # No real target — fake the bind() outputs. Route lm_head through the
    # centroid head when present (E2B / E4B) so this exercises the same path
    # real generation uses; otherwise fall back to the tied embed.
    if model.masked_embedding is not None:
        embed_w = model.model.embed_tokens.weight
        masked = model.masked_embedding
        model._lm_head_fn = lambda h: masked(h, embed_w)
    else:
        model._lm_head_fn = model.model.embed_tokens.as_linear
    model._input_embed = _FakeTargetEmbed(text_cfg.vocab_size, cfg.backbone_hidden_size)

    B = 1
    block = cfg.block_size
    backbone = cfg.backbone_hidden_size
    n_kv_heads = (
        text_cfg.num_global_key_value_heads
        if text_cfg.attention_k_eq_v and text_cfg.num_global_key_value_heads
        else text_cfg.num_key_value_heads
    )

    kv_len = 32
    # Provide one dict per layer-type. Use the global head dim for full
    # attention layers when present (the model code gates on layer_type ==
    # "full_attention" + global_head_dim, regardless of ``attention_k_eq_v``).
    full_head = text_cfg.global_head_dim or text_cfg.head_dim

    def _kv(seq_len: int, head_dim: int):
        return (
            mx.random.normal((B, n_kv_heads, seq_len, head_dim)) * 0.02,
            mx.random.normal((B, n_kv_heads, seq_len, head_dim)) * 0.02,
        )

    shared_kv = {
        "full_attention": _kv(kv_len, full_head),
        "sliding_attention": _kv(
            min(kv_len, text_cfg.sliding_window), text_cfg.head_dim
        ),
    }
    model.set_shared_kv(shared_kv, kv_offset=kv_len)

    # Single forward smoke
    inputs_embeds = mx.random.normal((B, 1, 2 * backbone)) * 0.02
    position_ids = mx.array([[kv_len]])
    h, logits = model(inputs_embeds, shared_kv, position_ids)
    mx.eval(h, logits)
    print(
        f"forward OK: logits shape={tuple(logits.shape)} "
        f"hidden shape={tuple(h.shape)} "
        f"mean={float(logits.mean()):.5f} std={float(logits.std()):.5f}"
    )

    # Multi-step draft_block smoke
    def _greedy(x):
        return mx.argmax(x[:, -1:, :], axis=-1).astype(mx.int32)

    drafted = model.draft_block(
        last_bonus=42,
        hidden=mx.random.normal((B, 1, backbone)) * 0.02,
        cache=None,
        block_size=block,
        sampler=_greedy,
        token_dtype=mx.int32,
    )
    mx.eval(drafted)
    print(f"draft_block OK: tokens shape={tuple(drafted.shape)} dtype={drafted.dtype}")


if __name__ == "__main__":
    main()
