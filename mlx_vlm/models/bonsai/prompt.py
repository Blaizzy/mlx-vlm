from __future__ import annotations

import mlx.core as mx

from mlx_vlm.models.bonsai.qwen.text_encoder import Qwen3TextEncoder
from mlx_vlm.models.bonsai.tokenizer import BonsaiTokenizer

DEFAULT_SEQ_LEN_BUCKETS: tuple[int, ...] = (32, 64, 128, 256, 512)


def _pick_bucket(true_len: int, buckets: tuple[int, ...], cap: int) -> int:
    eligible = [b for b in buckets if b <= cap]
    for b in sorted(eligible):
        if b >= true_len:
            return b
    return cap


def encode_prompt(
    *,
    prompt: str | list[str],
    tokenizer: BonsaiTokenizer,
    text_encoder: Qwen3TextEncoder,
    max_sequence_length: int = 512,
    hidden_state_layers: tuple[int, ...] = (9, 18, 27),
    bucketed: bool = True,
) -> tuple[mx.array, mx.array]:
    true_len = tokenizer.count_tokens(prompt) if bucketed else max_sequence_length
    effective_max = _pick_bucket(
        true_len, DEFAULT_SEQ_LEN_BUCKETS, cap=max_sequence_length
    )
    tokens = tokenizer.tokenize(prompt=prompt, max_length=effective_max)
    embeds = text_encoder.get_prompt_embeds(
        input_ids=tokens.input_ids,
        attention_mask=tokens.attention_mask,
        hidden_state_layers=hidden_state_layers,
    )
    return embeds, prepare_text_ids(embeds)


def prepare_text_ids(x: mx.array) -> mx.array:
    batch_size, seq_len, _ = x.shape
    out_ids = []
    for _ in range(batch_size):
        t = mx.zeros((seq_len,), dtype=mx.int32)
        h = mx.zeros((seq_len,), dtype=mx.int32)
        w = mx.zeros((seq_len,), dtype=mx.int32)
        token_ids = mx.arange(seq_len, dtype=mx.int32)
        out_ids.append(mx.stack([t, h, w, token_ids], axis=1))
    return mx.stack(out_ids, axis=0)
