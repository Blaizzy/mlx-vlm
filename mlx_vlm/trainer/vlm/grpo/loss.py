"""Completion-only VLM scoring for GRPO; the objective lives in ``losses.grpo``."""

import mlx.core as mx

from mlx_vlm.trainer.losses.cross_entropy import token_cross_entropy
from mlx_vlm.trainer.vlm.cross_entropy.runtime import (
    _model_type,
    forward_vlm_logits,
)


def completion_logps(model, batch, start, chunk_size=128):
    """Preserve full causal context; normalize only completion target positions.

    Qwen3.5's conventional body/head permits projecting completion hidden states
    in bounded vocabulary chunks. Other architectures retain their public forward.
    """
    ids, attention = batch["input_ids"], batch["attention_mask"]
    targets = ids[:, start + 1 :]
    if _model_type(model) == "qwen3_5":
        media = {
            k: v for k, v in batch.items() if k not in {"input_ids", "attention_mask"}
        }
        pixels = media.pop("pixel_values", None)
        embedding = model.get_input_embeddings(
            ids[:, :-1], pixels, mask=attention[:, :-1], **media
        )
        lm = model.language_model
        hidden = lm.model(
            ids[:, :-1],
            inputs_embeds=embedding.inputs_embeds,
            mask=attention[:, :-1],
            position_ids=embedding.position_ids,
        )[:, start:, :]
        head = (
            lm.model.embed_tokens.as_linear
            if lm.args.tie_word_embeddings
            else lm.lm_head
        )
        scores = []
        for position in range(0, targets.shape[1], chunk_size):
            logits = head(hidden[:, position : position + chunk_size]).astype(
                mx.float32
            )
            scores.append(
                -token_cross_entropy(
                    logits, targets[:, position : position + chunk_size]
                )
            )
        return mx.concatenate(scores, axis=1)
    logits = forward_vlm_logits(model, ids[:, :-1], attention[:, :-1], batch)
    return -token_cross_entropy(logits[:, start:], targets)
