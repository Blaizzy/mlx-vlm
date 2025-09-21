from __future__ import annotations

import numpy as np
import mlx.core as mx


def fuse_embeddings_from_image_tokens(
    embedding_weights: mx.array,
    input_ids: np.ndarray,
    image_token_id: int,
    projected_tokens: mx.array,
):
    """Replace a single <image> token with projected vision embeddings."""

    ids = [int(i) for i in input_ids]
    try:
        image_pos = ids.index(int(image_token_id))
    except ValueError as exc:
        raise ValueError("No <image> token found in input_ids") from exc

    text_tokens = ids[:image_pos] + ids[image_pos + 1 :]
    text_emb = embedding_weights[mx.array(text_tokens, dtype=mx.int32)]
    fused = mx.concatenate(
        [text_emb[:image_pos], projected_tokens, text_emb[image_pos:]], axis=0
    )
    return fused, image_pos
