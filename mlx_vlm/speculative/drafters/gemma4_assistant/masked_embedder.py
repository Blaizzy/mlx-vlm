"""Centroid-routed sparse LM head for Gemma 4 E2B / E4B drafters.

Mirrors HF's ``Gemma4AssistantMaskedEmbedder``
(``transformers/models/gemma4_assistant/modeling_gemma4_assistant.py:43``).

Idea: rather than computing ``hidden @ embed.T`` over the full 262144-vocab
(expensive for a tiny drafter with ``hidden_size=256``), the drafter learns a
``centroids`` Linear that scores 2048 token clusters, and a ``token_ordering``
buffer that maps each cluster to a contiguous block of canonical token IDs.
At inference, the top-K clusters' tokens (typ. 32×128 = 4096 of 262144) are
materialized and scored densely; the rest of the vocab is filled with a
sentinel ``min - 1`` so it loses any argmax / sampling competition.
"""

from typing import Any

import mlx.core as mx
import mlx.nn as nn


class MaskedEmbedder(nn.Module):
    """Centroid-routed sparse softmax for the assistant drafter's LM head."""

    def __init__(self, config: Any):
        super().__init__()
        text_cfg = config.text_config
        self.hidden_size = text_cfg.hidden_size
        self.vocab_size = text_cfg.vocab_size
        self.num_centroids = config.num_centroids
        self.top_k = config.centroid_intermediate_top_k
        self.vocab_size_per_centroid = self.vocab_size // self.num_centroids

        self.centroids = nn.Linear(self.hidden_size, self.num_centroids, bias=False)
        # ``token_ordering[c * vocab_size_per_centroid : (c+1) * vocab_size_per_centroid]``
        # holds the canonical token IDs assigned to centroid ``c``.
        # Loaded from checkpoint as int64; cast to int32 for indexing.
        self.token_ordering = mx.zeros((self.vocab_size,), dtype=mx.int32)

    def __call__(self, hidden_states: mx.array, lm_head_weight: mx.array) -> mx.array:
        """Compute sparse logits over the full vocab.

        ``hidden_states``: ``[B, L, hidden_size]``.
        ``lm_head_weight``: ``[vocab_size, hidden_size]`` (tied to the drafter's
        ``embed_tokens.weight``).
        Returns: ``[B, L, vocab_size]`` with non-selected positions masked
        to ``min(selected_logits) - 1``.
        """
        B, L = hidden_states.shape[:2]

        # Cluster scores → top-K cluster indices.
        centroid_logits = self.centroids(hidden_states)  # [B, L, num_centroids]
        topk_idx = mx.argpartition(centroid_logits, kth=-self.top_k, axis=-1)[
            ..., -self.top_k :
        ]  # [B, L, top_k]

        # Reshape token_ordering to [num_centroids, vocab_size_per_centroid].
        ordering = self.token_ordering.reshape(
            self.num_centroids, self.vocab_size_per_centroid
        )

        # For each selected cluster, fetch its canonical token IDs.
        # selected_canonical: [B, L, top_k, vocab_size_per_centroid]
        selected_canonical = ordering[topk_idx]

        # Gather embeddings: lm_head_weight[selected_canonical] → [B, L, top_k * vsc, hidden]
        flat_idx = selected_canonical.reshape(-1)
        selected_emb = lm_head_weight[flat_idx].reshape(
            B, L, self.top_k * self.vocab_size_per_centroid, self.hidden_size
        )

        # selected_logits = (h @ E.T)
        selected_logits = mx.matmul(
            hidden_states[..., None, :],  # [B, L, 1, hidden]
            selected_emb.swapaxes(-1, -2),  # [B, L, hidden, top_k*vsc]
        ).squeeze(
            -2
        )  # [B, L, top_k*vsc]

        mask_value = float(selected_logits.min().item()) - 1.0

        # Scatter selected_logits into a full-vocab tensor at canonical positions.
        scatter_idx = selected_canonical.reshape(B, L, -1)  # [B, L, top_k*vsc]
        out = mx.full(
            (B, L, self.vocab_size),
            vals=mask_value,
            dtype=hidden_states.dtype,
        )
        # mlx.put_along_axis writes ``src`` at ``index`` along ``axis``.
        return mx.put_along_axis(out, scatter_idx, selected_logits, axis=-1)
