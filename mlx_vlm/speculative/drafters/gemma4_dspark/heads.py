import mlx.core as mx
import mlx.nn as nn


class DSparkMarkovHead(nn.Module):
    """Low-rank (rank-r) token-transition logit bias.

    ``markov_w1`` embeds a token id into r dims; ``markov_w2`` projects back to
    vocab logits. Returns ``(logits, embed)`` so the confidence head can reuse the
    embedding. Weight layout matches the checkpoint: ``markov_w1`` is
    ``[vocab, rank]`` (embedding) and ``markov_w2`` is ``[vocab, rank]`` (the head,
    an ``nn.Linear(rank, vocab)`` whose MLX weight is ``[vocab, rank]``).
    """

    def __init__(self, vocab_size: int, rank: int):
        super().__init__()
        self.markov_w1 = nn.Embedding(vocab_size, rank)
        self.markov_w2 = nn.Linear(rank, vocab_size, bias=False)

    def __call__(self, token_ids: mx.array) -> tuple[mx.array, mx.array]:
        embed = self.markov_w1(token_ids)
        logits = self.markov_w2(embed.astype(mx.float32))
        return logits, embed


class DSparkConfidenceHead(nn.Module):
    """Per-draft-token acceptance score from ``[hidden ‖ markov_embed]``.

    Advisory only in the lossless verify path; it gates adaptive draft length
    when a confidence threshold is set. Gemma 4 DSpark uses a bias projection.
    """

    def __init__(self, input_dim: int, bias: bool = True):
        super().__init__()
        self.proj = nn.Linear(input_dim, 1, bias=bias)

    def __call__(self, hidden: mx.array, markov_embed: mx.array) -> mx.array:
        x = mx.concatenate([hidden, markov_embed], axis=-1)
        return self.proj(x.astype(mx.float32))[..., 0]
