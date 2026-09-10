import mlx.core as mx
import mlx.nn as nn

from .config import ModelConfig


def get_dspark_topk_idxs(
    window_size: int, bsz: int, block_size: int, start_pos: int
) -> mx.array:
    """Sliding-window plus draft-block indices for DSpark sparse attention."""
    assert start_pos > 0
    matrix = mx.concatenate(
        [
            mx.arange(min(window_size, start_pos + 1)),
            window_size + mx.arange(block_size),
        ]
    )
    return mx.broadcast_to(matrix.reshape(1, 1, -1), (bsz, block_size, matrix.shape[0]))


class DSparkMarkovHead(nn.Module):
    """Token-to-logits Markov draft head over a low-rank embedding."""

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.embed = nn.Embedding(config.vocab_size, config.dspark_markov_rank)
        self.head = nn.Linear(config.dspark_markov_rank, config.vocab_size, bias=False)

    def __call__(self, token_ids: mx.array):
        embed = self.embed(token_ids)
        return self.head(embed), embed


class DSparkConfidenceHead(nn.Module):
    """Acceptance-confidence head over concatenated hidden and Markov states.

    The projection stays in float32 for fp32 confidence scores.
    """

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.proj = nn.Linear(
            config.hidden_size + config.dspark_markov_rank, 1, bias=False
        )

    def __call__(self, hidden: mx.array, markov_embed: mx.array) -> mx.array:
        out = self.proj(
            mx.concatenate(
                [hidden.astype(mx.float32), markov_embed.astype(mx.float32)], axis=-1
            )
        )
        return out.squeeze(-1)
