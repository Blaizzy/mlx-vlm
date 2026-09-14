import mlx.core as mx
import mlx.nn as nn

from .config import ModelConfig


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
