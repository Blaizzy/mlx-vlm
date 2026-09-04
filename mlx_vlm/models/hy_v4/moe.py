from functools import partial

import mlx.core as mx


@partial(mx.compile, shapeless=True)
def weighted_expert_sum(expert_outputs: mx.array, routing_scores: mx.array) -> mx.array:
    return (
        (expert_outputs * routing_scores[..., None])
        .sum(axis=-2)
        .astype(expert_outputs.dtype)
    )
