from functools import partial

import mlx.core as mx
from mlx.nn.layers.distributed import sum_gradients

from ..deepseek_v32.language import DeepseekV32MoE


@partial(mx.compile, shapeless=True)
def weighted_expert_sum(expert_outputs: mx.array, routing_scores: mx.array) -> mx.array:
    return (
        (expert_outputs * routing_scores[..., None])
        .sum(axis=-2)
        .astype(expert_outputs.dtype)
    )


class HyV4MoE(DeepseekV32MoE):
    def __call__(self, x):
        if self.sharding_group is not None:
            x = sum_gradients(self.sharding_group)(x)

        indices, scores = self.gate(x)
        y = weighted_expert_sum(self.switch_mlp(x, indices), scores)
        if self.config.n_shared_experts is not None:
            y = y + self.shared_experts(x)

        if self.sharding_group is not None:
            y = mx.distributed.all_sum(y, group=self.sharding_group)
        return y
