"""Multi-view alternating-attention transformer.

Even layers attend over the tokens of all views plus the global tokens (the
scale token); odd layers attend within each view.
"""

from typing import List, Tuple

import mlx.core as mx
import mlx.nn as nn

from .config import InfoSharingConfig
from .layers import Block, layer_norm


class AlternatingAttentionTransformer(nn.Module):
    def __init__(self, config: InfoSharingConfig, input_dim: int, ffn: str):
        super().__init__()
        self.indices = frozenset(config.indices)
        self.norm_intermediate = config.norm_intermediate
        dim = config.dim
        if input_dim != dim:
            self.proj_embed = nn.Linear(input_dim, dim)
        self.self_attention_blocks = [
            Block(
                dim,
                config.num_heads,
                config.mlp_ratio,
                ffn,
                layer_scale=config.layer_scale,
                qkv_bias=config.qkv_bias,
            )
            for _ in range(config.depth)
        ]
        self.norm = nn.LayerNorm(dim, eps=1e-6)
        if config.distinguish_ref_and_non_ref_views:
            self.view_pos_table = mx.zeros((1, dim))

    def _embed(self, x: mx.array) -> mx.array:
        if "proj_embed" not in self:
            return x.astype(mx.float32)
        dtype = self.proj_embed.weight.dtype
        return self.proj_embed(x.astype(dtype)).astype(mx.float32)

    def __call__(
        self, views: mx.array, global_tokens: mx.array
    ) -> Tuple[List[mx.array], mx.array]:
        """views: (B, V, T, C) tokens of each view, the first view being the
        reference; global_tokens: (B, G, C).

        Returns the normed view tokens (B, V, T, D) of the intermediate layers
        and of the last layer, and the normed global tokens (B, G, D).
        """
        B, V, T, _ = views.shape
        x = self._embed(views.reshape(B, V * T, -1))
        g = self._embed(global_tokens)
        if "view_pos_table" in self:
            x = mx.concatenate([x[:, :T] + self.view_pos_table, x[:, T:]], axis=1)

        outputs = []
        for i, block in enumerate(self.self_attention_blocks):
            if i % 2 == 0:
                y = block(mx.concatenate([x, g], axis=1))
                x, g = y[:, : V * T], y[:, V * T :]
            else:
                x = block(x.reshape(B * V, T, -1)).reshape(B, V * T, -1)
            if i in self.indices:
                outputs.append(
                    layer_norm(self.norm, x) if self.norm_intermediate else x
                )
        outputs.append(layer_norm(self.norm, x))
        return [o.reshape(B, V, T, -1) for o in outputs], layer_norm(self.norm, g)
