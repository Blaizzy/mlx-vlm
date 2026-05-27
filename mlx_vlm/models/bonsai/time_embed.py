import math

import mlx.core as mx
from mlx import nn


class Flux2TimestepGuidanceEmbeddings(nn.Module):
    def __init__(self, in_channels: int = 256, embedding_dim: int = 6144, guidance_embeds: bool = True):
        super().__init__()
        self.in_channels = in_channels
        self.embedding_dim = embedding_dim
        self.guidance_embeds = guidance_embeds
        self.linear_1 = nn.Linear(in_channels, embedding_dim, bias=False)
        self.linear_2 = nn.Linear(embedding_dim, embedding_dim, bias=False)
        self.guidance_linear_1 = nn.Linear(in_channels, embedding_dim, bias=False) if guidance_embeds else None
        self.guidance_linear_2 = nn.Linear(embedding_dim, embedding_dim, bias=False) if guidance_embeds else None

    def __call__(self, timestep: mx.array, guidance: mx.array | None) -> mx.array:
        timestep = Flux2TimestepGuidanceEmbeddings._timestep_embedding(timestep.astype(mx.float32), self.in_channels)
        timesteps_emb = self.linear_2(nn.silu(self.linear_1(timestep)))
        if guidance is not None and self.guidance_linear_1 is not None and self.guidance_linear_2 is not None:
            guidance = Flux2TimestepGuidanceEmbeddings._timestep_embedding(
                guidance.astype(mx.float32), self.in_channels
            )
            guidance_emb = self.guidance_linear_2(nn.silu(self.guidance_linear_1(guidance)))
            return timesteps_emb + guidance_emb
        return timesteps_emb

    @staticmethod
    def _timestep_embedding(timesteps: mx.array, dim: int, flip_sin_to_cos: bool = True) -> mx.array:
        half = dim // 2
        freqs = mx.exp(-math.log(10000.0) * mx.arange(0, half, dtype=mx.float32) / half)
        args = timesteps[:, None] * freqs[None, :]
        emb = mx.concatenate([mx.sin(args), mx.cos(args)], axis=-1)
        if flip_sin_to_cos:
            emb = mx.concatenate([emb[:, half:], emb[:, :half]], axis=-1)
        if dim % 2 == 1:
            emb = mx.concatenate([emb, mx.zeros((emb.shape[0], 1), dtype=emb.dtype)], axis=-1)
        return emb
