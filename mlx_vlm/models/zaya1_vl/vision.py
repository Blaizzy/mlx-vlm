from functools import partial

import mlx.core as mx
import mlx.nn as nn

from ..qwen2_5_vl.vision import Attention, PatchEmbed
from ..qwen2_5_vl.vision import VisionModel as Qwen2_5VisionModel
from ..qwen2_5_vl.vision import VisionRotaryEmbedding
from .config import VisionConfig


@partial(mx.compile, shapeless=True)
def swiglu(gate: mx.array, x: mx.array) -> mx.array:
    return (nn.silu(gate.astype(mx.float32)) * x.astype(mx.float32)).astype(gate.dtype)


@partial(mx.compile, shapeless=True)
def gelu(x: mx.array) -> mx.array:
    return nn.gelu(x.astype(mx.float32)).astype(x.dtype)


class PatchMerger(nn.Module):
    def __init__(self, dim: int, context_dim: int, spatial_merge_size: int = 2) -> None:
        super().__init__()
        self.hidden_size = context_dim * (spatial_merge_size**2)
        self.ln_q = nn.RMSNorm(context_dim, eps=1e-6)
        self.mlp = [
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.GELU(),
            nn.Linear(self.hidden_size, dim),
        ]

    def __call__(self, x: mx.array) -> mx.array:
        x = self.ln_q(x).reshape(-1, self.hidden_size)
        return self.mlp[2](gelu(self.mlp[0](x)))


class MLP(nn.Module):
    def __init__(self, dim, hidden_dim):
        super().__init__()
        self.gate_proj = nn.Linear(dim, hidden_dim)
        self.up_proj = nn.Linear(dim, hidden_dim)
        self.down_proj = nn.Linear(hidden_dim, dim)

    def __call__(self, x: mx.array) -> mx.array:
        return self.down_proj(swiglu(self.gate_proj(x), self.up_proj(x)))


class Zaya1VLVisionBlock(nn.Module):
    def __init__(self, config: VisionConfig) -> None:
        super().__init__()
        self.norm1 = nn.RMSNorm(config.hidden_size, eps=1e-6)
        self.norm2 = nn.RMSNorm(config.hidden_size, eps=1e-6)

        self.attn = Attention(dim=config.hidden_size, num_heads=config.num_heads)
        self.mlp = MLP(dim=config.hidden_size, hidden_dim=config.intermediate_size)

    def __call__(self, hidden_states, cu_seqlens, rotary_pos_emb) -> mx.array:
        hidden_states = hidden_states + self.attn(
            self.norm1(hidden_states),
            cu_seqlens=cu_seqlens,
            rotary_pos_emb=rotary_pos_emb,
        )
        hidden_states = hidden_states + self.mlp(self.norm2(hidden_states))
        return hidden_states


class VisionModel(Qwen2_5VisionModel):
    def __init__(self, config: VisionConfig) -> None:
        nn.Module.__init__(self)
        self.config = config
        self.model_type = config.model_type
        if self.model_type != "qwen2_5_vl":
            raise ValueError(f"Unsupported model type: {self.model_type}")
        self.spatial_merge_size = config.spatial_merge_size

        self.patch_embed = PatchEmbed(
            patch_size=config.patch_size,
            temporal_patch_size=config.temporal_patch_size,
            in_channels=config.in_channels,
            hidden_size=config.hidden_size,
        )

        self.window_size = config.window_size
        self.patch_size = config.patch_size
        self.spatial_merge_unit = self.spatial_merge_size * self.spatial_merge_size
        self.fullatt_block_indexes = config.fullatt_block_indexes
        head_dim = config.hidden_size // config.num_heads
        self.rotary_pos_emb = VisionRotaryEmbedding(head_dim // 2)

        self.blocks = [Zaya1VLVisionBlock(config) for _ in range(config.depth)]
        self.merger = PatchMerger(
            dim=config.out_hidden_size,
            context_dim=config.hidden_size,
            spatial_merge_size=config.spatial_merge_size,
        )
