import inspect
import math
from dataclasses import dataclass
from typing import Optional

import mlx.core as mx
import mlx.nn as nn
import numpy as np


@dataclass
class VisionConfig:
    model_type: str = "qwen2_vl"
    depth: int = 32
    embed_dim: int = 768
    hidden_size: int = 3584
    num_heads: int = 16
    image_size: int = 384
    patch_size: int = 14
    vocab_size: int = 32000
    mlp_ratio: float = 4.0
    in_channels: int = 3
    layer_norm_eps: float = 1e-6
    spatial_merge_size: int = 2
    temporal_patch_size: int = 2

    @classmethod
    def from_dict(cls, params):
        return cls(
            **{
                k: v
                for k, v in params.items()
                if k in inspect.signature(cls).parameters
            }
        )


def check_array_shape(arr):
    shape = arr.shape

    # Check if the shape has 4 dimensions
    if len(shape) != 4:
        return False

    out_channels, kH, KW, _ = shape

    # Check if out_channels is the largest, and kH and KW are the same
    if (out_channels >= kH) and (out_channels >= KW) and (kH == KW):
        return True
    else:
        return False


class Qwen2RotaryEmbedding(nn.Module):
    def __init__(self, dim, max_position_embeddings=2048, base=10000):
        super().__init__()
        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        inv_freq = 1.0 / (
            self.base
            ** (mx.arange(0, self.dim, 2, dtype=mx.int64).astype(mx.float32) / self.dim)
        )
        self.inv_freq = mx.array(inv_freq)
        self._set_cos_sin_cache(seq_len=max_position_embeddings, dtype=mx.float32)

    def _set_cos_sin_cache(self, seq_len, dtype):
        self.max_seq_len_cached = seq_len
        t = mx.arange(self.max_seq_len_cached, dtype=mx.int64).astype(
            self.inv_freq.dtype
        )
        freqs = mx.outer(t, self.inv_freq)
        emb = mx.concatenate((freqs, freqs), axis=-1)
        self.cos_cached = mx.cos(emb).astype(dtype)
        self.sin_cached = mx.sin(emb).astype(dtype)

    def __call__(self, x, seq_len=None):
        if seq_len > self.max_seq_len_cached:
            self._set_cos_sin_cache(seq_len=seq_len, dtype=x.dtype)

        return (
            self.cos_cached[:seq_len].astype(x.dtype),
            self.sin_cached[:seq_len].astype(x.dtype),
        )


def rotate_half(x):
    x1, x2 = mx.split(x, 2, axis=-1)
    return mx.concatenate((-x2, x1), axis=-1)


def apply_multimodal_rotary_pos_emb(
    q, k, cos, sin, position_ids, mrope_section, unsqueeze_dim=1
):
    cos = cos[position_ids]
    sin = sin[position_ids]
    mrope_section = mrope_section * 2
    cos = mx.concatenate(
        [m[i % 3] for i, m in enumerate(mx.split(cos, mrope_section, axis=-1))], axis=-1
    )
    sin = mx.concatenate(
        [m[i % 3] for i, m in enumerate(mx.split(sin, mrope_section, axis=-1))], axis=-1
    )
    cos = mx.expand_dims(cos, axis=unsqueeze_dim)
    sin = mx.expand_dims(sin, axis=unsqueeze_dim)

    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


def apply_rotary_pos_emb_vision(tensor: mx.array, freqs: mx.array) -> mx.array:
    orig_dtype = tensor.dtype
    tensor = tensor.astype(mx.float32)
    cos = mx.cos(freqs)
    sin = mx.sin(freqs)
    cos = mx.expand_dims(mx.repeat(mx.expand_dims(cos, 1), 2, axis=1), 0)
    sin = mx.expand_dims(mx.repeat(mx.expand_dims(sin, 1), 2, axis=1), 0)
    output = (tensor * cos) + (rotate_half(tensor) * sin)
    output = output.astype(orig_dtype)
    return output


class VisionRotaryEmbedding(nn.Module):
    def __init__(self, dim: int, theta: float = 10000.0) -> None:
        super().__init__()
        self.dim = dim
        self.theta = theta

    def __call__(self, seqlen: int) -> mx.array:
        inv_freq = 1.0 / (
            self.theta ** (mx.arange(0, self.dim, 2, dtype=mx.float32) / self.dim)
        )
        inv_freq = mx.array(inv_freq)
        seq = mx.arange(seqlen, dtype=inv_freq.dtype)
        freqs = mx.outer(seq, inv_freq)
        return freqs


class PatchEmbed(nn.Module):
    def __init__(
        self,
        patch_size: int = 14,
        temporal_patch_size: int = 2,
        in_channels: int = 3,
        embed_dim: int = 1280,
    ) -> None:
        super().__init__()
        self.patch_size = patch_size
        self.temporal_patch_size = temporal_patch_size
        self.in_channels = in_channels
        self.embed_dim = embed_dim

        kernel_size = [temporal_patch_size, patch_size, patch_size]
        self.proj = nn.Conv3d(
            in_channels,
            embed_dim,
            kernel_size=kernel_size,
            stride=kernel_size,
            bias=False,
        )

    def __call__(self, hidden_states: mx.array) -> mx.array:
        hidden_states = hidden_states.reshape(
            -1,
            self.temporal_patch_size,
            self.patch_size,
            self.patch_size,
            self.in_channels,
        )
        hidden_states = self.proj(hidden_states).reshape(-1, self.embed_dim)
        return hidden_states


class PatchMerger(nn.Module):
    def __init__(self, dim: int, context_dim: int, spatial_merge_size: int = 2) -> None:
        super().__init__()
        self.hidden_size = context_dim * (spatial_merge_size**2)
        self.ln_q = nn.LayerNorm(context_dim, eps=1e-6)
        self.mlp = [
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.GELU(),
            nn.Linear(self.hidden_size, dim),
        ]

    def __call__(self, x: mx.array) -> mx.array:
        x = self.mlp(self.ln_q(x).reshape(-1, self.hidden_size))
        return x


class Attention(nn.Module):
    def __init__(self, dim: int, num_heads: int = 16) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.qkv = nn.Linear(dim, dim * 3, bias=True)
        self.proj = nn.Linear(dim, dim)

    def __call__(
        self, x: mx.array, cu_seqlens: mx.array, rotary_pos_emb: mx.array = None
    ) -> mx.array:
        seq_length = x.shape[0]
        qkv = (
            self.qkv(x).reshape(seq_length, 3, self.num_heads, -1).transpose(1, 0, 2, 3)
        )
        q, k, v = mx.split(qkv, 3)
        q = apply_rotary_pos_emb_vision(mx.expand_dims(q, 0), rotary_pos_emb)[0]
        k = apply_rotary_pos_emb_vision(mx.expand_dims(k, 0), rotary_pos_emb)[0]

        attention_mask = mx.zeros((1, seq_length, seq_length), dtype=mx.bool)
        for i in range(1, len(cu_seqlens)):
            attention_mask = mx.index_update(
                attention_mask,
                mx.index[
                    :,
                    cu_seqlens[i - 1] : cu_seqlens[i],
                    cu_seqlens[i - 1] : cu_seqlens[i],
                ],
                True,
            )

        q = q.transpose(0, 1)
        k = k.transpose(0, 1)
        v = v.transpose(0, 1)
        attn_weights = mx.matmul(q, k.transpose(0, 2, 1)) / math.sqrt(self.head_dim)
        attn_weights = attn_weights + attention_mask
        attn_weights = mx.softmax(attn_weights, axis=-1)
        attn_output = mx.matmul(attn_weights, v)
        attn_output = attn_output.transpose(0, 2, 1)
        attn_output = attn_output.reshape(seq_length, -1)
        attn_output = self.proj(attn_output)
        return attn_output


class MLP(nn.Module):
    def __init__(self, dim, hidden_dim):
        super().__init__()
        self.activation_fn = nn.GELU(approx="fast")
        self.fc1 = nn.Linear(dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, dim)

    def __call__(self, x: mx.array) -> mx.array:
        x = self.activation_fn(self.fc1(x))
        x = self.fc2(x)
        return x


class Qwen2VLVisionBlock(nn.Module):
    def __init__(self, config: VisionConfig) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(config.embed_dim, eps=1e-6)
        self.norm2 = nn.LayerNorm(config.embed_dim, eps=1e-6)
        mlp_hidden_dim = int(config.embed_dim * config.mlp_ratio)

        self.attn = Attention(dim=config.embed_dim, num_heads=config.num_heads)
        self.mlp = MLP(dim=config.embed_dim, hidden_dim=mlp_hidden_dim)

    def __call__(self, hidden_states, cu_seqlens, rotary_pos_emb) -> mx.array:
        hidden_states = hidden_states + self.attn(
            self.norm1(hidden_states),
            cu_seqlens=cu_seqlens,
            rotary_pos_emb=rotary_pos_emb,
        )
        hidden_states = hidden_states + self.mlp(self.norm2(hidden_states))
        return hidden_states


class VisionModel(nn.Module):

    def __init__(self, config: VisionConfig) -> None:
        super().__init__()
        self.config = config
        self.model_type = config.model_type
        if self.model_type != "qwen2_vl":
            raise ValueError(f"Unsupported model type: {self.model_type}")
        self.spatial_merge_size = config.spatial_merge_size

        self.patch_embed = PatchEmbed(
            patch_size=config.patch_size,
            temporal_patch_size=config.temporal_patch_size,
            in_channels=config.in_channels,
            embed_dim=config.embed_dim,
        )

        head_dim = config.embed_dim // config.num_heads
        self.rotary_pos_emb = VisionRotaryEmbedding(head_dim // 2)

        self.blocks = [Qwen2VLVisionBlock(config) for _ in range(config.depth)]
        self.merger = PatchMerger(dim=config.hidden_size, context_dim=config.embed_dim)

    import mlx.core as mx

    def rot_pos_emb(self, grid_thw):
        pos_ids = []
        for t, h, w in grid_thw:
            # Create hpos_ids
            hpos_ids = mx.repeat(mx.expand_dims(mx.arange(h), axis=1), w, axis=1)
            print(hpos_ids.shape, h, w)
            hpos_ids = hpos_ids.reshape(
                h // self.spatial_merge_size,
                self.spatial_merge_size,
                w // self.spatial_merge_size,
                self.spatial_merge_size,
            )
            hpos_ids = mx.transpose(hpos_ids, (0, 2, 1, 3))
            hpos_ids = hpos_ids.flatten()

            # Create wpos_ids
            wpos_ids = mx.broadcast_to(mx.arange(w)[None, :], (h, w))
            wpos_ids = wpos_ids.reshape(
                h // self.spatial_merge_size,
                self.spatial_merge_size,
                w // self.spatial_merge_size,
                self.spatial_merge_size,
            )
            wpos_ids = mx.transpose(wpos_ids, (0, 2, 1, 3))
            wpos_ids = wpos_ids.flatten()

            # Stack and tile (equivalent to repeat in PyTorch)
            stacked = mx.stack([hpos_ids, wpos_ids], axis=-1)
            tiled = mx.tile(stacked[None, ...], (t, 1, 1))
            pos_ids.append(tiled)

        pos_ids = mx.concatenate(pos_ids, axis=0)

        # Calculate max_grid_size
        max_grid_size = mx.max(
            mx.max(grid_thw[:, 1], axis=0), mx.max(grid_thw[:, 2], axis=0)
        )

        rotary_pos_emb_full = self.rotary_pos_emb(max_grid_size)
        rotary_pos_emb = rotary_pos_emb_full[pos_ids].reshape(
            -1, rotary_pos_emb_full.shape[-1]
        )

        return rotary_pos_emb

    def __call__(
        self,
        hidden_states: mx.array,
        grid_thw: mx.array,
        output_hidden_states: Optional[bool] = None,
    ) -> mx.array:
        hidden_states = self.patch_embed(hidden_states)
        rotary_pos_emb = self.rot_pos_emb(grid_thw)

        cu_seqlens = mx.cumsum(
            mx.repeat(grid_thw[:, 1] * grid_thw[:, 2], grid_thw[:, 0]), axis=0
        )
        cu_seqlens = mx.concatenate([mx.zeros(1, dtype=cu_seqlens.dtype), cu_seqlens])

        encoder_states = (hidden_states,) if output_hidden_states else None

        for blk in self.blocks:
            hidden_states = blk(
                hidden_states, cu_seqlens=cu_seqlens, rotary_pos_emb=rotary_pos_emb
            )
            if output_hidden_states:
                encoder_states = encoder_states + (hidden_states,)

        return self.merger(hidden_states)

    def sanitize(self, weights):
        sanitized_weights = {}
        for k, v in weights.items():
            if "position_ids" in k:
                # Remove unused position_ids
                continue
            elif "patch_embed.proj.weight" in k:
                # PyTorch conv2d weight tensors have shape:
                #   [out_channels, in_channels, kH, KW]
                # MLX conv2d expects the weight be of shape:
                #   [out_channels, kH, KW, in_channels]
                if check_array_shape(v):
                    sanitized_weights[k] = v
                else:
                    sanitized_weights[k] = v.transpose(0, 2, 3, 4, 1)
            else:
                sanitized_weights[k] = v

        return sanitized_weights
