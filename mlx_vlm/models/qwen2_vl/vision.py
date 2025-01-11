import inspect
from dataclasses import dataclass
from typing import Optional

import mlx.core as mx
import mlx.nn as nn


@dataclass
class VisionConfig:
    model_type: str = "qwen2_vl"
    depth: int = 32
    embed_dim: int = 1280
    hidden_size: int = 1536
    num_heads: int = 16
    image_size: int = 384
    patch_size: int = 14
    vocab_size: int = 32000
    mlp_ratio: float = 4.0
    in_channels: int = 3
    layer_norm_eps: float = 1e-6
    spatial_patch_size: int = 14
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
    if len(shape) not in [4, 5]:
        return False

    B, out_channels, kH, KW, t = shape

    if t == 3:
        return True

    # Check if out_channels is the largest, and kH and KW are the same
    if (out_channels >= kH) and (out_channels >= KW) and (kH == KW):
        return True
    else:
        return False


def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return mx.concatenate([-x2, x1], axis=-1)


def apply_rotary_pos_emb_vision(tensor, freqs) -> mx.array:
    orig_dtype = tensor.dtype

    cos = mx.cos(freqs)
    sin = mx.sin(freqs)

    cos = mx.expand_dims(cos, axis=1)  # Equivalent to unsqueeze(1)
    cos = mx.tile(cos, (1, 1, 2))  # Equivalent to repeat(1, 1, 2)
    cos = mx.expand_dims(cos, axis=0)  # Equivalent to [None, ...]

    sin = mx.expand_dims(sin, axis=1)  # Equivalent to unsqueeze(1)
    sin = mx.tile(sin, (1, 1, 2))  # Equivalent to repeat(1, 1, 2)
    sin = mx.expand_dims(sin, axis=0)  # Equivalent to [None, ...]

    output = (tensor * cos) + (rotate_half(tensor) * sin)
    return output.astype(orig_dtype)


class VisionRotaryEmbedding(nn.Module):
    def __init__(self, dim: int, theta: float = 10000.0) -> None:
        super().__init__()
        self.dim = dim
        self.theta = theta

    def __call__(self, seqlen: int) -> mx.array:
        inv_freq = 1.0 / (
            self.theta ** (mx.arange(0, self.dim, 2, dtype=mx.float32) / self.dim)
        )
        seq = mx.arange(seqlen.tolist(), dtype=inv_freq.dtype)
        freqs = mx.outer(seq, inv_freq)
        return freqs


class PatchEmbed(nn.Module):
    def __init__(
        self,
        patch_size: int = 14,
        temporal_patch_size: int = 2,
        in_channels: int = 3,
        embed_dim: int = 1152,
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
            self.in_channels,
            self.temporal_patch_size,
            self.patch_size,
            self.patch_size,
        ).moveaxis(1, 4)

        hidden_states = self.proj(hidden_states)
        hidden_states = hidden_states.reshape(-1, self.embed_dim)
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
        x = self.ln_q(x).reshape(-1, self.hidden_size)
        for layer in self.mlp:
            x = layer(x)
        return x


class Attention(nn.Module):
    def __init__(self, dim: int, num_heads: int = 16) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim = dim // num_heads
        self.scale = head_dim**-0.5
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
        attention_mask = mx.ones((1, seq_length, seq_length), dtype=x.dtype)

        for i in range(1, len(cu_seqlens)):
            start = int(cu_seqlens[i - 1])
            end = int(cu_seqlens[i])
            attention_mask[start:end, start:end] = 0

        q = q.transpose(0, 2, 1, 3)
        k = k.transpose(0, 2, 1, 3)
        v = v.transpose(0, 2, 1, 3)

        output = mx.fast.scaled_dot_product_attention(
            q, k, v, scale=self.scale, mask=attention_mask
        )
        output = output.transpose(0, 2, 1, 3)
        output = output.reshape(seq_length, -1)
        return self.proj(output)


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

    def rot_pos_emb(self, grid_thw):
        pos_ids = []

        for t, h, w in grid_thw:
            h, w = int(h), int(w)  # Ensure h and w are integers
            hpos_ids = mx.expand_dims(mx.arange(h), 1)
            hpos_ids = mx.repeat(hpos_ids, w, axis=1)
            hpos_ids = hpos_ids.reshape(
                h // self.spatial_merge_size,
                self.spatial_merge_size,
                w // self.spatial_merge_size,
                self.spatial_merge_size,
            )
            hpos_ids = mx.transpose(hpos_ids, (0, 2, 1, 3))
            hpos_ids = hpos_ids.flatten()

            wpos_ids = mx.expand_dims(mx.arange(w), 0)
            wpos_ids = mx.repeat(wpos_ids, h, axis=0)
            wpos_ids = wpos_ids.reshape(
                h // self.spatial_merge_size,
                self.spatial_merge_size,
                w // self.spatial_merge_size,
                self.spatial_merge_size,
            )
            wpos_ids = mx.transpose(wpos_ids, (0, 2, 1, 3))
            wpos_ids = wpos_ids.flatten()

            stacked_pos_ids = mx.stack([hpos_ids, wpos_ids], axis=-1)
            pos_ids.append(mx.repeat(stacked_pos_ids, t, axis=0))

        pos_ids = mx.concatenate(pos_ids, axis=0)
        max_grid_size = mx.max(grid_thw[:, 1:])
        rotary_pos_emb_full = self.rotary_pos_emb(max_grid_size)

        rotary_pos_emb_full = rotary_pos_emb_full[pos_ids]

        return rotary_pos_emb_full.reshape(pos_ids.shape[0], -1)

    def __call__(
        self,
        hidden_states: mx.array,
        grid_thw: mx.array,
        output_hidden_states: Optional[bool] = None,
    ) -> mx.array:

        hidden_states = self.patch_embed(hidden_states)
        rotary_pos_emb = self.rot_pos_emb(grid_thw)

        # Assuming grid_thw has shape (batch_size, 3)
        batch_size = grid_thw.shape[0]

        # Calculate cu_seqlens for each item in the batch
        cu_seqlens = []
        for i in range(batch_size):
            seq_len = grid_thw[i, 1] * grid_thw[i, 2]
            cu_seqlens.append(mx.repeat(seq_len, grid_thw[i, 0]))

        # Concatenate the cu_seqlens for all items in the batch
        cu_seqlens = mx.concatenate(cu_seqlens)

        cu_seqlens = mx.cumsum(cu_seqlens.astype(mx.int32), axis=0)
        cu_seqlens = mx.pad(cu_seqlens, (1, 0), mode="constant", constant_values=0)

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
