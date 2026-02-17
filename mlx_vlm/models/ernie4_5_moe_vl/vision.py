"""DFNRope Vision Transformer for ERNIE 4.5 VL."""

from typing import Optional

import mlx.core as mx
import mlx.nn as nn
import numpy as np

from .config import VisionConfig


def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return mx.concatenate([-x2, x1], axis=-1)


def apply_rotary_pos_emb_vision(tensor: mx.array, freqs: mx.array) -> mx.array:
    """Applies Rotary Position Embedding to the input tensors.

    Args:
        tensor: The input tensor.
        freqs: The frequencies used for the rotation.

    Returns:
        output: the tensor rotated using the Rotary Position Embedding.
    """
    orig_dtype = tensor.dtype
    tensor = tensor.astype(mx.float32)
    cos = mx.cos(freqs)
    sin = mx.sin(freqs)
    # cos [θ0,θ1,θ2......θd/2-1] -> cos_pos [θ0,θ0,θ1,θ1,θ2,θ2......θd/2-1,θd/2-1]
    cos = mx.expand_dims(cos, axis=1)
    cos = mx.tile(cos, (1, 1, 2))
    cos = mx.expand_dims(cos, axis=0)

    sin = mx.expand_dims(sin, axis=1)
    sin = mx.tile(sin, (1, 1, 2))
    sin = mx.expand_dims(sin, axis=0)

    output = tensor * cos + rotate_half(tensor) * sin
    return output.astype(orig_dtype)


class VisionRotaryEmbedding(nn.Module):
    """Rotary position embedding for vision transformer."""

    def __init__(self, dim: int, theta: float = 10000.0) -> None:
        super().__init__()
        self.dim = dim
        self.theta = theta

    def __call__(self, seqlen: int) -> mx.array:
        inv_freq = 1.0 / (
            self.theta ** (mx.arange(0, self.dim, 2, dtype=mx.float32) / self.dim)
        )
        if isinstance(seqlen, mx.array):
            seqlen = seqlen.item()
        seq = mx.arange(seqlen, dtype=inv_freq.dtype)
        freqs = mx.outer(seq, inv_freq)
        return freqs


class PatchEmbed(nn.Module):
    """Linear patch embedding for DFNRope Vision Transformer."""

    def __init__(
        self,
        patch_size: int = 14,
        in_channels: int = 3,
        embed_dim: int = 1152,
    ) -> None:
        super().__init__()
        self.patch_size = patch_size
        self.in_channels = in_channels
        self.embed_dim = embed_dim
        # Linear projection: in_channels * patch_size * patch_size -> embed_dim
        self.proj = nn.Linear(
            in_channels * patch_size * patch_size, embed_dim, bias=False
        )

    def __call__(self, hidden_states: mx.array) -> mx.array:
        """
        Args:
            hidden_states: Input tensor of shape [num_patches, in_channels * patch_size * patch_size]
        Returns:
            Output tensor of shape [num_patches, embed_dim]
        """
        target_dtype = self.proj.weight.dtype
        hidden_states = self.proj(hidden_states.astype(target_dtype))
        return hidden_states


class VisionMLP(nn.Module):
    """MLP for vision transformer block."""

    def __init__(
        self, dim: int, hidden_dim: int, hidden_act: str = "quick_gelu"
    ) -> None:
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, dim)
        self.hidden_act = hidden_act

    def __call__(self, x: mx.array) -> mx.array:
        x = self.fc1(x)
        if self.hidden_act == "quick_gelu":
            x = x * mx.sigmoid(1.702 * x)
        elif self.hidden_act == "gelu":
            x = nn.gelu(x)
        elif self.hidden_act == "silu":
            x = nn.silu(x)
        else:
            x = nn.gelu(x)
        return self.fc2(x)


class VisionAttention(nn.Module):
    """Multi-head attention for vision transformer."""

    def __init__(self, dim: int, num_heads: int = 16) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim**-0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=True)
        self.proj = nn.Linear(dim, dim)

    def __call__(
        self,
        x: mx.array,
        cu_seqlens: mx.array,
        rotary_pos_emb: Optional[mx.array] = None,
    ) -> mx.array:
        """Forward function for vision attention."""
        seq_length = x.shape[0]
        qkv = (
            self.qkv(x).reshape(seq_length, 3, self.num_heads, -1).transpose(1, 0, 2, 3)
        )
        q, k, v = mx.split(qkv, 3)

        q = apply_rotary_pos_emb_vision(mx.expand_dims(q, 0), rotary_pos_emb)[0]
        k = apply_rotary_pos_emb_vision(mx.expand_dims(k, 0), rotary_pos_emb)[0]

        q = q.transpose(0, 2, 1, 3)
        k = k.transpose(0, 2, 1, 3)
        v = v.transpose(0, 2, 1, 3)

        lengths = (cu_seqlens[1:] - cu_seqlens[:-1]).tolist()
        splits = [
            mx.split(tensor, [lengths[0], sum(lengths[:2])], axis=2)
            for tensor in (q, k, v)
        ]

        attn_outputs = []
        for q, k, v in zip(*splits):
            output = mx.fast.scaled_dot_product_attention(
                q, k, v, scale=self.scale, mask=None
            )
            attn_outputs.append(output)

        output = mx.concatenate(attn_outputs, axis=2)
        output = output.transpose(0, 2, 1, 3).reshape(seq_length, -1)
        return self.proj(output)


class DFNRopeVisionBlock(nn.Module):
    """DFNRope Vision Transformer block."""

    def __init__(self, config: VisionConfig) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(config.embed_dim, eps=config.layer_norm_eps)
        self.norm2 = nn.LayerNorm(config.embed_dim, eps=config.layer_norm_eps)

        mlp_hidden_dim = int(config.embed_dim * config.mlp_ratio)
        self.attn = VisionAttention(config.embed_dim, num_heads=config.num_heads)
        self.mlp = VisionMLP(
            dim=config.embed_dim,
            hidden_dim=mlp_hidden_dim,
            hidden_act=config.hidden_act,
        )

    def __call__(
        self, hidden_states: mx.array, cu_seqlens: mx.array, rotary_pos_emb: mx.array
    ) -> mx.array:
        hidden_states = hidden_states + self.attn(
            self.norm1(hidden_states),
            cu_seqlens=cu_seqlens,
            rotary_pos_emb=rotary_pos_emb,
        )
        hidden_states = hidden_states + self.mlp(self.norm2(hidden_states))
        return hidden_states


class VisionModel(nn.Module):
    """DFNRope Vision Transformer for ERNIE 4.5 VL."""

    def __init__(self, config: VisionConfig) -> None:
        super().__init__()
        self.config = config
        self.model_type = config.model_type
        self.spatial_merge_size = config.spatial_merge_size

        self.patch_embed = PatchEmbed(
            patch_size=config.patch_size,
            in_channels=config.in_channels,
            embed_dim=config.embed_dim,
        )

        head_dim = config.embed_dim // config.num_heads
        self.rotary_pos_emb = VisionRotaryEmbedding(head_dim // 2)

        self.blocks = [DFNRopeVisionBlock(config) for _ in range(config.depth)]
        self.ln = nn.LayerNorm(config.embed_dim, eps=config.layer_norm_eps)

    def rot_pos_emb(self, grid_thw: mx.array, num_pad: int = 0) -> mx.array:
        """Compute rotary position embedding for vision.

        Args:
            grid_thw: Grid dimensions [batch, 3] containing (t, h, w)
            num_pad: Number of padding tokens

        Returns:
            Rotary position embedding tensor
        """
        pos_ids = []
        grid_hw_array = np.array(grid_thw.tolist(), dtype=np.int64)

        for t, h, w in grid_hw_array:
            hpos_ids = np.arange(h).reshape(-1, 1)
            hpos_ids = np.tile(hpos_ids, (1, w))
            hpos_ids = hpos_ids.reshape(
                h // self.spatial_merge_size,
                self.spatial_merge_size,
                w // self.spatial_merge_size,
                self.spatial_merge_size,
            )
            hpos_ids = np.transpose(hpos_ids, (0, 2, 1, 3))
            hpos_ids = hpos_ids.flatten()

            wpos_ids = np.arange(w).reshape(1, -1)
            wpos_ids = np.tile(wpos_ids, (h, 1))
            wpos_ids = wpos_ids.reshape(
                h // self.spatial_merge_size,
                self.spatial_merge_size,
                w // self.spatial_merge_size,
                self.spatial_merge_size,
            )
            wpos_ids = np.transpose(wpos_ids, (0, 2, 1, 3))
            wpos_ids = wpos_ids.flatten()

            stacked_ids = np.stack([hpos_ids, wpos_ids], axis=-1)
            tiled_ids = np.tile(stacked_ids, (t, 1))
            pos_ids.append(tiled_ids)

        pos_ids = np.concatenate(pos_ids, axis=0)
        if num_pad > 0:
            pos_ids = np.concatenate(
                [pos_ids, np.zeros((num_pad, 2), dtype=pos_ids.dtype)], axis=0
            )

        max_grid_size = int(np.max(grid_hw_array[:, 1:]))
        rotary_pos_emb_full = self.rotary_pos_emb(max_grid_size)
        pos_ids_mx = mx.array(pos_ids, dtype=mx.int32)
        rotary_pos_emb = rotary_pos_emb_full[pos_ids_mx].reshape(pos_ids.shape[0], -1)

        return rotary_pos_emb

    def __call__(
        self,
        hidden_states: mx.array,
        grid_thw: mx.array,
        output_hidden_states: Optional[bool] = None,
        num_pad: int = 0,
    ) -> mx.array:
        """Forward pass through the vision model.

        Args:
            hidden_states: Input pixel values [num_patches, channels * patch_h * patch_w]
            grid_thw: Grid dimensions [batch, 3]
            output_hidden_states: Whether to output hidden states
            num_pad: Number of padding tokens

        Returns:
            Vision features
        """
        hidden_states = self.patch_embed(hidden_states)
        rotary_pos_emb = self.rot_pos_emb(grid_thw, num_pad=num_pad)

        # Compute cumulative sequence lengths
        cu_seqlens = mx.zeros(1, dtype=mx.int32)
        for i in range(grid_thw.shape[0]):
            t, h, w = grid_thw[i].tolist()
            seq_len = t * h * w
            cu_seqlens = mx.concatenate([cu_seqlens, cu_seqlens[-1:] + seq_len])

        if num_pad > 0:
            cu_seqlens = mx.concatenate([cu_seqlens, cu_seqlens[-1:] + num_pad])

        encoder_states = (hidden_states,) if output_hidden_states else None

        for blk in self.blocks:
            hidden_states = blk(
                hidden_states,
                cu_seqlens=cu_seqlens,
                rotary_pos_emb=rotary_pos_emb,
            )
            if output_hidden_states:
                encoder_states = encoder_states + (hidden_states,)

        hidden_states = self.ln(hidden_states)
        return hidden_states

    def sanitize(self, weights):
        """Sanitize weights for loading."""
        sanitized_weights = {}
        for k, v in weights.items():
            if "position_ids" in k:
                continue
            sanitized_weights[k] = v
        return sanitized_weights
