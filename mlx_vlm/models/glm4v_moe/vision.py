from typing import Optional

import mlx.core as mx
import mlx.nn as nn

from ..kernels import grid_sample
from .config import VisionConfig


def check_array_shape(arr):
    shape = arr.shape

    # Check if the shape has 4 or 5 dimensions
    if len(shape) == 4:
        out_channels, kH, KW, _ = shape
        # Check if out_channels is the largest, and kH and KW are the same
        return (out_channels >= kH) and (out_channels >= KW) and (kH == KW)
    elif len(shape) == 5:
        B, out_channels, kH, KW, t = shape
        # Special case for temporal dimension
        if t == 3:
            return True
        # Check if out_channels is the largest, and kH and KW are the same
        return (out_channels >= kH) and (out_channels >= KW) and (kH == KW)
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


class Glm4vMoeVisionRotaryEmbedding(nn.Module):
    def __init__(self, dim: int, theta: float = 10000.0) -> None:
        super().__init__()
        self.dim = dim
        self.theta = theta

    def __call__(self, seqlen: int) -> mx.array:
        inv_freq = 1.0 / (
            self.theta ** (mx.arange(0, self.dim, 2, dtype=mx.float32) / self.dim)
        )
        seq = mx.arange(seqlen.item(), dtype=inv_freq.dtype)
        freqs = mx.outer(seq, inv_freq)
        return freqs


class Glm4vVisionEmbeddings(nn.Module):
    def __init__(self, config: VisionConfig):
        super().__init__()
        self.config = config
        self.embed_dim = config.hidden_size
        self.image_size = config.image_size
        self.patch_size = config.patch_size

        self.num_patches = (self.image_size // self.patch_size) ** 2
        self.num_positions = self.num_patches
        self.position_embedding = nn.Embedding(self.num_positions, self.embed_dim)

    def __call__(self, embeddings, lengths, image_shapes, h_coords, w_coords):

        # Get position embedding parameters
        pos_embed_weight = self.position_embedding.weight
        hidden_size = pos_embed_weight.shape[1]
        total_seq = h_coords.shape[0]

        # Handle empty sequence case
        if total_seq == 0:
            adapted_pos_embed = mx.empty(0, hidden_size, dtype=pos_embed_weight.dtype)
        else:
            # Convert inputs to tensors if needed
            if isinstance(lengths, list):
                lengths = mx.array(lengths, dtype=mx.int32)
            if not isinstance(image_shapes, mx.array):
                image_shapes = mx.array(image_shapes, dtype=mx.int32)

            # Prepare 2D position embedding
            orig_size_sq = pos_embed_weight.shape[0]
            orig_size = int(orig_size_sq**0.5)
            pos_embed_2d = (
                pos_embed_weight.reshape(orig_size, orig_size, hidden_size)
                .transpose(2, 0, 1)[None, ...]
                .astype(mx.float32)
            )

            # Calculate target dimensions for each patch
            target_h = mx.concatenate(
                [mx.repeat(image_shapes[i, 1], lengths[i]) for i in range(len(lengths))]
            ).astype(mx.float32)
            target_w = mx.concatenate(
                [mx.repeat(image_shapes[i, 2], lengths[i]) for i in range(len(lengths))]
            ).astype(mx.float32)

            # Normalize coordinates to [-1, 1] range for grid_sample
            h_coords = h_coords.astype(mx.float32)
            w_coords = w_coords.astype(mx.float32)
            norm_w = ((w_coords + 0.5) / target_w) * 2 - 1
            norm_h = ((h_coords + 0.5) / target_h) * 2 - 1

            # Create sampling grid
            grid = mx.stack((norm_w, norm_h), axis=-1)[None, :, None, ...]

            # Perform bicubic interpolation
            interpolated_embed_fp32 = grid_sample(
                pos_embed_2d.transpose(0, 2, 3, 1),
                grid,
            )

            # Reshape and convert back to original dtype
            adapted_pos_embed_fp32 = interpolated_embed_fp32.squeeze(0).squeeze(1)
            adapted_pos_embed = adapted_pos_embed_fp32.astype(pos_embed_weight.dtype)

        # Add adapted position encoding to embeddings
        embeddings = embeddings + adapted_pos_embed
        return embeddings


class Glm4vMoeVisionPatchEmbed(nn.Module):
    def __init__(self, config: VisionConfig) -> None:
        super().__init__()
        self.config = config
        self.patch_size = config.patch_size
        self.temporal_patch_size = config.temporal_patch_size
        self.in_channels = config.in_channels
        self.embed_dim = config.hidden_size

        kernel_size = [self.temporal_patch_size, self.patch_size, self.patch_size]
        self.proj = nn.Conv3d(
            self.in_channels,
            self.embed_dim,
            kernel_size=kernel_size,
            stride=kernel_size,
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


class Glm4vMoeVisionPatchMerger(nn.Module):
    def __init__(self, dim: int, context_dim: int, bias: bool = False) -> None:
        super().__init__()

        self.proj = nn.Linear(dim, dim, bias=bias)
        self.post_projection_norm = nn.LayerNorm(dim)
        self.gate_proj = nn.Linear(dim, context_dim, bias=bias)
        self.up_proj = nn.Linear(dim, context_dim, bias=bias)
        self.down_proj = nn.Linear(context_dim, dim, bias=bias)

    def __call__(self, x: mx.array) -> mx.array:
        x = self.proj(x)
        x = nn.gelu(self.post_projection_norm(x))
        return self.down_proj(nn.silu(self.gate_proj(x)) * self.up_proj(x))


class Glm4vMoeVisionAttention(nn.Module):
    def __init__(self, dim: int, num_heads: int = 16) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim = dim // num_heads
        self.scale = head_dim**-0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=False)
        self.proj = nn.Linear(dim, dim, bias=False)

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

        attention_mask = mx.full(
            (1, seq_length, seq_length), mx.finfo(q.dtype).min, dtype=q.dtype
        )

        for i in range(1, len(cu_seqlens)):
            start = int(cu_seqlens[i - 1])
            end = int(cu_seqlens[i])
            attention_mask[..., start:end, start:end] = 0

        q = q.transpose(0, 2, 1, 3)
        k = k.transpose(0, 2, 1, 3)
        v = v.transpose(0, 2, 1, 3)

        output = mx.fast.scaled_dot_product_attention(
            q, k, v, scale=self.scale, mask=attention_mask
        )
        output = output.transpose(0, 2, 1, 3)
        output = output.reshape(seq_length, -1)
        return self.proj(output)


class Glm4vMoeVisionMLP(nn.Module):
    def __init__(self, dim, hidden_dim):
        super().__init__()
        self.gate_proj = nn.Linear(dim, hidden_dim, bias=False)
        self.up_proj = nn.Linear(dim, hidden_dim, bias=False)
        self.down_proj = nn.Linear(hidden_dim, dim, bias=False)

    def __call__(self, x: mx.array) -> mx.array:
        return self.down_proj(nn.silu(self.gate_proj(x)) * self.up_proj(x))


class Glm4vMoeVisionBlock(nn.Module):
    def __init__(self, config: VisionConfig) -> None:
        super().__init__()
        self.norm1 = nn.RMSNorm(config.hidden_size, eps=1e-6)
        self.norm2 = nn.RMSNorm(config.hidden_size, eps=1e-6)

        self.attn = Glm4vMoeVisionAttention(
            dim=config.hidden_size, num_heads=config.num_heads
        )
        self.mlp = Glm4vMoeVisionMLP(
            dim=config.hidden_size, hidden_dim=config.out_hidden_size
        )

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
        if self.model_type != "glm4v_moe":
            raise ValueError(f"Unsupported model type: {self.model_type}")
        self.spatial_merge_size = config.spatial_merge_size

        self.embeddings = Glm4vVisionEmbeddings(config)
        self.patch_embed = Glm4vMoeVisionPatchEmbed(
            config=config,
        )

        self.window_size = config.window_size
        self.patch_size = config.patch_size
        self.spatial_merge_unit = self.spatial_merge_size * self.spatial_merge_size

        head_dim = config.hidden_size // config.num_heads
        self.rotary_pos_emb = Glm4vMoeVisionRotaryEmbedding(head_dim // 2)

        self.blocks = [Glm4vMoeVisionBlock(config) for _ in range(config.depth)]
        self.merger = Glm4vMoeVisionPatchMerger(
            dim=config.out_hidden_size, context_dim=config.intermediate_size
        )

        self.post_conv_layernorm = nn.RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )
        self.downsample = nn.Conv2d(
            in_channels=config.hidden_size,
            out_channels=config.out_hidden_size,
            kernel_size=config.spatial_merge_size,
            stride=config.spatial_merge_size,
        )
        self.post_layernorm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def rot_pos_emb(self, grid_thw):
        pos_ids = []

        for t, h, w in grid_thw.tolist():
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
            pos_ids.append(mx.tile(stacked_pos_ids, (t, 1)))

        pos_ids = mx.concatenate(pos_ids, axis=0)
        max_grid_size = mx.max(grid_thw[:, 1:])
        rotary_pos_emb_full = self.rotary_pos_emb(max_grid_size)
        rotary_pos_emb = rotary_pos_emb_full[pos_ids]

        return rotary_pos_emb.reshape(pos_ids.shape[0], -1), pos_ids

    def __call__(
        self,
        hidden_states: mx.array,
        grid_thw: mx.array,
        output_hidden_states: Optional[bool] = None,
    ) -> mx.array:

        hidden_states = self.patch_embed(hidden_states)
        hidden_states = self.post_conv_layernorm(hidden_states)
        rotary_pos_emb, image_type_ids = self.rot_pos_emb(grid_thw)

        seq_lens = grid_thw[:, 1] * grid_thw[:, 2]
        repeats = grid_thw[:, 0]
        repeated_values = []
        for i, (seq_len, repeat_count) in enumerate(
            zip(seq_lens.tolist(), repeats.tolist())
        ):
            repeated_values.extend([seq_len] * repeat_count)

        cu_seqlens = mx.array(repeated_values).cumsum(axis=0)
        cu_seqlens = mx.pad(cu_seqlens, (1, 0), constant_values=0)
        seqlens = (cu_seqlens[1:] - cu_seqlens[:-1]).tolist()
        hidden_states = self.embeddings(
            hidden_states, seqlens, grid_thw, image_type_ids[:, 0], image_type_ids[:, 1]
        )

        for blk in self.blocks:
            hidden_states = blk(
                hidden_states,
                cu_seqlens=cu_seqlens,
                rotary_pos_emb=rotary_pos_emb,
            )

        hidden_states = self.post_layernorm(hidden_states)

        hidden_states = hidden_states.reshape(
            -1,
            self.spatial_merge_size,
            self.spatial_merge_size,
            hidden_states.shape[-1],
        )
        hidden_states = self.downsample(hidden_states).reshape(
            -1, self.config.out_hidden_size
        )

        hidden_states = self.merger(hidden_states)
        return hidden_states

    def sanitize(self, weights):
        sanitized_weights = {}
        for k, v in weights.items():
            if "position_ids" in k:
                # Remove unused position_ids
                continue
            elif "patch_embed.proj.weight" in k or "downsample.weight" in k:
                # PyTorch conv2d weight tensors have shape:
                #   [out_channels, in_channels, kH, KW]
                # MLX conv2d expects the weight be of shape:
                #   [out_channels, kH, KW, in_channels]
                if check_array_shape(v):
                    sanitized_weights[k] = v
                else:
                    if v.ndim == 5:
                        sanitized_weights[k] = v.transpose(0, 2, 3, 4, 1)
                    if v.ndim == 4:
                        sanitized_weights[k] = v.transpose(0, 2, 3, 1)
            else:
                sanitized_weights[k] = v

        return sanitized_weights
