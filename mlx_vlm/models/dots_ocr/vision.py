from typing import Optional

import mlx.core as mx
import mlx.nn as nn

from .config import VisionConfig


def check_array_shape(arr):
    shape = arr.shape

    if len(shape) != 4:
        return False

    out_channels, k_h, k_w, _ = shape
    return (out_channels >= k_h) and (out_channels >= k_w) and (k_h == k_w)


def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return mx.concatenate([-x2, x1], axis=-1)


def apply_rotary_pos_emb_vision(tensor, freqs) -> mx.array:
    orig_dtype = tensor.dtype

    cos = mx.cos(freqs)
    sin = mx.sin(freqs)

    cos = mx.expand_dims(cos, axis=1)
    cos = mx.tile(cos, (1, 1, 2))
    cos = mx.expand_dims(cos, axis=0)

    sin = mx.expand_dims(sin, axis=1)
    sin = mx.tile(sin, (1, 1, 2))
    sin = mx.expand_dims(sin, axis=0)

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
        seq = mx.arange(int(seqlen), dtype=inv_freq.dtype)
        return mx.outer(seq, inv_freq)


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.weight = mx.ones((dim,))
        self.eps = eps

    def __call__(self, x: mx.array) -> mx.array:
        input_dtype = x.dtype
        x = x.astype(mx.float32)
        x = x * mx.rsqrt(mx.mean(x**2, axis=-1, keepdims=True) + self.eps)
        return x.astype(input_dtype) * self.weight.astype(input_dtype)


class PatchMerger(nn.Module):
    def __init__(
        self,
        dim: int,
        context_dim: int,
        spatial_merge_size: int = 2,
    ) -> None:
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


class VisionAttention(nn.Module):
    def __init__(self, config: VisionConfig) -> None:
        super().__init__()
        self.num_heads = config.num_attention_heads
        self.head_dim = config.embed_dim // config.num_attention_heads
        self.scale = self.head_dim**-0.5

        self.qkv = nn.Linear(
            config.embed_dim, config.embed_dim * 3, bias=config.use_bias
        )
        self.proj = nn.Linear(config.embed_dim, config.embed_dim, bias=config.use_bias)

    def __call__(
        self,
        hidden_states: mx.array,
        cu_seqlens: mx.array,
        rotary_pos_emb: mx.array,
    ) -> mx.array:
        seq_length = hidden_states.shape[0]

        qkv = self.qkv(hidden_states).reshape(seq_length, 3, self.num_heads, -1)
        qkv = qkv.transpose(1, 0, 2, 3)
        q, k, v = mx.split(qkv, 3)

        q = apply_rotary_pos_emb_vision(mx.expand_dims(q, 0), rotary_pos_emb)[0]
        k = apply_rotary_pos_emb_vision(mx.expand_dims(k, 0), rotary_pos_emb)[0]

        q = q.transpose(0, 2, 1, 3)
        k = k.transpose(0, 2, 1, 3)
        v = v.transpose(0, 2, 1, 3)

        splits = [
            mx.split(tensor, cu_seqlens[1:-1].tolist(), axis=2) for tensor in (q, k, v)
        ]

        attn_outputs = []
        for q_chunk, k_chunk, v_chunk in zip(*splits):
            output = mx.fast.scaled_dot_product_attention(
                q_chunk, k_chunk, v_chunk, scale=self.scale
            )
            attn_outputs.append(output)

        output = mx.concatenate(attn_outputs, axis=2)
        output = output.transpose(0, 2, 1, 3).reshape(seq_length, -1)
        return self.proj(output)


class DotsSwiGLUFFN(nn.Module):
    def __init__(self, config: VisionConfig):
        super().__init__()
        self.fc1 = nn.Linear(
            config.embed_dim, config.intermediate_size, bias=config.use_bias
        )
        self.fc2 = nn.Linear(
            config.intermediate_size, config.embed_dim, bias=config.use_bias
        )
        self.fc3 = nn.Linear(
            config.embed_dim, config.intermediate_size, bias=config.use_bias
        )

    def __call__(self, x: mx.array) -> mx.array:
        return self.fc2(nn.silu(self.fc1(x)) * self.fc3(x))


class DotsPatchEmbed(nn.Module):
    def __init__(self, config: VisionConfig):
        super().__init__()
        self.num_channels = config.num_channels
        self.patch_size = config.patch_size
        self.temporal_patch_size = config.temporal_patch_size
        self.embed_dim = config.embed_dim
        self.proj = nn.Conv2d(
            config.num_channels,
            config.embed_dim,
            kernel_size=(config.patch_size, config.patch_size),
            stride=(config.patch_size, config.patch_size),
            bias=True,
        )
        self.norm = RMSNorm(config.embed_dim, eps=config.rms_norm_eps)

    def __call__(self, x: mx.array, grid_thw=None) -> mx.array:
        x = x.reshape(
            -1,
            self.num_channels,
            self.temporal_patch_size,
            self.patch_size,
            self.patch_size,
        )[:, :, 0]
        x = x.transpose(0, 2, 3, 1)
        x = self.proj(x).reshape(-1, self.embed_dim)
        return self.norm(x)


class DotsViTPreprocessor(nn.Module):
    def __init__(self, config: VisionConfig):
        super().__init__()
        self.patchifier = DotsPatchEmbed(config)

    def __call__(self, x: mx.array, grid_thw=None) -> mx.array:
        return self.patchifier(x, grid_thw)


class DotsVisionBlock(nn.Module):
    def __init__(self, config: VisionConfig):
        super().__init__()
        self.attn = VisionAttention(config)
        self.norm1 = RMSNorm(config.embed_dim, eps=config.rms_norm_eps)
        self.mlp = DotsSwiGLUFFN(config)
        self.norm2 = RMSNorm(config.embed_dim, eps=config.rms_norm_eps)

    def __call__(
        self,
        hidden_states: mx.array,
        cu_seqlens: mx.array,
        rotary_pos_emb: mx.array,
    ) -> mx.array:
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
        self.spatial_merge_size = config.spatial_merge_size

        self.patch_embed = DotsViTPreprocessor(config)
        head_dim = config.embed_dim // config.num_attention_heads
        self.rotary_pos_emb = VisionRotaryEmbedding(head_dim // 2)

        self.blocks = [DotsVisionBlock(config) for _ in range(config.num_hidden_layers)]

        if self.config.post_norm:
            self.post_trunk_norm = RMSNorm(config.embed_dim, eps=config.rms_norm_eps)

        self.merger = PatchMerger(
            dim=config.hidden_size,
            context_dim=config.embed_dim,
            spatial_merge_size=config.spatial_merge_size,
        )

    def get_pos_ids_by_grid(self, grid_thw: mx.array):
        pos_ids = []
        for t, h, w in grid_thw.tolist():
            hpos_ids = mx.arange(h).reshape(h, 1)
            hpos_ids = mx.repeat(hpos_ids, w, axis=1)
            hpos_ids = hpos_ids.reshape(
                h // self.spatial_merge_size,
                self.spatial_merge_size,
                w // self.spatial_merge_size,
                self.spatial_merge_size,
            )
            hpos_ids = hpos_ids.transpose(0, 2, 1, 3).flatten()

            wpos_ids = mx.arange(w).reshape(1, w)
            wpos_ids = mx.repeat(wpos_ids, h, axis=0)
            wpos_ids = wpos_ids.reshape(
                h // self.spatial_merge_size,
                self.spatial_merge_size,
                w // self.spatial_merge_size,
                self.spatial_merge_size,
            )
            wpos_ids = wpos_ids.transpose(0, 2, 1, 3).flatten()

            pos_ids.append(mx.tile(mx.stack([hpos_ids, wpos_ids], axis=-1), (t, 1)))
        return pos_ids

    def rot_pos_emb(self, grid_thw: mx.array) -> mx.array:
        pos_ids = mx.concatenate(self.get_pos_ids_by_grid(grid_thw), axis=0)
        max_grid_size = int(mx.max(grid_thw[:, 1:]).item())
        rotary_pos_emb_full = self.rotary_pos_emb(max_grid_size)
        rotary_pos_emb = rotary_pos_emb_full[pos_ids].reshape(pos_ids.shape[0], -1)
        return rotary_pos_emb

    def __call__(
        self,
        hidden_states: mx.array,
        grid_thw: mx.array,
        output_hidden_states: Optional[bool] = None,
    ) -> mx.array:
        hidden_states = self.patch_embed(hidden_states, grid_thw)
        rotary_pos_emb = self.rot_pos_emb(grid_thw)

        cu_seqlens = []
        for i in range(grid_thw.shape[0]):
            seq_len = grid_thw[i, 1] * grid_thw[i, 2]
            cu_seqlens.append(mx.repeat(seq_len, grid_thw[i, 0]))
        cu_seqlens = mx.concatenate(cu_seqlens).astype(mx.int32)
        cu_seqlens = mx.cumsum(cu_seqlens, axis=0)
        cu_seqlens = mx.pad(cu_seqlens, (1, 0), constant_values=0)

        for block in self.blocks:
            hidden_states = block(
                hidden_states, cu_seqlens=cu_seqlens, rotary_pos_emb=rotary_pos_emb
            )

        if self.config.post_norm:
            hidden_states = self.post_trunk_norm(hidden_states)

        hidden_states = self.merger(hidden_states)
        return hidden_states

    def sanitize(self, weights):
        sanitized_weights = {}
        for key, value in weights.items():
            if "position_ids" in key:
                continue
            if "vision_tower.patch_embed.patchifier.proj.weight" in key:
                if check_array_shape(value):
                    sanitized_weights[key] = value
                else:
                    sanitized_weights[key] = value.transpose(0, 2, 3, 1)
            else:
                sanitized_weights[key] = value
        return sanitized_weights
