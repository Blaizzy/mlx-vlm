from typing import Optional

import mlx.core as mx
import mlx.nn as nn

from .config import VisionConfig


def _limited_swiglu(gate: mx.array, up: mx.array, limit: float) -> mx.array:
    gate = mx.minimum(gate, limit)
    up = mx.clip(up, -limit, limit)
    return nn.silu(gate) * up


def _rotate_half(x: mx.array) -> mx.array:
    x1, x2 = mx.split(x, 2, axis=-1)
    return mx.concatenate([-x2, x1], axis=-1)


def _apply_rotary(
    q: mx.array, k: mx.array, cos: mx.array, sin: mx.array
) -> tuple[mx.array, mx.array]:
    q_dtype, k_dtype = q.dtype, k.dtype
    q, k = q.astype(mx.float32), k.astype(mx.float32)
    cos = mx.expand_dims(cos, axis=-2).astype(mx.float32)
    sin = mx.expand_dims(sin, axis=-2).astype(mx.float32)
    q = q * cos + _rotate_half(q) * sin
    k = k * cos + _rotate_half(k) * sin
    return q.astype(q_dtype), k.astype(k_dtype)


class VisionRotaryEmbedding(nn.Module):
    def __init__(self, dim: int, theta: float = 10000.0):
        super().__init__()
        self.dim = dim
        self.theta = theta

    def __call__(self, seqlen: int) -> mx.array:
        inv_freq = 1.0 / (
            self.theta ** (mx.arange(0, self.dim, 2, dtype=mx.float32) / self.dim)
        )
        return mx.outer(mx.arange(seqlen, dtype=mx.float32), inv_freq)


class VisionPatchEmbed(nn.Module):
    def __init__(self, config: VisionConfig):
        super().__init__()
        self.patch_size = config.patch_size
        self.temporal_patch_size = config.temporal_patch_size
        self.in_channels = config.in_channels
        self.embed_dim = config.hidden_size
        kernel = [self.temporal_patch_size, self.patch_size, self.patch_size]
        self.proj = nn.Conv3d(
            self.in_channels,
            self.embed_dim,
            kernel_size=kernel,
            stride=kernel,
            bias=True,
        )

    def __call__(self, hidden_states: mx.array) -> mx.array:
        hidden_states = hidden_states.reshape(
            -1,
            self.in_channels,
            self.temporal_patch_size,
            self.patch_size,
            self.patch_size,
        ).moveaxis(1, 4)
        return self.proj(hidden_states).reshape(-1, self.embed_dim)


class VisionAttention(nn.Module):
    def __init__(self, config: VisionConfig):
        super().__init__()
        self.num_heads = config.num_heads
        self.head_dim = config.hidden_size // config.num_heads
        self.scale = self.head_dim**-0.5
        self.qkv = nn.Linear(
            config.hidden_size,
            config.hidden_size * 3,
            bias=config.attention_bias,
        )
        self.proj = nn.Linear(
            config.hidden_size, config.hidden_size, bias=config.attention_bias
        )
        self.q_norm = nn.RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.k_norm = nn.RMSNorm(self.head_dim, eps=config.rms_norm_eps)

    def __call__(self, x, cu_seqlens, position_embeddings):
        length = x.shape[0]
        qkv = self.qkv(x).reshape(length, 3, self.num_heads, self.head_dim)
        q, k, v = qkv.transpose(1, 0, 2, 3)
        q, k = self.q_norm(q), self.k_norm(k)
        q, k = _apply_rotary(q, k, *position_embeddings)
        q = q.transpose(1, 0, 2)[None]
        k = k.transpose(1, 0, 2)[None]
        v = v.transpose(1, 0, 2)[None]

        lengths = (cu_seqlens[1:] - cu_seqlens[:-1]).tolist()
        split_indices = []
        offset = 0
        for chunk_length in lengths[:-1]:
            offset += chunk_length
            split_indices.append(offset)

        outputs = []
        for q_chunk, k_chunk, v_chunk in zip(
            mx.split(q, split_indices, axis=2),
            mx.split(k, split_indices, axis=2),
            mx.split(v, split_indices, axis=2),
        ):
            outputs.append(
                mx.fast.scaled_dot_product_attention(
                    q_chunk, k_chunk, v_chunk, scale=self.scale
                )
            )
        out = mx.concatenate(outputs, axis=2)
        out = out.transpose(0, 2, 1, 3).reshape(length, -1)
        return self.proj(out)


class VisionMLP(nn.Module):
    def __init__(self, config: VisionConfig):
        super().__init__()
        self.gate_proj = nn.Linear(
            config.hidden_size, config.intermediate_size, bias=config.attention_bias
        )
        self.up_proj = nn.Linear(
            config.hidden_size, config.intermediate_size, bias=config.attention_bias
        )
        self.down_proj = nn.Linear(
            config.intermediate_size, config.hidden_size, bias=config.attention_bias
        )
        self.swiglu_limit = config.swiglu_limit

    def __call__(self, x):
        return self.down_proj(
            _limited_swiglu(self.gate_proj(x), self.up_proj(x), self.swiglu_limit)
        )


class VisionBlock(nn.Module):
    def __init__(self, config: VisionConfig):
        super().__init__()
        self.norm1 = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.norm2 = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.attn = VisionAttention(config)
        self.mlp = VisionMLP(config)

    def __call__(self, x, cu_seqlens, position_embeddings):
        x = x + self.attn(self.norm1(x), cu_seqlens, position_embeddings)
        return x + self.mlp(self.norm2(x))


class VisionPatchMerger(nn.Module):
    def __init__(self, config: VisionConfig):
        super().__init__()
        dim = config.out_hidden_size
        self.proj = nn.Linear(dim, dim, bias=False)
        self.post_projection_norm = nn.LayerNorm(dim)
        self.gate_proj = nn.Linear(dim, config.projection_intermediate_size, bias=False)
        self.up_proj = nn.Linear(dim, config.projection_intermediate_size, bias=False)
        self.down_proj = nn.Linear(config.projection_intermediate_size, dim, bias=False)
        self.swiglu_limit = config.swiglu_limit

    def __call__(self, x):
        x = nn.gelu(self.post_projection_norm(self.proj(x)))
        return self.down_proj(
            _limited_swiglu(self.gate_proj(x), self.up_proj(x), self.swiglu_limit)
        )


class VisionModel(nn.Module):
    def __init__(self, config: VisionConfig):
        super().__init__()
        self.config = config
        self.model_type = config.model_type
        self.spatial_merge_size = config.spatial_merge_size
        self.patch_size = config.patch_size
        self.patch_embed = VisionPatchEmbed(config)
        head_dim = config.hidden_size // config.num_heads
        self.rotary_pos_emb = VisionRotaryEmbedding(head_dim // 2)
        self.blocks = [VisionBlock(config) for _ in range(config.depth)]
        self.merger = VisionPatchMerger(config)
        self.downsample = nn.Conv2d(
            config.hidden_size,
            config.out_hidden_size,
            kernel_size=config.spatial_merge_size,
            stride=config.spatial_merge_size,
            bias=True,
        )
        self.post_layernorm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def _rotary_embeddings(self, grid_thw: mx.array):
        pos_ids = []
        merge = self.spatial_merge_size
        for t, h, w in grid_thw.tolist():
            h_ids = mx.repeat(mx.arange(h)[:, None], w, axis=1)
            w_ids = mx.repeat(mx.arange(w)[None, :], h, axis=0)
            h_ids = h_ids.reshape(h // merge, merge, w // merge, merge)
            w_ids = w_ids.reshape(h // merge, merge, w // merge, merge)
            h_ids = h_ids.transpose(0, 2, 1, 3).flatten()
            w_ids = w_ids.transpose(0, 2, 1, 3).flatten()
            pos_ids.append(mx.tile(mx.stack([h_ids, w_ids], axis=-1), (t, 1)))

        pos_ids = mx.concatenate(pos_ids, axis=0)
        max_grid = int(mx.max(grid_thw[:, 1:]).item())
        rotary = self.rotary_pos_emb(max_grid)[pos_ids].reshape(pos_ids.shape[0], -1)
        emb = mx.concatenate([rotary, rotary], axis=-1)
        return mx.cos(emb), mx.sin(emb)

    def __call__(
        self,
        hidden_states: mx.array,
        grid_thw: mx.array,
        output_hidden_states: Optional[bool] = None,
    ) -> mx.array:
        del output_hidden_states
        hidden_states = self.patch_embed(hidden_states)
        position_embeddings = self._rotary_embeddings(grid_thw)

        repeated = []
        for t, h, w in grid_thw.tolist():
            repeated.extend([h * w] * t)
        cu_seqlens = mx.pad(mx.cumsum(mx.array(repeated)), (1, 0))

        for block in self.blocks:
            hidden_states = block(hidden_states, cu_seqlens, position_embeddings)

        hidden_states = self.post_layernorm(hidden_states)
        merge = self.spatial_merge_size
        hidden_states = hidden_states.reshape(-1, merge, merge, hidden_states.shape[-1])
        hidden_states = self.downsample(hidden_states).reshape(
            -1, self.config.out_hidden_size
        )
        return self.merger(hidden_states)

    def sanitize(self, weights):
        def is_mlx_conv2d(value):
            out_channels, kernel_h, kernel_w, _ = value.shape
            return (
                out_channels >= kernel_h
                and out_channels >= kernel_w
                and kernel_h == kernel_w
            )

        sanitized = {}
        for key, value in weights.items():
            if "position_ids" in key:
                continue
            if key.endswith(("patch_embed.proj.weight", "downsample.weight")):
                if value.ndim == 5 and value.shape[-1] != self.config.in_channels:
                    value = value.transpose(0, 2, 3, 4, 1)
                elif value.ndim == 4 and not is_mlx_conv2d(value):
                    value = value.transpose(0, 2, 3, 1)
            sanitized[key] = value
        return sanitized
