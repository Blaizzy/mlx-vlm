from itertools import accumulate

import mlx.core as mx
import mlx.nn as nn

from ..base import ensure_fused_sdpa
from .config import VisionConfig


def _rotate_half(x):
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return mx.concatenate([-x2, x1], axis=-1)


def _apply_rotary_pos_emb_vision(q, k, cos, sin):
    dtype_q, dtype_k = q.dtype, k.dtype
    q = q.astype(mx.float32)
    k = k.astype(mx.float32)
    cos = mx.expand_dims(cos.astype(mx.float32), axis=-2)
    sin = mx.expand_dims(sin.astype(mx.float32), axis=-2)
    q = q * cos + _rotate_half(q) * sin
    k = k * cos + _rotate_half(k) * sin
    return q.astype(dtype_q), k.astype(dtype_k)


class VisionRotaryEmbedding(nn.Module):
    def __init__(self, dim: int, theta: float = 10000.0):
        super().__init__()
        self.dim = dim
        self.theta = theta

    def __call__(self, position_ids: mx.array) -> mx.array:
        inv_freq = 1.0 / (
            self.theta ** (mx.arange(0, self.dim, 2, dtype=mx.float32) / self.dim)
        )
        return (position_ids[..., None] * inv_freq).reshape(position_ids.shape[0], -1)


class PatchEmbed(nn.Module):
    def __init__(self, config: VisionConfig):
        super().__init__()
        self.patch_size = config.patch_size
        self.temporal_patch_size = config.temporal_patch_size
        self.in_channels = config.in_channels
        self.hidden_size = config.hidden_size
        kernel_size = (
            config.temporal_patch_size,
            config.patch_size,
            config.patch_size,
        )
        self.proj = nn.Conv3d(
            config.in_channels,
            config.hidden_size,
            kernel_size=kernel_size,
            stride=kernel_size,
            bias=True,
        )

    def __call__(self, hidden_states: mx.array) -> mx.array:
        hidden_states = hidden_states.reshape(
            -1,
            self.in_channels,
            self.temporal_patch_size,
            self.patch_size,
            self.patch_size,
        )
        # MLX Conv3d uses channels-last tensors.
        hidden_states = hidden_states.transpose(0, 2, 3, 4, 1)
        hidden_states = self.proj(hidden_states)
        return hidden_states.reshape(-1, self.hidden_size)


class PatchMerger(nn.Module):
    def __init__(self, config: VisionConfig, use_postshuffle_norm=False):
        super().__init__()
        self.hidden_size = config.hidden_size * (config.spatial_merge_size**2)
        self.use_postshuffle_norm = use_postshuffle_norm
        self.norm = nn.LayerNorm(
            self.hidden_size if use_postshuffle_norm else config.hidden_size,
            eps=1e-6,
        )
        self.linear_fc1 = nn.Linear(self.hidden_size, self.hidden_size)
        self.act_fn = nn.GELU(approx="tanh")
        self.linear_fc2 = nn.Linear(self.hidden_size, config.out_hidden_size)

    def __call__(self, x: mx.array) -> mx.array:
        if self.use_postshuffle_norm:
            x = self.norm(x.reshape(-1, self.hidden_size))
        else:
            x = self.norm(x)
        x = x.reshape(-1, self.hidden_size)
        return self.linear_fc2(self.act_fn(self.linear_fc1(x)))


class Attention(nn.Module):
    def __init__(self, config: VisionConfig):
        super().__init__()
        self.num_heads = config.num_heads
        self.head_dim = config.hidden_size // config.num_heads
        self.scale = self.head_dim**-0.5
        self.qkv = nn.Linear(config.hidden_size, config.hidden_size * 3, bias=True)
        self.proj = nn.Linear(config.hidden_size, config.hidden_size)

    def __call__(
        self,
        x: mx.array,
        cu_seqlens: mx.array,
        rotary_pos_emb: mx.array | None = None,
    ) -> mx.array:
        seq_length = x.shape[0]
        qkv = self.qkv(x).reshape(seq_length, 3, self.num_heads, self.head_dim)
        q, k, v = mx.split(qkv, 3, axis=1)
        q, k, v = q[:, 0], k[:, 0], v[:, 0]

        if rotary_pos_emb is not None:
            cos = mx.cos(rotary_pos_emb)
            sin = mx.sin(rotary_pos_emb)
            cos = mx.repeat(cos, 2, axis=-1)
            sin = mx.repeat(sin, 2, axis=-1)
            q, k = _apply_rotary_pos_emb_vision(q, k, cos, sin)

        q = q.transpose(1, 0, 2)[None]
        k = k.transpose(1, 0, 2)[None]
        v = v.transpose(1, 0, 2)[None]

        splits = [
            mx.split(tensor, cu_seqlens[1:-1].tolist(), axis=2) for tensor in (q, k, v)
        ]
        outputs = []
        for q_i, k_i, v_i in zip(*splits):
            outputs.append(ensure_fused_sdpa(q_i, k_i, v_i, self.scale))
        output = mx.concatenate(outputs, axis=2)
        output = output.transpose(0, 2, 1, 3).reshape(seq_length, -1)
        return self.proj(output)


class VisionMLP(nn.Module):
    def __init__(self, config: VisionConfig):
        super().__init__()
        self.linear_fc1 = nn.Linear(config.hidden_size, config.intermediate_size)
        self.linear_fc2 = nn.Linear(config.intermediate_size, config.hidden_size)
        self.act_fn = nn.GELU(approx="tanh")

    def __call__(self, x: mx.array) -> mx.array:
        return self.linear_fc2(self.act_fn(self.linear_fc1(x)))


class VisionBlock(nn.Module):
    def __init__(self, config: VisionConfig):
        super().__init__()
        self.norm1 = nn.LayerNorm(config.hidden_size, eps=1e-6)
        self.norm2 = nn.LayerNorm(config.hidden_size, eps=1e-6)
        self.attn = Attention(config)
        self.mlp = VisionMLP(config)

    def __call__(self, hidden_states, cu_seqlens, rotary_pos_emb):
        hidden_states = hidden_states + self.attn(
            self.norm1(hidden_states), cu_seqlens, rotary_pos_emb
        )
        return hidden_states + self.mlp(self.norm2(hidden_states))


class VisionModel(nn.Module):
    def __init__(self, config: VisionConfig):
        super().__init__()
        self.config = config
        self.model_type = config.model_type
        if self.model_type != "cohere_compass_vision":
            raise ValueError(f"Unsupported model type: {self.model_type}")

        self.spatial_merge_size = config.spatial_merge_size
        self.spatial_merge_unit = self.spatial_merge_size**2
        self.patch_embed = PatchEmbed(config)
        self.pos_embed = nn.Embedding(
            config.num_position_embeddings, config.hidden_size
        )
        self.num_grid_per_side = int(config.num_position_embeddings**0.5)
        head_dim = config.hidden_size // config.num_heads
        self.rotary_pos_emb = VisionRotaryEmbedding(head_dim // 2)
        self.blocks = [VisionBlock(config) for _ in range(config.depth)]
        self.merger = PatchMerger(config)
        self.deepstack_visual_indexes = config.deepstack_visual_indexes
        self.deepstack_merger_list = [
            PatchMerger(config, use_postshuffle_norm=True)
            for _ in self.deepstack_visual_indexes
        ]
        self.use_rope = config.use_rope

    def fast_pos_embed_interpolate(self, grid_thw: mx.array) -> mx.array:
        idx_list = [[] for _ in range(4)]
        weight_list = [[] for _ in range(4)]
        for _, h, w in grid_thw.tolist():
            h, w = int(h), int(w)
            h_idxs = mx.linspace(0, self.num_grid_per_side - 1, h)
            w_idxs = mx.linspace(0, self.num_grid_per_side - 1, w)
            h_floor = h_idxs.astype(mx.int32)
            w_floor = w_idxs.astype(mx.int32)
            h_ceil = mx.minimum(h_floor + 1, self.num_grid_per_side - 1)
            w_ceil = mx.minimum(w_floor + 1, self.num_grid_per_side - 1)
            dh = h_idxs - h_floor.astype(mx.float32)
            dw = w_idxs - w_floor.astype(mx.float32)
            base_h = h_floor * self.num_grid_per_side
            base_h_ceil = h_ceil * self.num_grid_per_side
            indices = [
                (base_h[:, None] + w_floor[None, :]).flatten(),
                (base_h[:, None] + w_ceil[None, :]).flatten(),
                (base_h_ceil[:, None] + w_floor[None, :]).flatten(),
                (base_h_ceil[:, None] + w_ceil[None, :]).flatten(),
            ]
            weights = [
                ((1 - dh)[:, None] * (1 - dw)[None, :]).flatten(),
                ((1 - dh)[:, None] * dw[None, :]).flatten(),
                (dh[:, None] * (1 - dw)[None, :]).flatten(),
                (dh[:, None] * dw[None, :]).flatten(),
            ]
            for i in range(4):
                idx_list[i].extend(indices[i].tolist())
                weight_list[i].extend(weights[i].tolist())

        idx_tensor = mx.array(idx_list, dtype=mx.int32)
        weight_tensor = mx.array(weight_list, dtype=self.pos_embed.weight.dtype)
        pos_embeds = self.pos_embed(idx_tensor) * weight_tensor[..., None]
        patch_pos_embeds = pos_embeds.sum(axis=0)

        split_sizes = [int(h * w) for _, h, w in grid_thw.tolist()]
        splits = (
            mx.split(patch_pos_embeds, list(accumulate(split_sizes[:-1])), axis=0)
            if len(split_sizes) > 1
            else [patch_pos_embeds]
        )
        output = []
        for pos_embed, (t, h, w) in zip(splits, grid_thw.tolist()):
            t, h, w = int(t), int(h), int(w)
            pos_embed = mx.tile(pos_embed, (t, 1)).reshape(t, h, w, -1)
            pos_embed = pos_embed.reshape(
                t,
                h // self.spatial_merge_size,
                self.spatial_merge_size,
                w // self.spatial_merge_size,
                self.spatial_merge_size,
                pos_embed.shape[-1],
            )
            output.append(
                pos_embed.transpose(0, 1, 3, 2, 4, 5).reshape(-1, pos_embed.shape[-1])
            )
        return mx.concatenate(output, axis=0)

    def rot_pos_emb(self, grid_thw: mx.array) -> mx.array:
        max_hw = int(mx.max(grid_thw[:, 1:]).item())
        freq_table = self.rotary_pos_emb(mx.arange(max_hw, dtype=mx.int32))
        positions = []
        merge = self.spatial_merge_size
        for t, h, w in grid_thw.tolist():
            t, h, w = int(t), int(h), int(w)
            merged_h, merged_w = h // merge, w // merge
            rows = mx.arange(merged_h)[:, None, None, None] * merge
            rows = rows + mx.arange(merge)[None, None, :, None]
            cols = mx.arange(merged_w)[None, :, None, None] * merge
            cols = cols + mx.arange(merge)[None, None, None, :]
            rows = mx.broadcast_to(rows, (merged_h, merged_w, merge, merge)).reshape(-1)
            cols = mx.broadcast_to(cols, (merged_h, merged_w, merge, merge)).reshape(-1)
            coords = mx.stack([rows, cols], axis=-1)
            positions.append(mx.tile(coords, (t, 1)) if t > 1 else coords)
        pos_ids = mx.concatenate(positions, axis=0)
        return mx.concatenate(
            [freq_table[pos_ids[:, 0]], freq_table[pos_ids[:, 1]]], axis=-1
        )

    def _forward_single(self, hidden_states: mx.array, grid_thw: mx.array):
        hidden_states = self.patch_embed(hidden_states)
        hidden_states = hidden_states + self.fast_pos_embed_interpolate(
            grid_thw
        ).astype(hidden_states.dtype)

        rotary_pos_emb = None
        if self.use_rope:
            rotary_pos_emb = self.rot_pos_emb(grid_thw)

        sequence_lengths = []
        for t, h, w in grid_thw.tolist():
            sequence_lengths.extend([int(h) * int(w)] * int(t))
        cu_seqlens = mx.pad(
            mx.cumsum(mx.array(sequence_lengths, dtype=mx.int32)), (1, 0)
        )

        deepstack_features = []
        for layer_num, block in enumerate(self.blocks):
            hidden_states = block(hidden_states, cu_seqlens, rotary_pos_emb)
            if layer_num in self.deepstack_visual_indexes:
                index = self.deepstack_visual_indexes.index(layer_num)
                deepstack_features.append(
                    self.deepstack_merger_list[index](hidden_states)
                )

        return self.merger(hidden_states), deepstack_features

    def __call__(self, hidden_states: mx.array, grid_thw: mx.array, **kwargs):
        if grid_thw.shape[0] == 1:
            return self._forward_single(hidden_states, grid_thw)

        # Keep each image's attention graph independent.  The flattened
        # multi-image path uses split views inside fused SDPA; on bfloat16
        # hardware, that changes accumulation order enough for small visual
        # errors to compound through the 27-layer tower and alter decoding.
        image_lengths = [int(t) * int(h) * int(w) for t, h, w in grid_thw.tolist()]
        image_splits = mx.split(
            hidden_states,
            list(accumulate(image_lengths[:-1])),
            axis=0,
        )
        visual_features = []
        deepstack_features = [[] for _ in self.deepstack_visual_indexes]
        for image_hidden_states, image_grid in zip(image_splits, grid_thw):
            features, image_deepstack = self._forward_single(
                image_hidden_states, image_grid[None]
            )
            visual_features.append(features)
            for index, feature in enumerate(image_deepstack):
                deepstack_features[index].append(feature)

        return mx.concatenate(visual_features, axis=0), [
            mx.concatenate(features, axis=0) for features in deepstack_features
        ]

    def sanitize(self, weights):
        sanitized = {}
        for key, value in weights.items():
            if "position_ids" in key:
                continue
            if key.endswith("patch_embed.proj.weight") and value.ndim == 5:
                if value.shape[1] == self.config.in_channels:
                    value = value.transpose(0, 2, 3, 4, 1)
            sanitized[key] = value
        return sanitized
