from typing import Optional

import mlx.core as mx
import mlx.nn as nn
import numpy as np

from .config import VisionConfig


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
        seq = mx.arange(seqlen.item(), dtype=inv_freq.dtype)
        freqs = mx.outer(seq, inv_freq)
        return freqs


class PatchEmbed(nn.Module):
    def __init__(
        self,
        patch_size: int = 14,
        temporal_patch_size: int = 2,
        in_channels: int = 3,
        hidden_size: int = 1152,
    ) -> None:
        super().__init__()
        self.patch_size = patch_size
        self.temporal_patch_size = temporal_patch_size
        self.in_channels = in_channels
        self.hidden_size = hidden_size

        kernel_size = [temporal_patch_size, patch_size, patch_size]
        self.proj = nn.Conv3d(
            in_channels,
            hidden_size,
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
        ).moveaxis(1, 4)

        hidden_states = self.proj(hidden_states)
        hidden_states = hidden_states.reshape(-1, self.hidden_size)
        return hidden_states


class PatchMerger(nn.Module):
    def __init__(self, config: VisionConfig, use_postshuffle_norm=False) -> None:
        super().__init__()
        self.hidden_size = config.hidden_size * (config.spatial_merge_size**2)
        self.use_postshuffle_norm = use_postshuffle_norm
        self.norm = nn.LayerNorm(
            self.hidden_size if use_postshuffle_norm else config.hidden_size, eps=1e-6
        )
        self.linear_fc1 = nn.Linear(self.hidden_size, self.hidden_size)
        self.act_fn = nn.GELU()
        self.linear_fc2 = nn.Linear(self.hidden_size, config.out_hidden_size)

    def __call__(self, x: mx.array) -> mx.array:
        x = self.norm(
            x.reshape(-1, self.hidden_size) if self.use_postshuffle_norm else x
        ).reshape(-1, self.hidden_size)
        x = self.linear_fc2(self.act_fn(self.linear_fc1(x)))
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


class MLP(nn.Module):
    def __init__(self, dim, hidden_dim):
        super().__init__()
        self.linear_fc1 = nn.Linear(dim, hidden_dim, bias=True)
        self.linear_fc2 = nn.Linear(hidden_dim, dim, bias=True)
        self.act_fn = nn.GELU(approx="fast")

    def call(self, x: mx.array) -> mx.array:
        return self.linear_fc2(self.act_fn(self.linear_fc1(x)))


class Qwen3VLMoEVisionBlock(nn.Module):
    def __init__(self, config: VisionConfig) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(config.hidden_size, eps=1e-6)
        self.norm2 = nn.LayerNorm(config.hidden_size, eps=1e-6)

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


class VisionModel(nn.Module):

    def __init__(self, config: VisionConfig) -> None:
        super().__init__()
        self.config = config
        self.model_type = config.model_type
        if self.model_type != "qwen3_vl_moe":
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

        self.pos_embed = nn.Embedding(
            config.num_position_embeddings, config.hidden_size
        )
        self.num_grid_per_side = int(config.num_position_embeddings**0.5)

        self.blocks = [Qwen3VLMoEVisionBlock(config) for _ in range(config.depth)]
        self.merger = PatchMerger(config=config, use_postshuffle_norm=False)

        self.deepstack_visual_indexes = config.deepstack_visual_indexes
        self.deepstack_merger_list = [
            PatchMerger(
                config=config,
                use_postshuffle_norm=True,
            )
            for _ in range(len(config.deepstack_visual_indexes))
        ]

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

        return rotary_pos_emb.reshape(pos_ids.shape[0], -1)

    def get_window_index(self, grid_thw):
        window_index = []
        cu_window_seqlens = [0]
        window_index_id = 0
        vit_merger_window_size = (
            self.window_size // self.spatial_merge_size // self.patch_size
        )

        for grid_t, grid_h, grid_w in grid_thw.tolist():
            llm_grid_h = grid_h // self.spatial_merge_size
            llm_grid_w = grid_w // self.spatial_merge_size

            index = mx.arange(grid_t * llm_grid_h * llm_grid_w).reshape(
                grid_t, llm_grid_h, llm_grid_w
            )

            pad_h = vit_merger_window_size - llm_grid_h % vit_merger_window_size
            pad_w = vit_merger_window_size - llm_grid_w % vit_merger_window_size
            num_windows_h = (llm_grid_h + pad_h) // vit_merger_window_size
            num_windows_w = (llm_grid_w + pad_w) // vit_merger_window_size

            # Replace F.pad with np.pad
            index_padded = mx.pad(
                index,
                ((0, 0), (0, pad_h), (0, pad_w)),
                mode="constant",
                constant_values=-100,
            )

            index_padded = index_padded.reshape(
                grid_t,
                num_windows_h,
                vit_merger_window_size,
                num_windows_w,
                vit_merger_window_size,
            )

            # Replace permute with np.transpose
            index_padded = mx.transpose(index_padded, (0, 1, 3, 2, 4)).reshape(
                grid_t,
                num_windows_h * num_windows_w,
                vit_merger_window_size,
                vit_merger_window_size,
            )

            # Replace torch operations with numpy
            seqlens = mx.sum(index_padded != -100, axis=(2, 3)).reshape(-1)
            index_padded = index_padded.reshape(-1)
            index = np.where(index_padded != -100)[
                0
            ].tolist()  # [i for i, x in enumerate(index_padded) if x != -100]
            index_new = index_padded[index]

            window_index.append(index_new + window_index_id)
            cu_seqlens_tmp = (
                mx.cumsum(seqlens, axis=0) * self.spatial_merge_unit
                + cu_window_seqlens[-1]
            )
            cu_window_seqlens.extend(cu_seqlens_tmp.tolist())
            window_index_id += int(grid_t * llm_grid_h * llm_grid_w)

        # Replace torch.cat with np.concatenate
        window_index = mx.concatenate(window_index, axis=0)
        cu_window_seqlens = mx.array(cu_window_seqlens)

        return window_index, cu_window_seqlens

    def fast_pos_embed_interpolate(self, grid_thw):
        grid_ts = grid_thw[:, 0]
        grid_hs = grid_thw[:, 1]
        grid_ws = grid_thw[:, 2]

        idx_list = [[] for _ in range(4)]
        weight_list = [[] for _ in range(4)]

        for t, h, w in zip(grid_ts.tolist(), grid_hs.tolist(), grid_ws.tolist()):
            t, h, w = int(t), int(h), int(w)

            # Create linearly spaced indices
            h_idxs = mx.linspace(0, self.num_grid_per_side - 1, h)
            w_idxs = mx.linspace(0, self.num_grid_per_side - 1, w)

            # Floor and ceiling indices
            h_idxs_floor = h_idxs.astype(mx.int32)
            w_idxs_floor = w_idxs.astype(mx.int32)
            h_idxs_ceil = mx.clip(h_idxs_floor + 1, 0, self.num_grid_per_side - 1)
            w_idxs_ceil = mx.clip(w_idxs_floor + 1, 0, self.num_grid_per_side - 1)

            # Compute interpolation weights
            dh = h_idxs - h_idxs_floor
            dw = w_idxs - w_idxs_floor

            # Base indices for height
            base_h = h_idxs_floor * self.num_grid_per_side
            base_h_ceil = h_idxs_ceil * self.num_grid_per_side

            # Four corner indices for bilinear interpolation
            indices = [
                (base_h[:, None] + w_idxs_floor[None, :]).flatten(),
                (base_h[:, None] + w_idxs_ceil[None, :]).flatten(),
                (base_h_ceil[:, None] + w_idxs_floor[None, :]).flatten(),
                (base_h_ceil[:, None] + w_idxs_ceil[None, :]).flatten(),
            ]

            # Four corner weights for bilinear interpolation
            weights = [
                ((1 - dh)[:, None] * (1 - dw)[None, :]).flatten(),
                ((1 - dh)[:, None] * dw[None, :]).flatten(),
                (dh[:, None] * (1 - dw)[None, :]).flatten(),
                (dh[:, None] * dw[None, :]).flatten(),
            ]

            for i in range(4):
                idx_list[i].extend(indices[i].tolist())
                weight_list[i].extend(weights[i].tolist())

        # Convert to arrays
        idx_tensor = mx.array(idx_list, dtype=mx.int32)
        weight_tensor = mx.array(weight_list, dtype=self.pos_embed.weight.dtype)

        # Lookup embeddings and apply weights
        pos_embeds = self.pos_embed(idx_tensor) * weight_tensor[:, :, None]

        # Sum the four corners
        patch_pos_embeds = pos_embeds[0] + pos_embeds[1] + pos_embeds[2] + pos_embeds[3]

        # Split by sample
        split_sizes = [int(h * w) for h, w in zip(grid_hs.tolist(), grid_ws.tolist())]
        patch_pos_embeds_list = mx.split(
            patch_pos_embeds, mx.cumsum(mx.array(split_sizes))[:-1]
        )

        # Apply temporal repetition and spatial merging
        patch_pos_embeds_permute = []
        merge_size = self.spatial_merge_size

        for pos_embed, t, h, w in zip(
            patch_pos_embeds_list, grid_ts.tolist(), grid_hs.tolist(), grid_ws.tolist()
        ):
            t, h, w = int(t), int(h), int(w)

            # Repeat for temporal dimension
            pos_embed = mx.repeat(pos_embed, t, axis=0)

            # Reshape and permute for spatial merging
            embed_dim = pos_embed.shape[-1]
            pos_embed = pos_embed.reshape(
                t, h // merge_size, merge_size, w // merge_size, merge_size, embed_dim
            )

            # Permute: (0, 1, 3, 2, 4, 5) -> (t, h//merge, w//merge, merge, merge, embed_dim)
            pos_embed = mx.transpose(pos_embed, (0, 1, 3, 2, 4, 5))

            # Flatten spatial dimensions
            pos_embed = pos_embed.reshape(-1, embed_dim)

            patch_pos_embeds_permute.append(pos_embed)

        # Concatenate all samples
        patch_pos_embeds = mx.concatenate(patch_pos_embeds_permute, axis=0)

        return patch_pos_embeds

    def __call__(
        self,
        hidden_states: mx.array,
        grid_thw: mx.array,
        output_hidden_states: Optional[bool] = None,
    ) -> mx.array:

        hidden_states = self.patch_embed(hidden_states)
        pos_embeds = self.fast_pos_embed_interpolate(grid_thw)
        hidden_states = hidden_states + pos_embeds
        rotary_pos_emb = self.rot_pos_emb(grid_thw)
        window_index, cu_window_seqlens = self.get_window_index(grid_thw)

        # Get indices of first occurrence of each unique value
        seen = set()
        idx = []
        for i, x in enumerate(cu_window_seqlens):
            if x not in seen:
                seen.add(x)
                idx.append(i)

        idx = mx.array(idx, dtype=mx.int32)
        cu_window_seqlens = cu_window_seqlens[idx]

        seq_len, _ = hidden_states.shape
        hidden_states = hidden_states.reshape(
            seq_len // self.spatial_merge_unit, self.spatial_merge_unit, -1
        )
        hidden_states = hidden_states[window_index, :, :]
        hidden_states = hidden_states.reshape(seq_len, -1)
        rotary_pos_emb = rotary_pos_emb.reshape(
            seq_len // self.spatial_merge_unit, self.spatial_merge_unit, -1
        )
        rotary_pos_emb = rotary_pos_emb[window_index, :, :]
        rotary_pos_emb = rotary_pos_emb.reshape(seq_len, -1)

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

        deepstack_feature_lists = []
        for layer_num, blk in enumerate(self.blocks):
            if layer_num in self.fullatt_block_indexes:
                cu_seqlens_now = cu_seqlens
            else:
                cu_seqlens_now = cu_window_seqlens

            hidden_states = blk(
                hidden_states, cu_seqlens=cu_seqlens_now, rotary_pos_emb=rotary_pos_emb
            )
            if layer_num in self.deepstack_visual_indexes:
                deepstack_feature = self.deepstack_merger_list[
                    self.deepstack_visual_indexes.index(layer_num)
                ](hidden_states)
                deepstack_feature_lists.append(deepstack_feature)

            if output_hidden_states:
                encoder_states = encoder_states + (hidden_states,)

        hidden_states = self.merger(hidden_states)
        reverse_indices = mx.argsort(window_index, axis=0)
        hidden_states = hidden_states[reverse_indices, :]
        return hidden_states, deepstack_feature_lists

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
