
import mlx.core as mx
import mlx.nn as nn
import numpy as np

from .config import VisionConfig


def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return mx.concatenate([-x2, x1], axis=-1)


def apply_rotary_pos_emb_vision(q, k, cos, sin):
    """Apply rotary position embeddings to vision q, k tensors.
    cos, sin: (seq_len, head_dim) with unsqueeze(-2) -> (seq_len, 1, head_dim)
    q, k: (seq_len, num_heads, head_dim)
    """
    orig_dtype = q.dtype
    cos = mx.expand_dims(cos, axis=-2)
    sin = mx.expand_dims(sin, axis=-2)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed.astype(orig_dtype), k_embed.astype(orig_dtype)


class VisionRoPE(nn.Module):
    def __init__(self, dim: int, theta: float = 10000.0) -> None:
        super().__init__()
        self.dim = dim
        self.theta = theta

    def __call__(self, seqlen) -> mx.array:
        if isinstance(seqlen, mx.array):
            seqlen = seqlen.item()
        inv_freq = 1.0 / (
            self.theta ** (mx.arange(0, self.dim, 2, dtype=mx.float32) / self.dim)
        )
        seq = mx.arange(seqlen, dtype=mx.float32)
        freqs = mx.outer(seq, inv_freq)
        return freqs


class Siglip2VisionEmbeddings(nn.Module):
    """Patch embedding without positional embedding (RoPE is used in encoder)."""

    def __init__(self, config: VisionConfig):
        super().__init__()
        self.config = config
        self.embed_dim = config.hidden_size
        self.patch_size = config.patch_size
        self.in_features = config.num_channels * self.patch_size * self.patch_size
        self.patch_embedding = nn.Linear(self.in_features, self.embed_dim)

    def __call__(self, pixel_values: mx.array) -> mx.array:
        # pixel_values: (batch, num_patches, patch_dim)
        # Flatten to (total_patches, embed_dim)
        patch_embeds = self.patch_embedding(pixel_values)
        patch_embeds = patch_embeds.reshape(-1, self.embed_dim)
        return patch_embeds


class Siglip2Attention(nn.Module):
    """Vision attention with RoPE and windowed attention via cu_seqlens."""

    def __init__(self, config: VisionConfig) -> None:
        super().__init__()
        dim = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = dim // self.num_heads
        self.scale = self.head_dim**-0.5

        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.out_proj = nn.Linear(dim, dim)

    def __call__(
        self,
        hidden_states: mx.array,
        cu_seqlens: mx.array,
        position_embeddings=None,
    ) -> mx.array:
        seq_length = hidden_states.shape[0]

        q = self.q_proj(hidden_states).reshape(seq_length, self.num_heads, -1)
        k = self.k_proj(hidden_states).reshape(seq_length, self.num_heads, -1)
        v = self.v_proj(hidden_states).reshape(seq_length, self.num_heads, -1)

        # Apply rotary position embeddings
        if position_embeddings is not None:
            cos, sin = position_embeddings
            q, k = apply_rotary_pos_emb_vision(q, k, cos, sin)

        # Windowed attention: split by cu_seqlens and attend within each window
        # Transpose to (num_heads, seq_len, head_dim) for attention
        q = q.transpose(1, 0, 2)  # (num_heads, seq_len, head_dim)
        k = k.transpose(1, 0, 2)
        v = v.transpose(1, 0, 2)

        cu_seqlens_list = cu_seqlens.tolist()
        if isinstance(cu_seqlens_list[0], list):
            cu_seqlens_list = [int(x) for x in cu_seqlens_list]

        # Split by windows and attend
        attn_outputs = []
        for i in range(len(cu_seqlens_list) - 1):
            start = int(cu_seqlens_list[i])
            end = int(cu_seqlens_list[i + 1])
            q_win = q[:, start:end, :]
            k_win = k[:, start:end, :]
            v_win = v[:, start:end, :]
            out = mx.fast.scaled_dot_product_attention(
                mx.expand_dims(q_win, 0),
                mx.expand_dims(k_win, 0),
                mx.expand_dims(v_win, 0),
                scale=self.scale,
                mask=None,
            )
            attn_outputs.append(out[0])

        output = mx.concatenate(attn_outputs, axis=1)  # (num_heads, seq_len, head_dim)
        output = output.transpose(1, 0, 2)  # (seq_len, num_heads, head_dim)
        output = output.reshape(seq_length, -1)
        return self.out_proj(output)


class Siglip2MLP(nn.Module):
    def __init__(self, config: VisionConfig):
        super().__init__()
        self.fc1 = nn.Linear(config.hidden_size, config.intermediate_size)
        self.fc2 = nn.Linear(config.intermediate_size, config.hidden_size)

    def __call__(self, x: mx.array) -> mx.array:
        x = self.fc1(x)
        x = nn.gelu_approx(x)  # gelu_pytorch_tanh
        x = self.fc2(x)
        return x


class Siglip2EncoderLayer(nn.Module):
    def __init__(self, config: VisionConfig):
        super().__init__()
        self.layer_norm1 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.self_attn = Siglip2Attention(config)
        self.layer_norm2 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.mlp = Siglip2MLP(config)

    def __call__(
        self,
        hidden_states: mx.array,
        cu_seqlens: mx.array,
        position_embeddings=None,
    ) -> mx.array:
        residual = hidden_states
        hidden_states = self.layer_norm1(hidden_states)
        hidden_states = self.self_attn(
            hidden_states,
            cu_seqlens=cu_seqlens,
            position_embeddings=position_embeddings,
        )
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.layer_norm2(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        return hidden_states


class Siglip2Encoder(nn.Module):
    def __init__(self, config: VisionConfig):
        super().__init__()
        self.config = config
        self.layers = [
            Siglip2EncoderLayer(config) for _ in range(config.num_hidden_layers)
        ]
        self.spatial_merge_size = config.spatial_merge_size
        self.spatial_merge_unit = self.spatial_merge_size * self.spatial_merge_size
        self.patch_size = config.patch_size
        self.window_size = config.window_size
        self.fullatt_block_indexes = config.fullatt_block_indexes

        head_dim = config.hidden_size // config.num_attention_heads
        self.rotary_pos_emb = VisionRoPE(head_dim // 2)

    def rot_pos_emb(self, spatial_shapes):
        """Compute rotary position embeddings for vision tokens."""
        pos_ids = []

        for h, w in spatial_shapes.tolist():
            h, w = int(h), int(w)
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
            pos_ids.append(stacked_pos_ids)  # t=1, no repeat needed

        pos_ids = mx.concatenate(pos_ids, axis=0)
        max_grid_size = mx.max(spatial_shapes)
        rotary_pos_emb_full = self.rotary_pos_emb(max_grid_size)
        rotary_pos_emb = rotary_pos_emb_full[pos_ids]
        return rotary_pos_emb.reshape(pos_ids.shape[0], -1)

    def get_window_index(self, spatial_shapes):
        """Compute window indices for windowed attention."""
        window_index = []
        cu_window_seqlens = [0]
        window_index_id = 0
        vit_merger_window_size = (
            self.window_size // self.spatial_merge_size // self.patch_size
        )

        for grid_h, grid_w in spatial_shapes.tolist():
            grid_h, grid_w = int(grid_h), int(grid_w)
            grid_t = 1
            llm_grid_h = grid_h // self.spatial_merge_size
            llm_grid_w = grid_w // self.spatial_merge_size

            index = mx.arange(grid_t * llm_grid_h * llm_grid_w).reshape(
                grid_t, llm_grid_h, llm_grid_w
            )

            pad_h = (
                vit_merger_window_size - llm_grid_h % vit_merger_window_size
            ) % vit_merger_window_size
            pad_w = (
                vit_merger_window_size - llm_grid_w % vit_merger_window_size
            ) % vit_merger_window_size
            num_windows_h = (llm_grid_h + pad_h) // vit_merger_window_size
            num_windows_w = (llm_grid_w + pad_w) // vit_merger_window_size

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
            index_padded = mx.transpose(index_padded, (0, 1, 3, 2, 4)).reshape(
                grid_t,
                num_windows_h * num_windows_w,
                vit_merger_window_size,
                vit_merger_window_size,
            )

            seqlens = mx.sum(index_padded != -100, axis=(2, 3)).reshape(-1)
            index_padded = index_padded.reshape(-1)
            # Get indices where value is not -100
            valid_mask = np.where(np.array(index_padded) != -100)[0].tolist()
            index_new = index_padded[valid_mask]

            window_index.append(index_new + window_index_id)
            cu_seqlens_tmp = (
                mx.cumsum(seqlens, axis=0) * self.spatial_merge_unit
                + cu_window_seqlens[-1]
            )
            cu_window_seqlens.extend(cu_seqlens_tmp.tolist())
            window_index_id += int(grid_t * llm_grid_h * llm_grid_w)

        window_index = mx.concatenate(window_index, axis=0)
        cu_window_seqlens = mx.array(cu_window_seqlens, dtype=mx.int32)

        return window_index, cu_window_seqlens

    def __call__(
        self,
        inputs_embeds: mx.array,
        spatial_shapes: mx.array,
    ) -> mx.array:
        hidden_states = inputs_embeds
        rotary_pos_emb = self.rot_pos_emb(spatial_shapes)
        window_index, cu_window_seqlens = self.get_window_index(spatial_shapes)

        # Deduplicate cu_window_seqlens
        seen = set()
        idx = []
        for i, x in enumerate(cu_window_seqlens.tolist()):
            x_int = int(x)
            if x_int not in seen:
                seen.add(x_int)
                idx.append(i)
        idx = mx.array(idx, dtype=mx.int32)
        cu_window_seqlens = cu_window_seqlens[idx]

        seq_len = hidden_states.shape[0]

        # Reorder by window index
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

        # Precompute position embeddings (cos, sin) from rotary freqs
        emb = mx.concatenate([rotary_pos_emb, rotary_pos_emb], axis=-1)
        position_embeddings = (mx.cos(emb), mx.sin(emb))

        # Compute full cu_seqlens for full attention layers
        cu_seqlens = (
            (spatial_shapes[:, 0] * spatial_shapes[:, 1])
            .cumsum(axis=0)
            .astype(mx.int32)
        )
        cu_seqlens = mx.pad(cu_seqlens, (1, 0), mode="constant", constant_values=0)

        for layer_num, layer in enumerate(self.layers):
            if layer_num in self.fullatt_block_indexes:
                cu_seqlens_now = cu_seqlens
            else:
                cu_seqlens_now = cu_window_seqlens

            hidden_states = layer(
                hidden_states,
                cu_seqlens=cu_seqlens_now,
                position_embeddings=position_embeddings,
            )

        # Reverse window reordering
        hidden_states = hidden_states.reshape(
            seq_len // self.spatial_merge_unit, self.spatial_merge_unit, -1
        )
        reverse_indices = mx.argsort(window_index, axis=0)
        hidden_states = hidden_states[reverse_indices, :, :]
        hidden_states = hidden_states.reshape(seq_len, -1)

        return hidden_states


class VLPatchMerger(nn.Module):
    """Merge vision patches to match language model hidden size."""

    def __init__(
        self,
        dim: int,
        context_dim: int,
        spatial_merge_size: int = 2,
    ) -> None:
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
        for layer in self.mlp:
            x = layer(x)
        return x


class VisionModel(nn.Module):
    """Top-level Siglip2 vision model for YoutuVL."""

    def __init__(self, config: VisionConfig) -> None:
        super().__init__()
        self.config = config
        self.model_type = config.model_type
        self.embeddings = Siglip2VisionEmbeddings(config)
        self.encoder = Siglip2Encoder(config)
        self.post_layernorm = nn.LayerNorm(
            config.hidden_size, eps=config.layer_norm_eps
        )
        self.merger = VLPatchMerger(
            dim=config.out_hidden_size,
            context_dim=config.hidden_size,
            spatial_merge_size=config.spatial_merge_size,
        )

    def __call__(
        self,
        pixel_values: mx.array,
        spatial_shapes: mx.array,
    ) -> mx.array:
        hidden_states = self.embeddings(pixel_values)
        hidden_states = self.encoder(hidden_states, spatial_shapes)
        hidden_states = self.post_layernorm(hidden_states)
        hidden_states = self.merger(hidden_states)
        return hidden_states
