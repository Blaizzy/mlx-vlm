import mlx.core as mx
import mlx.nn as nn

from ..base import chunked_attention
from .config import VisionConfig


class VisionMLP(nn.Module):
    def __init__(self, config: VisionConfig):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.dense_h_to_4h = nn.Linear(
            self.hidden_size, self.intermediate_size, bias=True
        )
        self.dense_4h_to_h = nn.Linear(
            self.intermediate_size, self.hidden_size, bias=True
        )

    def __call__(self, x: mx.array) -> mx.array:
        x = self.dense_h_to_4h(x)
        x = nn.gelu(x)
        x = self.dense_4h_to_h(x)
        return x


class VisionAttention(nn.Module):

    def __init__(self, config: VisionConfig):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.scale = self.head_dim**-0.5

        self.q_proj = nn.Linear(
            config.hidden_size, self.num_heads * self.head_dim, bias=True
        )
        self.k_proj = nn.Linear(
            config.hidden_size, self.num_heads * self.head_dim, bias=True
        )
        self.v_proj = nn.Linear(
            config.hidden_size, self.num_heads * self.head_dim, bias=True
        )
        self.o_proj = nn.Linear(
            self.num_heads * self.head_dim, config.hidden_size, bias=True
        )

    def __call__(self, x: mx.array, chunk_size: int = 1024) -> mx.array:
        B, L, _ = x.shape

        queries = self.q_proj(x)
        keys = self.k_proj(x)
        values = self.v_proj(x)

        # Reshape to (B, n_heads, L, head_dim)
        queries = queries.reshape(B, L, self.num_heads, self.head_dim).transpose(
            0, 2, 1, 3
        )
        keys = keys.reshape(B, L, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)
        values = values.reshape(B, L, self.num_heads, self.head_dim).transpose(
            0, 2, 1, 3
        )

        output = chunked_attention(
            queries,
            keys,
            values,
            scale=self.scale,
            chunk_size=chunk_size,
        )

        output = output.transpose(0, 2, 1, 3).reshape(B, L, -1)
        return self.o_proj(output)


class VisionBlock(nn.Module):

    def __init__(self, config: VisionConfig):
        super().__init__()
        self.input_layernorm = nn.LayerNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = nn.LayerNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )
        self.self_attn = VisionAttention(config)
        self.mlp = VisionMLP(config)

    def __call__(self, x: mx.array) -> mx.array:
        # Self-attention with residual
        h = x + self.self_attn(self.input_layernorm(x))
        # MLP with residual
        out = h + self.mlp(self.post_attention_layernorm(h))
        return out


class PatchEmbed(nn.Module):

    def __init__(self, config: VisionConfig):
        super().__init__()
        self.config = config
        self.embed_dim = config.hidden_size
        self.patch_size = config.patch_size
        self.num_channels = config.num_channels
        self.spatial_merge_size = config.spatial_merge_size
        self.interpolate_mode = config.interpolate_mode

        self.patch_embedding = nn.Conv2d(
            in_channels=config.num_channels,
            out_channels=self.embed_dim,
            kernel_size=self.patch_size,
            stride=self.patch_size,
            bias=True,
        )

        self.max_num_patches = (config.max_image_size // self.patch_size) ** 2
        self.num_positions = self.max_num_patches + 1
        self.position_edge = int(self.num_positions**0.5)
        self.position_embedding = nn.Embedding(self.num_positions, self.embed_dim)

    def __call__(self, pixel_values: mx.array, grid_thw: list) -> mx.array:
        num_patches = pixel_values.shape[0]
        # Reshape: (num_patches, C*P*P) -> (num_patches, C, P, P) -> (num_patches, P, P, C) for MLX conv
        pixel_values = pixel_values.reshape(
            num_patches, self.num_channels, self.patch_size, self.patch_size
        )
        pixel_values = pixel_values.transpose(0, 2, 3, 1)  # NCHW -> NHWC for MLX

        # Apply patch embedding
        patch_embeds = self.patch_embedding(pixel_values)  # (N, 1, 1, embed_dim)
        patch_embeds = patch_embeds.reshape(1, num_patches, self.embed_dim)

        # Get position embeddings and interpolate for each grid
        pos_embed_weights = self.position_embedding.weight[1:, :]  # Skip cls token
        base_pos_embed = pos_embed_weights.reshape(
            1, self.position_edge, self.position_edge, self.embed_dim
        )

        patch_pos_embed_list = []
        for grid in grid_thw:
            t, h, w = grid
            h_float = float(h) + 0.1
            w_float = float(w) + 0.1

            target_h = int(h)
            target_w = int(w)

            # Simple bilinear interpolation
            pos_embed = self._interpolate_pos_embed(base_pos_embed, target_h, target_w)
            pos_embed = pos_embed.reshape(1, -1, self.embed_dim)
            patch_pos_embed_list.append(pos_embed)

        patch_pos_embed = mx.concatenate(patch_pos_embed_list, axis=1)
        embeddings = patch_embeds + patch_pos_embed

        return embeddings

    def _interpolate_pos_embed(
        self, pos_embed: mx.array, target_h: int, target_w: int
    ) -> mx.array:
        dtype = pos_embed.dtype
        src_h, src_w = pos_embed.shape[1], pos_embed.shape[2]

        if src_h == target_h and src_w == target_w:
            return pos_embed

        # Create coordinate grids
        h_scale = src_h / (target_h + 0.1)
        w_scale = src_w / (target_w + 0.1)
        h_coords = (mx.arange(target_h) + 0.5) * h_scale - 0.5
        w_coords = (mx.arange(target_w) + 0.5) * w_scale - 0.5

        i0 = h_coords.astype(mx.int32)
        j0 = w_coords.astype(mx.int32)
        i1 = mx.minimum(i0 + 1, src_h - 1)
        j1 = mx.minimum(j0 + 1, src_w - 1)

        di = (h_coords - i0.astype(mx.float32))[:, None, None]
        dj = (w_coords - j0.astype(mx.float32))[None, :, None]

        # Gather corners and interpolate
        p00 = pos_embed[0, i0][:, j0]
        p01 = pos_embed[0, i0][:, j1]
        p10 = pos_embed[0, i1][:, j0]
        p11 = pos_embed[0, i1][:, j1]

        result = (
            (1 - di) * (1 - dj) * p00
            + (1 - di) * dj * p01
            + di * (1 - dj) * p10
            + di * dj * p11
        )

        return result[None].astype(dtype)


class PatchMerger(nn.Module):
    def __init__(
        self,
        config: VisionConfig,
    ):
        super().__init__()
        self.config = config
        self.spatial_merge_size = config.spatial_merge_size
        self.hidden_size = config.hidden_size
        self.out_hidden_size = config.out_hidden_size

        merge_hidden = config.hidden_size * 2  # 2304
        final_hidden = config.hidden_size * 4  # 4608

        self.before_rms = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.after_rms = nn.RMSNorm(config.out_hidden_size, eps=config.rms_norm_eps)

        self.proj = [
            nn.Conv2d(
                config.hidden_size,
                merge_hidden,
                kernel_size=config.spatial_merge_size,
                stride=config.spatial_merge_size,
                bias=True,
            ),
            nn.GELU(),
            nn.Conv2d(merge_hidden, final_hidden, kernel_size=1, bias=True),
        ]

        self.mlp = nn.Linear(final_hidden, config.out_hidden_size, bias=True)

        self.image_newline = mx.zeros((final_hidden,))
        self.image_begin = mx.zeros((config.out_hidden_size,))
        self.image_end = mx.zeros((config.out_hidden_size,))
        self.image_sep = mx.zeros((config.out_hidden_size,))

    def __call__(self, hidden_states: mx.array, grid_h: int, grid_w: int) -> mx.array:

        B = hidden_states.shape[0]
        final_hidden = self.config.hidden_size * 4  # 4608

        x = self.before_rms(hidden_states)

        x = x.reshape(B, grid_h, grid_w, self.hidden_size)

        for layer in self.proj:
            x = layer(x)

        merged_h = grid_h // self.spatial_merge_size
        merged_w = grid_w // self.spatial_merge_size

        x = x.reshape(B, merged_h, merged_w, final_hidden)

        newlines = mx.broadcast_to(
            self.image_newline[None, None, None, :], (B, merged_h, 1, final_hidden)
        )

        x = mx.concatenate(
            [x, newlines], axis=2
        )  # (B, merged_h, merged_w+1, final_hidden)
        x = x.reshape(B, merged_h * (merged_w + 1), final_hidden)

        x = self.mlp(x)

        begin = mx.broadcast_to(
            self.image_begin[None, None, :], (B, 1, self.out_hidden_size)
        )
        end = mx.broadcast_to(
            self.image_end[None, None, :], (B, 1, self.out_hidden_size)
        )

        x = mx.concatenate([begin, x, end], axis=1)

        x = self.after_rms(x)

        return x


class VisionModel(nn.Module):
    def __init__(self, config: VisionConfig):
        super().__init__()
        self.config = config
        self.model_type = config.model_type
        if self.model_type != "hunyuan_vl":
            raise ValueError(f"Unsupported model type: {self.model_type}")
        self.embeddings = PatchEmbed(config)
        self.layers = [VisionBlock(config) for _ in range(config.num_hidden_layers)]
        self.perceive = PatchMerger(
            config=config,
        )

    def __call__(
        self,
        pixel_values: mx.array,
        grid_thw: list,
    ) -> mx.array:
        """
        Args:
            pixel_values: Flattened pixel values of shape (total_patches, C*P*P)
            grid_thw: List of [t, h, w] for each image

        Returns:
            Image features of shape (1, total_tokens, text_hidden_size)
        """
        hidden_states = self.embeddings(pixel_values, grid_thw)

        for layer in self.layers:
            hidden_states = layer(hidden_states)

        # Calculate cumulative sequence lengths
        cu_seqlens = [0]
        for t, h, w in grid_thw:
            cu_seqlens.append(int(h * w))
        cu_seqlens = mx.cumsum(mx.array(cu_seqlens, dtype=mx.int32))

        # Split and process each image
        processed_items = []
        for i, grid in enumerate(grid_thw):
            t, h, w = grid
            start_idx = int(cu_seqlens[i])
            end_idx = int(cu_seqlens[i + 1])
            item = hidden_states[:, start_idx:end_idx, :]
            processed = self.perceive(item, int(h), int(w))
            processed_items.append(processed)

        hidden_states = mx.concatenate(processed_items, axis=1)
        return hidden_states
