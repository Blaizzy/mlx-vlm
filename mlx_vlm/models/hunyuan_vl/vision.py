from typing import Optional, Tuple

import mlx.core as mx
import mlx.nn as nn

from ..interpolate import resize_bilinear
from .config import VisionConfig


class VisionMLP(nn.Module):

    def __init__(self, config: VisionConfig):
        super().__init__()
        self.dense_h_to_4h = nn.Linear(
            config.hidden_size, config.intermediate_size, bias=True
        )
        self.dense_4h_to_h = nn.Linear(
            config.intermediate_size, config.hidden_size, bias=True
        )

    def __call__(self, x: mx.array) -> mx.array:
        return self.dense_4h_to_h(nn.gelu(self.dense_h_to_4h(x)))


class VisionAttention(nn.Module):

    def __init__(self, config: VisionConfig):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.scale = self.head_dim**-0.5

        # All projections have bias in HunyuanOCR vision tower
        self.q_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=True)
        self.k_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=True)
        self.v_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=True)
        self.o_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=True)

    def __call__(self, x: mx.array) -> mx.array:
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

        # Scaled dot-product attention (no mask for vision)
        output = mx.fast.scaled_dot_product_attention(
            queries, keys, values, scale=self.scale
        )

        # Reshape and project
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


class VisionEmbeddings(nn.Module):

    def __init__(self, config: VisionConfig):
        super().__init__()
        self.config = config
        self.patch_size = config.patch_size
        self.hidden_size = config.hidden_size
        self.num_channels = config.num_channels

        self.patch_embedding = nn.Conv2d(
            in_channels=config.num_channels,
            out_channels=config.hidden_size,
            kernel_size=config.patch_size,
            stride=config.patch_size,
            bias=config.add_patchemb_bias,
        )

        # Learned position embedding - will be interpolated to match grid size
        # Max grid size based on max_image_size / patch_size
        # +1 for potential CLS token (HunyuanOCR has 16385 = 128*128 + 1)
        max_grid = config.max_image_size // config.patch_size
        self.position_edge = int((max_grid * max_grid + 1) ** 0.5)
        self.position_embedding = nn.Embedding(
            max_grid * max_grid + 1, config.hidden_size
        )

        self._patch_pos_embed = None

    def __call__(self, pixel_values: mx.array, grid_thw: list) -> mx.array:

        num_patches, hidden_dim = pixel_values.shape

        pixel_values = pixel_values.reshape(
            num_patches, self.num_channels, self.patch_size, self.patch_size
        )

        pixel_values = pixel_values.transpose(0, 2, 3, 1)
        patch_embeds = self.patch_embedding(pixel_values)

        # Squeeze spatial dims and add batch dim: (1, num_patches, hidden_size)
        patch_embeds = patch_embeds.squeeze(1).squeeze(1)[None, :, :]

        # Get position embeddings with interpolation for each image
        patch_pos_embed_list = []
        for grid in grid_thw:
            t, h, w = grid
            h, w = int(h), int(w)
            pos_embed = self._get_position_embedding(h, w, patch_embeds.dtype)
            patch_pos_embed_list.append(pos_embed)

        # Concatenate position embeddings for all images
        patch_pos_embed = mx.concatenate(patch_pos_embed_list, axis=1)

        # Add position embeddings
        embeddings = patch_embeds + patch_pos_embed

        return embeddings

    def _manual_bilinear(self, x: mx.array, size: Tuple[int, int]) -> mx.array:
        """
        Perform bilinear interpolation matching PyTorch's align_corners=False.
        x: (B, H_in, W_in, C)
        size: (H_out, W_out)
        """
        B, H_in, W_in, C = x.shape
        H_out, W_out = size

        # Map to input coordinates
        # x_in = (x_out + 0.5) * (src_size / dst_size) - 0.5
        h_idx = mx.arange(H_out, dtype=mx.float32)
        w_idx = mx.arange(W_out, dtype=mx.float32)

        # HunyuanOCR uses a specific hack in HF implementation:
        # scale_factor = (size + 0.1) / src_size
        # So src_size / dst_size becomes src_size / (size + 0.1)
        h_scale = H_in / (H_out + 0.1)
        w_scale = W_in / (W_out + 0.1)

        h_in = (h_idx + 0.5) * h_scale - 0.5
        w_in = (w_idx + 0.5) * w_scale - 0.5

        # Clamp to valid range
        h_in = mx.clip(h_in, 0, H_in - 1)
        w_in = mx.clip(w_in, 0, W_in - 1)

        # Get integer parts and next neighbors
        h0 = mx.floor(h_in).astype(mx.int32)
        w0 = mx.floor(w_in).astype(mx.int32)
        h1 = mx.minimum(h0 + 1, H_in - 1)
        w1 = mx.minimum(w0 + 1, W_in - 1)

        # Get fractional parts (weights)
        h_lambda = (h_in - h0).astype(x.dtype)
        w_lambda = (w_in - w0).astype(x.dtype)

        # Reshape weights for broadcasting: (1, H_out, 1, 1) and (1, 1, W_out, 1)
        h_lambda = h_lambda.reshape(1, H_out, 1, 1)
        w_lambda = w_lambda.reshape(1, 1, W_out, 1)

        # Gather values from input tensor
        # We need to gather (B, H_out, W_out, C)
        # x is (B, H_in, W_in, C)

        # Gather along height: result is (B, H_out, W_in, C)
        x_h0 = mx.take(x, h0, axis=1)
        x_h1 = mx.take(x, h1, axis=1)

        # Gather along width from the height-gathered tensors: result is (B, H_out, W_out, C)
        q00 = mx.take(x_h0, w0, axis=2)
        q01 = mx.take(x_h0, w1, axis=2)
        q10 = mx.take(x_h1, w0, axis=2)
        q11 = mx.take(x_h1, w1, axis=2)

        # Bilinear interpolation
        r0 = q00 * (1 - w_lambda) + q01 * w_lambda
        r1 = q10 * (1 - w_lambda) + q11 * w_lambda
        result = r0 * (1 - h_lambda) + r1 * h_lambda

        return result

    def _get_position_embedding(self, grid_h: int, grid_w: int, dtype) -> mx.array:
        """Get position embedding with interpolation if necessary."""
        # Get base position embeddings (skip first token which is CLS)
        pos_ids = mx.arange(1, self.position_edge * self.position_edge + 1)
        patch_pos_embed = self.position_embedding(pos_ids)

        # Reshape to 2D grid: (position_edge, position_edge, hidden_size)
        patch_pos_embed = patch_pos_embed.reshape(
            self.position_edge, self.position_edge, self.hidden_size
        )

        # Interpolate to target grid size if needed
        if grid_h != self.position_edge or grid_w != self.position_edge:
            # Add batch dim for interpolation: (1, position_edge, position_edge, hidden_size)
            patch_pos_embed = patch_pos_embed[None, :, :, :]

            # Use manual bilinear interpolation to match HF/PyTorch exactly
            patch_pos_embed = resize_bilinear(
                patch_pos_embed.transpose(0, 3, 1, 2), (grid_h, grid_w), antialias=False
            ).transpose(0, 2, 3, 1)
        else:
            patch_pos_embed = patch_pos_embed[None, :, :, :]

        # Reshape to (1, num_patches, hidden_size)
        patch_pos_embed = patch_pos_embed.reshape(1, -1, self.hidden_size)

        return patch_pos_embed.astype(dtype)


class VisionPatchMerger(nn.Module):

    def __init__(self, config: VisionConfig):
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

        rows_with_newline = []
        for row_idx in range(merged_h):
            row = x[:, row_idx, :, :]  # (B, merged_w, final_hidden)
            newline = mx.broadcast_to(
                self.image_newline[None, None, :], (B, 1, final_hidden)
            )
            rows_with_newline.append(mx.concatenate([row, newline], axis=1))

        x = mx.concatenate(rows_with_newline, axis=1)

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

        self.embeddings = VisionEmbeddings(config)

        self.layers = [VisionBlock(config) for _ in range(config.num_hidden_layers)]

        # Patch merger (perceive module)
        self.perceive = VisionPatchMerger(config)

    def __call__(
        self,
        pixel_values: mx.array,
        image_grid_thw: Optional[mx.array] = None,
        **kwargs,
    ) -> mx.array:

        if image_grid_thw is not None:
            if hasattr(image_grid_thw, "tolist"):
                grid_thw = image_grid_thw.tolist()
            else:
                grid_thw = [list(g) for g in image_grid_thw]
        else:
            raise ValueError("image_grid_thw is required for HunyuanOCR vision model")

        hidden_states = self.embeddings(pixel_values, grid_thw)

        for layer in self.layers:
            hidden_states = layer(hidden_states)

        cu_seqlens = [0]
        for t, h, w in grid_thw:
            cu_seqlens.append(cu_seqlens[-1] + int(h) * int(w))

        processed_items = []
        for i, (t, h, w) in enumerate(grid_thw):
            start_idx = cu_seqlens[i]
            end_idx = cu_seqlens[i + 1]
            item = hidden_states[:, start_idx:end_idx, :]
            processed = self.perceive(item, int(h), int(w))
            processed_items.append(processed)

        vision_features = mx.concatenate(processed_items, axis=1)

        return vision_features

    def get_num_tokens(self, grid_h: int, grid_w: int) -> int:

        merge = self.config.spatial_merge_size
        merged_h = grid_h // merge
        merged_w = grid_w // merge
        return merged_h * (merged_w + 1) + 2
