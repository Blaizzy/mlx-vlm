from typing import Optional

import mlx.core as mx
import mlx.nn as nn

from ..interpolate import bilinear_interpolate
from .config import VisionConfig


def check_array_shape(arr):
    out_channels, kH, KW, t = arr.shape

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


class PaddleOCRVisionEmbeddings(nn.Module):
    def __init__(
        self,
        patch_size: int = 14,
        image_size: int = 384,
        in_channels: int = 3,
        embed_dim: int = 1152,
    ) -> None:
        super().__init__()
        self.patch_size = patch_size
        self.in_channels = in_channels
        self.embed_dim = embed_dim

        self.patch_embedding = nn.Conv2d(
            in_channels=in_channels,
            out_channels=embed_dim,
            kernel_size=patch_size,
            stride=patch_size,
        )

        num_patches = (image_size // patch_size) ** 2
        self.position_embedding = nn.Embedding(num_patches, self.embed_dim)

    def interpolate_pos_encoding(self, height: int, width: int) -> mx.array:
        # Get the number of positions and embedding dimension
        num_positions = self.position_embedding.weight.shape[0]

        # Get all position embeddings (this will dequantize if quantized)
        position_ids = mx.arange(num_positions)
        patch_pos_embed = self.position_embedding(position_ids)
        dim = patch_pos_embed.shape[-1]

        # Reshape to 2D grid
        sqrt_num_positions = int(num_positions**0.5)
        patch_pos_embed = patch_pos_embed.reshape(
            1, sqrt_num_positions, sqrt_num_positions, dim
        )

        # Interpolate to target size
        patch_pos_embed = bilinear_interpolate(
            patch_pos_embed[0],
            height,
            width,
        ).astype(patch_pos_embed.dtype)
        patch_pos_embed = patch_pos_embed.reshape(-1, dim)
        return patch_pos_embed

    def __call__(self, hidden_states: mx.array, grid_thw: mx.array) -> mx.array:
        batch_size, squence_len, channel, patch_size, patch_size = hidden_states.shape
        target_dtype = self.patch_embedding.weight.dtype
        hidden_states = hidden_states.reshape(
            batch_size * squence_len, channel, patch_size, patch_size
        )
        # For MLX-Conv2d
        hidden_states = hidden_states.transpose(0, 2, 3, 1)
        patch_embeds = self.patch_embedding(hidden_states).astype(target_dtype)
        patch_embeds = patch_embeds.transpose(0, 3, 1, 2)
        embeddings = patch_embeds.flatten(-2).squeeze(-1)
        embeddings = embeddings.reshape(batch_size, squence_len, -1)

        start = 0
        embeddings = embeddings.squeeze(0)
        tmp_embeddings = []
        for image_grid in grid_thw:
            t, h, w = image_grid.tolist()
            end = start + t * h * w
            image_embeddings = embeddings[start:end, :]
            position_embedding = self.interpolate_pos_encoding(h, w)
            image_embeddings = image_embeddings + position_embedding
            tmp_embeddings.append(image_embeddings)
            start = end
        embeddings = mx.concatenate(tmp_embeddings, axis=0)

        return embeddings


class PaddleOCRProjector(nn.Module):
    def __init__(self, dim, context_dim, spatial_merge_size) -> None:
        super().__init__()

        hidden_size = dim * (spatial_merge_size**2)
        self.spatial_merge_size = spatial_merge_size
        self.pre_norm = nn.LayerNorm(dim, eps=1e-6)
        self.linear_1 = nn.Linear(hidden_size, hidden_size, bias=True)
        self.act = nn.GELU()
        self.linear_2 = nn.Linear(hidden_size, context_dim, bias=True)

    def __call__(self, x: mx.array, grid_thw: mx.array) -> mx.array:
        x_chunks = x.split(grid_thw.prod(axis=1).tolist(), axis=0)

        processed_features = []
        for x, image_grid in zip(x_chunks, grid_thw):
            x = self.pre_norm(x)
            t, h, w = image_grid.tolist()
            d = x.shape[-1]
            h_block = h // self.spatial_merge_size
            w_block = w // self.spatial_merge_size

            x = x.reshape(
                t, h_block, self.spatial_merge_size, w_block, self.spatial_merge_size, d
            )
            x = x.transpose(0, 1, 3, 2, 4, 5)
            x = x.reshape(
                t * h_block * w_block,
                self.spatial_merge_size * self.spatial_merge_size * d,
            )

            hidden_states = self.linear_1(x)
            hidden_states = self.act(hidden_states)
            hidden_states = self.linear_2(hidden_states)
            processed_features.append(hidden_states)

        return mx.concatenate(processed_features, axis=0)


class Attention(nn.Module):
    def __init__(self, dim: int, num_heads: int = 16) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim = dim // num_heads
        self.scale = head_dim**-0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=True)
        self.out_proj = nn.Linear(dim, dim)

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
        return self.out_proj(output)


class MLP(nn.Module):
    def __init__(self, dim, hidden_dim):
        super().__init__()
        self.activation_fn = nn.GELU(approx="precise")
        self.fc1 = nn.Linear(dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, dim)

    def __call__(self, x: mx.array) -> mx.array:
        x = self.activation_fn(self.fc1(x))
        x = self.fc2(x)
        return x


class PaddleOCRVisionEncoderLayer(nn.Module):
    def __init__(self, config: VisionConfig) -> None:
        super().__init__()
        self.layer_norm1 = nn.LayerNorm(config.hidden_size, eps=1e-6)
        self.layer_norm2 = nn.LayerNorm(config.hidden_size, eps=1e-6)

        self.self_attn = Attention(
            dim=config.hidden_size, num_heads=config.num_attention_heads
        )
        self.mlp = MLP(dim=config.hidden_size, hidden_dim=config.intermediate_size)

    def __call__(self, hidden_states, cu_seqlens, rotary_pos_emb) -> mx.array:
        hidden_states = hidden_states + self.self_attn(
            self.layer_norm1(hidden_states),
            cu_seqlens=cu_seqlens,
            rotary_pos_emb=rotary_pos_emb,
        )
        hidden_states = hidden_states + self.mlp(self.layer_norm2(hidden_states))
        return hidden_states


class VisionModel(nn.Module):
    def __init__(self, config: VisionConfig) -> None:
        super().__init__()
        self.config = config
        self.model_type = config.model_type
        if self.model_type != "paddleocr_vl":
            raise ValueError(f"Unsupported model type: {self.model_type}")

        self.embeddings = PaddleOCRVisionEmbeddings(
            patch_size=config.patch_size,
            image_size=config.image_size,
            in_channels=config.num_channels,
            embed_dim=config.hidden_size,
        )

        head_dim = config.hidden_size // config.num_attention_heads
        self.rotary_pos_emb = VisionRotaryEmbedding(head_dim // 2)

        self.layers = [
            PaddleOCRVisionEncoderLayer(config) for _ in range(config.num_hidden_layers)
        ]
        self.post_layernorm = nn.LayerNorm(
            config.hidden_size, eps=config.layer_norm_eps
        )
        self.projector = PaddleOCRProjector(
            dim=config.hidden_size,
            context_dim=1024,
            spatial_merge_size=config.spatial_merge_size,
        )

    def rot_pos_emb(self, grid_thw):
        pos_ids = []

        split_hids = []
        split_wids = []
        for t, h, w in grid_thw:
            image_pids = mx.arange(int(t * h * w)) % (h * w)
            sample_hids = image_pids // w
            sample_wids = image_pids % w
            split_hids.append(sample_hids)
            split_wids.append(sample_wids)

        height_position_ids = mx.concatenate(split_hids, axis=0)
        width_position_ids = mx.concatenate(split_wids, axis=0)

        pos_ids = mx.stack([height_position_ids, width_position_ids], axis=-1)
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
        hidden_states = self.embeddings(hidden_states, grid_thw)
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
        for layer in self.layers:
            hidden_states = layer(
                hidden_states, cu_seqlens=cu_seqlens, rotary_pos_emb=rotary_pos_emb
            )
            if output_hidden_states:
                encoder_states = encoder_states + (hidden_states,)

        hidden_states = self.post_layernorm(hidden_states)
        hidden_states = self.projector(hidden_states, grid_thw)
        return hidden_states

    def sanitize(self, weights):
        sanitized_weights = {}
        for k, v in weights.items():
            if "position_ids" in k:
                # Remove unused position_ids
                continue
            elif "patch_embedding.weight" in k:
                # PyTorch conv2d weight tensors have shape:
                #   [out_channels, in_channels, kH, KW]
                # MLX conv2d expects the weight be of shape:
                #   [out_channels, kH, KW, in_channels]
                if check_array_shape(v):
                    sanitized_weights[k] = v
                else:
                    sanitized_weights[k] = v.transpose(0, 2, 3, 1)
            else:
                sanitized_weights[k] = v

        return sanitized_weights
