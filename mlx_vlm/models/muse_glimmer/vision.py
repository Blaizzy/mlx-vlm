from typing import List

import mlx.core as mx
import mlx.nn as nn
import numpy as np

from ..base import ensure_fused_sdpa
from .config import VisionConfig


def rotate_half(x: mx.array) -> mx.array:
    half = x.shape[-1] // 2
    return mx.concatenate((-x[..., half:], x[..., :half]), axis=-1)


def apply_rotary(q: mx.array, k: mx.array, cos: mx.array, sin: mx.array):
    cos = cos[None, :, None, :].astype(mx.float32)
    sin = sin[None, :, None, :].astype(mx.float32)
    q_dtype, k_dtype = q.dtype, k.dtype
    q = q.astype(mx.float32)
    k = k.astype(mx.float32)
    q = (q * cos + rotate_half(q) * sin).astype(q_dtype)
    k = (k * cos + rotate_half(k) * sin).astype(k_dtype)
    return q, k


def _cu_seqlens(grid_thw: mx.array) -> mx.array:
    lengths: List[int] = []
    for temporal, height, width in grid_thw.tolist():
        lengths.extend([int(height) * int(width)] * int(temporal))
    return mx.array([0] + np.cumsum(lengths).tolist(), dtype=mx.int32)


def _window_index(grid_thw: mx.array, window_size: int):
    indices = []
    cumulative = [0]
    offset = 0
    for temporal, height, width in grid_thw.tolist():
        temporal, height, width = int(temporal), int(height), int(width)
        index = np.arange(temporal * height * width).reshape(temporal, height, width)
        pad_h = window_size - height % window_size
        pad_w = window_size - width % window_size
        n_h = (height + pad_h) // window_size
        n_w = (width + pad_w) // window_size
        padded = np.pad(
            index,
            ((0, 0), (0, pad_h), (0, pad_w)),
            constant_values=-100,
        )
        padded = padded.reshape(temporal, n_h, window_size, n_w, window_size).transpose(
            0, 1, 3, 2, 4
        )
        lengths = (padded != -100).sum(axis=(3, 4)).reshape(-1)
        flattened = padded.reshape(-1)
        indices.append(flattened[flattened != -100] + offset)
        for length in lengths:
            cumulative.append(cumulative[-1] + int(length))
        offset += temporal * height * width

    cumulative = np.asarray(cumulative, dtype=np.int32)
    keep = np.concatenate(([True], cumulative[1:] != cumulative[:-1]))
    return (
        mx.array(np.concatenate(indices), dtype=mx.int32),
        mx.array(cumulative[keep], dtype=mx.int32),
    )


def _position_ids(grid_thw: mx.array) -> mx.array:
    result = []
    for temporal, height, width in grid_thw.tolist():
        temporal, height, width = int(temporal), int(height), int(width)
        rows, cols = np.meshgrid(np.arange(height), np.arange(width), indexing="ij")
        # Reference order after flip(-1) + 1 is (width, height), one-indexed.
        coords = np.stack((cols.reshape(-1), rows.reshape(-1)), axis=-1) + 1
        result.append(np.tile(coords, (temporal, 1)))
    return mx.array(np.concatenate(result), dtype=mx.int32)


class PatchEmbedder(nn.Module):
    def __init__(self, config: VisionConfig):
        super().__init__()
        patch_dim = config.patch_temporal * 3 * config.patch_size**2
        self.patch_embedding = nn.Linear(patch_dim, config.hidden_size, bias=False)
        self.position_embedding_table = nn.Embedding(
            config.pos_emb_height * config.pos_emb_width, config.hidden_size
        )
        self.num_grid_per_side = config.pos_emb_height

    def _position_embeddings(self, grid_thw: mx.array) -> mx.array:
        side = self.num_grid_per_side
        all_indices = [[] for _ in range(4)]
        all_weights = [[] for _ in range(4)]

        for temporal, height, width in grid_thw.tolist():
            temporal, height, width = int(temporal), int(height), int(width)
            h_grid = (np.arange(height, dtype=np.float32) + 0.5) * (side / height) - 0.5
            w_grid = (np.arange(width, dtype=np.float32) + 0.5) * (side / width) - 0.5
            h0, w0 = np.floor(h_grid).astype(np.int32), np.floor(w_grid).astype(
                np.int32
            )
            h1, w1 = h0 + 1, w0 + 1
            dh, dw = h_grid - h0, w_grid - w0
            vh0, vh1 = (h0 >= 0) & (h0 < side), (h1 >= 0) & (h1 < side)
            vw0, vw1 = (w0 >= 0) & (w0 < side), (w1 >= 0) & (w1 < side)
            h0, h1 = np.clip(h0, 0, side - 1), np.clip(h1, 0, side - 1)
            w0, w1 = np.clip(w0, 0, side - 1), np.clip(w1, 0, side - 1)

            indices = [
                (h0[:, None] * side + w0[None, :]).reshape(-1),
                (h0[:, None] * side + w1[None, :]).reshape(-1),
                (h1[:, None] * side + w0[None, :]).reshape(-1),
                (h1[:, None] * side + w1[None, :]).reshape(-1),
            ]
            weights = [
                (
                    (1 - dh)[:, None]
                    * (1 - dw)[None, :]
                    * (vh0[:, None] & vw0[None, :])
                ).reshape(-1),
                (
                    (1 - dh)[:, None] * dw[None, :] * (vh0[:, None] & vw1[None, :])
                ).reshape(-1),
                (
                    dh[:, None] * (1 - dw)[None, :] * (vh1[:, None] & vw0[None, :])
                ).reshape(-1),
                (dh[:, None] * dw[None, :] * (vh1[:, None] & vw1[None, :])).reshape(-1),
            ]
            for idx in range(4):
                all_indices[idx].append(np.tile(indices[idx], temporal))
                all_weights[idx].append(np.tile(weights[idx], temporal))

        indices = mx.array(
            np.stack([np.concatenate(part) for part in all_indices]),
            dtype=mx.int32,
        )
        weights = mx.array(
            np.stack([np.concatenate(part) for part in all_weights]),
            dtype=mx.float32,
        )
        table = self.position_embedding_table(indices)
        return mx.sum(table * weights[..., None], axis=0)

    def __call__(self, pixel_values: mx.array, grid_thw: mx.array) -> mx.array:
        embeddings = self.patch_embedding(pixel_values)
        positions = self._position_embeddings(grid_thw).astype(embeddings.dtype)
        return embeddings + positions


class VisionAttention(nn.Module):
    def __init__(self, config: VisionConfig):
        super().__init__()
        self.num_heads = config.num_attention_heads
        self.head_dim = config.hidden_size // config.num_attention_heads
        self.scale = self.head_dim**-0.5
        self.q_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=True)
        self.k_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=True)
        self.v_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=True)
        self.proj = nn.Linear(config.hidden_size, config.hidden_size, bias=True)

    def __call__(
        self,
        hidden_states: mx.array,
        cu_seqlens: mx.array,
        cos: mx.array,
        sin: mx.array,
    ) -> mx.array:
        length = hidden_states.shape[0]
        q = self.q_proj(hidden_states).reshape(1, length, self.num_heads, -1)
        k = self.k_proj(hidden_states).reshape(1, length, self.num_heads, -1)
        v = self.v_proj(hidden_states).reshape(1, length, self.num_heads, -1)
        q, k = apply_rotary(q, k, cos, sin)
        q, k, v = (tensor.transpose(0, 2, 1, 3) for tensor in (q, k, v))

        split_points = cu_seqlens[1:-1].tolist()
        chunks = [mx.split(tensor, split_points, axis=2) for tensor in (q, k, v)]
        output = mx.concatenate(
            [
                ensure_fused_sdpa(q_chunk, k_chunk, v_chunk, self.scale)
                for q_chunk, k_chunk, v_chunk in zip(*chunks)
            ],
            axis=2,
        )
        output = output.transpose(0, 2, 1, 3).reshape(length, -1)
        return self.proj(output)


class VisionMLP(nn.Module):
    def __init__(self, config: VisionConfig):
        super().__init__()
        self.fc1 = nn.Linear(config.hidden_size, config.intermediate_size, bias=True)
        self.fc2 = nn.Linear(config.intermediate_size, config.hidden_size, bias=True)

    def __call__(self, x: mx.array) -> mx.array:
        return self.fc2(nn.gelu(self.fc1(x)))


class VisionBlock(nn.Module):
    def __init__(self, config: VisionConfig):
        super().__init__()
        self.norm1 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.norm2 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.attn = VisionAttention(config)
        self.mlp = VisionMLP(config)

    def __call__(self, x, cu_seqlens, cos, sin):
        x = x + self.attn(self.norm1(x), cu_seqlens, cos, sin)
        return x + self.mlp(self.norm2(x))


class VisionModel(nn.Module):
    def __init__(self, config: VisionConfig):
        super().__init__()
        self.config = config
        self.patch_embedder = PatchEmbedder(config)
        self.ln_pre = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.layers = [VisionBlock(config) for _ in range(config.num_hidden_layers)]
        self.ln_post = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

        head_dim = config.hidden_size // config.num_attention_heads
        spatial_dim = head_dim // 2
        self.rope_theta = float(config.rope_parameters.get("rope_theta", 10000.0))
        self.rope_spatial_dim = spatial_dim

    def _rotary(self, position_ids: mx.array):
        inv_freq = 1.0 / (
            self.rope_theta
            ** (
                mx.arange(0, self.rope_spatial_dim, 2, dtype=mx.float32)
                / self.rope_spatial_dim
            )
        )
        width = position_ids[:, 0].astype(mx.float32)
        height = position_ids[:, 1].astype(mx.float32)
        freq_w = width[:, None] * inv_freq[None, :]
        freq_h = height[:, None] * inv_freq[None, :]
        freqs = mx.concatenate((freq_w, freq_h, freq_w, freq_h), axis=-1)
        return mx.cos(freqs), mx.sin(freqs)

    def _pixel_shuffle(self, hidden_states: mx.array, grid_thw: mx.array):
        merge = self.config.merge_size
        dim = hidden_states.shape[-1]
        outputs = []
        offset = 0
        for temporal, height, width in grid_thw.tolist():
            temporal, height, width = int(temporal), int(height), int(width)
            count = temporal * height * width
            chunk = hidden_states[offset : offset + count]
            chunk = chunk.reshape(temporal, height, width, dim)
            chunk = chunk.reshape(
                temporal,
                height // merge,
                merge,
                width // merge,
                merge,
                dim,
            ).transpose(0, 1, 3, 5, 2, 4)
            outputs.append(chunk.reshape(-1, dim * merge * merge))
            offset += count
        return mx.concatenate(outputs, axis=0)

    def __call__(self, pixel_values: mx.array, grid_thw: mx.array) -> mx.array:
        full_cu = _cu_seqlens(grid_thw)
        window_side = self.config.pos_emb_height
        window_index, window_cu = _window_index(grid_thw, window_side)

        hidden_states = self.patch_embedder(pixel_values, grid_thw)
        hidden_states = self.ln_pre(hidden_states)[window_index]
        positions = _position_ids(grid_thw)[window_index]
        cos, sin = self._rotary(positions)

        for idx, block in enumerate(self.layers):
            cu = (
                full_cu
                if self.config.layer_types[idx] == "full_attention"
                else window_cu
            )
            hidden_states = block(hidden_states, cu, cos, sin)

        hidden_states = hidden_states[mx.argsort(window_index)]
        hidden_states = self.ln_post(hidden_states)
        return self._pixel_shuffle(hidden_states, grid_thw)
