"""Temporal motion module (AnimateDiff-style) for Video Depth Anything."""

import math

import mlx.core as mx
import mlx.nn as nn


def sinusoidal_table(d_model: int, max_len: int) -> mx.array:
    """Sinusoidal position table of shape (1, max_len, d_model)."""
    position = mx.arange(max_len).astype(mx.float32)[:, None]
    div_term = mx.exp(
        mx.arange(0, d_model, 2).astype(mx.float32) * (-math.log(10000.0) / d_model)
    )
    pe = mx.zeros((1, max_len, d_model))
    pe[0, :, 0::2] = mx.sin(position * div_term)
    pe[0, :, 1::2] = mx.cos(position * div_term)
    return pe


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int):
        super().__init__()
        self.pe = sinusoidal_table(d_model, max_len)

    def __call__(self, x: mx.array) -> mx.array:
        return x + self.pe[:, : x.shape[1]].astype(x.dtype)


class TemporalAttention(nn.Module):
    """Self-attention along the temporal axis."""

    def __init__(
        self,
        query_dim: int,
        heads: int,
        dim_head: int,
        temporal_max_len: int,
        pos_embedding_type: str = "ape",
    ):
        super().__init__()
        inner_dim = heads * dim_head
        self.heads = heads
        self.scale = dim_head**-0.5
        self.pos_embedding_type = pos_embedding_type

        self.to_q = nn.Linear(query_dim, inner_dim, bias=False)
        self.to_k = nn.Linear(query_dim, inner_dim, bias=False)
        self.to_v = nn.Linear(query_dim, inner_dim, bias=False)
        self.to_out = [nn.Linear(inner_dim, query_dim, bias=True), nn.Dropout(0.0)]

        if pos_embedding_type == "ape":
            self.pos_encoder = PositionalEncoding(query_dim, temporal_max_len)
        else:
            raise NotImplementedError(f"pos_embedding_type {pos_embedding_type!r}")

    def __call__(self, hidden_states: mx.array, video_length: int) -> mx.array:
        # (B*T, D, C) -> (B*D, T, C)
        BT, D, C = hidden_states.shape
        B = BT // video_length
        x = hidden_states.reshape(B, video_length, D, C).transpose(0, 2, 1, 3)
        x = x.reshape(B * D, video_length, C)

        # The position table is added before the query/key/value projections
        # and the queries are taken from the position-encoded states.
        x = self.pos_encoder(x)

        q = self.to_q(x)
        k = self.to_k(x)
        v = self.to_v(x)

        def split_heads(t):
            return t.reshape(
                B * D, video_length, self.heads, C // self.heads
            ).transpose(0, 2, 1, 3)

        out = mx.fast.scaled_dot_product_attention(
            split_heads(q), split_heads(k), split_heads(v), scale=self.scale
        )
        out = out.transpose(0, 2, 1, 3).reshape(B * D, video_length, C)
        out = self.to_out[0](out)

        # (B*D, T, C) -> (B*T, D, C)
        out = out.reshape(B, D, video_length, C).transpose(0, 2, 1, 3)
        return out.reshape(BT, D, C)


class GEGLU(nn.Module):
    def __init__(self, dim_in: int, dim_out: int):
        super().__init__()
        self.proj = nn.Linear(dim_in, dim_out * 2, bias=True)

    def __call__(self, x: mx.array) -> mx.array:
        x, gate = mx.split(self.proj(x), 2, axis=-1)
        return x * nn.gelu(gate)


class FeedForward(nn.Module):
    def __init__(self, dim: int, mult: int = 4):
        super().__init__()
        inner_dim = int(dim * mult)
        self.net = [GEGLU(dim, inner_dim), nn.Dropout(0.0), nn.Linear(inner_dim, dim)]

    def __call__(self, x: mx.array) -> mx.array:
        for module in self.net:
            x = module(x)
        return x


class TemporalTransformerBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        num_attention_heads: int,
        attention_head_dim: int,
        num_attention_blocks: int,
        temporal_max_len: int,
        pos_embedding_type: str,
    ):
        super().__init__()
        self.attention_blocks = [
            TemporalAttention(
                query_dim=dim,
                heads=num_attention_heads,
                dim_head=attention_head_dim,
                temporal_max_len=temporal_max_len,
                pos_embedding_type=pos_embedding_type,
            )
            for _ in range(num_attention_blocks)
        ]
        self.norms = [nn.LayerNorm(dim) for _ in range(num_attention_blocks)]
        self.ff = FeedForward(dim)
        self.ff_norm = nn.LayerNorm(dim)

    def __call__(self, hidden_states: mx.array, video_length: int) -> mx.array:
        for attention_block, norm in zip(self.attention_blocks, self.norms):
            hidden_states = (
                attention_block(norm(hidden_states), video_length) + hidden_states
            )
        return self.ff(self.ff_norm(hidden_states)) + hidden_states


class TemporalTransformer3DModel(nn.Module):
    def __init__(
        self,
        in_channels: int,
        num_attention_heads: int,
        attention_head_dim: int,
        num_layers: int,
        num_attention_blocks: int,
        norm_num_groups: int,
        temporal_max_len: int,
        pos_embedding_type: str,
    ):
        super().__init__()
        inner_dim = num_attention_heads * attention_head_dim
        self.norm = nn.GroupNorm(
            norm_num_groups, in_channels, eps=1e-6, pytorch_compatible=True
        )
        self.proj_in = nn.Linear(in_channels, inner_dim)
        self.transformer_blocks = [
            TemporalTransformerBlock(
                dim=inner_dim,
                num_attention_heads=num_attention_heads,
                attention_head_dim=attention_head_dim,
                num_attention_blocks=num_attention_blocks,
                temporal_max_len=temporal_max_len,
                pos_embedding_type=pos_embedding_type,
            )
            for _ in range(num_layers)
        ]
        self.proj_out = nn.Linear(inner_dim, in_channels)

    def __call__(self, hidden_states: mx.array) -> mx.array:
        # hidden_states: (B, T, H, W, C) channel-last
        B, T, H, W, C = hidden_states.shape
        x = hidden_states.reshape(B * T, H, W, C)
        residual = x

        x = self.norm(x)
        x = x.reshape(B * T, H * W, C)
        x = self.proj_in(x)

        for block in self.transformer_blocks:
            x = block(x, video_length=T)

        x = self.proj_out(x)
        x = x.reshape(B * T, H, W, C)
        return (x + residual).reshape(B, T, H, W, C)


class TemporalModule(nn.Module):
    def __init__(
        self,
        in_channels: int,
        num_attention_heads: int = 8,
        num_transformer_block: int = 1,
        num_attention_blocks: int = 2,
        norm_num_groups: int = 32,
        temporal_max_len: int = 32,
        pos_embedding_type: str = "ape",
    ):
        super().__init__()
        self.temporal_transformer = TemporalTransformer3DModel(
            in_channels=in_channels,
            num_attention_heads=num_attention_heads,
            attention_head_dim=in_channels // num_attention_heads,
            num_layers=num_transformer_block,
            num_attention_blocks=num_attention_blocks,
            norm_num_groups=norm_num_groups,
            temporal_max_len=temporal_max_len,
            pos_embedding_type=pos_embedding_type,
        )

    def __call__(self, input_tensor: mx.array) -> mx.array:
        return self.temporal_transformer(input_tensor)
