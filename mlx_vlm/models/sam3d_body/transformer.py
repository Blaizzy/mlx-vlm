"""Transformer decoder layer for SAM 3D Body."""

import mlx.core as mx
import mlx.nn as nn

from .layers import LayerNorm32


class DecoderAttention(nn.Module):
    """Multi-head attention with separate q/k/v input dimensions."""

    def __init__(
        self,
        embed_dims: int,
        num_heads: int,
        query_dims: int,
        key_dims: int,
        value_dims: int,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.head_dims = embed_dims // num_heads
        self.q_proj = nn.Linear(query_dims, embed_dims)
        self.k_proj = nn.Linear(key_dims, embed_dims)
        self.v_proj = nn.Linear(value_dims, embed_dims)
        self.proj = nn.Linear(embed_dims, query_dims)

    def __call__(self, q: mx.array, k: mx.array, v: mx.array) -> mx.array:
        B, N, _ = q.shape
        q = (
            self.q_proj(q)
            .reshape(B, N, self.num_heads, self.head_dims)
            .transpose(0, 2, 1, 3)
        )
        k = (
            self.k_proj(k)
            .reshape(B, -1, self.num_heads, self.head_dims)
            .transpose(0, 2, 1, 3)
        )
        v = (
            self.v_proj(v)
            .reshape(B, -1, self.num_heads, self.head_dims)
            .transpose(0, 2, 1, 3)
        )
        out = mx.fast.scaled_dot_product_attention(q, k, v, scale=self.head_dims**-0.5)
        out = out.transpose(0, 2, 1, 3).reshape(B, N, -1)
        return self.proj(out)


class DecoderFFN(nn.Module):
    """ReLU FFN matching weight keys ffn.layers.0.0 and ffn.layers.1."""

    def __init__(self, embed_dims: int, hidden_dims: int):
        super().__init__()
        # layers[0] is a list with one Linear -> keys: layers.0.0.*
        # layers[1] is a Linear -> keys: layers.1.*
        self.layers = [
            [nn.Linear(embed_dims, hidden_dims)],
            nn.Linear(hidden_dims, embed_dims),
        ]

    def __call__(self, x: mx.array) -> mx.array:
        x = nn.relu(self.layers[0][0](x))
        return self.layers[1](x)


class TransformerDecoderLayer(nn.Module):
    """Single decoder layer: self-attn + cross-attn + FFN with LaPE norms.

    Returns (tokens, context) tuple — both are updated each layer,
    matching the PyTorch decoder which returns (token_embedding, image_embedding).
    """

    def __init__(
        self,
        token_dims: int = 1024,
        context_dims: int = 1280,
        num_heads: int = 8,
        head_dims: int = 64,
        mlp_dims: int = 1024,
        repeat_pe: bool = True,
        skip_first_pe: bool = False,
    ):
        super().__init__()
        embed_dims = num_heads * head_dims  # 512
        self.repeat_pe = repeat_pe
        self.skip_first_pe = skip_first_pe

        # LaPE norms
        self.ln_pe_1 = LayerNorm32(token_dims, eps=1e-6)
        self.ln_pe_2 = LayerNorm32(context_dims, eps=1e-6)

        # Self-attention
        self.ln1 = LayerNorm32(token_dims, eps=1e-6)
        self.self_attn = DecoderAttention(
            embed_dims=embed_dims,
            num_heads=num_heads,
            query_dims=token_dims,
            key_dims=token_dims,
            value_dims=token_dims,
        )

        # Cross-attention
        self.ln2_1 = LayerNorm32(token_dims, eps=1e-6)
        self.ln2_2 = LayerNorm32(context_dims, eps=1e-6)
        self.cross_attn = DecoderAttention(
            embed_dims=embed_dims,
            num_heads=num_heads,
            query_dims=token_dims,
            key_dims=context_dims,
            value_dims=context_dims,
        )

        # FFN
        self.ln3 = LayerNorm32(token_dims, eps=1e-6)
        self.ffn = DecoderFFN(token_dims, mlp_dims)

    def __call__(
        self,
        x: mx.array,
        context: mx.array,
        x_pe: mx.array = None,
        context_pe: mx.array = None,
    ) -> tuple[mx.array, mx.array]:
        # LaPE normalization
        if self.repeat_pe and context_pe is not None:
            x_pe = self.ln_pe_1(x_pe)
            context_pe = self.ln_pe_2(context_pe)

        # Self-attention
        if self.repeat_pe and not self.skip_first_pe and x_pe is not None:
            ln1_x = self.ln1(x)
            q = k = ln1_x + x_pe
            v = ln1_x
        else:
            q = k = v = self.ln1(x)
        x = x + self.self_attn(q, k, v)

        # Cross-attention
        if self.repeat_pe and context_pe is not None:
            q = self.ln2_1(x) + x_pe
            k = self.ln2_2(context) + context_pe
            v = self.ln2_2(context)
        else:
            q = self.ln2_1(x)
            k = v = self.ln2_2(context)
        x = x + self.cross_attn(q, k, v)

        # FFN with residual
        x = x + self.ffn(self.ln3(x))

        return x, context
