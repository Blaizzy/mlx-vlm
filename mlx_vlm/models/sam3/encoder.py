"""SAM3 DETR Transformer Encoder with text cross-attention fusion.

Weight keys: detector_model.detr_encoder.layers.*
"""

from typing import Optional

import mlx.core as mx
import mlx.nn as nn

from .config import DETREncoderConfig


class MultiheadAttention(nn.Module):
    """Standard multi-head attention with separate q/k/v/o projections."""

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        dropout: float = 0.0,
        kv_dim: Optional[int] = None,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.scale = self.head_dim**-0.5
        kv_dim = kv_dim if kv_dim is not None else hidden_size

        self.q_proj = nn.Linear(hidden_size, hidden_size)
        self.k_proj = nn.Linear(kv_dim, hidden_size)
        self.v_proj = nn.Linear(kv_dim, hidden_size)
        self.o_proj = nn.Linear(hidden_size, hidden_size)

    def __call__(
        self,
        query: mx.array,
        key: mx.array,
        value: mx.array,
        mask: Optional[mx.array] = None,
    ) -> mx.array:
        B, N_q, _ = query.shape
        N_k = key.shape[1]

        q = (
            self.q_proj(query)
            .reshape(B, N_q, self.num_heads, self.head_dim)
            .transpose(0, 2, 1, 3)
        )
        k = (
            self.k_proj(key)
            .reshape(B, N_k, self.num_heads, self.head_dim)
            .transpose(0, 2, 1, 3)
        )
        v = (
            self.v_proj(value)
            .reshape(B, N_k, self.num_heads, self.head_dim)
            .transpose(0, 2, 1, 3)
        )

        out = mx.fast.scaled_dot_product_attention(q, k, v, scale=self.scale, mask=mask)
        out = out.transpose(0, 2, 1, 3).reshape(B, N_q, -1)
        return self.o_proj(out)


class DETREncoderLayer(nn.Module):
    """Single DETR encoder layer: self-attn + text cross-attn + FFN.

    Weight keys per layer:
        self_attn.{q,k,v,o}_proj.{weight,bias}
        cross_attn.{q,k,v,o}_proj.{weight,bias}
        layer_norm1.{weight,bias}  (self-attn)
        layer_norm2.{weight,bias}  (cross-attn)
        layer_norm3.{weight,bias}  (FFN)
        mlp.fc1.{weight,bias}
        mlp.fc2.{weight,bias}
    """

    def __init__(self, config: DETREncoderConfig):
        super().__init__()
        d = config.hidden_size

        self.self_attn = MultiheadAttention(
            d, config.num_attention_heads, config.dropout
        )
        self.cross_attn = MultiheadAttention(
            d, config.num_attention_heads, config.dropout
        )

        self.layer_norm1 = nn.LayerNorm(d, eps=config.layer_norm_eps)
        self.layer_norm2 = nn.LayerNorm(d, eps=config.layer_norm_eps)
        self.layer_norm3 = nn.LayerNorm(d, eps=config.layer_norm_eps)

        self.mlp = MLP(d, config.intermediate_size, config.hidden_act)

    def __call__(
        self,
        src: mx.array,
        pos: mx.array,
        text_memory: mx.array,
        text_mask: Optional[mx.array] = None,
    ) -> mx.array:
        """Pre-norm encoder layer matching HF Sam3DetrEncoderLayer.

        Args:
            src: (B, HW, D) image features
            pos: (B, HW, D) positional encoding
            text_memory: (B, T, D) text features
            text_mask: (B, T) text padding mask (True=valid, False=pad)
        Returns:
            (B, HW, D) updated features
        """
        # 1. Self-attention with pre-norm; pos added to q/k only
        residual = src
        hidden = self.layer_norm1(src)
        hidden_with_pos = hidden + pos
        src2 = self.self_attn(hidden_with_pos, hidden_with_pos, hidden)
        src = residual + src2

        # 2. Cross-attention to text with pre-norm
        cross_mask = None
        if text_mask is not None:
            cross_mask = (1 - text_mask[:, None, None, :].astype(src.dtype)) * -1e9

        residual = src
        hidden = self.layer_norm2(src)
        src2 = self.cross_attn(hidden, text_memory, text_memory, mask=cross_mask)
        src = residual + src2

        # 3. FFN with pre-norm
        residual = src
        hidden = self.layer_norm3(src)
        src2 = self.mlp(hidden)
        src = residual + src2

        return src


class MLP(nn.Module):
    def __init__(self, hidden_size: int, intermediate_size: int, act: str = "relu"):
        super().__init__()
        self.fc1 = nn.Linear(hidden_size, intermediate_size)
        self.fc2 = nn.Linear(intermediate_size, hidden_size)
        self.act = act

    def __call__(self, x: mx.array) -> mx.array:
        x = self.fc1(x)
        if self.act == "relu":
            x = nn.relu(x)
        else:
            x = nn.gelu(x)
        return self.fc2(x)


class DETREncoder(nn.Module):
    """DETR Transformer Encoder with text fusion.

    Weight keys: detector_model.detr_encoder.*
    """

    def __init__(self, config: DETREncoderConfig):
        super().__init__()
        self.layers = [DETREncoderLayer(config) for _ in range(config.num_layers)]

    def __call__(
        self,
        src: mx.array,
        pos: mx.array,
        text_memory: mx.array,
        text_mask: Optional[mx.array] = None,
    ) -> mx.array:
        """
        Args:
            src: (B, HW, D) flattened multi-scale features
            pos: (B, HW, D) positional encoding
            text_memory: (B, T, D) text features
            text_mask: (B, T) text padding mask
        Returns:
            (B, HW, D) encoded features
        """
        output = src
        for layer in self.layers:
            output = layer(output, pos, text_memory, text_mask)
            mx.eval(output)
        return output
