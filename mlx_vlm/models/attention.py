from dataclasses import dataclass
from functools import cached_property
from typing import Optional

import mlx.core as mx
import mlx.nn as nn

from .cache import create_causal_mask


class VisionAttention(nn.Module):
    def __init__(
        self,
        dims: int,
        num_heads: int,
        query_input_dims: Optional[int] = None,
        key_input_dims: Optional[int] = None,
        value_input_dims: Optional[int] = None,
        value_dims: Optional[int] = None,
        value_output_dims: Optional[int] = None,
        bias: bool = True,
    ):
        super().__init__()
        if dims % num_heads != 0:
            raise ValueError(
                f"The input feature dimensions should be divisible by the "
                f"number of heads ({dims} % {num_heads}) != 0"
            )
        query_input_dims = query_input_dims or dims
        key_input_dims = key_input_dims or dims
        value_input_dims = value_input_dims or key_input_dims
        value_dims = value_dims or dims
        value_output_dims = value_output_dims or dims
        self.num_heads = num_heads
        head_dim = dims // num_heads
        self.scale = head_dim ** (-0.5)
        self.q_proj = nn.Linear(query_input_dims, dims, bias=bias)
        self.k_proj = nn.Linear(key_input_dims, dims, bias=bias)
        self.v_proj = nn.Linear(value_input_dims, value_dims, bias=bias)
        self.out_proj = nn.Linear(value_dims, value_output_dims, bias=bias)

    def __call__(self, x, mask=None):
        queries = self.q_proj(x)
        keys = self.k_proj(x)
        values = self.v_proj(x)
        num_heads = self.num_heads
        B, L, D = queries.shape
        _, S, _ = keys.shape
        queries = queries.reshape(B, L, num_heads, -1).transpose(0, 2, 1, 3)
        keys = keys.reshape(B, S, num_heads, -1).transpose(0, 2, 1, 3)
        values = values.reshape(B, S, num_heads, -1).transpose(0, 2, 1, 3)
        output = mx.fast.scaled_dot_product_attention(
            queries, keys, values, scale=self.scale, mask=mask
        )
        output = output.transpose(0, 2, 1, 3).reshape(B, L, -1)
        return self.out_proj(output)


@dataclass(frozen=True)
class BatchAttentionMask:
    """Causal bounds shared across layers for one batched model call."""

    query_length: int
    offset: int
    left_padding: tuple[int, ...]

    @cached_property
    def decode_mask(self):
        return create_causal_mask(
            self.query_length,
            offset=self.offset,
            left_padding=mx.array(self.left_padding),
        )

    def apply(self, queries, keys, values, scale):
        if (
            queries.ndim != 4
            or keys.ndim != 4
            or values.ndim != 4
            or queries.shape[2] != self.query_length
            or keys.shape[2] != self.offset + self.query_length
            or queries.shape[0] != len(self.left_padding)
            or keys.shape[0] != queries.shape[0]
            or values.shape[:3] != keys.shape[:3]
            or queries.shape[-1] != keys.shape[-1]
            or queries.shape[1] % keys.shape[1]
        ):
            raise ValueError("Batched attention dimensions do not match the mask")
        if self.query_length < 16:
            # Keep short GQA queries on the native masked kernel.
            return mx.fast.scaled_dot_product_attention(
                queries, keys, values, scale=scale, mask=self.decode_mask
            )

        outputs = []
        start = 0
        while start < len(self.left_padding):
            end = start + 1
            pad = max(0, self.left_padding[start])
            while end < len(self.left_padding) and self.left_padding[end] == pad:
                end += 1
            query_start = min(self.query_length, max(0, pad - self.offset))
            if query_start == self.query_length:
                output = mx.zeros(
                    (
                        end - start,
                        queries.shape[1],
                        self.query_length,
                        values.shape[-1],
                    ),
                    dtype=queries.dtype,
                )
            else:
                q = queries[start:end, :, query_start:]
                k, v = keys[start:end, :, pad:], values[start:end, :, pad:]
                mask = create_causal_mask(q.shape[2]) if q.shape[2] < 16 else "causal"
                output = mx.fast.scaled_dot_product_attention(
                    q, k, v, scale=scale, mask=mask
                )
                if query_start:
                    output = mx.pad(output, [(0, 0), (0, 0), (query_start, 0), (0, 0)])
            outputs.append(output)
            start = end
        return outputs[0] if len(outputs) == 1 else mx.concatenate(outputs, axis=0)
