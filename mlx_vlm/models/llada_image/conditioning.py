# Copyright 2026 The HuggingFace Team. All rights reserved.
# Adapted from inclusionAI/LLaDA-Image for MLX.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from __future__ import annotations

import mlx.core as mx
from mlx import nn


class RMSNorm(nn.Module):
    """The image components use RMS normalization without learned weights."""

    def __init__(self, eps: float):
        super().__init__()
        self.eps = eps

    def __call__(self, x: mx.array) -> mx.array:
        return mx.fast.rms_norm(x, None, self.eps)


class MLP(nn.Module):
    def __init__(self, hidden_size: int, intermediate_size: int):
        super().__init__()
        self.fc1 = nn.Linear(hidden_size, intermediate_size)
        self.fc2 = nn.Linear(intermediate_size, hidden_size)

    def __call__(self, x: mx.array) -> mx.array:
        return self.fc2(nn.gelu_approx(self.fc1(x)))


def attention(q, k, v, heads, mask=None):
    batch, length, dim = q.shape
    head_dim = dim // heads
    q = q.reshape(batch, length, heads, head_dim).transpose(0, 2, 1, 3)
    k = k.reshape(batch, -1, heads, head_dim).transpose(0, 2, 1, 3)
    v = v.reshape(batch, -1, heads, head_dim).transpose(0, 2, 1, 3)
    x = mx.fast.scaled_dot_product_attention(q, k, v, scale=head_dim**-0.5, mask=mask)
    return x.transpose(0, 2, 1, 3).reshape(batch, length, dim)


class QueryAttention(nn.Module):
    def __init__(self, hidden_size: int, num_heads: int):
        super().__init__()
        self.heads = num_heads
        self.in_proj_weight = mx.zeros((3 * hidden_size, hidden_size))
        self.in_proj_bias = mx.zeros((3 * hidden_size,))
        self.out_proj = nn.Linear(hidden_size, hidden_size)

    def __call__(self, queries, context, mask):
        weights = mx.split(self.in_proj_weight, 3, axis=0)
        biases = mx.split(self.in_proj_bias, 3)
        q = mx.addmm(biases[0], queries, weights[0].T)
        k = mx.addmm(biases[1], context, weights[1].T)
        v = mx.addmm(biases[2], context, weights[2].T)
        return self.out_proj(attention(q, k, v, self.heads, mask[:, None, None, :]))


class QueryFormerBlock(nn.Module):
    def __init__(self, config: dict):
        super().__init__()
        dim = config["hidden_size"]
        eps = config.get("norm_eps", 1e-6)
        self.norm_q = nn.LayerNorm(dim, eps=eps, affine=False)
        self.norm_k = nn.LayerNorm(dim, eps=eps, affine=False)
        self.norm1 = nn.LayerNorm(dim, eps=eps, affine=False)
        self.cross_attn = QueryAttention(dim, config["num_attention_heads"])
        self.mlp = MLP(dim, config["intermediate_size"])

    def __call__(self, queries, context, mask):
        # Both residuals use normalized queries in the reference implementation.
        queries = self.norm_q(queries)
        queries = queries + self.cross_attn(queries, self.norm_k(context), mask)
        queries = self.norm1(queries)
        return queries + self.mlp(queries)


class QueryFormer(nn.Module):
    def __init__(self, config: dict):
        super().__init__()
        self.meta_queries = mx.zeros((config["num_queries"], config["hidden_size"]))
        self.query_blocks = [
            QueryFormerBlock(config) for _ in range(config["num_hidden_layers"])
        ]

    def __call__(self, embeddings: mx.array, mask: mx.array) -> mx.array:
        queries = mx.broadcast_to(
            self.meta_queries[None], (embeddings.shape[0], *self.meta_queries.shape)
        )
        for block in self.query_blocks:
            queries = block(queries, embeddings, mask)
        return queries


class ProjectionAttention(nn.Module):
    def __init__(self, config: dict):
        super().__init__()
        dim = config["hidden_size"]
        self.heads = config["num_attention_heads"]
        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.out_proj = nn.Linear(dim, dim)
        self.q_norm = RMSNorm(config.get("norm_eps", 1e-6))
        self.k_norm = RMSNorm(config.get("norm_eps", 1e-6))

    def __call__(self, x: mx.array) -> mx.array:
        shape = (*x.shape[:2], self.heads, -1)
        q = self.q_norm(self.q_proj(x).reshape(shape)).reshape(x.shape)
        k = self.k_norm(self.k_proj(x).reshape(shape)).reshape(x.shape)
        return self.out_proj(attention(q, k, self.v_proj(x), self.heads))


class ProjectionBlock(nn.Module):
    def __init__(self, config: dict):
        super().__init__()
        self.self_attn = ProjectionAttention(config)
        self.layer_norm1 = RMSNorm(config.get("norm_eps", 1e-6))
        self.layer_norm2 = RMSNorm(config.get("norm_eps", 1e-6))
        self.mlp = MLP(config["hidden_size"], config["intermediate_size"])

    def __call__(self, x: mx.array) -> mx.array:
        x = x + self.self_attn(self.layer_norm1(x))
        return x + self.mlp(self.layer_norm2(x))


class TextProjection(nn.Module):
    def __init__(self, config: dict):
        super().__init__()
        self.layers = [
            ProjectionBlock(config) for _ in range(config["num_hidden_layers"])
        ]
        self.projector = nn.Linear(config["hidden_size"], config["projection_dim"])

    def __call__(self, x: mx.array) -> mx.array:
        for layer in self.layers:
            x = layer(x)
        return self.projector(x)


def format_prompt(prompt: str | None) -> str:
    instruction = "Generate an image."
    if prompt is not None:
        instruction = f"Generate an image: {prompt.strip()}"
    return f"<role>HUMAN</role> {instruction}\n<role>ASSISTANT</role>\n<IMAGE1>"


def query_attention_mask(text_length: int, query_length: int) -> mx.array:
    """Text attends text; appended image queries attend the entire sequence."""
    positions = mx.arange(text_length + query_length)
    allowed = (positions[:, None] >= text_length) | (positions[None, :] < text_length)
    return allowed[None, None]
