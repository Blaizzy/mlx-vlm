from typing import Optional

import mlx.core as mx
import mlx.nn as nn
import numpy as np

from ..base import scaled_dot_product_attention
from ..pooling import EmbeddingOutput, normalize_embeddings
from .config import ModelConfig, TextConfig


class Attention(nn.Module):
    def __init__(self, dims: int, num_heads: int, bias: bool = True):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dims // num_heads
        self.scale = head_dim**-0.5
        self.q_proj = nn.Linear(dims, dims, bias=bias)
        self.k_proj = nn.Linear(dims, dims, bias=bias)
        self.v_proj = nn.Linear(dims, dims, bias=bias)
        self.out_proj = nn.Linear(dims, dims, bias=bias)

    def __call__(self, x, mask=None):
        B, L, _ = x.shape
        q = self.q_proj(x).reshape(B, L, self.num_heads, -1).transpose(0, 2, 1, 3)
        k = self.k_proj(x).reshape(B, L, self.num_heads, -1).transpose(0, 2, 1, 3)
        v = self.v_proj(x).reshape(B, L, self.num_heads, -1).transpose(0, 2, 1, 3)
        out = scaled_dot_product_attention(q, k, v, None, self.scale, mask)
        out = out.transpose(0, 2, 1, 3).reshape(B, L, -1)
        return self.out_proj(out)


class MLP(nn.Module):
    def __init__(self, config: TextConfig):
        super().__init__()
        self.activation_fn = nn.GELU(approx="precise")
        self.fc1 = nn.Linear(config.hidden_size, config.intermediate_size, bias=True)
        self.fc2 = nn.Linear(config.intermediate_size, config.hidden_size, bias=True)

    def __call__(self, x):
        return self.fc2(self.activation_fn(self.fc1(x)))


class EncoderLayer(nn.Module):
    def __init__(self, config: TextConfig):
        super().__init__()
        self.self_attn = Attention(config.hidden_size, config.num_attention_heads)
        self.layer_norm1 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.mlp = MLP(config)
        self.layer_norm2 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

    def __call__(self, x, mask=None):
        h = x + self.self_attn(self.layer_norm1(x), mask)
        return h + self.mlp(self.layer_norm2(h))


class Encoder(nn.Module):
    def __init__(self, config: TextConfig):
        super().__init__()
        self.layers = [EncoderLayer(config) for _ in range(config.num_hidden_layers)]

    def __call__(self, x, mask=None):
        for layer in self.layers:
            x = layer(x, mask)
        return x


class SiglipTextEmbeddings(nn.Module):
    def __init__(self, config: TextConfig):
        super().__init__()
        self.token_embedding = nn.Embedding(config.vocab_size, config.hidden_size)
        self.position_embedding = nn.Embedding(
            config.max_position_embeddings, config.hidden_size
        )

    def __call__(self, input_ids):
        seq_length = input_ids.shape[-1]
        position_ids = mx.array(np.arange(seq_length)[None, :])
        return self.token_embedding(input_ids) + self.position_embedding(position_ids)


class SiglipTextTransformer(nn.Module):
    def __init__(self, config: TextConfig):
        super().__init__()
        self.embeddings = SiglipTextEmbeddings(config)
        self.encoder = Encoder(config)
        self.final_layer_norm = nn.LayerNorm(
            config.hidden_size, eps=config.layer_norm_eps
        )
        self.head = nn.Linear(config.hidden_size, config.projection_size)

    def __call__(self, input_ids, mask=None):
        x = self.embeddings(input_ids)
        x = self.encoder(x, mask)
        x = self.final_layer_norm(x)
        return self.head(x[:, -1, :])


class SiglipTextModel(nn.Module):
    def __init__(self, config: TextConfig):
        super().__init__()
        self.text_model = SiglipTextTransformer(config)

    def __call__(self, input_ids, mask=None):
        return self.text_model(input_ids, mask)


class Model(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.model_type = config.model_type
        text_config = config.text_config
        if isinstance(text_config, dict):
            text_config = TextConfig.from_dict(text_config)
        self.text_model = SiglipTextModel(text_config)

    def __call__(
        self,
        input_ids: mx.array,
        attention_mask: Optional[mx.array] = None,
        **kwargs,
    ):
        pooled = self.text_model(input_ids)
        return EmbeddingOutput(text_embeds=normalize_embeddings(pooled))

    def sanitize(self, weights):
        out = {}
        for k, v in weights.items():
            if k.startswith("vision_model") or k.startswith("logit_"):
                continue
            if "position_ids" in k:
                continue
            if k.startswith("text_model") and not k.startswith("text_model.text_model"):
                k = "text_model." + k
            out[k] = v
        return out
