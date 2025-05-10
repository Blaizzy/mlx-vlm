import inspect
import math
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Dict, List, Optional, Tuple, Union

import mlx.core as mx
import mlx.nn as nn
import numpy as np

from ..base import LanguageModelOutput, create_attention_mask
from ..cache import KVCache
from .language import LanguageModel, TextConfig
from .su_rope import Phi3SuScaledRotaryEmbedding
from .vision import VisionConfig, VisionModel


@dataclass
class ModelConfig:
    text_config: TextConfig
    vision_config: VisionConfig
    model_type: str
    vocab_size: int

    num_hidden_layers: int
    intermediate_size: int
    num_attention_heads: int
    rms_norm_eps: float

    ignore_index: int = -100
    image_token_index: int = 257152
    hidden_size: int = 2048
    pad_token_id: int = 0

    num_key_value_heads: int = None
    rope_theta: float = 10000
    rope_traditional: bool = False
    rope_scaling: Optional[Dict[str, Union[float, str]]] = None
    max_position_embeddings: int = 131072
    original_max_position_embeddings: int = 4096
    eos_token_id: Optional[List[int]] = None

    @classmethod
    def from_dict(cls, params):
        return cls(
            **{
                k: v
                for k, v in params.items()
                if k in inspect.signature(cls).parameters
            }
        )


class Attention(nn.Module):
    def __init__(self, config: TextConfig):
        super().__init__()

        dim = config.hidden_size
        self.n_heads = n_heads = config.num_attention_heads
        self.n_kv_heads = n_kv_heads = config.num_key_value_heads
        self.num_hidden_layers = config.num_hidden_layers

        self.head_dim = head_dim = config.hidden_size // n_heads
        self.scale = head_dim**-0.5

        op_size = n_heads * head_dim + 2 * (n_kv_heads * head_dim)
        self.qkv_proj = nn.Linear(dim, op_size, bias=False)
        self.o_proj = nn.Linear(n_heads * head_dim, dim, bias=False)

        rope_scale = 1.0
        if config.rope_scaling and config.rope_scaling["type"] == "su":
            self.rope = Phi3SuScaledRotaryEmbedding(
                head_dim,
                traditional=False,
                base=config.rope_theta,
                scale=rope_scale,
                max_position_embeddings=config.max_position_embeddings,
                original_max_position_embeddings=config.original_max_position_embeddings,
                short_factor=config.rope_scaling["short_factor"],
                long_factor=config.rope_scaling["long_factor"],
            )
        else:
            if config.rope_scaling and config.rope_scaling["type"] == "linear":
                rope_scale = 1 / config.rope_scaling["factor"]
            self.rope = nn.RoPE(
                head_dim,
                traditional=config.rope_traditional,
                base=config.rope_theta,
                scale=rope_scale,
            )

    def __call__(
        self,
        x: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional[KVCache] = None,
    ) -> mx.array:
        B, L, D = x.shape

        qkv = self.qkv_proj(x)
        query_pos = self.n_heads * self.head_dim
        queries, keys, values = mx.split(
            qkv, [query_pos, query_pos + self.n_kv_heads * self.head_dim], axis=-1
        )

        queries = queries.reshape(B, L, self.n_heads, -1).transpose(0, 2, 1, 3)
        keys = keys.reshape(B, L, self.n_kv_heads, -1).transpose(0, 2, 1, 3)
        values = values.reshape(B, L, self.n_kv_heads, -1).transpose(0, 2, 1, 3)

        if cache is not None:
            queries = self.rope(queries, offset=cache.offset)
            keys = self.rope(keys, offset=cache.offset)
            keys, values = cache.update_and_fetch(keys, values)
        else:
            queries = self.rope(queries)
            keys = self.rope(keys)

        output = mx.fast.scaled_dot_product_attention(
            queries, keys, values, scale=self.scale, mask=mask
        )
        output = output.transpose(0, 2, 1, 3).reshape(B, L, -1)
        return self.o_proj(output)


class MLP(nn.Module):
    def __init__(self, dim, hidden_dim):
        super().__init__()
        self.gate_up_proj = nn.Linear(dim, 2 * hidden_dim, bias=False)
        self.down_proj = nn.Linear(hidden_dim, dim, bias=False)

    def __call__(self, x) -> mx.array:
        x = self.gate_up_proj(x)
        gate, x = mx.split(x, 2, axis=-1)
        return self.down_proj(nn.silu(gate) * x)


class TransformerBlock(nn.Module):
    def __init__(self, config: TextConfig):
        super().__init__()
        self.num_attention_heads = config.num_attention_heads
        self.hidden_size = config.hidden_size
        self.self_attn = Attention(config)
        self.mlp = MLP(config.hidden_size, config.intermediate_size)
        self.input_layernorm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = nn.RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )
        self.config = config

    def __call__(
        self,
        x: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional[KVCache] = None,
    ) -> mx.array:
        r = self.self_attn(self.input_layernorm(x), mask, cache)
        h = x + r
        r = self.mlp(self.post_attention_layernorm(h))
        out = h + r
        return out


class Phi3V(nn.Module):
    def __init__(self, config: TextConfig):
        super().__init__()
        self.config = config
        self.vocab_size = config.vocab_size
        self.num_hidden_layers = config.num_hidden_layers
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.vision_embed_tokens = VisionModel(config)
        self.layers = [
            TransformerBlock(config=config) for _ in range(config.num_hidden_layers)
        ]
        self.norm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def __call__(
        self,
        inputs: mx.array,
        pixel_values=None,
        image_sizes=None,
        mask: Optional[mx.array] = None,
        cache=None,
    ):
        h = self.embed_tokens(inputs)
        p = np.argwhere(inputs < 0).tolist()

        if pixel_values is not None:
            h = self.vision_embed_tokens(pixel_values, h, image_sizes, p)

        if cache is None:
            cache = [None] * len(self.layers)

        if mask is None:
            mask = create_attention_mask(h, cache)

        for layer, c in zip(self.layers, cache):
            h = layer(h, mask, c)

        return self.norm(h)


class Model(nn.Module):
    def __init__(self, config: TextConfig):
        super().__init__()
        self.model_type = config.model_type
        self.model = Phi3V(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.config = config

    def __call__(
        self,
        inputs: mx.array,
        pixel_values=None,
        mask=None,
        cache=None,
        image_sizes=None,
        **kwargs,
    ):
        out = self.model(inputs, pixel_values, image_sizes, mask=mask, cache=cache)
        logits = self.lm_head(out)
        if self.lm_head.weight.dtype in [mx.float16, mx.bfloat16, mx.float32]:
            logits = logits.astype(self.lm_head.weight.dtype)
        return LanguageModelOutput(logits=logits)

    @property
    def layers(self):
        return self.model.layers

    @property
    def head_dim(self):
        return self.config.hidden_size // self.config.num_attention_heads

    @property
    def n_kv_heads(self):
        return self.config.num_key_value_heads

    @property
    def language_model(self):
        return self

    @property
    def vision_model(self):
        return self.model.vision_embed_tokens
