import inspect
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

import mlx.core as mx
import mlx.nn as nn

from ..base import KVCache, LanguageModelOutput, create_attention_mask


@dataclass
class TextConfig:
    model_type: str = "molmo"
    max_position_embeddings: int = 4096
    d_model: int = 3584
    n_heads: int = 28
    n_kv_heads: int = 4
    n_layers: int = 28
    mlp_ratio: int = 4
    max_sequence_length: int = 1024
    act_output_multiplier: int = 0.5
    mlp_hidden_size: int = 37888
    vocab_size: int = 152064
    embedding_size: Optional[int] = 152064
    additional_vocab_size: Optional[int] = None
    attention_dropout: float = 0.1
    residual_dropout: float = 0.1
    embedding_dropout: float = 0.1
    layer_norm_eps: float = 1e-5
    initializer_range: float = 0.02
    pad_token_id: int = -1
    rope: bool = True
    rope_theta: float = 1000000.0
    weight_tying: bool = False
    rope_full_precision: bool = True
    rope_impl: str = "interleave"
    additional_vocab_size: Optional[int] = 128

    @classmethod
    def from_dict(cls, params):
        return cls(
            **{
                k: v
                for k, v in params.items()
                if k in inspect.signature(cls).parameters
            }
        )


class SwiGLU(nn.Module):
    def __call__(self, x: mx.array) -> mx.array:
        x, gate = mx.split(x, 2, axis=-1)
        return nn.silu(gate) * x


class MolmoBlock(nn.Module):
    def __init__(self, config: TextConfig):
        super().__init__()
        self.attn_out = nn.Linear(config.d_model, config.d_model, bias=False)
        self.ff_out = nn.Linear(
            int(config.act_output_multiplier * config.mlp_hidden_size),
            config.d_model,
            bias=False,
        )
        self.attn_norm = nn.RMSNorm(config.d_model, eps=config.layer_norm_eps)
        self.ff_norm = nn.RMSNorm(config.d_model, eps=config.layer_norm_eps)
        self.ff_proj = nn.Linear(config.d_model, config.mlp_hidden_size, bias=False)
        head_dim = config.d_model // config.n_heads
        self.rotary_emb = nn.RoPE(head_dim, base=config.rope_theta)
        self.scale = head_dim**-0.5
        self.n_heads = config.n_heads
        self.n_kv_heads = config.n_kv_heads
        self.fused_dims = (
            config.d_model,
            config.n_kv_heads * head_dim,
            config.n_kv_heads * head_dim,
        )
        self.att_proj = nn.Linear(config.d_model, sum(self.fused_dims), bias=True)
        self.act = SwiGLU()

    def __call__(self, x, mask=None, cache=None):
        batch_size, seq_len, D = x.shape
        attn_in = self.attn_norm(x)

        qkv = self.att_proj(attn_in)

        q, k, v = mx.split(
            qkv, [self.fused_dims[0], self.fused_dims[0] + self.fused_dims[1]], axis=-1
        )

        q = q.reshape(batch_size, seq_len, self.n_heads, D // self.n_heads).transpose(
            0, 2, 1, 3
        )
        k = k.reshape(
            batch_size, seq_len, self.n_kv_heads, D // self.n_heads
        ).transpose(0, 2, 1, 3)
        v = v.reshape(
            batch_size, seq_len, self.n_kv_heads, D // self.n_heads
        ).transpose(0, 2, 1, 3)

        if cache is not None:
            q = self.rotary_emb(q, offset=cache.offset)
            k = self.rotary_emb(k, offset=cache.offset)
            k, v = cache.update_and_fetch(k, v)
        else:
            q = self.rotary_emb(q)
            k = self.rotary_emb(k)

        # Perform attention
        att = mx.fast.scaled_dot_product_attention(q, k, v, scale=self.scale, mask=mask)
        att = att.transpose(0, 2, 1, 3).reshape(batch_size, seq_len, D)
        att = self.attn_out(att)

        # Add attention scores
        # shape: (batch_size, seq_len, d_model)
        x = x + att

        # Feed-forward layer
        og_x = x
        x = self.ff_norm(x)
        x = self.ff_proj(x)
        x = self.act(x)
        x = self.ff_out(x)
        x = og_x + x

        return x


class Embedding(nn.Module):
    def __init__(
        self,
        num_embeddings: int,
        num_new_embeddings: int,
        features: int,
        initializer_range: float = 0.02,
        new_embed_initializer_range: float = 0.02,
    ):
        super().__init__()
        self.initializer_range = initializer_range
        self.new_embed_initializer_range = new_embed_initializer_range

        # Initialize embeddings
        self.embedding = mx.random.normal(
            (num_embeddings, features), scale=self.initializer_range
        )
        self.new_embedding = mx.random.normal(
            (num_new_embeddings, features), scale=self.new_embed_initializer_range
        )

    def __call__(self, x: mx.array) -> mx.array:
        return mx.concat([self.embedding, self.new_embedding], axis=0)[x]


class Molmo(nn.Module):
    def __init__(self, config: TextConfig):
        super().__init__()
        self.config = config

        self.wte = Embedding(
            config.embedding_size, config.additional_vocab_size, config.d_model
        )

        self.blocks = [MolmoBlock(config) for _ in range(config.n_layers)]

        self.ln_f = nn.RMSNorm(config.d_model, eps=config.layer_norm_eps)

        if not config.weight_tying:
            self.ff_out = nn.Linear(config.d_model, config.vocab_size, bias=False)

    def __call__(
        self,
        input_ids: mx.array,
        inputs_embeds: Optional[mx.array] = None,
        mask: Optional[mx.array] = None,
        cache: Optional[KVCache] = None,
    ) -> LanguageModelOutput:

        if inputs_embeds is None:
            h = self.wte(input_ids)
        else:
            h = inputs_embeds

        if cache is None:
            cache = [None] * self.config.n_layers

        if mask is None:
            mask = create_attention_mask(h)

        for block, c in zip(self.blocks, cache):
            h = block(h, mask, c)

        h = self.ln_f(h)

        if self.config.weight_tying:
            logits = mx.matmul(h, self.wte.weight.T)
        else:
            logits = self.ff_out(h)

        return LanguageModelOutput(logits=logits)


class LanguageModel(nn.Module):
    def __init__(self, config: TextConfig):
        super().__init__()
        self.config = config
        self.model_type = config.model_type
        if self.model_type != "molmo":
            raise ValueError(
                f"Model type {self.model_type} not supported. Currently only 'molmo' is supported"
            )
        self.model = Molmo(config)

    def __call__(
        self,
        input_ids: mx.array,
        inputs_embeds: Optional[mx.array] = None,
        mask: Optional[mx.array] = None,
        cache: Optional[KVCache] = None,
    ) -> LanguageModelOutput:
        outputs = self.model(input_ids, inputs_embeds, mask, cache)
        return outputs

    @staticmethod
    def sanitize(weights):
        # Remove unused precomputed rotary freqs
        return {k: v for k, v in weights.items() if "rotary_emb.inv_freq" not in k}

    @property
    def layers(self):
        return self.model.blocks

    @property
    def head_dim(self):
        return self.config.d_model // self.config.n_heads

    @property
    def n_kv_heads(self):
        return self.config.n_kv_heads
