import inspect
import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

import mlx.core as mx
import mlx.nn as nn

from ..base import KVCache, LanguageModelOutput


@dataclass
class TextConfig:
    d_model: int = 768
    n_heads: int = 12
    n_layers: int = 12
    mlp_ratio: int = 4
    max_sequence_length: int = 1024
    vocab_size: int = 50257
    embedding_size: Optional[int] = 50304
    additional_vocab_size: Optional[int] = None
    attention_dropout: float = 0.1
    residual_dropout: float = 0.1
    embedding_dropout: float = 0.1
    layer_norm_eps: float = 1e-5
    initializer_range: float = 0.02
    use_cache: bool = True
    pad_token_id: int = -1
    rope: bool = True
    rope_theta: float = 10000.0
    weight_tying: bool = True

    @classmethod
    def from_dict(cls, params):
        return cls(
            **{
                k: v
                for k, v in params.items()
                if k in inspect.signature(cls).parameters
            }
        )


class RotaryEmbedding(nn.Module):
    def __init__(self, config: TextConfig):
        super().__init__()
        self.config = config
        self.inv_freq = None

    def _compute_inv_freq(self, seq_len):
        dim = self.config.d_model // self.config.n_heads
        inv_freq = 1.0 / (self.config.rope_theta ** (mx.arange(0, dim, 2) / dim))
        return inv_freq

    def _get_rotary_embedding(self, seq_len):
        if self.inv_freq is None or self.inv_freq.shape[0] < seq_len:
            self.inv_freq = self._compute_inv_freq(seq_len)

        t = mx.arange(seq_len)
        freqs = mx.einsum("i,j->ij", t, self.inv_freq)
        emb = mx.concat([freqs, freqs], axis=-1)
        return emb.cos(), emb.sin()

    def rotate_half(self, x):
        x1, x2 = mx.split(x, 2, axis=-1)
        return mx.concat([-x2, x1], axis=-1)

    def apply_rotary_pos_emb(self, q, k, cos, sin):
        q = (q * cos) + (self.rotate_half(q) * sin)
        k = (k * cos) + (self.rotate_half(k) * sin)
        return q, k

    def __call__(self, q, k, seq_len):
        cos, sin = self._get_rotary_embedding(seq_len)
        return self.apply_rotary_pos_emb(q, k, cos, sin)


class MolmoAttention(nn.Module):
    def __init__(self, config: TextConfig):
        super().__init__()
        self.config = config
        self.n_heads = config.n_heads
        self.d_head = config.d_model // config.n_heads

        self.q_proj = nn.Linear(config.d_model, config.d_model, bias=False)
        self.k_proj = nn.Linear(config.d_model, config.d_model, bias=False)
        self.v_proj = nn.Linear(config.d_model, config.d_model, bias=False)
        self.o_proj = nn.Linear(config.d_model, config.d_model, bias=False)

        self.rotary_emb = RotaryEmbedding(config)
        self.attention_dropout = nn.Dropout(config.attention_dropout)

    def __call__(
        self,
        x: mx.array,
        attention_mask: Optional[mx.array] = None,
        cache: Optional[Tuple[mx.array, mx.array]] = None,
    ) -> mx.array:
        batch_size, seq_len, _ = x.shape

        q = (
            self.q_proj(x)
            .reshape(batch_size, seq_len, self.n_heads, self.d_head)
            .transpose(0, 2, 1, 3)
        )
        k = (
            self.k_proj(x)
            .reshape(batch_size, seq_len, self.n_heads, self.d_head)
            .transpose(0, 2, 1, 3)
        )
        v = (
            self.v_proj(x)
            .reshape(batch_size, seq_len, self.n_heads, self.d_head)
            .transpose(0, 2, 1, 3)
        )

        if cache is not None:
            q, k = self.rotary_emb(q, k, cache.offset)
            k, v = cache.update_and_fetch(k, v)
        else:
            if self.config.rope:
                q, k = self.rotary_emb(q, k, k.shape[2])

        attn = mx.matmul(q, k.transpose(0, 1, 3, 2)) / math.sqrt(self.d_head)

        if attention_mask is not None:
            attn = attn + attention_mask

        attn = mx.softmax(attn, axis=-1)
        attn = self.attention_dropout(attn)

        output = mx.matmul(attn, v)
        output = output.transpose(0, 2, 1, 3).reshape(batch_size, seq_len, -1)
        output = self.o_proj(output)
        return output


class MolmoMLP(nn.Module):
    def __init__(self, config: TextConfig):
        super().__init__()
        self.fc1 = nn.Linear(config.d_model, config.d_model * config.mlp_ratio)
        self.fc2 = nn.Linear(config.d_model * config.mlp_ratio, config.d_model)
        self.act = nn.SiLU()
        self.dropout = nn.Dropout(config.residual_dropout)

    def __call__(self, x: mx.array) -> mx.array:
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x


class MolmoBlock(nn.Module):
    def __init__(self, config: TextConfig, layer_id: int):
        super().__init__()
        self.config = config
        self.layer_id = layer_id

        self.ln1 = nn.LayerNorm(config.d_model, eps=config.layer_norm_eps)
        self.attn = MolmoAttention(config)
        self.ln2 = nn.LayerNorm(config.d_model, eps=config.layer_norm_eps)
        self.mlp = MolmoMLP(config)

    def __call__(
        self,
        x: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional[Tuple[mx.array, mx.array]] = None,
    ) -> mx.array:
        residual = x
        x = self.ln1(x)
        attn_output = self.attn(x, mask, cache)
        x = residual + attn_output
        residual = x
        x = self.ln2(x)
        x = self.mlp(x)
        x = residual + x
        return x


class Molmo(nn.Module):
    def __init__(self, config: TextConfig):
        super().__init__()
        self.config = config

        self.wte = nn.Embedding(config.embedding_size, config.d_model)
        if config.additional_vocab_size:
            self.new_wte = nn.Embedding(config.additional_vocab_size, config.d_model)
        self.drop = nn.Dropout(config.embedding_dropout)

        self.blocks = [MolmoBlock(config, i) for i in range(config.n_layers)]

        self.ln_f = nn.LayerNorm(config.d_model, eps=config.layer_norm_eps)

        if not config.weight_tying:
            self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)

    def __call__(
        self,
        input_ids: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional[List[Tuple[mx.array, mx.array]]] = None,
    ) -> LanguageModelOutput:
        if cache is None:
            cache = [None] * self.config.n_layers

        if self.config.additional_vocab_size:
            emb = mx.where(
                input_ids < self.config.embedding_size,
                self.wte(input_ids),
                self.new_wte(input_ids - self.config.embedding_size),
            )
        else:
            emb = self.wte(input_ids)

        hidden_states = self.drop(emb)

        for i, (block, c) in enumerate(zip(self.blocks, cache)):
            hidden_states = block(hidden_states, mask, c)

        hidden_states = self.ln_f(hidden_states)

        if self.config.weight_tying:
            logits = mx.matmul(hidden_states, self.wte.weight.T)
        else:
            logits = self.lm_head(hidden_states)

        return LanguageModelOutput(logits=logits)


class LanguageModel(nn.Module):
    def __init__(self, config: TextConfig):
        super().__init__()
        self.config = config
        self.model = Molmo(config)

    def __call__(
        self,
        input_ids: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional[List[Tuple[mx.array, mx.array]]] = None,
    ) -> LanguageModelOutput:
        outputs = self.model(input_ids, mask, cache)

        return outputs

    @property
    def layers(self):
        return self.model.layers

    @property
    def head_dim(self):
        return self.args.hidden_size // self.args.num_attention_heads

    @property
    def n_kv_heads(self):
        return self.args.num_key_value_heads
