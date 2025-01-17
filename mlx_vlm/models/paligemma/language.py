import inspect
from dataclasses import dataclass
from typing import Optional, Tuple

import mlx.core as mx
import mlx.nn as nn

from ..base import KVCache, LanguageModelOutput, create_attention_mask


@dataclass
class TextConfig:
    model_type: str
    hidden_size: int
    num_hidden_layers: int
    intermediate_size: int
    num_attention_heads: int
    num_key_value_heads: int
    vocab_size: int
    head_dim: int = 256
    rms_norm_eps: float = 1e-6
    rope_theta: float = 10000
    rope_traditional: bool = False
    attn_logit_softcapping: Optional[float] = None
    final_logit_softcapping: Optional[float] = None
    query_pre_attn_scalar: Optional[float] = None

    @classmethod
    def from_dict(cls, params):
        return cls(
            **{
                k: v
                for k, v in params.items()
                if k in inspect.signature(cls).parameters
            }
        )


class RMSNorm(nn.Module):
    def __init__(self, dims: int, eps: float = 1e-6):
        super().__init__()
        self.weight = mx.ones((dims,))
        self.eps = eps

    def __call__(self, x):
        return mx.fast.rms_norm(x, 1.0 + self.weight, self.eps)


class Attention(nn.Module):
    def __init__(self, config: TextConfig):
        super().__init__()

        dim = config.hidden_size
        self.n_heads = n_heads = config.num_attention_heads
        self.n_kv_heads = n_kv_heads = config.num_key_value_heads
        self.model_type = config.model_type
        self.attn_logit_softcapping = config.attn_logit_softcapping
        self.repeats = n_heads // n_kv_heads
        self.head_dim = head_dim = (
            config.hidden_size // n_heads
            if self.model_type == "gemma"
            else config.head_dim
        )
        self.scale = (
            head_dim**-0.5
            if self.model_type == "gemma"
            else 1.0 / (config.query_pre_attn_scalar**0.5)
        )

        self.q_proj = nn.Linear(dim, n_heads * head_dim, bias=False)
        self.k_proj = nn.Linear(dim, n_kv_heads * head_dim, bias=False)
        self.v_proj = nn.Linear(dim, n_kv_heads * head_dim, bias=False)
        self.o_proj = nn.Linear(n_heads * head_dim, dim, bias=False)

        self.rope = nn.RoPE(
            head_dim,
            traditional=config.rope_traditional,
            base=config.rope_theta,
        )

    def __call__(
        self,
        x: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional[KVCache] = None,
    ) -> mx.array:
        B, L, D = x.shape

        queries, keys, values = self.q_proj(x), self.k_proj(x), self.v_proj(x)

        # Prepare the queries, keys and values for the attention computation
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

        if self.model_type == "gemma":
            output = mx.fast.scaled_dot_product_attention(
                queries, keys, values, scale=self.scale, mask=mask
            )
        else:
            queries = queries * self.scale

            if self.repeats > 1:
                queries = queries.reshape(
                    B, self.n_kv_heads, self.repeats, L, self.head_dim
                )
                keys = mx.expand_dims(keys, 2)
                values = mx.expand_dims(values, 2)

            scores = queries @ keys.swapaxes(-1, -2)
            scores = mx.tanh(scores / self.attn_logit_softcapping)
            scores *= self.attn_logit_softcapping

            if mask is not None:
                scores = scores + mask
            scores = mx.softmax(scores, precise=True, axis=-1)
            output = scores @ values
            if self.repeats > 1:
                output = output.reshape(B, self.n_heads, L, self.head_dim)

        output = output.transpose(0, 2, 1, 3).reshape(B, L, -1)
        return self.o_proj(output)


class MLP(nn.Module):
    def __init__(self, dim, hidden_dim, model_type):
        super().__init__()
        self.gate_proj = nn.Linear(dim, hidden_dim, bias=False)
        self.down_proj = nn.Linear(hidden_dim, dim, bias=False)
        self.up_proj = nn.Linear(dim, hidden_dim, bias=False)
        self.gelu = nn.GELU() if model_type == "gemma" else nn.GELU(approx="precise")

    def __call__(self, x) -> mx.array:
        return self.down_proj(self.gelu(self.gate_proj(x)) * self.up_proj(x))


class TransformerBlock(nn.Module):
    def __init__(self, config: TextConfig):
        super().__init__()
        self.model_type = config.model_type
        self.num_attention_heads = config.num_attention_heads
        self.hidden_size = config.hidden_size
        self.self_attn = Attention(config)
        self.mlp = MLP(config.hidden_size, config.intermediate_size, config.model_type)
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )
        self.config = config

        if config.model_type == "gemma2":
            self.pre_feedforward_layernorm = RMSNorm(
                config.hidden_size, eps=config.rms_norm_eps
            )
            self.post_feedforward_layernorm = RMSNorm(
                config.hidden_size, eps=config.rms_norm_eps
            )

    def __call__(
        self,
        x: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional[KVCache] = None,
    ) -> mx.array:
        # Self attention block
        r = self.self_attn(self.input_layernorm(x), mask, cache)

        if self.model_type == "gemma":
            # Gemma: Post-attention residual connection then MLP
            h = x + r
            r = self.mlp(self.post_attention_layernorm(h))
            out = h + r
        else:
            # Gemma2: Normalized residual connections with pre/post norms
            h = x + self.post_attention_layernorm(r)
            r = self.mlp(self.pre_feedforward_layernorm(h))
            out = h + self.post_feedforward_layernorm(r)
        return out


class GemmaModel(nn.Module):
    def __init__(self, config: TextConfig):
        super().__init__()
        self.config = config
        self.vocab_size = config.vocab_size
        self.num_hidden_layers = config.num_hidden_layers
        assert self.vocab_size > 0
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = [
            TransformerBlock(config=config) for _ in range(config.num_hidden_layers)
        ]
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def __call__(
        self,
        inputs: mx.array,
        cache=None,
        inputs_embeds=None,
        mask: Optional[mx.array] = None,
    ):
        # for passing merged input embeddings
        if inputs_embeds is None:
            h = self.embed_tokens(inputs)
        else:
            h = inputs_embeds

        h *= self.config.hidden_size**0.5
        if mask is None or cache[0].offset > 0:
            mask = create_attention_mask(h)

        if cache is None:
            cache = [None] * len(self.layers)

        for layer, c in zip(self.layers, cache):
            h = layer(h, mask, c)

        return self.norm(h)


class LanguageModel(nn.Module):
    def __init__(self, config: TextConfig):
        super().__init__()
        self.config = config
        self.final_logit_softcapping = config.final_logit_softcapping
        self.model_type = config.model_type
        self.model = GemmaModel(config)

        if self.model_type not in ["gemma", "gemma2"]:
            raise ValueError(
                f"Model type {self.model_type} not supported. Currently only 'gemma' is supported"
            )

    def __call__(
        self,
        inputs: mx.array,
        cache=None,
        inputs_embeds=None,
        mask: Optional[mx.array] = None,
    ):
        out = self.model(inputs, cache, inputs_embeds=inputs_embeds, mask=mask)
        out = self.model.embed_tokens.as_linear(out)

        if self.model_type == "gemma2":
            out = mx.tanh(out / self.final_logit_softcapping)
            out = out * self.final_logit_softcapping
        return LanguageModelOutput(logits=out)

    def sanitize(self, weights):
        return {
            k: v for k, v in weights.items() if "self_attn.rotary_emb.inv_freq" not in k
        }

    @property
    def layers(self):
        return self.model.layers

    @property
    def head_dim(self):
        return (
            self.config.hidden_size // self.config.num_attention_heads
            if self.model_type == "gemma"
            else self.config.head_dim
        )

    @property
    def n_kv_heads(self):
        return self.config.num_key_value_heads
