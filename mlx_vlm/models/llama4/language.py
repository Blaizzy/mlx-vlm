import inspect
import re
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple, Union

import mlx.core as mx
import mlx.nn as nn
from mlx_lm.models.cache import KVCache, RotatingKVCache
from mlx_lm.models.rope_utils import initialize_rope
from mlx_lm.models.switch_layers import SwitchGLU

from ..base import LanguageModelOutput, create_attention_mask


@dataclass
class TextConfig:
    model_type: str
    hidden_size: int
    intermediate_size: int
    num_attention_heads: int
    rms_norm_eps: float
    vocab_size: int
    num_key_value_heads: int
    rope_theta: float = 500000.0
    num_hidden_layers: int = 48
    rope_traditional: bool = False
    rope_scaling: Optional[dict] = None  # Add missing rope_scaling attribute
    tie_word_embeddings: bool = False
    head_dim: int = 128
    hidden_act: str = "silu"
    intermediate_size_mlp: int = 16384
    max_position_embeddings: int = 10485760
    num_experts_per_tok: int = 1
    num_local_experts: int = 16
    attention_dropout: float = 0.0
    use_qk_norm: bool = True
    bos_token_id: int = 200000
    eos_token_id: list = None
    pad_token_id: int = 200018
    attention_chunk_size: int = 8192
    attention_bias: bool = False
    interleave_moe_layer_step: int = 1
    no_rope_layers: list = 4
    output_router_logits: bool = False
    router_aux_loss_coef: float = 0.001
    router_jitter_noise: float = 0.0
    attn_temperature_tuning: int = 4
    floor_scale: float = 8192
    attn_scale: float = 0.1
    moe_layers: list = None

    @classmethod
    def from_dict(cls, params):
        return cls(
            **{
                k: v
                for k, v in params.items()
                if k in inspect.signature(cls).parameters
            }
        )

    def __post_init__(self):
        if self.num_key_value_heads is None:
            self.num_key_value_heads = self.num_attention_heads

        if self.moe_layers is None:
            self.moe_layers = [
                i
                for i in range(self.num_hidden_layers)
                if i % self.interleave_moe_layer_step == 0
            ]


class Llama4TextMLP(nn.Module):
    def __init__(self, config, intermediate_size=None):
        super().__init__()

        if intermediate_size is None:
            intermediate_size = config.intermediate_size

        self.config = config
        self.gate_proj = nn.Linear(config.hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(config.hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, config.hidden_size, bias=False)
        self.activation_fn = nn.SiLU()

    def __call__(self, x):
        down_proj = self.activation_fn(self.gate_proj(x)) * self.up_proj(x)
        return self.down_proj(down_proj)


class Llama4TextMoe(nn.Module):
    def __init__(self, config: TextConfig):
        super().__init__()
        self.top_k = config.num_experts_per_tok
        self.num_experts = config.num_local_experts
        self.experts = SwitchGLU(
            config.hidden_size, config.intermediate_size, self.num_experts
        )
        self.router = nn.Linear(
            config.hidden_size, config.num_local_experts, bias=False
        )
        self.shared_expert = Llama4TextMLP(config)

    def __call__(self, x) -> mx.array:
        logits = self.router(x)
        k = self.top_k
        indices = mx.argpartition(-logits, kth=k - 1, axis=-1)[..., :k]
        scores = mx.take_along_axis(logits, indices, axis=-1)
        scores = mx.sigmoid(scores.astype(mx.float32)).astype(x.dtype)

        out = self.experts(x * scores, indices).squeeze(2)
        return out + self.shared_expert(x)


class Llama4TextAttention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(self, config: TextConfig, layer_idx):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.head_dim = head_dim = getattr(
            config, "head_dim", config.hidden_size // config.num_attention_heads
        )
        self.attention_bias = attention_bias = getattr(config, "attention_bias", False)
        dim = config.hidden_size
        self.n_heads = n_heads = config.num_attention_heads
        self.n_kv_heads = n_kv_heads = config.num_key_value_heads

        self.scale = head_dim**-0.5

        self.use_rope = int((layer_idx + 1) % 4 != 0)  # rope unused for dense layers
        self.attn_temperature_tuning = config.attn_temperature_tuning
        self.floor_scale = config.floor_scale
        self.attn_scale = config.attn_scale

        self.q_proj = nn.Linear(dim, n_heads * head_dim, bias=attention_bias)
        self.k_proj = nn.Linear(dim, n_kv_heads * head_dim, bias=attention_bias)
        self.v_proj = nn.Linear(dim, n_kv_heads * head_dim, bias=attention_bias)
        self.o_proj = nn.Linear(n_heads * head_dim, dim, bias=attention_bias)

        if self.use_rope:
            self.rope = initialize_rope(
                self.head_dim,
                self.config.rope_theta,
                self.config.rope_traditional,
                self.config.rope_scaling,
                self.config.max_position_embeddings,
            )
        self.use_qk_norm = self.config.use_qk_norm and self.use_rope

    def __call__(
        self,
        x: mx.array,
        mask: Optional[mx.array],
        cache: Optional[Any] = None,
    ):
        B, L, D = x.shape
        queries, keys, values = self.q_proj(x), self.k_proj(x), self.v_proj(x)

        queries = queries.reshape(B, L, self.n_heads, -1).transpose(0, 2, 1, 3)
        keys = keys.reshape(B, L, self.n_kv_heads, -1).transpose(0, 2, 1, 3)
        values = values.reshape(B, L, self.n_kv_heads, -1).transpose(0, 2, 1, 3)

        if cache is not None:
            offset = cache.offset
        else:
            offset = 0

        if self.use_rope:  # the 16E model skips rope for long context on certain layers
            queries = self.rope(queries, offset=offset)
            keys = self.rope(keys, offset=offset)

        if self.use_qk_norm:
            queries = mx.fast.rms_norm(
                queries, weight=None, eps=self.config.rms_norm_eps
            )
            keys = mx.fast.rms_norm(keys, weight=None, eps=self.config.rms_norm_eps)

        # Use temperature tuning from https://arxiv.org/abs/2501.19399) to NoROPE layers
        if self.attn_temperature_tuning and not self.use_rope:
            attn_scales = (
                mx.log(
                    mx.floor(mx.arange(offset + 1, offset + L + 1) / self.floor_scale)
                    + 1.0
                )
                * self.attn_scale
                + 1.0
            )
            attn_scales = attn_scales[:, None]
            queries = (queries * attn_scales).astype(queries.dtype)

        if cache is not None:
            keys, values = cache.update_and_fetch(keys, values)

        attn_output = mx.fast.scaled_dot_product_attention(
            queries, keys, values, mask=mask, scale=self.scale
        )

        attn_output = attn_output.transpose(0, 2, 1, 3).reshape(B, L, -1)
        return self.o_proj(attn_output)


class Llama4TextDecoderLayer(nn.Module):
    def __init__(self, config, layer_idx):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.self_attn = Llama4TextAttention(config, layer_idx)
        self.use_chunked_attention = int((layer_idx + 1) % 4 != 0)  # <=> use rope
        self.is_moe_layer = layer_idx in config.moe_layers
        if self.is_moe_layer:  # the 128E model interleaves dense / sparse
            self.feed_forward = Llama4TextMoe(config)
        else:
            self.feed_forward = Llama4TextMLP(
                config, intermediate_size=config.intermediate_size_mlp
            )

        self.input_layernorm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = nn.RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )

        self.layer_idx = layer_idx

    def __call__(
        self,
        x: mx.array,
        mask: Optional[mx.array] = None,
        chunk_causal_mask: Optional[mx.array] = None,
        cache: Optional[Any] = None,
    ):
        r = self.self_attn(self.input_layernorm(x), mask, cache)
        h = x + r
        r = self.feed_forward(self.post_attention_layernorm(h))
        out = h + r

        return out


class Llama4TextModel(nn.Module):
    def __init__(self, config: TextConfig):
        super().__init__()
        self.config = config
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size
        assert self.vocab_size > 0
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = [
            Llama4TextDecoderLayer(config, layer_idx)
            for layer_idx in range(config.num_hidden_layers)
        ]

        self.norm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def get_input_embeddings(self):
        return self.embed_tokens

    def set_input_embeddings(self, value):
        self.embed_tokens = value

    def create_chunked_attention_mask(
        self, seq_len: int, attention_chunk_size: int
    ) -> mx.array:
        """
        Generate the following:

        'What'      :  0 ■ ⬚ ⬚ ⬚ ⬚ ⬚    |
        '▁is'       :  1 ■ ■ ⬚ ⬚ ⬚ ⬚     |
        '▁ch'       :  2 ■ ■ ■ ⬚ ⬚ ⬚     |
        'unked'     :  3 ⬚ ⬚ ⬚ ■ ⬚ ⬚    |
        '▁attention':  4 ⬚ ⬚ ⬚ ■ ■ ⬚    |
        '?'         :  5 ⬚ ⬚ ⬚ ■ ■ ■     |

        If the chunk size is 3.
        This can just be appplied over the already created attention mask
        """
        block_pos = mx.abs(
            (mx.expand_dims(mx.arange(seq_len), 0) // attention_chunk_size)
            - (mx.expand_dims(mx.arange(seq_len), 1) // attention_chunk_size)
        )
        token_pos = mx.expand_dims(mx.arange(seq_len), 0) - mx.expand_dims(
            mx.arange(seq_len), 1
        )
        mask = (block_pos == 0) & (token_pos <= 0)
        return mask

    def __call__(
        self,
        input_ids: mx.array = None,
        input_embeds: mx.array = None,
        mask: Optional[mx.array] = None,
        cache: Optional[Any] = None,
        output_hidden_states: Optional[bool] = None,
    ):

        if input_ids is None:
            raise ValueError("You must specify input_ids")

        if input_embeds is None:
            h = self.embed_tokens(input_ids.astype(self.embed_tokens.weight.dtype))
        else:
            h = input_embeds

        if mask is None:
            mask = create_attention_mask(h, cache)

        if cache is None:
            cache = [None] * len(self.layers)

        chunk_causal_mask = None
        if chunk_size := self.config.attention_chunk_size:
            chunk_causal_mask = self.create_chunked_attention_mask(
                h.shape[1], chunk_size
            )

        for layer, c in zip(self.layers, cache):

            h = layer(
                h,
                mask=mask,
                chunk_causal_mask=chunk_causal_mask,
                cache=c,
            )

        return self.norm(h)


class LanguageModel(nn.Module):

    def __init__(self, config: TextConfig):
        super().__init__()
        self.config = config
        self.model = Llama4TextModel(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def set_input_embeddings(self, value):
        self.model.embed_tokens = value

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def set_decoder(self, decoder):
        self.model = decoder

    def get_decoder(self):
        return self.model

    def __call__(
        self,
        input_ids: mx.array = None,
        input_embeds: mx.array = None,
        mask: Optional[mx.array] = None,
        cache: Optional[Any] = None,
    ):
        outputs = self.model(
            input_ids=input_ids,
            input_embeds=input_embeds,
            mask=mask,
            cache=cache,
        )

        logits = self.lm_head(outputs)

        return LanguageModelOutput(logits=logits)

    # def make_cache(self):
    #     caches = []
    #     for i in range(self.config.num_hidden_layers):
    #         if int((i + 1) % 4 != 0):
    #             caches.append(KVCache())
    #         else:
    #             caches.append(RotatingKVCache())
    #     return caches

    @property
    def n_kv_heads(self):
        return self.config.num_key_value_heads

    @property
    def head_dim(self):
        return self.config.hidden_size // self.config.num_attention_heads

    @property
    def layers(self):
        return self.model.layers

    def sanitize(self, weights):

        # Rename expert weights for SwitchGLU
        for l in range(self.config.num_hidden_layers):
            prefix = f"language_model.model.layers.{l}.feed_forward.experts"
            if f"{prefix}.gate_up_proj" in weights:
                v = weights.pop(f"{prefix}.gate_up_proj")
                gate_k = f"{prefix}.gate_proj.weight"
                up_k = f"{prefix}.up_proj.weight"
                gate_proj, up_proj = mx.split(v, 2, axis=-1)
                weights[gate_k] = mx.swapaxes(gate_proj, 1, 2)
                weights[up_k] = mx.swapaxes(up_proj, 1, 2)
            if f"{prefix}.down_proj" in weights:
                down_proj = weights.pop(f"{prefix}.down_proj")
                weights[f"{prefix}.down_proj.weight"] = mx.swapaxes(down_proj, 1, 2)
        return weights
