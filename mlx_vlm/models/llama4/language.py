import inspect
from dataclasses import dataclass
from typing import Any, Optional

import mlx.core as mx
import mlx.nn as nn
from mlx_lm.models.rope_utils import initialize_rope
from mlx_lm.models.switch_layers import SwitchGLU

from ..base import (
    LanguageModelOutput,
    create_attention_mask,
    scaled_dot_product_attention,
)
from ..cache import ChunkedKVCache, KVCache


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


class Attention(nn.Module):
    def __init__(self, config: TextConfig, layer_idx: int):
        super().__init__()

        dim = config.hidden_size
        self.n_heads = n_heads = config.num_attention_heads
        self.n_kv_heads = n_kv_heads = config.num_key_value_heads

        self.use_rope = int((layer_idx + 1) % 4 != 0)  # rope unused for dense layers
        self.attn_temperature_tuning = config.attn_temperature_tuning
        self.floor_scale = config.floor_scale
        self.attn_scale = config.attn_scale

        self.head_dim = head_dim = config.head_dim or config.hidden_size // n_heads

        self.scale = head_dim**-0.5
        if hasattr(config, "attention_bias"):
            attention_bias = config.attention_bias
        else:
            attention_bias = False

        self.q_proj = nn.Linear(dim, n_heads * head_dim, bias=attention_bias)
        self.k_proj = nn.Linear(dim, n_kv_heads * head_dim, bias=attention_bias)
        self.v_proj = nn.Linear(dim, n_kv_heads * head_dim, bias=attention_bias)
        self.o_proj = nn.Linear(n_heads * head_dim, dim, bias=attention_bias)

        self.use_qk_norm = config.use_qk_norm and self.use_rope

        if self.use_rope:
            self.rope = initialize_rope(
                head_dim,
                config.rope_theta,
                traditional=True,
                scaling_config=config.rope_scaling,
                max_position_embeddings=config.max_position_embeddings,
            )

    def __call__(
        self,
        x: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional[Any] = None,
    ) -> mx.array:
        B, L, D = x.shape

        queries, keys, values = self.q_proj(x), self.k_proj(x), self.v_proj(x)

        queries = queries.reshape(B, L, self.n_heads, -1).transpose(0, 2, 1, 3)
        keys = keys.reshape(B, L, self.n_kv_heads, -1).transpose(0, 2, 1, 3)
        values = values.reshape(B, L, self.n_kv_heads, -1).transpose(0, 2, 1, 3)

        if cache is not None:
            offset = cache.offset
        else:
            offset = 0

        if self.use_rope:
            queries = self.rope(queries, offset=offset)
            keys = self.rope(keys, offset=offset)

        if self.use_qk_norm:
            queries = mx.fast.rms_norm(queries, weight=None, eps=1e-6)
            keys = mx.fast.rms_norm(keys, weight=None, eps=1e-6)

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

        if self.use_rope and isinstance(mask, mx.array):
            key_len = keys.shape[-2]
            if mask.shape[-1] != key_len:
                mask = mask[..., -key_len:]

        output = scaled_dot_product_attention(
            queries, keys, values, cache, scale=self.scale, mask=mask
        )
        output = output.transpose(0, 2, 1, 3).reshape(B, L, -1)
        return self.o_proj(output)


class MLP(nn.Module):
    def __init__(self, config: TextConfig, intermediate_size: int = None):
        super().__init__()

        dim = config.hidden_size
        hidden_dim = intermediate_size or config.intermediate_size

        self.gate_proj = nn.Linear(dim, hidden_dim, bias=False)
        self.down_proj = nn.Linear(hidden_dim, dim, bias=False)
        self.up_proj = nn.Linear(dim, hidden_dim, bias=False)

    def __call__(self, x) -> mx.array:
        return self.down_proj(nn.silu(self.gate_proj(x)) * self.up_proj(x))


class MoE(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.top_k = config.num_experts_per_tok
        self.num_experts = config.num_local_experts
        self.experts = SwitchGLU(
            config.hidden_size, config.intermediate_size, self.num_experts
        )
        self.router = nn.Linear(
            config.hidden_size, config.num_local_experts, bias=False
        )
        self.shared_expert = MLP(config)

    def __call__(self, x) -> mx.array:
        logits = self.router(x)
        k = self.top_k
        indices = mx.argpartition(-logits, kth=k - 1, axis=-1)[..., :k]
        scores = mx.take_along_axis(logits, indices, axis=-1)
        scores = mx.sigmoid(scores.astype(mx.float32)).astype(x.dtype)

        out = self.experts(x * scores, indices).squeeze(2)
        return out + self.shared_expert(x)


class TransformerBlock(nn.Module):
    def __init__(self, config: TextConfig, layer_idx: int):
        super().__init__()
        self.num_attention_heads = config.num_attention_heads
        self.hidden_size = config.hidden_size
        self.self_attn = Attention(config, layer_idx)
        self.use_chunked_attention = int((layer_idx + 1) % 4 != 0)
        self.is_moe_layer = (layer_idx % config.interleave_moe_layer_step) == (
            config.interleave_moe_layer_step - 1
        )
        if self.is_moe_layer:
            self.feed_forward = MoE(config)
        else:
            self.feed_forward = MLP(config, config.intermediate_size_mlp)

        self.input_layernorm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = nn.RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )
        self.config = config

        self.use_chunked_attention = int((layer_idx + 1) % 4 != 0)  # <=> use rope

    def __call__(
        self,
        x: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional[Any] = None,
    ) -> mx.array:

        r = self.self_attn(self.input_layernorm(x), mask, cache)
        h = x + r
        r = self.feed_forward(self.post_attention_layernorm(h))
        out = h + r
        return out


class LlamaModel(nn.Module):
    def __init__(self, config: TextConfig):
        super().__init__()
        self.config = config
        self.vocab_size = config.vocab_size
        self.num_hidden_layers = config.num_hidden_layers
        assert self.vocab_size > 0
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = [
            TransformerBlock(config, i) for i in range(config.num_hidden_layers)
        ]
        self.norm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def create_chunked_attention_mask(
        self, seq_len: int, attention_chunk_size: int, start: int = 0, offset: int = 0
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

        end = offset + seq_len
        linds = mx.arange(start, end)
        rinds = mx.arange(offset, end)[:, None]
        block_pos = mx.abs(
            (linds // attention_chunk_size) - (rinds // attention_chunk_size)
        )
        token_pos = linds <= rinds
        mask = (block_pos == 0) & (token_pos)
        return mask

    def __call__(
        self,
        input_ids: mx.array = None,
        input_embeds: mx.array = None,
        mask: mx.array = None,
        cache=None,
    ):
        if input_embeds is None:
            h = self.embed_tokens(input_ids)
        else:
            h = input_embeds

        if mask is None:
            mask = create_attention_mask(h, cache)

        if cache is not None:
            for idx, c in enumerate(cache):
                if (idx + 1) % 4 != 0:
                    c.maybe_trim_front()
            start = cache[0].start_position
            offset = cache[0].offset
        else:
            start = 0
            offset = 0

        # Create a mask for the chunked attention
        chunk_mask = self.create_chunked_attention_mask(
            h.shape[1], self.config.attention_chunk_size, start, offset
        )

        if cache is None:
            cache = [None] * len(self.layers)

        for idx, (layer, c) in enumerate(zip(self.layers, cache)):
            use_chunked_attention = (idx + 1) % 4 != 0
            if use_chunked_attention:
                local_mask = chunk_mask
            else:
                local_mask = mask
            h = layer(h, local_mask, cache=c)

        return self.norm(h)


class LanguageModel(nn.Module):
    def __init__(self, config: TextConfig):
        super().__init__()
        self.config = config
        self.model_type = config.model_type
        self.model = LlamaModel(self.config)
        self.lm_head = nn.Linear(
            self.config.hidden_size, self.config.vocab_size, bias=False
        )

    def __call__(
        self,
        input_ids: mx.array = None,
        input_embeds: mx.array = None,
        mask: mx.array = None,
        cache=None,
    ):
        out = self.model(
            input_ids=input_ids,
            input_embeds=input_embeds,
            mask=mask,
            cache=cache,
        )
        out = self.lm_head(out)
        return LanguageModelOutput(logits=out)

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

    @property
    def layers(self):
        return self.model.layers

    @property
    def n_kv_heads(self):
        return self.config.num_key_value_heads

    @property
    def head_dim(self):
        return (
            self.config.head_dim
            if self.config.head_dim
            else self.config.hidden_size // self.config.num_attention_heads
        )

    def make_cache(self):
        caches = []
        for i in range(self.config.num_hidden_layers):
            if (i + 1) % 4 != 0:
                caches.append(ChunkedKVCache(self.config.attention_chunk_size))
            else:
                caches.append(KVCache())  # no chunking for dense layers
        return caches
