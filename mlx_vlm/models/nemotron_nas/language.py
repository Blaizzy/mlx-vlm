from typing import Any, List, Optional

import mlx.core as mx
import mlx.nn as nn

from ..base import (
    LanguageModelOutput,
    create_attention_mask,
    scaled_dot_product_attention,
)
from ..cache import KVCache
from ..rope_utils import initialize_rope
from .config import AttentionConfig, FFNConfig, ModelConfig


def _find_multiple(n: int, k: int) -> int:
    if n % k == 0:
        return n
    return n + k - (n % k)


def _ffn_mult_to_intermediate_size(ffn_mult: float, n_embd: int) -> int:
    intermediate_size = int(2 * ffn_mult * n_embd / 3)
    return _find_multiple(intermediate_size, 256)


_ACT2FN = {
    "silu": nn.silu,
    "relu": nn.relu,
    "gelu": nn.gelu,
    "gelu_new": nn.gelu_approx,
    "gelu_fast": nn.gelu_approx,
}


class Attention(nn.Module):
    def __init__(self, args: ModelConfig, attention_config: AttentionConfig):
        super().__init__()

        dim = args.hidden_size
        self.n_heads = n_heads = args.num_attention_heads
        self.n_kv_heads = n_kv_heads = n_heads // attention_config.n_heads_in_group

        self.head_dim = head_dim = args.hidden_size // n_heads
        if (self.head_dim * n_heads) != dim:
            raise ValueError(
                f"hidden_size ({dim}) must be divisible by "
                f"num_attention_heads ({n_heads})"
            )

        self.scale = head_dim**-0.5

        self.q_proj = nn.Linear(dim, n_heads * head_dim, bias=args.attention_bias)
        self.k_proj = nn.Linear(dim, n_kv_heads * head_dim, bias=args.attention_bias)
        self.v_proj = nn.Linear(dim, n_kv_heads * head_dim, bias=args.attention_bias)
        self.o_proj = nn.Linear(n_heads * head_dim, dim, bias=args.attention_bias)

        self.rope = initialize_rope(
            self.head_dim,
            args.rope_theta,
            False,
            args.rope_scaling,
            args.max_position_embeddings,
        )

    def __call__(
        self,
        x: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional[Any] = None,
    ) -> mx.array:
        B, L, D = x.shape

        queries, keys, values = self.q_proj(x), self.k_proj(x), self.v_proj(x)

        queries = queries.reshape(B, L, self.n_heads, self.head_dim).transpose(
            0, 2, 1, 3
        )
        keys = keys.reshape(B, L, self.n_kv_heads, self.head_dim).transpose(0, 2, 1, 3)
        values = values.reshape(B, L, self.n_kv_heads, self.head_dim).transpose(
            0, 2, 1, 3
        )

        if cache is not None:
            queries = self.rope(queries, offset=cache.offset)
            keys = self.rope(keys, offset=cache.offset)
            keys, values = cache.update_and_fetch(keys, values)
        else:
            queries = self.rope(queries)
            keys = self.rope(keys)

        output = scaled_dot_product_attention(
            queries, keys, values, cache=cache, scale=self.scale, mask=mask
        )
        output = output.transpose(0, 2, 1, 3).reshape(B, L, -1)
        return self.o_proj(output)


class MLP(nn.Module):
    def __init__(self, args: ModelConfig, ffn_config: FFNConfig):
        super().__init__()

        dim = args.hidden_size
        hidden_dim = _ffn_mult_to_intermediate_size(ffn_config.ffn_mult, dim)

        self.gate_proj = nn.Linear(dim, hidden_dim, bias=args.mlp_bias)
        self.down_proj = nn.Linear(hidden_dim, dim, bias=args.mlp_bias)
        self.up_proj = nn.Linear(dim, hidden_dim, bias=args.mlp_bias)

        self.act_fn = args.hidden_act
        if self.act_fn not in _ACT2FN:
            raise ValueError(f"Unknown activation function: {args.hidden_act}")

    def __call__(self, x) -> mx.array:
        act_fn = _ACT2FN[self.act_fn]
        return self.down_proj(act_fn(self.gate_proj(x)) * self.up_proj(x))


class LinearSubblockReplacement(nn.Module):
    def __init__(self, hidden_size: int, bias: bool):
        super().__init__()
        self.linear = nn.Linear(hidden_size, hidden_size, bias=bias)

    def __call__(self, x: mx.array, *args, **kwargs) -> mx.array:
        return self.linear(x)


class TransformerBlock(nn.Module):
    def __init__(self, args: ModelConfig, layer_idx: int):
        super().__init__()
        self.hidden_size = args.hidden_size
        block_config = args.block_configs[layer_idx]
        self.attention_config = block_config.attention
        self.ffn_config = block_config.ffn

        if not self.attention_config.no_op:
            self.input_layernorm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)
        else:
            self.input_layernorm = None

        if self.attention_config.no_op:
            self.self_attn = None
        elif self.attention_config.replace_with_linear:
            self.self_attn = LinearSubblockReplacement(
                args.hidden_size, args.attention_bias
            )
        else:
            self.self_attn = Attention(args, self.attention_config)

        if not self.ffn_config.no_op:
            self.post_attention_layernorm = nn.RMSNorm(
                args.hidden_size, eps=args.rms_norm_eps
            )
        else:
            self.post_attention_layernorm = None

        if self.ffn_config.no_op:
            self.mlp = None
        elif self.ffn_config.replace_with_linear:
            self.mlp = LinearSubblockReplacement(args.hidden_size, args.mlp_bias)
        else:
            self.mlp = MLP(args, self.ffn_config)

    def __call__(
        self,
        x: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional[Any] = None,
    ) -> mx.array:
        if self.self_attn is not None:
            residual = x
            h = self.input_layernorm(x)
            attn_out = self.self_attn(h, mask=mask, cache=cache)
            x = residual + attn_out

        if self.mlp is not None:
            residual = x
            h = self.post_attention_layernorm(x)
            mlp_out = self.mlp(h)
            x = residual + mlp_out

        return x


class NemotronNASModel(nn.Module):
    def __init__(self, args: ModelConfig):
        super().__init__()
        self.args = args
        self.vocab_size = args.vocab_size
        self.num_hidden_layers = args.num_hidden_layers
        self.embed_tokens = nn.Embedding(args.vocab_size, args.hidden_size)
        self.layers = [
            TransformerBlock(args=args, layer_idx=i)
            for i in range(args.num_hidden_layers)
        ]
        self.norm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)
        self.num_attn_layers = sum(
            1 for layer in self.layers if layer.self_attn is not None
        )

    def __call__(self, inputs, cache=None, input_embeddings=None):
        h = (
            input_embeddings
            if input_embeddings is not None
            else self.embed_tokens(inputs)
        )

        if cache is None:
            cache = [None] * self.num_attn_layers

        mask = create_attention_mask(h, cache[0])

        cache_idx = 0
        for layer in self.layers:
            if layer.self_attn is not None:
                c = cache[cache_idx]
                cache_idx += 1
            else:
                c = None
            h = layer(h, mask, cache=c)

        return self.norm(h)


class LanguageModel(nn.Module):
    def __init__(self, args: ModelConfig):
        super().__init__()
        self.args = args
        self.config = args
        self.model_type = args.model_type
        self.model = NemotronNASModel(args)
        if not args.tie_word_embeddings:
            self.lm_head = nn.Linear(args.hidden_size, args.vocab_size, bias=False)
        else:
            self.lm_head = None

    def __call__(
        self,
        inputs: Optional[mx.array] = None,
        cache=None,
        input_embeddings: Optional[mx.array] = None,
        inputs_embeds: Optional[mx.array] = None,
        **kwargs,
    ) -> LanguageModelOutput:
        if inputs is None:
            inputs = kwargs.get("input_ids")
        if inputs_embeds is None:
            inputs_embeds = input_embeddings
        out = self.model(inputs, cache=cache, input_embeddings=inputs_embeds)
        if self.args.tie_word_embeddings:
            out = self.model.embed_tokens.as_linear(out)
        else:
            out = self.lm_head(out)
        return LanguageModelOutput(logits=out)

    def sanitize(self, weights):
        if self.args.tie_word_embeddings:
            weights.pop("lm_head.weight", None)
        return weights

    def make_cache(self):
        return [KVCache() for layer in self.layers if layer.self_attn is not None]

    @property
    def layers(self) -> List[nn.Module]:
        return self.model.layers

    @property
    def head_dim(self):
        return self.args.hidden_size // self.args.num_attention_heads

    @property
    def n_kv_heads(self):
        return self.args.num_attention_heads
