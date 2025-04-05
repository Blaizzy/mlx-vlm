import inspect
import re
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple, Union

import mlx.core as mx
import mlx.nn as nn
from mlx_lm.models.cache import KVCache, RotatingKVCache

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


def bmm(a: mx.array, b: mx.array) -> mx.array:
    """
    Performs a batch matrix-matrix product using einsum.

    Args:
        a: Tensor of shape (b, n, m)
        b: Tensor of shape (b, m, p)

    Returns:
        Tensor of shape (b, n, p)
    """
    return mx.einsum("bij,bjk->bik", a, b)


class Llama4TextExperts(nn.Module):
    def __init__(self, config: TextConfig):
        super().__init__()
        self.num_experts = config.num_local_experts
        self.intermediate_size = config.intermediate_size
        self.hidden_size = config.hidden_size
        self.expert_dim = self.intermediate_size
        self.gate_up_proj = mx.ones(
            (self.num_experts, self.hidden_size, 2 * self.expert_dim)
        )
        self.down_proj = mx.ones((self.num_experts, self.expert_dim, self.hidden_size))
        self.act_fn = nn.SiLU()

    def __call__(self, hidden_states: mx.array) -> mx.array:

        hidden_states = hidden_states.reshape(self.num_experts, -1, self.hidden_size)
        gate_up = bmm(hidden_states, self.gate_up_proj)
        gate, up = mx.split(gate_up, 2, axis=-1)
        next_states = bmm((up * self.act_fn(gate)), self.down_proj)
        next_states = next_states.reshape(-1, self.hidden_size)
        return next_states


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


class Llama4TextL2Norm(nn.Module):
    def __init__(self, dim: int = None, eps: float = 1e-6):
        super().__init__()
        self.eps = eps

    def _norm(self, x):
        residual = mx.array(x**2)
        residual = mx.mean(residual, axis=-1, keepdims=True)
        return x * mx.rsqrt(residual + self.eps)

    def __call__(self, x):
        return self._norm(x.float()).astype(x.dtype)

    def extra_repr(self):
        return f"eps={self.eps}"


class Llama4TextRMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-5):
        """
        Llama4RMSNorm is equivalent to T5LayerNorm
        """
        super().__init__()
        self.eps = eps
        self.weight = mx.ones(hidden_size)

    def _norm(self, x):
        residual = mx.array(x**2)
        residual = mx.mean(residual, axis=-1, keepdims=True)
        return x * mx.rsqrt(residual + self.eps)

    def __call__(self, x):
        output = self._norm(x.float()).astype(x.dtype)
        return output * self.weight

    def extra_repr(self):
        return f"{tuple(self.weight.shape)}, eps={self.eps}"


class Llama4TextMoe(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.top_k = config.num_experts_per_tok
        self.hidden_dim = config.hidden_size
        self.num_experts = config.num_local_experts
        self.experts = Llama4TextExperts(config)
        self.router = nn.Linear(
            config.hidden_size, config.num_local_experts, bias=False
        )
        self.shared_expert = Llama4TextMLP(config)

    def __call__(self, hidden_states):
        batch, seq_len, hidden_dim = hidden_states.shape
        hidden_states = hidden_states.reshape(-1, self.hidden_dim)
        router_logits = self.router(hidden_states).transpose(0, 2, 1)
        tokens_per_expert = batch * seq_len

        router_top_value, router_indices = mx.argpartition(
            router_logits.transpose(0, 2, 1), self.top_k, axis=1
        )
        router_scores = scatter(
            mx.full(router_logits.transpose(0, 2, 1), float("-inf")),
            1,
            router_indices,
            router_top_value,
        ).transpose(0, 2, 1)
        # We do this to make sure we have -inf for non topK tokens before going through the !
        # Here we are just creating a tensor to index each and every single one of the hidden states. Let s maybe register a buffer for this!
        router_indices = (
            mx.arange(tokens_per_expert)
            .reshape(1, -1)
            .expand(router_scores.shape[0], -1)
        )
        router_scores = mx.sigmoid(router_scores.float()).to(hidden_states.dtype)

        router_indices = router_indices.reshape(-1, 1).expand(-1, hidden_dim)
        routed_in = mx.take_along_axis(hidden_states, router_indices, axis=0)
        # we gather inputs corresponding to each expert based on the router indices
        routed_in = routed_in * router_scores.reshape(-1, 1)
        routed_out = self.experts(routed_in)
        out = self.shared_expert(hidden_states)
        # now that we finished expert computation -> we scatter add because we gathered previously
        # we have to do this because we used all experts on all tokens. This is faster than the for loop, tho you are compute bound
        # this scales a lot better if you do EP!
        out = scatter_add(out, 0, router_indices, routed_out.reshape(-1, hidden_dim))
        return out, router_scores


class Llama4TextAttention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(self, config: TextConfig, layer_idx):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.head_dim = getattr(
            config, "head_dim", config.hidden_size // config.num_attention_heads
        )
        self.num_attention_heads = config.num_attention_heads
        self.num_key_value_groups = (
            config.num_attention_heads // config.num_key_value_heads
        )
        self.num_key_value_heads = config.num_key_value_heads
        self.scale = self.head_dim**-0.5
        self.attn_scale = config.attn_scale
        self.floor_scale = config.floor_scale
        self.attn_temperature_tuning = config.attn_temperature_tuning
        self.attention_dropout = config.attention_dropout
        self.is_causal = True
        self.use_rope = int((layer_idx + 1) % 4 != 0)  # rope unused for dense layers
        self.q_proj = nn.Linear(
            config.hidden_size,
            config.num_attention_heads * self.head_dim,
            bias=config.attention_bias,
        )
        self.k_proj = nn.Linear(
            config.hidden_size,
            config.num_key_value_heads * self.head_dim,
            bias=config.attention_bias,
        )
        self.v_proj = nn.Linear(
            config.hidden_size,
            config.num_key_value_heads * self.head_dim,
            bias=config.attention_bias,
        )
        self.o_proj = nn.Linear(
            config.num_attention_heads * self.head_dim,
            config.hidden_size,
            bias=config.attention_bias,
        )

        self.rope = initialize_rope(
            self.head_dim,
            self.config.rope_theta,
            self.config.rope_traditional,
            self.config.rope_scaling,
            self.config.max_position_embeddings,
        )
        if self.config.use_qk_norm and self.use_rope:
            self.qk_norm = Llama4TextL2Norm()

    def __call__(
        self,
        hidden_states: mx.array,
        attention_mask: Optional[mx.array],
        cache: Optional[Any] = None,
    ):
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        queries = self.q_proj(hidden_states).reshape(hidden_shape)
        keys = self.k_proj(hidden_states).reshape(hidden_shape)
        values = self.v_proj(hidden_states).reshape(hidden_shape).transpose(1, 2)

        if self.use_rope:  # the 16E model skips rope for long context on certain layers
            queries = self.rope(queries, offset=cache.offset)
            keys = self.rope(keys, offset=cache.offset)

        if hasattr(self, "qk_norm"):  # the 128E model does not use qk_norm
            queries = self.qk_norm(queries)
            keys = self.qk_norm(keys)

        # Use temperature tuning from https://arxiv.org/abs/2501.19399) to NoROPE layers
        if self.attn_temperature_tuning and not self.use_rope:
            attn_scales = (
                mx.log(mx.floor((float(cache.offset) + 1.0) / self.floor_scale) + 1.0)
                * self.attn_scale
                + 1.0
            )
            attn_scales = attn_scales.reshape((*input_shape, 1, 1))
            queries = (queries * attn_scales).to(queries.dtype)

        queries = queries.transpose(1, 2)
        keys = keys.transpose(1, 2)

        if cache is not None:
            keys, values = cache.update_and_fetch(keys, values)

        attn_output = mx.fast.scaled_dot_product_attention(
            queries, keys, values, mask=attention_mask, scale=self.scale
        )

        attn_output = attn_output.reshape(*input_shape, -1)
        attn_output = self.o_proj(attn_output)
        return attn_output


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

        self.input_layernorm = Llama4TextRMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )
        self.post_attention_layernorm = Llama4TextRMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )

        self.layer_idx = layer_idx

    def __call__(
        self,
        hidden_states: mx.array,
        attention_mask: Optional[mx.array] = None,
        chunk_causal_mask: Optional[mx.array] = None,
        cache: Optional[Any] = None,
    ):
        residual = hidden_states

        hidden_states = self.input_layernorm(hidden_states)

        # use local attention mask for ROPE layers
        if self.use_chunked_attention and chunk_causal_mask is not None:
            attention_mask = chunk_causal_mask

        # Self Attention
        attention_states = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            cache=cache,
        )
        hidden_states = residual + attention_states

        # Fully Connected
        residual = hidden_states

        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.feed_forward(hidden_states)
        if self.is_moe_layer:
            hidden_states, _ = hidden_states

        hidden_states = residual + hidden_states.reshape(residual.shape)

        return hidden_states


class Llama4TextModel(nn.Module):
    def __init__(self, config: TextConfig):
        super().__init__()
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = [
            Llama4TextDecoderLayer(config, layer_idx)
            for layer_idx in range(config.num_hidden_layers)
        ]

        self.norm = Llama4TextRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

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
            (mx.arange(seq_len).unsqueeze(0) // attention_chunk_size)
            - (mx.arange(seq_len).unsqueeze(1) // attention_chunk_size)
        )
        token_pos = mx.arange(seq_len).unsqueeze(0) - mx.arange(seq_len).unsqueeze(1)
        mask = (block_pos == 0) & (token_pos <= 0)
        return mask

    def __call__(
        self,
        input_ids: mx.array = None,
        attention_mask: Optional[mx.array] = None,
        cache: Optional[Any] = None,
        output_hidden_states: Optional[bool] = None,
    ):

        if input_ids is None:
            raise ValueError("You must specify input_ids")

        inputs_embeds = self.embed_tokens(
            input_ids.astype(self.embed_tokens.weight.dtype)
        )

        if mask is None:
            mask = create_attention_mask(h, cache)
        if cache is None:
            cache = [None] * len(self.layers)

        if chunk_size := self.config.attention_chunk_size:
            chunk_causal_mask = self.create_chunked_attention_mask(
                inputs_embeds.shape[1], chunk_size
            )

        hidden_states = inputs_embeds

        for decoder_layer in self.layers[: self.config.num_hidden_layers]:

            layer_outputs = decoder_layer(
                hidden_states,
                attention_mask=mask,
                chunk_causal_mask=chunk_causal_mask,
                cache=cache,
            )

            hidden_states = layer_outputs[0]

        hidden_states = self.norm(hidden_states)

        return hidden_states


class LanguageModel(nn.Module):

    def __init__(self, config: TextConfig):
        super().__init__()
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

    def forward(
        self,
        input_ids: mx.array = None,
        mask: Optional[mx.array] = None,
        cache: Optional[Any] = None,
    ):
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=mask,
            cache=cache,
        )

        hidden_states = outputs[0]

        logits = self.lm_head(hidden_states)

        return LanguageModelOutput(logits=logits)

    def make_cache(self):
        caches = []
        for i in range(self.args.num_hidden_layers):
            if int((i + 1) % 4 != 0):
                caches.append(KVCache())
            else:
                caches.append(RotatingKVCache())
        return caches

    @property
    def layers(self):
        return self.model.layers


def scatter(input_tensor, dim, index, src):
    """
    Writes values from src into input_tensor at the indices specified in index (functional version).

    Args:
        input_tensor: The tensor to be updated
        dim: The dimension along which to index
        index: The indices into input_tensor where values from src will be written
        src: The tensor containing values to write into input_tensor

    Returns:
        Updated tensor
    """
    result = input_tensor

    # Handle different dimensions
    if dim == 0:
        # For each index, update the corresponding row
        for i in range(index.shape[0]):
            for j in range(index.shape[1] if index.ndim > 1 else 1):
                idx = index[i, j] if index.ndim > 1 else index[i]
                src_val = src[i, j] if src.ndim > 1 else src[i]
                result = result.at[idx].set(src_val)
    elif dim == 1:
        # For each index, update the corresponding column
        for i in range(index.shape[0]):
            for j in range(index.shape[1] if index.ndim > 1 else 1):
                idx = index[i, j] if index.ndim > 1 else index[i]
                src_val = src[i, j] if src.ndim > 1 else src[i]
                result = result.at[i, idx].set(src_val)

    return result


def scatter_add(input_tensor, dim, index, src):
    """
    Adds values from src to input_tensor at the indices specified in index (functional version).

    Args:
        input_tensor: The tensor to be updated
        dim: The dimension along which to index
        index: The indices into input_tensor where values from src will be added
        src: The tensor containing values to add to input_tensor

    Returns:
        Updated tensor
    """
    result = input_tensor

    # Handle different dimensions
    if dim == 0:
        # For each index, add to the corresponding row
        for i in range(index.shape[0]):
            for j in range(index.shape[1] if index.ndim > 1 else 1):
                idx = index[i, j] if index.ndim > 1 else index[i]
                src_val = src[i, j] if src.ndim > 1 else src[i]
                result = result.at[idx].add(src_val)
    elif dim == 1:
        # For each index, add to the corresponding column
        for i in range(index.shape[0]):
            for j in range(index.shape[1] if index.ndim > 1 else 1):
                idx = index[i, j] if index.ndim > 1 else index[i]
                src_val = src[i, j] if src.ndim > 1 else src[i]
                result = result.at[i, idx].add(src_val)

    return result
