import inspect
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Union

import mlx.core as mx
import mlx.nn as nn

from ..base import KVCache, LanguageModelOutput, create_attention_mask


@dataclass
class TextConfig:
    model_type: str = "mllama"
    vocab_size: int = 32000
    hidden_size: int = 4096
    intermediate_size: int = 14336
    num_hidden_layers: int = 40
    num_attention_heads: int = 32
    num_key_value_heads: int = 8
    hidden_act: str = "silu"
    max_position_embeddings: int = 131072
    initializer_range: float = 0.02
    rms_norm_eps: float = 1e-6
    tie_word_embeddings: bool = False
    rope_theta: float = 10000.0
    rope_traditional: bool = False
    rope_scaling: Optional[Dict[str, Union[float, str]]] = None
    cross_attention_layers: List[int] = field(
        default_factory=lambda: [3, 8, 13, 18, 23, 28, 33, 38]
    )

    def __post_init__(self):
        if self.num_key_value_heads is None:
            self.num_key_value_heads = self.num_attention_heads

    @classmethod
    def from_dict(cls, params):
        return cls(
            **{
                k: v
                for k, v in params.items()
                if k in inspect.signature(cls).parameters
            }
        )


class MllamaTextCrossAttention(nn.Module):
    def __init__(self, config: TextConfig, layer_idx: Optional[int] = None):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.layer_idx = layer_idx
        self.scale = self.head_dim**-0.5
        self.q_proj = nn.Linear(
            self.hidden_size, self.num_heads * self.head_dim, bias=False
        )
        self.k_proj = nn.Linear(
            self.hidden_size, self.num_key_value_heads * self.head_dim, bias=False
        )
        self.v_proj = nn.Linear(
            self.hidden_size, self.num_key_value_heads * self.head_dim, bias=False
        )
        self.o_proj = nn.Linear(
            self.num_heads * self.head_dim, self.hidden_size, bias=False
        )

        self.q_norm = nn.RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.k_norm = nn.RMSNorm(self.head_dim, eps=config.rms_norm_eps)

    def __call__(
        self,
        hidden_states: mx.array,
        cross_attention_states: Optional[mx.array] = None,
        attention_mask: Optional[mx.array] = None,
        cache: Optional[KVCache] = None,
    ) -> mx.array:

        bsz, q_len, _ = hidden_states.shape
        query = (
            self.q_proj(hidden_states)
            .reshape(bsz, q_len, self.num_heads, self.head_dim)
            .transpose(0, 2, 1, 3)
        )
        query_states = self.q_norm(query)

        if cross_attention_states is not None:
            key_states = (
                self.k_proj(cross_attention_states)
                .reshape(bsz, -1, self.num_key_value_heads, self.head_dim)
                .transpose(0, 2, 1, 3)
            )
            value_states = (
                self.v_proj(cross_attention_states)
                .reshape(bsz, -1, self.num_key_value_heads, self.head_dim)
                .transpose(0, 2, 1, 3)
            )
            key_states = self.k_norm(key_states)
        elif cache is not None and cache.offset > 0:
            key_states, value_states = cache.fetch()
        else:
            key_states, value_states = mx.split(query, 2, axis=1)
            key_states = self.k_norm(key_states)

        attn_output = mx.fast.scaled_dot_product_attention(
            query_states,
            key_states,
            value_states,
            scale=self.scale,
            mask=attention_mask,  # add a dim for batch processing
        )
        attn_output = attn_output.transpose(0, 2, 1, 3).reshape(
            bsz, q_len, self.hidden_size
        )
        return self.o_proj(attn_output)


class MllamaTextSelfAttention(nn.Module):
    def __init__(self, config: TextConfig, layer_idx: int):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.scale = self.head_dim**-0.5
        self.layer_idx = layer_idx

        self.q_proj = nn.Linear(
            self.hidden_size, self.num_heads * self.head_dim, bias=False
        )
        self.k_proj = nn.Linear(
            self.hidden_size, self.num_key_value_heads * self.head_dim, bias=False
        )
        self.v_proj = nn.Linear(
            self.hidden_size, self.num_key_value_heads * self.head_dim, bias=False
        )
        self.o_proj = nn.Linear(
            self.num_heads * self.head_dim, self.hidden_size, bias=False
        )

        self.rope = nn.RoPE(
            self.head_dim,
            traditional=config.rope_traditional,
            base=config.rope_theta,
            scale=1,
        )

    def __call__(
        self,
        x: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional[KVCache] = None,
    ) -> mx.array:
        bsz, q_len, _ = x.shape
        query_states = (
            self.q_proj(x).reshape(bsz, q_len, self.num_heads, -1).transpose(0, 2, 1, 3)
        )
        key_states = (
            self.k_proj(x)
            .reshape(bsz, q_len, self.num_key_value_heads, -1)
            .transpose(0, 2, 1, 3)
        )
        value_states = (
            self.v_proj(x)
            .reshape(bsz, q_len, self.num_key_value_heads, -1)
            .transpose(0, 2, 1, 3)
        )

        if cache is not None:
            query_states = self.rope(query_states, offset=cache.offset)
            key_states = self.rope(key_states, offset=cache.offset)
            key_states, value_states = cache.update_and_fetch(key_states, value_states)
        else:
            query_states = self.rope(query_states)
            key_states = self.rope(key_states)

        attn_output = mx.fast.scaled_dot_product_attention(
            query_states, key_states, value_states, scale=self.scale, mask=mask
        )
        attn_output = attn_output.transpose(0, 2, 1, 3).reshape(
            bsz, q_len, self.hidden_size
        )
        return self.o_proj(attn_output)


class MllamaTextMLP(nn.Module):
    def __init__(self, config: TextConfig):
        super().__init__()
        self.gate_proj = nn.Linear(
            config.hidden_size, config.intermediate_size, bias=False
        )
        self.up_proj = nn.Linear(
            config.hidden_size, config.intermediate_size, bias=False
        )
        self.down_proj = nn.Linear(
            config.intermediate_size, config.hidden_size, bias=False
        )
        self.act_fn = lambda x: x * mx.sigmoid(x)

    def __call__(self, x):
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))


class MllamaSelfAttentionDecoderLayer(nn.Module):
    def __init__(self, config: TextConfig, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.self_attn = MllamaTextSelfAttention(config, layer_idx=layer_idx)
        self.mlp = MllamaTextMLP(config)
        self.input_layernorm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = nn.RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )

    def __call__(
        self,
        hidden_states: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional[KVCache] = None,
    ) -> mx.array:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.self_attn(
            x=hidden_states,
            mask=mask,
            cache=cache,
        )
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        return hidden_states


class MllamaCrossAttentionDecoderLayer(nn.Module):
    def __init__(self, config: TextConfig, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.cross_attn = MllamaTextCrossAttention(config, layer_idx=layer_idx)
        self.mlp = MllamaTextMLP(config)
        self.input_layernorm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = nn.RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )
        self.cross_attn_attn_gate = mx.zeros(1)
        self.cross_attn_mlp_gate = mx.zeros(1)

    def __call__(
        self,
        hidden_states: mx.array,
        cross_attention_states: mx.array,
        attention_mask: Optional[mx.array] = None,
        full_text_row_masked_out_mask: Optional[mx.array] = None,
        cache: Optional[KVCache] = None,
    ) -> mx.array:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.cross_attn(
            hidden_states=hidden_states,
            cross_attention_states=cross_attention_states,
            attention_mask=attention_mask,
            cache=cache,
        )
        hidden_states = residual + mx.tanh(self.cross_attn_attn_gate) * hidden_states

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        if full_text_row_masked_out_mask is not None:
            hidden_states = full_text_row_masked_out_mask[:, 0] * hidden_states
        hidden_states = residual + mx.tanh(self.cross_attn_mlp_gate) * hidden_states

        return hidden_states


class MllamaTextModel(nn.Module):
    def __init__(self, config: TextConfig):
        super().__init__()
        self.config = config
        self.vocab_size = config.vocab_size
        self.hidden_size = config.hidden_size

        self.embed_tokens = nn.Embedding(config.vocab_size + 8, config.hidden_size)
        self.layers = [
            (
                MllamaCrossAttentionDecoderLayer(config, layer_idx)
                if layer_idx in config.cross_attention_layers
                else MllamaSelfAttentionDecoderLayer(config, layer_idx)
            )
            for layer_idx in range(config.num_hidden_layers)
        ]
        self.norm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def __call__(
        self,
        input_ids: Optional[mx.array] = None,
        mask: Optional[mx.array] = None,
        position_ids: Optional[mx.array] = None,
        cross_attention_states: Optional[mx.array] = None,
        cross_attention_mask: Optional[mx.array] = None,
        full_text_row_masked_out_mask: Optional[mx.array] = None,
        inputs_embeds: Optional[mx.array] = None,
        cache: Optional[KVCache] = None,
    ) -> mx.array:
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError(
                "You cannot specify both input_ids and inputs_embeds at the same time"
            )
        elif input_ids is not None:
            batch_size, seq_length = input_ids.shape
            inputs_embeds = self.embed_tokens(input_ids)
        elif inputs_embeds is not None:
            batch_size, seq_length, _ = inputs_embeds.shape
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        if position_ids is None:
            position_ids = mx.expand_dims(mx.arange(seq_length), 0)
            position_ids = mx.repeat(position_ids, batch_size, axis=0)

        hidden_states = inputs_embeds

        if cache is None:
            cache = [None] * len(self.layers)

        mask = create_attention_mask(hidden_states)

        for idx, (decoder_layer, c) in enumerate(zip(self.layers, cache)):
            if idx in self.config.cross_attention_layers:
                layer_outputs = decoder_layer(
                    hidden_states,
                    cross_attention_states=cross_attention_states,
                    attention_mask=cross_attention_mask,
                    full_text_row_masked_out_mask=full_text_row_masked_out_mask,
                    cache=c,
                )
            else:
                layer_outputs = decoder_layer(
                    hidden_states,
                    mask=mask,
                    cache=c,
                )
            hidden_states = layer_outputs

        hidden_states = self.norm(hidden_states)

        return hidden_states


class LanguageModel(nn.Module):
    def __init__(self, config: TextConfig):
        super().__init__()
        self.config = config
        self.model = MllamaTextModel(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

    def __call__(
        self,
        input_ids: Optional[mx.array] = None,
        mask: Optional[mx.array] = None,
        cross_attention_states: Optional[mx.array] = None,
        cross_attention_mask: Optional[mx.array] = None,
        full_text_row_masked_out_mask: Optional[mx.array] = None,
        inputs_embeds: Optional[mx.array] = None,
        cache: Optional[KVCache] = None,
    ) -> Tuple[mx.array, Optional[mx.array]]:

        hidden_states = self.model(
            input_ids=input_ids,
            mask=mask,
            cross_attention_states=cross_attention_states,
            cross_attention_mask=cross_attention_mask,
            full_text_row_masked_out_mask=full_text_row_masked_out_mask,
            inputs_embeds=inputs_embeds,
            cache=cache,
        )

        logits = self.lm_head(hidden_states)

        return LanguageModelOutput(
            logits=logits, cross_attention_states=cross_attention_states
        )

    @staticmethod
    def sanitize(weights):
        # Remove unused precomputed rotary freqs
        return {
            k: v for k, v in weights.items() if "self_attn.rotary_emb.inv_freq" not in k
        }

    @property
    def layers(self):
        return self.model.layers

    @property
    def head_dim(self):
        return self.config.hidden_size // self.config.num_attention_heads

    @property
    def n_kv_heads(self):
        return self.config.num_key_value_heads
