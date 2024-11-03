import inspect
import math
from dataclasses import dataclass
from typing import List, Optional, Tuple, Union

import mlx.core as mx
import mlx.nn as nn

from ..base import KVCache, LanguageModelOutput, create_attention_mask


@dataclass
class TextConfig:
    d_model: int = 768
    encoder_attention_heads: int = 8
    decoder_attention_heads: int = 8
    encoder_ffn_dim: int = 3072
    decoder_ffn_dim: int = 3072
    dropout: float = 0.1
    attention_dropout: float = 0.0
    activation_dropout: float = 0.0
    activation_function: str = "gelu"
    init_std: float = 0.02
    encoder_layerdrop: float = 0.0
    decoder_layerdrop: float = 0.0
    scale_embedding: bool = False
    use_cache: bool = True
    max_position_embeddings: int = 1024
    vocab_size: int = 51289
    pad_token_id: int = 1
    bos_token_id: int = 0
    eos_token_id: int = 2
    encoder_layers: int = 6
    decoder_layers: int = 6

    @classmethod
    def from_dict(cls, params):
        return cls(
            **{
                k: v
                for k, v in params.items()
                if k in inspect.signature(cls).parameters
            }
        )


class Florence2LearnedPositionalEmbedding(nn.Embedding):
    """
    This module learns positional embeddings up to a fixed maximum size.
    """

    def __init__(self, num_embeddings: int, embedding_dim: int):
        # Florence2 is set up so that if padding_idx is specified then offset the embedding ids by 2
        # and adjust num_embeddings appropriately. Other models don't have this hack
        self.offset = 2
        super().__init__(num_embeddings + self.offset, embedding_dim)

    def __call__(self, input_ids: mx.array, past_key_values_length: int = 0):
        """`input_ids' shape is expected to be [bsz x seqlen]."""

        bsz, seq_len = input_ids.shape[:2]
        positions = mx.arange(
            past_key_values_length,
            past_key_values_length + seq_len,
            dtype=mx.int64,
        )
        positions = mx.expand_dims(positions, axis=0)

        return super().__call__(positions + self.offset)


class Florence2Attention(nn.Module):
    def __init__(
        self, config: TextConfig, is_decoder: bool = False, is_causal: bool = False
    ):
        super().__init__()
        self.embed_dim = config.d_model
        self.num_heads = (
            config.decoder_attention_heads
            if is_decoder
            else config.encoder_attention_heads
        )
        self.is_decoder = is_decoder
        self.is_causal = is_causal
        self.head_dim = self.embed_dim // self.num_heads
        self.scaling = self.head_dim**-0.5

        self.k_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.v_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.q_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.out_proj = nn.Linear(self.embed_dim, self.embed_dim)

    def __call__(
        self,
        hidden_states,
        key_value_states=None,
        cache: Optional[Tuple[mx.array, mx.array]] = None,
        attention_mask=None,
        layer_head_mask=None,
    ):
        batch_size, tgt_len, _ = hidden_states.shape

        q = (
            self.q_proj(hidden_states)
            .reshape(batch_size, tgt_len, self.num_heads, self.head_dim)
            .transpose(0, 2, 1, 3)
        )

        is_cross_attention = key_value_states is not None

        batch_size, tgt_len, _ = hidden_states.shape
        src_len = (
            key_value_states.shape[1]
            if key_value_states is not None
            else hidden_states.shape[1]
        )

        if (
            is_cross_attention
            and cache is not None
            and cache[0].shape[2] == key_value_states.shape[1]
        ):
            k = cache[0]
            v = cache[1]

        elif is_cross_attention:
            # Cross attention
            k = (
                self.k_proj(key_value_states)
                .reshape(batch_size, src_len, self.num_heads, self.head_dim)
                .transpose(0, 2, 1, 3)
            )
            v = (
                self.v_proj(key_value_states)
                .reshape(batch_size, src_len, self.num_heads, self.head_dim)
                .transpose(0, 2, 1, 3)
            )
        elif cache is not None:
            # reuse k, v, self_attention
            k = (
                self.k_proj(hidden_states)
                .reshape(batch_size, src_len, self.num_heads, self.head_dim)
                .transpose(0, 2, 1, 3)
            )
            v = (
                self.v_proj(hidden_states)
                .reshape(batch_size, src_len, self.num_heads, self.head_dim)
                .transpose(0, 2, 1, 3)
            )
            k = mx.concatenate([cache[0], k], axis=2)
            v = mx.concatenate([cache[1], v], axis=2)
        else:
            # Self attention
            k = (
                self.k_proj(hidden_states)
                .reshape(batch_size, src_len, self.num_heads, self.head_dim)
                .transpose(0, 2, 1, 3)
            )
            v = (
                self.v_proj(hidden_states)
                .reshape(batch_size, src_len, self.num_heads, self.head_dim)
                .transpose(0, 2, 1, 3)
            )

        if self.is_decoder:
            cache = (k, v)

        if self.is_causal and self.is_decoder:
            causal_mask = create_attention_mask(hidden_states)
            attention_mask = causal_mask

        attn_output = (
            mx.fast.scaled_dot_product_attention(
                q, k, v, scale=self.scaling, mask=attention_mask
            )
            .transpose(0, 2, 1, 3)
            .reshape(batch_size, tgt_len, -1)
        )

        attn_output = self.out_proj(attn_output)

        return attn_output, cache


class Florence2EncoderLayer(nn.Module):
    def __init__(self, config: TextConfig):
        super().__init__()
        self.embed_dim = config.d_model
        self.self_attn = Florence2Attention(config, is_decoder=False, is_causal=False)
        self.self_attn_layer_norm = nn.LayerNorm(self.embed_dim)
        self.activation_fn = nn.GELU()
        self.fc1 = nn.Linear(self.embed_dim, config.encoder_ffn_dim)
        self.fc2 = nn.Linear(config.encoder_ffn_dim, self.embed_dim)
        self.final_layer_norm = nn.LayerNorm(self.embed_dim)

    def __call__(self, hidden_states, attention_mask=None):
        residual = hidden_states
        hidden_states, _ = self.self_attn(hidden_states, attention_mask=attention_mask)
        hidden_states = residual + hidden_states
        hidden_states = self.self_attn_layer_norm(hidden_states)

        residual = hidden_states
        hidden_states = self.activation_fn(self.fc1(hidden_states))
        hidden_states = self.fc2(hidden_states)
        hidden_states = residual + hidden_states
        hidden_states = self.final_layer_norm(hidden_states)

        return hidden_states


class Florence2DecoderLayer(nn.Module):
    def __init__(self, config: TextConfig):
        super().__init__()
        self.embed_dim = config.d_model
        self.self_attn = Florence2Attention(config, is_decoder=True, is_causal=True)
        self.dropout = config.dropout
        self.activation_fn = nn.GELU()
        self.activation_dropout = config.activation_dropout

        self.self_attn_layer_norm = nn.LayerNorm(self.embed_dim)
        self.encoder_attn = Florence2Attention(config, is_decoder=True)
        self.encoder_attn_layer_norm = nn.LayerNorm(self.embed_dim)
        self.fc1 = nn.Linear(self.embed_dim, config.decoder_ffn_dim)
        self.fc2 = nn.Linear(config.decoder_ffn_dim, self.embed_dim)
        self.final_layer_norm = nn.LayerNorm(self.embed_dim)

    def __call__(
        self,
        hidden_states,
        encoder_hidden_states,
        attention_mask=None,
        encoder_attention_mask=None,
        cache: Optional[Tuple[mx.array, mx.array, mx.array, mx.array]] = None,
    ):
        residual = hidden_states

        # decoder uni-directional self-attention cached key/values tuple is at positions 1,2
        self_attn_cache = cache[:2] if cache is not None else None

        hidden_states, present_cache = self.self_attn(
            hidden_states, attention_mask=attention_mask, cache=self_attn_cache
        )

        hidden_states = residual + hidden_states
        hidden_states = self.self_attn_layer_norm(hidden_states)

        cross_attn_present_cache = None

        if encoder_hidden_states is not None:
            residual = hidden_states

            mask = create_attention_mask(hidden_states)

            # cross_attn cached key/values tuple is at positions 3,4 of cache tuple
            cross_attn_cache = cache[-2:] if cache is not None else None

            hidden_states, cross_attn_present_cache = self.encoder_attn(
                hidden_states,
                key_value_states=encoder_hidden_states,
                attention_mask=mask,
                cache=cross_attn_cache,
            )
            hidden_states = residual + hidden_states
            hidden_states = self.encoder_attn_layer_norm(hidden_states)

            # add cross-attn to positions 3,4 of cache tuple
            present_cache += cross_attn_present_cache

        # Fully Connected
        residual = hidden_states
        hidden_states = self.activation_fn(self.fc1(hidden_states))
        hidden_states = self.fc2(hidden_states)
        hidden_states = residual + hidden_states
        hidden_states = self.final_layer_norm(hidden_states)

        return hidden_states, present_cache


class Florence2Encoder(nn.Module):
    def __init__(self, config: TextConfig):
        super().__init__()
        self.config = config
        self.dropout = config.dropout
        self.layerdrop = config.encoder_layerdrop

        embed_dim = config.d_model
        self.embed_scale = math.sqrt(embed_dim) if config.scale_embedding else 1.0

        self.embed_positions = Florence2LearnedPositionalEmbedding(
            config.max_position_embeddings, embed_dim
        )
        self.layers = [
            Florence2EncoderLayer(config) for _ in range(config.encoder_layers)
        ]
        self.layernorm_embedding = nn.LayerNorm(embed_dim)

    def __call__(self, input_ids=None, inputs_embeds=None, attention_mask=None):

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)
            input_shape = inputs_embeds.shape
        else:
            input_shape = inputs_embeds.shape

        positions = mx.arange(input_shape[1])

        if positions.ndim == 1:
            positions = mx.expand_dims(positions, axis=0)

        embed_pos = self.embed_positions(positions)

        hidden_states = inputs_embeds + embed_pos
        hidden_states = self.layernorm_embedding(hidden_states)

        for encoder_layer in self.layers:
            # Add LayerDrop (see https://arxiv.org/abs/1909.11556 for description)
            dropout_probability = mx.random.uniform()
            if self.training and (dropout_probability < self.layerdrop):
                continue
            hidden_states = encoder_layer(hidden_states, attention_mask)

        return hidden_states


class Florence2Decoder(nn.Module):
    def __init__(self, config: TextConfig):
        super().__init__()
        self.config = config
        self.dropout = config.dropout
        self.layerdrop = config.decoder_layerdrop
        self.padding_idx = config.pad_token_id
        self.max_target_positions = config.max_position_embeddings
        self.embed_scale = math.sqrt(config.d_model) if config.scale_embedding else 1.0

        self.embed_positions = Florence2LearnedPositionalEmbedding(
            config.max_position_embeddings, config.d_model
        )
        self.layers = [
            Florence2DecoderLayer(config) for _ in range(config.decoder_layers)
        ]
        self.layernorm_embedding = nn.LayerNorm(config.d_model)

    def __call__(
        self,
        input_ids=None,
        attention_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        head_mask=None,
        cross_attn_head_mask=None,
        inputs_embeds=None,
        cache=None,
    ):
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError(
                "You cannot specify both decoder_input_ids and decoder_inputs_embeds at the same time"
            )
        elif input_ids is not None:
            inputs_embeds = self.embed_tokens(input_ids)
            input_shape = inputs_embeds.shape  # for 2d masks
            positions = input_ids
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.shape[:-1]  # for 4d masks
            positions = inputs_embeds[:, :, -1]
        else:
            raise ValueError(
                "You have to specify either decoder_input_ids or decoder_inputs_embeds"
            )

        if positions.ndim == 1:
            positions = mx.expand_dims(positions, axis=0)

        cache_length = cache[0][0].shape[2] if cache[0] is not None else 0
        embed_pos = self.embed_positions(positions, cache_length)

        hidden_states = inputs_embeds + embed_pos
        hidden_states = self.layernorm_embedding(hidden_states)
        next_decoder_cache = []
        for e, decoder_layer in enumerate(self.layers):
            # add LayerDrop (see https://arxiv.org/abs/1909.11556 for description)
            dropout_probability = mx.random.uniform()
            if self.training and (dropout_probability < self.layerdrop):
                continue
            hidden_states, decoder_cache = decoder_layer(
                hidden_states=hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                attention_mask=attention_mask,
                encoder_attention_mask=encoder_attention_mask,
                cache=cache[e],
            )
            next_decoder_cache += (decoder_cache,)

        next_cache = next_decoder_cache
        return hidden_states, next_cache


class Florence2LanguageModel(nn.Module):
    def __init__(self, config: TextConfig):
        super().__init__()
        self.config = config
        self.shared = nn.Embedding(config.vocab_size, config.d_model)
        self.encoder = Florence2Encoder(config)
        self.decoder = Florence2Decoder(config)
        if config.scale_embedding:
            self.embed_scale = math.sqrt(config.d_model)
        else:
            self.embed_scale = 1.0

    def __call__(
        self,
        input_ids=None,
        inputs_embeds=None,
        decoder_input_ids=None,
        decoder_inputs_embeds=None,
        attention_mask=None,
        decoder_attention_mask=None,
        encoder_outputs=None,
        cache=None,
    ):
        self.encoder.embed_tokens = self.shared
        self.decoder.embed_tokens = self.shared

        if decoder_input_ids is None and decoder_inputs_embeds is None:
            if input_ids is None:
                raise ValueError(
                    "If no `decoder_input_ids` or `decoder_inputs_embeds` are "
                    "passed, `input_ids` cannot be `None`. Please pass either "
                    "`input_ids` or `decoder_input_ids` or `decoder_inputs_embeds`."
                )

            decoder_input_ids = mx.zeros_like(input_ids)
            decoder_input_ids[:, 1:] = input_ids[:, :-1]
            decoder_input_ids[:, 0] = self.config.bos_token_id

        if inputs_embeds is not None:
            inputs_embeds = inputs_embeds * self.embed_scale

        if cache is None:
            cache = [None] * len(self.decoder.layers)

        if encoder_outputs is None:
            encoder_outputs = self.encoder(
                input_ids=input_ids,
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
            )

        decoder_outputs, next_cache = self.decoder(
            input_ids=decoder_input_ids,
            attention_mask=decoder_attention_mask,
            encoder_hidden_states=encoder_outputs,
            encoder_attention_mask=attention_mask,
            inputs_embeds=decoder_inputs_embeds,
            cache=cache,
        )
        return decoder_outputs, encoder_outputs, next_cache


class LanguageModel(nn.Module):
    def __init__(self, config: TextConfig):
        super().__init__()
        self.config = config
        self.model = Florence2LanguageModel(config)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)

    def __call__(
        self,
        input_ids=None,
        inputs_embeds=None,
        decoder_input_ids=None,
        decoder_inputs_embeds=None,
        attention_mask=None,
        decoder_attention_mask=None,
        encoder_outputs=None,
        cache=None,
    ):
        decoder_outputs, encoder_outputs, next_cache = self.model(
            input_ids,
            inputs_embeds,
            decoder_input_ids,
            decoder_inputs_embeds,
            attention_mask,
            decoder_attention_mask,
            encoder_outputs,
            cache,
        )
        out = self.lm_head(decoder_outputs)
        return LanguageModelOutput(logits=out), encoder_outputs, next_cache

    @property
    def layers(self):
        return range(
            self.model.config.encoder_layers + self.model.config.decoder_layers
        )

    @property
    def head_dim(self):
        return self.config.d_model // self.config.decoder_attention_heads

    @property
    def n_kv_heads(self):
        return self.config.decoder_attention_heads
