import inspect
import math
from typing import List, Optional, Tuple, Union

import mlx.core as mx
import mlx.nn as nn

from ..base import KVCache, LanguageModelOutput


class TextConfig:
    def __init__(self, **kwargs):
        self.d_model = kwargs.get("d_model", 768)
        self.encoder_attention_heads = kwargs.get("encoder_attention_heads", 8)
        self.decoder_attention_heads = kwargs.get("decoder_attention_heads", 8)
        self.encoder_ffn_dim = kwargs.get("encoder_ffn_dim", 3072)
        self.decoder_ffn_dim = kwargs.get("decoder_ffn_dim", 3072)
        self.dropout = kwargs.get("dropout", 0.1)
        self.attention_dropout = kwargs.get("attention_dropout", 0.0)
        self.activation_dropout = kwargs.get("activation_dropout", 0.0)
        self.activation_function = kwargs.get("activation_function", "gelu")
        self.init_std = kwargs.get("init_std", 0.02)
        self.encoder_layerdrop = kwargs.get("encoder_layerdrop", 0.0)
        self.decoder_layerdrop = kwargs.get("decoder_layerdrop", 0.0)
        self.scale_embedding = kwargs.get("scale_embedding", False)
        self.use_cache = kwargs.get("use_cache", True)
        self.max_position_embeddings = kwargs.get("max_position_embeddings", 1026)
        self.vocab_size = kwargs.get("vocab_size", 51289)
        self.pad_token_id = kwargs.get("pad_token_id", 1)
        self.bos_token_id = kwargs.get("bos_token_id", 0)
        self.eos_token_id = kwargs.get("eos_token_id", 2)
        self.encoder_layers = kwargs.get("encoder_layers", 6)
        self.decoder_layers = kwargs.get("decoder_layers", 6)

    @classmethod
    def from_dict(cls, params):
        return cls(
            **{
                k: v
                for k, v in params.items()
                if k in inspect.signature(cls).parameters
            }
        )


def gelu(x):
    return 0.5 * x * (1 + mx.tanh(mx.sqrt(2 / mx.pi) * (x + 0.044715 * mx.power(x, 3))))


class Florence2Attention(nn.Module):
    def __init__(self, config: TextConfig, is_decoder: bool = False):
        super().__init__()
        self.embed_dim = config.d_model
        self.num_heads = (
            config.decoder_attention_heads
            if is_decoder
            else config.encoder_attention_heads
        )
        self.head_dim = self.embed_dim // self.num_heads
        self.scaling = self.head_dim**-0.5

        self.k_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.v_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.q_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.out_proj = nn.Linear(self.embed_dim, self.embed_dim)

    def __call__(self, hidden_states, key_value_states=None, attention_mask=None):
        query_states = self.q_proj(hidden_states) * self.scaling
        if key_value_states is None:
            key_states = self.k_proj(hidden_states)
            value_states = self.v_proj(hidden_states)
        else:
            key_states = self.k_proj(key_value_states)
            value_states = self.v_proj(key_value_states)

        batch_size, tgt_len, _ = hidden_states.shape
        src_len = key_states.shape[1]

        q = query_states.reshape(
            batch_size, tgt_len, self.num_heads, self.head_dim
        ).transpose(0, 2, 1, 3)
        k = key_states.reshape(
            batch_size, src_len, self.num_heads, self.head_dim
        ).transpose(0, 2, 3, 1)
        v = value_states.reshape(
            batch_size, src_len, self.num_heads, self.head_dim
        ).transpose(0, 2, 1, 3)

        attn_weights = mx.matmul(q, k)

        if attention_mask is not None:
            attn_weights = mx.where(attention_mask, attn_weights, float("-inf"))

        attn_weights = mx.softmax(attn_weights, axis=-1)
        attn_output = mx.matmul(attn_weights, v)

        attn_output = attn_output.transpose(0, 2, 1, 3).reshape(
            batch_size, tgt_len, self.embed_dim
        )
        attn_output = self.out_proj(attn_output)

        return attn_output, attn_weights


class Florence2EncoderLayer(nn.Module):
    def __init__(self, config: TextConfig):
        super().__init__()
        self.embed_dim = config.d_model
        self.self_attn = Florence2Attention(config, is_decoder=False)
        self.self_attn_layer_norm = nn.LayerNorm(self.embed_dim)
        self.dropout = config.dropout
        self.activation_fn = gelu
        self.activation_dropout = config.activation_dropout
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
        self.self_attn = Florence2Attention(config, is_decoder=True)
        self.dropout = config.dropout
        self.activation_fn = gelu
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
    ):
        residual = hidden_states
        hidden_states, _ = self.self_attn(hidden_states, attention_mask=attention_mask)

        hidden_states = residual + hidden_states
        hidden_states = self.self_attn_layer_norm(hidden_states)

        residual = hidden_states
        hidden_states, _ = self.encoder_attn(
            hidden_states,
            key_value_states=encoder_hidden_states,
            attention_mask=encoder_attention_mask,
        )
        hidden_states = residual + hidden_states
        hidden_states = self.encoder_attn_layer_norm(hidden_states)

        residual = hidden_states
        hidden_states = self.activation_fn(self.fc1(hidden_states))
        hidden_states = self.fc2(hidden_states)
        hidden_states = residual + hidden_states
        hidden_states = self.final_layer_norm(hidden_states)

        return hidden_states


class Florence2Encoder(nn.Module):
    def __init__(self, config: TextConfig):
        super().__init__()
        self.config = config
        self.dropout = config.dropout
        self.layerdrop = config.encoder_layerdrop

        embed_dim = config.d_model
        self.embed_scale = math.sqrt(embed_dim) if config.scale_embedding else 1.0

        self.embed_positions = nn.Embedding(config.max_position_embeddings, embed_dim)
        self.layers = [
            Florence2EncoderLayer(config) for _ in range(config.encoder_layers)
        ]
        self.layernorm_embedding = nn.LayerNorm(embed_dim)

    def __call__(self, input_ids, attention_mask=None):
        inputs_embeds = self.embed_tokens(input_ids) * self.embed_scale
        embed_pos = self.embed_positions(mx.arange(input_ids.shape[1]))
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

        self.embed_positions = nn.Embedding(
            config.max_position_embeddings, config.d_model
        )
        self.layers = [
            Florence2DecoderLayer(config) for _ in range(config.decoder_layers)
        ]
        self.layernorm_embedding = nn.LayerNorm(config.d_model)

    def __call__(
        self,
        input_ids,
        encoder_hidden_states,
        attention_mask=None,
        encoder_attention_mask=None,
    ):
        input_shape = input_ids.shape
        inputs_embeds = self.embed_tokens(input_ids) * self.embed_scale

        # Embed positions
        positions = self.embed_positions(mx.arange(input_shape[1]))

        hidden_states = inputs_embeds + positions
        hidden_states = self.layernorm_embedding(hidden_states)

        for decoder_layer in self.layers:
            # add LayerDrop (see https://arxiv.org/abs/1909.11556 for description)
            dropout_probability = mx.random.uniform()
            if self.training and (dropout_probability < self.layerdrop):
                continue
            hidden_states = decoder_layer(
                hidden_states,
                encoder_hidden_states,
                attention_mask,
                encoder_attention_mask,
            )

        return hidden_states


class Florence2LanguageModel(nn.Module):
    def __init__(self, config: TextConfig):
        super().__init__()
        self.config = config
        self.shared = nn.Embedding(config.vocab_size, config.d_model)
        self.encoder = Florence2Encoder(config)
        self.decoder = Florence2Decoder(config)

    def __call__(
        self,
        input_ids,
        decoder_input_ids,
        attention_mask=None,
        decoder_attention_mask=None,
    ):
        self.encoder.embed_tokens = self.shared
        self.decoder.embed_tokens = self.shared

        encoder_outputs = self.encoder(input_ids, attention_mask)
        decoder_outputs = self.decoder(
            decoder_input_ids, encoder_outputs, decoder_attention_mask, attention_mask
        )
        return decoder_outputs


class LanguageModel(nn.Module):
    def __init__(self, config: TextConfig):
        super().__init__()
        self.config = config
        self.model = Florence2LanguageModel(config)

        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)

    def __call__(
        self,
        input_ids,
        decoder_input_ids,
        attention_mask=None,
        decoder_attention_mask=None,
    ):
        out = self.model(
            input_ids, decoder_input_ids, attention_mask, decoder_attention_mask
        )
        out = self.lm_head(out)
        return LanguageModelOutput(logits=out)

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
