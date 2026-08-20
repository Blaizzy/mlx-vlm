import math
from typing import Optional, Tuple

import mlx.core as mx
import mlx.nn as nn

from ..base import (
    LanguageModelOutput,
    create_attention_mask,
    scaled_dot_product_attention,
)
from ..cache import SimpleKVCache
from .config import TextConfig


class NemotronParseAttention(nn.Module):
    def __init__(
        self, config: TextConfig, is_decoder: bool = False, is_causal: bool = False
    ):
        super().__init__()
        self.embed_dim = config.d_model
        self.num_heads = config.decoder_attention_heads
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
        cache: Optional[SimpleKVCache] = None,
        attention_mask=None,
    ):
        batch_size, tgt_len, _ = hidden_states.shape

        q = (
            self.q_proj(hidden_states)
            .reshape(batch_size, tgt_len, self.num_heads, self.head_dim)
            .transpose(0, 2, 1, 3)
        )

        is_cross_attention = key_value_states is not None

        src_len = (
            key_value_states.shape[1]
            if key_value_states is not None
            else hidden_states.shape[1]
        )

        if is_cross_attention and cache is not None and cache.keys is not None:
            k, v = cache.keys, cache.values

        elif is_cross_attention:
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
            if cache is not None:
                cache.update_and_fetch(k, v)

        elif cache is not None:
            k = (
                self.k_proj(hidden_states)
                .reshape(batch_size, src_len, self.num_heads, -1)
                .transpose(0, 2, 1, 3)
            )
            v = (
                self.v_proj(hidden_states)
                .reshape(batch_size, src_len, self.num_heads, -1)
                .transpose(0, 2, 1, 3)
            )
            k, v = cache.update_and_fetch(k, v)

        else:
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

        # The decoder is causal: replace any caller-provided mask with the
        # causal one (the HF reference only ever passes a causal mask).
        if self.is_causal and self.is_decoder:
            attention_mask = create_attention_mask(hidden_states, cache=cache)

        attn_output = (
            scaled_dot_product_attention(
                q, k, v, cache=cache, scale=self.scaling, mask=attention_mask
            )
            .transpose(0, 2, 1, 3)
            .reshape(batch_size, tgt_len, -1)
        )

        attn_output = self.out_proj(attn_output)
        return attn_output


class NemotronParseDecoderLayer(nn.Module):
    def __init__(self, config: TextConfig):
        super().__init__()
        self.embed_dim = config.d_model
        self.self_attn = NemotronParseAttention(config, is_decoder=True, is_causal=True)
        self.dropout = config.dropout
        self.activation_fn = nn.GELU()
        self.activation_dropout = config.activation_dropout

        self.self_attn_layer_norm = nn.LayerNorm(self.embed_dim)
        self.encoder_attn = NemotronParseAttention(config, is_decoder=True)
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
        cache: Optional[Tuple[SimpleKVCache, SimpleKVCache]] = None,
    ):
        # mBART decoder layers are pre-norm: normalize, then attention, then residual.
        residual = hidden_states
        hidden_states = self.self_attn_layer_norm(hidden_states)
        self_attn_cache = cache[0] if cache[0] is not None else None
        hidden_states = self.self_attn(
            hidden_states, attention_mask=attention_mask, cache=self_attn_cache
        )
        hidden_states = residual + hidden_states

        if encoder_hidden_states is not None:
            residual = hidden_states
            hidden_states = self.encoder_attn_layer_norm(hidden_states)
            cross_attn_cache = cache[-1] if cache[-1] is not None else None
            hidden_states = self.encoder_attn(
                hidden_states,
                key_value_states=encoder_hidden_states,
                attention_mask=encoder_attention_mask,
                cache=cross_attn_cache,
            )
            hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.final_layer_norm(hidden_states)
        hidden_states = self.activation_fn(self.fc1(hidden_states))
        hidden_states = self.fc2(hidden_states)
        hidden_states = residual + hidden_states

        return hidden_states


class NemotronParseDecoder(nn.Module):
    def __init__(self, config: TextConfig):
        super().__init__()
        self.config = config
        self.dropout = config.dropout
        self.layerdrop = config.decoder_layerdrop
        self.padding_idx = config.pad_token_id
        self.max_target_positions = config.max_position_embeddings
        self.embed_scale = math.sqrt(config.d_model) if config.scale_embedding else 1.0

        self.layers = [
            NemotronParseDecoderLayer(config) for _ in range(config.decoder_layers)
        ]
        self.layernorm_embedding = nn.LayerNorm(config.d_model)
        self.layer_norm = nn.LayerNorm(config.d_model)

    def __call__(
        self,
        input_ids=None,
        attention_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        inputs_embeds=None,
        cache=None,
    ):
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError(
                "You cannot specify both decoder_input_ids and decoder_inputs_embeds at the same time"
            )
        elif input_ids is not None:
            inputs_embeds = self.embed_tokens(input_ids) * self.embed_scale
        elif inputs_embeds is None:
            raise ValueError(
                "You have to specify either decoder_input_ids or decoder_inputs_embeds"
            )
        else:
            inputs_embeds = inputs_embeds * self.embed_scale

        hidden_states = self.layernorm_embedding(inputs_embeds)

        for decoder_layer, c in zip(self.layers, cache):
            hidden_states = decoder_layer(
                hidden_states=hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                attention_mask=attention_mask,
                encoder_attention_mask=encoder_attention_mask,
                cache=c,
            )

        hidden_states = self.layer_norm(hidden_states)
        return hidden_states


class NemotronParseLanguageModel(nn.Module):
    def __init__(self, config: TextConfig):
        super().__init__()
        self.config = config
        self.shared = nn.Embedding(config.vocab_size, config.d_model)
        self.decoder = NemotronParseDecoder(config)
        # The encoder runs once in the vision tower; the decoder only ever
        # sees single-token steps, so chunked prefill over encoder states
        # does not apply.
        self.no_chunked_prefill = True

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
        self.decoder.embed_tokens = self.shared

        # The vision encoder output is passed as `inputs_embeds` (mlx-vlm's
        # encoder-decoder convention, same as florence2); there is no separate
        # text encoder, so the states ARE the encoder output.
        if encoder_outputs is None:
            encoder_outputs = inputs_embeds

        if decoder_input_ids is None and decoder_inputs_embeds is None:
            raise ValueError(
                "You have to specify either decoder_input_ids or decoder_inputs_embeds"
            )

        if cache is None:
            # A comprehension, not list multiplication: each layer must own a
            # distinct (self-attn, cross-attn) cache pair.
            cache = [(SimpleKVCache(), SimpleKVCache()) for _ in self.decoder.layers]

        # Prefer the tokenized prompt as the decoder seed when the caller
        # provided one (the HF reference routes the prompt into
        # decoder_input_ids); otherwise fall back to the start-token
        # embedding. Single-token decoder steps from the generation loop
        # pass decoder_input_ids directly and never hit this branch.
        if (
            decoder_inputs_embeds is not None
            and input_ids is not None
            and input_ids.shape[-1] > 1
        ):
            decoder_input_ids = input_ids
            # The tokenizer wraps the prompt with BOS/EOS by default, but the
            # HF reference seeds the decoder with the raw prompt; a leading
            # BOS or trailing EOS in the seed flips the next-token prediction
            # (the model emits an immediate stop). Strip the automatic
            # wrapper, keeping any specials that are part of the prompt.
            if decoder_input_ids.shape[-1] > 2:
                if decoder_input_ids[0, -1].item() == self.config.eos_token_id:
                    decoder_input_ids = decoder_input_ids[:, :-1]
                if decoder_input_ids[0, 0].item() == self.config.bos_token_id:
                    decoder_input_ids = decoder_input_ids[:, 1:]
            decoder_inputs_embeds = None
        elif decoder_inputs_embeds is not None:
            decoder_input_ids = None

        decoder_outputs = self.decoder(
            input_ids=decoder_input_ids,
            attention_mask=decoder_attention_mask,
            encoder_hidden_states=encoder_outputs,
            encoder_attention_mask=None,
            inputs_embeds=decoder_inputs_embeds,
            cache=cache,
        )
        return decoder_outputs, encoder_outputs


class LanguageModel(nn.Module):
    def __init__(self, config: TextConfig):
        super().__init__()
        self.config = config
        self.model = NemotronParseLanguageModel(config)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        # The encoder runs once in the vision tower (see NemotronParseLanguageModel).
        self.no_chunked_prefill = True

    @staticmethod
    def _to_decoder_step_ids(inputs: mx.array) -> mx.array:
        """Normalize token tensors to one-step decoder ids with shape [batch, 1]."""
        if inputs.ndim == 0:
            return inputs[None, None]
        if inputs.ndim == 1:
            return inputs[:, None]
        if inputs.ndim == 2:
            return inputs[:, -1:] if inputs.shape[-1] != 1 else inputs
        return mx.reshape(inputs, (inputs.shape[0], -1))[:, -1:]

    def __call__(
        self,
        inputs=None,
        inputs_embeds=None,
        decoder_input_ids=None,
        decoder_inputs_embeds=None,
        attention_mask=None,
        decoder_attention_mask=None,
        encoder_outputs=None,
        cache=None,
        **kwargs,
    ):
        cross_attention_states = kwargs.get("cross_attention_states", None)
        if encoder_outputs is None and cross_attention_states is not None:
            encoder_outputs = cross_attention_states

        if (
            encoder_outputs is not None
            and decoder_input_ids is None
            and decoder_inputs_embeds is None
            and inputs is not None
        ):
            decoder_input_ids = self._to_decoder_step_ids(inputs)
            inputs = None

        decoder_outputs, encoder_outputs = self.model(
            input_ids=inputs,
            inputs_embeds=inputs_embeds,
            decoder_input_ids=decoder_input_ids,
            decoder_inputs_embeds=decoder_inputs_embeds,
            attention_mask=attention_mask,
            decoder_attention_mask=decoder_attention_mask,
            encoder_outputs=encoder_outputs,
            cache=cache,
        )
        out = self.lm_head(decoder_outputs)
        return LanguageModelOutput(
            logits=out,
            encoder_outputs=encoder_outputs,
            cross_attention_states=encoder_outputs,
        )

    @property
    def layers(self):
        return range(self.model.config.decoder_layers)

    @property
    def head_dim(self):
        return self.config.d_model // self.config.decoder_attention_heads

    @property
    def n_kv_heads(self):
        return self.config.decoder_attention_heads

    def make_cache(self):
        return [(SimpleKVCache(), SimpleKVCache()) for n in self.layers]
