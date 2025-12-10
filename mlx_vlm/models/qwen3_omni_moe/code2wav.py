import math
from typing import Optional, Tuple

import mlx.core as mx
import mlx.nn as nn
import numpy as np
from mlx_lm.models.base import scaled_dot_product_attention

from mlx_vlm.models.qwen3_omni_moe.config import Code2WavConfig


class SnakeBeta(nn.Module):
    def __init__(self, in_features, alpha=1.0):
        super().__init__()
        self.in_features = in_features
        self.alpha = mx.zeros((in_features,)) * alpha
        self.beta = mx.zeros((in_features,)) * alpha
        self.no_div_by_zero = 0.000000001

    def __call__(self, hidden_states):
        alpha = mx.expand_dims(mx.expand_dims(self.alpha, axis=0), axis=-1)
        beta = mx.expand_dims(mx.expand_dims(self.beta, axis=0), axis=-1)
        alpha = mx.exp(alpha)
        beta = mx.exp(beta)
        hidden_states = hidden_states + (1.0 / (beta + self.no_div_by_zero)) * mx.power(
            mx.sin(hidden_states * alpha), 2
        )
        return hidden_states


class LayerScale(nn.Module):
    def __init__(self, config: Code2WavConfig):
        super().__init__()
        channels = config.hidden_size
        initial_scale = config.layer_scale_initial_scale
        self.scale = mx.full((channels,), initial_scale)

    def __call__(self, x: mx.array):
        return self.scale * x


class RoPE(nn.Module):
    def __init__(self, config: Code2WavConfig):
        super().__init__()
        self.config = config
        head_dim = config.hidden_size // config.num_attention_heads
        dim = head_dim
        inv_freq = 1.0 / (
            config.rope_theta ** (np.arange(0, dim, 2, dtype=np.float32) / dim)
        )
        self.inv_freq = inv_freq
        self.attention_scaling = 1.0

    def __call__(self, x: mx.array, position_ids: mx.array):
        batch_size = position_ids.shape[0]
        inv_freq_mx = mx.array(self.inv_freq)
        inv_freq_expanded = mx.broadcast_to(
            inv_freq_mx[None, :, None].astype(mx.float32),
            (batch_size, inv_freq_mx.shape[0], 1),
        )
        position_ids_expanded = mx.expand_dims(position_ids.astype(mx.float32), axis=1)
        freqs = inv_freq_expanded @ position_ids_expanded
        freqs = mx.swapaxes(freqs, 1, 2)
        emb = mx.concatenate([freqs, freqs], axis=-1)
        cos = mx.cos(emb) * self.attention_scaling
        sin = mx.sin(emb) * self.attention_scaling
        return cos.astype(x.dtype), sin.astype(x.dtype)


class CausalConvNet(nn.Module):
    def __init__(self, in_chn, out_chn, kernel_sz, dilation=1, stride=1, groups=1):
        super().__init__()
        self.conv = nn.Conv1d(
            in_chn, out_chn, kernel_sz, stride=stride, dilation=dilation, groups=groups
        )
        self.stride = stride
        self.kernel_size = (kernel_sz - 1) * dilation + 1
        self.dilation = dilation
        self.padding = self.kernel_size - self.stride

    def _get_extra_padding_for_conv1d(self, length: int) -> int:
        n_frames = (length - self.kernel_size + self.padding) / self.stride + 1
        ideal_length = (math.ceil(n_frames) - 1) * self.stride + (
            self.kernel_size - self.padding
        )
        return int(ideal_length - length)

    def __call__(self, hidden_state: mx.array) -> mx.array:
        length = hidden_state.shape[-1]
        extra_padding = self._get_extra_padding_for_conv1d(length)
        hidden_state = hidden_state.transpose(0, 2, 1)
        pad_width = [(0, 0), (self.padding, extra_padding), (0, 0)]
        hidden_state = mx.pad(
            hidden_state, pad_width, mode="constant", constant_values=0
        )
        output = self.conv(hidden_state)
        return output.transpose(0, 2, 1)


class CausalTransConvNet(nn.Module):
    def __init__(self, in_chn, out_chn, kernel_sz, stride=1):
        super().__init__()
        self.conv = nn.ConvTranspose1d(in_chn, out_chn, kernel_sz, stride=stride)
        pad = kernel_sz - stride
        self.left_pad = 0
        self.right_pad = pad

    def __call__(self, hidden_state: mx.array) -> mx.array:
        hidden_state = hidden_state.transpose(0, 2, 1)
        hidden_state = self.conv(hidden_state)
        length = hidden_state.shape[-2]
        hidden_state = hidden_state[:, self.left_pad : length - self.right_pad, :]
        return hidden_state.transpose(0, 2, 1)


class ConvNeXtBlock(nn.Module):
    def __init__(self, dim: int):
        super().__init__()

        self.dwconv = CausalConvNet(dim, dim, kernel_sz=7, groups=dim, dilation=1)
        self.norm = nn.LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, 4 * dim)
        self.pwconv2 = nn.Linear(4 * dim, dim)
        self.gamma = mx.full((dim,), 1e-6)

    def __call__(self, hidden_states: mx.array) -> mx.array:
        input = hidden_states
        hidden_states = self.dwconv(hidden_states)
        hidden_states = hidden_states.transpose(0, 2, 1)
        hidden_states = self.norm(hidden_states)
        hidden_states = self.pwconv1(hidden_states)
        hidden_states = nn.gelu(hidden_states)
        hidden_states = self.pwconv2(hidden_states)
        hidden_states = self.gamma * hidden_states
        hidden_states = hidden_states.transpose(0, 2, 1)
        hidden_states = input + hidden_states
        return hidden_states


class Code2WavDecoderResUnit(nn.Module):
    def __init__(self, dim: int, dilation: int = 1):
        super().__init__()

        self.act1 = SnakeBeta(dim)
        self.conv1 = CausalConvNet(dim, dim, kernel_sz=7, dilation=dilation)
        self.act2 = SnakeBeta(dim)
        self.conv2 = CausalConvNet(dim, dim, kernel_sz=1)

    def __call__(self, hidden_state: mx.array) -> mx.array:
        residual = hidden_state
        hidden_state = self.act1(hidden_state)
        hidden_state = self.conv1(hidden_state)
        hidden_state = self.act2(hidden_state)
        hidden_state = self.conv2(hidden_state)
        return hidden_state + residual


class Code2WavDecoderBlock(nn.Module):
    def __init__(self, config: Code2WavConfig, idx: int):
        super().__init__()

        in_dim = config.decoder_dim // 2**idx
        out_dim = config.decoder_dim // 2 ** (idx + 1)
        upsample_rate = config.upsample_rates[idx]

        self.block = [
            SnakeBeta(in_dim),
            CausalTransConvNet(in_dim, out_dim, 2 * upsample_rate, upsample_rate),
        ]
        self.block.extend(
            [Code2WavDecoderResUnit(out_dim, dilation) for dilation in (1, 3, 9)]
        )

    def __call__(self, hidden: mx.array) -> mx.array:
        for block in self.block:
            hidden = block(hidden)
        return hidden


def rotate_half(x):
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return mx.concatenate([-x2, x1], axis=-1)


def apply_rotary_pos_emb(q, k, cos, sin):
    cos = mx.expand_dims(cos, axis=1)
    sin = mx.expand_dims(sin, axis=1)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


class Code2WavAttention(nn.Module):
    def __init__(self, config: Code2WavConfig, idx: int):
        super().__init__()

        self.config = config
        self.layer_idx = idx
        self.head_dim = getattr(
            config, "head_dim", config.hidden_size // config.num_attention_heads
        )
        self.num_key_value_groups = (
            config.num_attention_heads // config.num_key_value_heads
        )
        self.scaling = self.head_dim**-0.5
        self.attention_dropout = config.attention_dropout
        self.is_causal = True

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
        self.q_norm = nn.Identity()
        self.k_norm = nn.Identity()
        self.sliding_window = config.sliding_window
        self.rotary_emb = RoPE(config)

    def __call__(
        self,
        hidden_states: mx.array,
        position_embeddings: Optional[Tuple[mx.array, mx.array]] = None,
        attention_mask: Optional[mx.array] = None,
        position_ids: Optional[mx.array] = None,
    ) -> Tuple[mx.array, Optional[mx.array]]:
        B, L, D = hidden_states.shape
        hidden_shape = (B, L, -1, self.head_dim)

        query_states = self.q_norm(
            self.q_proj(hidden_states).reshape(*hidden_shape)
        ).transpose(0, 2, 1, 3)
        key_states = self.k_norm(
            self.k_proj(hidden_states).reshape(*hidden_shape)
        ).transpose(0, 2, 1, 3)
        value_states = (
            self.v_proj(hidden_states).reshape(*hidden_shape).transpose(0, 2, 1, 3)
        )

        if position_embeddings is None:
            if position_ids is None:
                position_ids = mx.arange(L)
                position_ids = mx.expand_dims(position_ids, axis=0)
            cos, sin = self.rotary_emb(hidden_states, position_ids)
        else:
            cos, sin = position_embeddings

        query_states, key_states = apply_rotary_pos_emb(
            query_states, key_states, cos, sin
        )

        if attention_mask is not None and isinstance(attention_mask, mx.array):
            kv_seq_len = key_states.shape[-2]
            if attention_mask.shape[-1] != kv_seq_len:
                attention_mask = attention_mask[..., :kv_seq_len]

        if self.is_causal and attention_mask is None:
            attention_mask = nn.MultiHeadAttention.create_additive_causal_mask(L)
            attention_mask = attention_mask.astype(query_states.dtype)

        attn_output = scaled_dot_product_attention(
            query_states,
            key_states,
            value_states,
            None,
            scale=self.scaling,
            mask=attention_mask,
        )

        attn_output = attn_output.transpose(0, 2, 1, 3).reshape(B, L, -1)
        attn_output = self.o_proj(attn_output)
        return attn_output, None


class Code2WavMlp(nn.Module):
    def __init__(self, config: Code2WavConfig):
        super().__init__()

        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size

        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)

        if config.hidden_act == "silu":
            self.act_fn = nn.silu
        elif config.hidden_act == "gelu":
            self.act_fn = nn.gelu
        elif config.hidden_act == "gelu_pytorch_tanh":
            self.act_fn = nn.GELU(approx="precise")
        else:
            self.act_fn = nn.silu

    def __call__(self, x: mx.array) -> mx.array:
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))


class Code2WavTransformerLayer(nn.Module):
    def __init__(self, config: Code2WavConfig, idx: int):
        super().__init__()
        self.self_attn = Code2WavAttention(config, idx)
        self.mlp = Code2WavMlp(config)
        self.input_layernorm = nn.RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.post_attention_layernorm = nn.RMSNorm(
            config.hidden_size, config.rms_norm_eps
        )
        self.self_attn_layer_scale = LayerScale(config)
        self.mlp_layer_scale = LayerScale(config)

    def __call__(
        self,
        hidden_states: mx.array,
        attention_mask: Optional[mx.array] = None,
        position_ids: Optional[mx.array] = None,
        position_embeddings: Optional[Tuple[mx.array, mx.array]] = None,
    ) -> mx.array:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states, _ = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            position_embeddings=position_embeddings,
        )
        hidden_states = residual + self.self_attn_layer_scale(hidden_states)

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + self.mlp_layer_scale(hidden_states)

        return hidden_states


class Code2WavTransformerModel(nn.Module):
    def __init__(self, config: Code2WavConfig):
        super().__init__()

        self.layers = [
            Code2WavTransformerLayer(config, idx)
            for idx in range(config.num_hidden_layers)
        ]
        self.norm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = RoPE(config)

    def __call__(
        self,
        inputs_embeds: mx.array,
        attention_mask: Optional[mx.array] = None,
        position_ids: Optional[mx.array] = None,
    ) -> mx.array:
        hidden_states = inputs_embeds

        if position_ids is None:
            position_ids = mx.arange(hidden_states.shape[1])
            position_ids = mx.expand_dims(position_ids, axis=0)

        position_embeddings = self.rotary_emb(hidden_states, position_ids)

        for layer in self.layers:
            hidden_states = layer(
                hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                position_embeddings=position_embeddings,
            )

        hidden_states = self.norm(hidden_states)
        return hidden_states


class Code2WavModel(nn.Module):
    def __init__(self, config: Code2WavConfig):
        super().__init__()

        self.pre_transformer = Code2WavTransformerModel(config)
        self.code_embedding = nn.Embedding(
            config.codebook_size * config.num_quantizers, config.hidden_size
        )
        self.upsample = [
            [
                CausalTransConvNet(
                    config.hidden_size, config.hidden_size, factor, factor
                ),
                ConvNeXtBlock(config.hidden_size),
            ]
            for factor in config.upsampling_ratios
        ]
        self.decoder = [CausalConvNet(config.hidden_size, config.decoder_dim, 7)]
        self.decoder.extend(
            [
                Code2WavDecoderBlock(config, idx)
                for idx in range(len(config.upsample_rates))
            ]
        )
        output_dim = config.decoder_dim // 2 ** len(config.upsample_rates)
        self.decoder.extend([SnakeBeta(output_dim), CausalConvNet(output_dim, 1, 7)])
        self.config = config
        self.code_offset = (
            np.arange(config.num_quantizers).reshape(1, -1, 1) * config.codebook_size
        )

    def __call__(
        self, codes: mx.array = None, input_embeds: mx.array = None
    ) -> mx.array:
        if input_embeds is not None:
            hidden = input_embeds
        elif codes is not None:
            if codes.shape[1] != self.config.num_quantizers:
                raise ValueError(
                    f"Expected {self.config.num_quantizers} layer of codes, got {codes.shape[1]}"
                )
            hidden = self.code_embedding(codes + mx.array(self.code_offset)).mean(1)
        else:
            raise ValueError("Must provide codes or input_embeds")

        hidden = self.pre_transformer(inputs_embeds=hidden)
        hidden = hidden.transpose(0, 2, 1)
        for blocks in self.upsample:
            for block in blocks:
                hidden = block(hidden)
        wav = hidden
        for block in self.decoder:
            wav = block(wav)
        return mx.clip(wav, -1, 1)

    def chunked_decode(self, codes, chunk_size=300, left_context_size=25):
        total_upsample_factor = 1
        for r in self.config.upsampling_ratios:
            total_upsample_factor *= r
        for r in self.config.upsample_rates:
            total_upsample_factor *= r

        B, Q, L = codes.shape
        final_wav_list = []

        for start in range(0, L, chunk_size):
            end = min(start + chunk_size, L)
            context_start = max(0, start - left_context_size)
            chunk_codes = codes[:, :, context_start:end]
            wav_chunk = self(codes=chunk_codes)
            context_len_tokens = start - context_start
            valid_start_sample = context_len_tokens * total_upsample_factor
            current_chunk_valid_len_tokens = end - start
            valid_len_samples = current_chunk_valid_len_tokens * total_upsample_factor
            chunk_valid_wav = wav_chunk[
                :, :, valid_start_sample : valid_start_sample + valid_len_samples
            ]
            final_wav_list.append(chunk_valid_wav)

        return mx.concatenate(final_wav_list, axis=-1)

    def stream_decode(
        self, codes_buffer, chunk_size=300, left_context_size=25, decoded_len=0
    ):
        total_upsample_factor = 1
        for r in self.config.upsampling_ratios:
            total_upsample_factor *= r
        for r in self.config.upsample_rates:
            total_upsample_factor *= r

        L = codes_buffer.shape[2]
        start = decoded_len
        context_start = max(0, start - left_context_size)
        context_len = start - context_start
        new_tokens = chunk_size - context_len
        if L - start < new_tokens:
            return None, decoded_len

        end = start + new_tokens
        chunk_codes = codes_buffer[:, :, context_start:end]
        wav_chunk = self(codes=chunk_codes)
        context_len_tokens = start - context_start
        valid_start_sample = context_len_tokens * total_upsample_factor
        current_chunk_valid_len_tokens = end - start
        valid_len_samples = current_chunk_valid_len_tokens * total_upsample_factor
        chunk_valid_wav = wav_chunk[
            :, :, valid_start_sample : valid_start_sample + valid_len_samples
        ]
        return chunk_valid_wav, end

    def flush_decode(self, codes_buffer, left_context_size=25, decoded_len=0):
        total_upsample_factor = 1
        for r in self.config.upsampling_ratios:
            total_upsample_factor *= r
        for r in self.config.upsample_rates:
            total_upsample_factor *= r

        L = codes_buffer.shape[2]
        if decoded_len >= L:
            return None

        start = decoded_len
        context_start = max(0, start - left_context_size)
        chunk_codes = codes_buffer[:, :, context_start:]
        wav_chunk = self(codes=chunk_codes)
        context_len_tokens = start - context_start
        valid_start_sample = context_len_tokens * total_upsample_factor
        return wav_chunk[:, :, valid_start_sample:]

    def sanitize(self, weights):
        sanitized_weights = {}
        for k, v in weights.items():
            if ("upsample" in k and "conv.weight" in k and "dwconv" not in k) or (
                "decoder" in k
                and "block" in k
                and "conv.weight" in k
                and "conv1" not in k
                and "conv2" not in k
            ):
                sanitized_weights[k] = v.transpose(1, 2, 0)
            elif (
                ("dwconv.conv.weight" in k)
                or ("decoder" in k and "conv.weight" in k and "block" not in k)
                or (
                    "decoder" in k
                    and "block" in k
                    and ("conv1.conv.weight" in k or "conv2.conv.weight" in k)
                )
            ):
                sanitized_weights[k] = v.transpose(0, 2, 1)
            else:
                sanitized_weights[k] = v

        return sanitized_weights
