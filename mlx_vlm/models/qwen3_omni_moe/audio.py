import math
from typing import Optional

import mlx.core as mx
import mlx.nn as nn
import numpy as np

from mlx_vlm.models.qwen3_omni_moe.config import AudioConfig


def _get_feat_extract_output_lengths(input_lengths):
    input_lengths_leave = input_lengths % 100
    feat_lengths = (input_lengths_leave - 1) // 2 + 1
    output_lengths = (
        ((feat_lengths - 1) // 2 + 1 - 1) // 2 + 1 + (input_lengths // 100) * 13
    )
    return output_lengths


class Attention(nn.Module):
    def __init__(self, config: AudioConfig):
        super().__init__()
        self.embed_dim = config.d_model
        self.num_heads = config.encoder_attention_heads
        self.head_dim = self.embed_dim // self.num_heads
        self.scaling = self.head_dim**-0.5

        self.k_proj = nn.Linear(config.d_model, config.d_model, bias=True)
        self.v_proj = nn.Linear(config.d_model, config.d_model, bias=True)
        self.q_proj = nn.Linear(config.d_model, config.d_model, bias=True)
        self.out_proj = nn.Linear(config.d_model, config.d_model, bias=True)

    def __call__(
        self,
        hidden_states: mx.array,
        cu_seqlens: mx.array,
        attention_mask: Optional[mx.array] = None,
    ) -> mx.array:
        seq_length = hidden_states.shape[0]

        query_states = self.q_proj(hidden_states).reshape(
            seq_length, self.num_heads, -1
        )
        key_states = self.k_proj(hidden_states).reshape(seq_length, self.num_heads, -1)
        value_states = self.v_proj(hidden_states).reshape(
            seq_length, self.num_heads, -1
        )

        query_states = query_states.transpose(1, 0, 2)[None]
        key_states = key_states.transpose(1, 0, 2)[None]
        value_states = value_states.transpose(1, 0, 2)[None]

        lengths = (cu_seqlens[1:] - cu_seqlens[:-1]).tolist()
        if len(lengths) == 0:
            lengths = [seq_length]

        attn_outputs = []
        offset = 0
        for length in lengths:
            if length <= 0:
                continue
            end = offset + length
            q_chunk = query_states[:, :, offset:end]
            k_chunk = key_states[:, :, offset:end]
            v_chunk = value_states[:, :, offset:end]
            attn_weights = (q_chunk @ k_chunk.swapaxes(-2, -1)) * self.scaling
            attn_weights = mx.softmax(attn_weights, axis=-1)
            attn_output = attn_weights @ v_chunk
            attn_outputs.append(attn_output)
            offset = end

        attn_output = (
            mx.concatenate(attn_outputs, axis=2)
            if attn_outputs
            else mx.zeros_like(query_states)
        )
        attn_output = attn_output.transpose(0, 2, 1, 3).reshape(seq_length, -1)
        attn_output = self.out_proj(attn_output)

        return attn_output


class AudioEncoderLayer(nn.Module):
    def __init__(self, config: AudioConfig, idx: int):
        super().__init__()
        self.embed_dim = config.d_model
        self.self_attn = Attention(config)
        self.self_attn_layer_norm = nn.LayerNorm(config.d_model)
        self.fc1 = nn.Linear(config.d_model, config.encoder_ffn_dim)
        self.fc2 = nn.Linear(config.encoder_ffn_dim, config.d_model)
        self.final_layer_norm = nn.LayerNorm(config.d_model)

        if config.activation_function == "gelu":
            self.activation_fn = nn.gelu
        else:
            raise ValueError(f"Unsupported activation: {config.activation_function}")

    def __call__(
        self,
        hidden_states: mx.array,
        cu_seqlens: mx.array,
        attention_mask: Optional[mx.array] = None,
    ) -> mx.array:
        residual = hidden_states
        hidden_states = self.self_attn_layer_norm(hidden_states)
        hidden_states = self.self_attn(
            hidden_states=hidden_states,
            cu_seqlens=cu_seqlens,
            attention_mask=attention_mask,
        )
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.final_layer_norm(hidden_states)
        hidden_states = self.fc1(hidden_states)
        hidden_states = self.activation_fn(hidden_states)
        hidden_states = self.fc2(hidden_states)
        hidden_states = residual + hidden_states

        return hidden_states


class SinusoidsPositionEmbedding(nn.Module):
    def __init__(self, length, channels, max_timescale=10000):
        super().__init__()
        if channels % 2 != 0:
            raise ValueError("SinusoidsPositionEmbedding needs even channels input")
        log_timescale_increment = np.log(max_timescale) / (channels // 2 - 1)
        inv_timescales = np.exp(
            -log_timescale_increment * np.arange(channels // 2, dtype=np.float32)
        )
        scaled_time = np.arange(length)[:, np.newaxis] * inv_timescales[np.newaxis, :]
        positional_embedding = np.concatenate(
            [np.sin(scaled_time), np.cos(scaled_time)], axis=1
        )
        self.positional_embedding = positional_embedding

    def __call__(self, seqlen: int) -> mx.array:
        return mx.array(self.positional_embedding[:seqlen, :])


class AudioModel(nn.Module):
    def __init__(self, config: AudioConfig):
        super().__init__()

        self.dropout = config.dropout
        embed_dim = config.d_model
        self.num_mel_bins = config.num_mel_bins
        self.max_source_positions = config.max_source_positions
        self.embed_scale = math.sqrt(embed_dim) if config.scale_embedding else 1.0
        self.n_window = config.n_window
        self.n_window_infer = config.n_window_infer
        self.conv_chunksize = config.conv_chunksize

        self.positional_embedding = SinusoidsPositionEmbedding(
            self.max_source_positions, embed_dim
        )
        self.layers = [
            AudioEncoderLayer(config, idx) for idx in range(config.encoder_layers)
        ]
        self.ln_post = nn.LayerNorm(config.d_model)

        self.conv2d1 = nn.Conv2d(1, config.downsample_hidden_size, 3, 2, padding=1)
        self.conv2d2 = nn.Conv2d(
            config.downsample_hidden_size,
            config.downsample_hidden_size,
            3,
            2,
            padding=1,
        )
        self.conv2d3 = nn.Conv2d(
            config.downsample_hidden_size,
            config.downsample_hidden_size,
            3,
            2,
            padding=1,
        )
        self.conv_out = nn.Linear(
            config.downsample_hidden_size
            * ((((config.num_mel_bins + 1) // 2 + 1) // 2 + 1) // 2),
            config.d_model,
            bias=False,
        )
        self.proj1 = nn.Linear(config.d_model, config.d_model)

        if config.activation_function == "gelu":
            self.act = nn.gelu
        else:
            raise ValueError(f"Unsupported activation: {config.activation_function}")

        self.proj2 = nn.Linear(config.d_model, config.output_dim)

    def __call__(
        self,
        input_features: mx.array,
        feature_lens: Optional[mx.array] = None,
        aftercnn_lens: Optional[mx.array] = None,
    ):
        if feature_lens is None:
            feature_lens = mx.array([input_features.shape[-1]], dtype=mx.int32)

        aftercnn_lens = _get_feat_extract_output_lengths(feature_lens)
        feature_lens_np = np.array(feature_lens).astype(np.int32)
        n_window_step = self.n_window * 2
        chunk_num = np.ceil(feature_lens_np / n_window_step).astype(np.int32)

        chunk_lengths_list = []
        tail_chunk_info = []
        cumsum = 0
        for sample_idx, num_chunks in enumerate(chunk_num.tolist()):
            num_int = int(num_chunks)
            chunk_lengths_list.extend([n_window_step] * num_int)
            if num_int > 0:
                tail_chunk_info.append((cumsum + num_int - 1, sample_idx))
            cumsum += num_int

        for tail_idx, sample_idx in tail_chunk_info:
            remainder = feature_lens_np[sample_idx] % n_window_step
            if remainder == 0:
                remainder = n_window_step
            chunk_lengths_list[tail_idx] = int(remainder)

        chunk_lengths = mx.array(chunk_lengths_list, dtype=mx.int32)

        total_chunks = len(chunk_lengths_list)
        max_chunk_len = int(chunk_lengths.max())
        padded_feature = mx.zeros(
            (total_chunks, self.num_mel_bins, max_chunk_len), dtype=input_features.dtype
        )

        start_idx = 0
        for i, chunk_len in enumerate(chunk_lengths_list):
            end_idx = start_idx + chunk_len
            padded_feature[i, :, :chunk_len] = input_features[:, start_idx:end_idx]
            start_idx = end_idx

        padded_feature = padded_feature[:, None, :, :]

        feature_lens_after_cnn = _get_feat_extract_output_lengths(feature_lens)
        max_len_after_cnn = int(feature_lens_after_cnn.max())
        padded_mask_after_cnn = mx.zeros(
            (total_chunks, max_len_after_cnn), dtype=mx.bool_
        )
        for i, length in enumerate(feature_lens_after_cnn):
            padded_mask_after_cnn[i, : int(length)] = True

        padded_embeds = []
        for i in range(0, total_chunks, self.conv_chunksize):
            end_idx = min(i + self.conv_chunksize, total_chunks)
            chunk = padded_feature[i:end_idx]
            chunk = chunk.transpose(0, 2, 3, 1)
            padded_embed = nn.gelu(self.conv2d1(chunk))
            padded_embed = nn.gelu(self.conv2d2(padded_embed))
            padded_embed = nn.gelu(self.conv2d3(padded_embed))
            padded_embeds.append(padded_embed)

        padded_embed = mx.concatenate(padded_embeds, axis=0)
        b, h, w, c = padded_embed.shape
        padded_embed = padded_embed.transpose(0, 2, 3, 1).reshape(b, w, c * h)
        padded_embed = self.conv_out(padded_embed)

        seq_len = padded_embed.shape[1]
        positional_embedding = self.positional_embedding(seq_len)[None]
        padded_embed = padded_embed + positional_embedding

        linear_indices = []
        for i in range(total_chunks):
            mask_array = np.array(padded_mask_after_cnn[i])
            chunk_indices = np.where(mask_array)[0]
            linear_indices.extend([i * seq_len + idx for idx in chunk_indices])

        padded_embed_flat = padded_embed.reshape(-1, padded_embed.shape[-1])
        hidden_states = mx.take(
            padded_embed_flat,
            mx.array(np.array(linear_indices, dtype=np.int32)),
            axis=0,
        )

        cu_chunk_lens = []
        window_aftercnn = max_len_after_cnn * (
            self.n_window_infer // (self.n_window * 2)
        )
        for cnn_len in feature_lens_after_cnn:
            cnn_len_int = int(cnn_len)
            num_windows = cnn_len_int // window_aftercnn
            cu_chunk_lens.extend([window_aftercnn] * num_windows)
            remainder = cnn_len_int % window_aftercnn
            if remainder != 0:
                cu_chunk_lens.append(remainder)

        cu_seqlens = mx.cumsum(mx.array(cu_chunk_lens, dtype=mx.int32), axis=0)
        cu_seqlens = mx.pad(cu_seqlens, (1, 0), constant_values=0)

        for i, encoder_layer in enumerate(self.layers):
            hidden_states = encoder_layer(
                hidden_states,
                cu_seqlens,
            )
            if i % 2 == 0:
                mx.eval(hidden_states)

        hidden_states = self.ln_post(hidden_states)
        hidden_states = self.proj1(hidden_states)
        hidden_states = self.act(hidden_states)
        hidden_states = self.proj2(hidden_states)

        return hidden_states

    def sanitize(self, weights):
        sanitized_weights = {}
        for k, v in weights.items():
            if "audio_tower.conv2d" in k and "weight" in k:
                sanitized_weights[k] = v.transpose(0, 2, 3, 1)
            else:
                sanitized_weights[k] = v

        return sanitized_weights
