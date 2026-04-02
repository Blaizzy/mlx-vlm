"""Gemma 4 Conformer audio encoder, ported from PyTorch to MLX.

Architecture:
- SSCP (SubSample Conv Projection): 2x Conv2d blocks + Linear
- 12 Conformer blocks: FFW -> Attention -> LightConv1d -> FFW -> Clamp -> RMSNorm
- Chunked local attention with relative position embeddings and logit softcapping
- Output projection to text_hidden_size
"""

import math

import mlx.core as mx
import mlx.nn as nn

from .config import AudioConfig
from .vision import ClippableLinear


class AudioRMSNorm(nn.Module):
    """RMSNorm with weight applied directly (no offset)."""

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.weight = mx.ones((dim,))
        self.eps = eps

    def __call__(self, x: mx.array) -> mx.array:
        return mx.fast.rms_norm(x, self.weight, self.eps)


class SSCPConvBlock(nn.Module):
    """Conv2d -> LayerNorm(channels) -> ReLU with symmetric padding."""

    def __init__(self, config: AudioConfig, idx: int):
        super().__init__()
        in_channels = 1 if idx == 0 else config.subsampling_conv_channels[idx - 1]
        out_channels = config.subsampling_conv_channels[idx]
        self.time_stride = 2
        self.padding = (1, 1, 1, 1)

        # MLX Conv2d: input [B, H, W, C], weight [C_out, kH, kW, C_in]
        self.conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=(3, 3),
            stride=(2, 2),
            padding=0,
            bias=False,
        )

        # LayerNorm over channel dim (last dim in MLX channel-last format)
        self.norm = nn.LayerNorm(out_channels, eps=config.rms_norm_eps, bias=False)

    def __call__(self, x: mx.array, mask: mx.array):
        # x: [B, T, F, C] (MLX channel-last)
        # mask: [B, T] (True = invalid/padding)

        # Zero out invalid positions
        x = mx.where(mask[:, :, None, None], 0.0, x)

        # Manual padding on T and F dims
        pad_t_top, pad_t_bottom, pad_f_left, pad_f_right = self.padding
        x = mx.pad(
            x,
            [
                (0, 0),
                (pad_t_top, pad_t_bottom),
                (pad_f_left, pad_f_right),
                (0, 0),
            ],
        )

        x = self.conv(x)  # [B, T_out, F_out, C_out]

        # Downsample mask by time stride
        t_out = x.shape[1]
        output_mask = mask[:, :: self.time_stride][:, :t_out]

        # LayerNorm over channels (last dim) - works directly in MLX channel-last
        x = self.norm(x)
        x = nn.relu(x)

        return x, output_mask


class SubSampleConvProjection(nn.Module):
    """SSCP: 2 Conv2d blocks -> flatten(F, C) -> Linear projection to hidden_size."""

    INPUT_FEAT_SIZE = 128

    def __init__(self, config: AudioConfig):
        super().__init__()

        self.layer0 = SSCPConvBlock(config, idx=0)
        self.layer1 = SSCPConvBlock(config, idx=1)

        freq = self.INPUT_FEAT_SIZE
        for _ in range(2):
            freq = (freq + 2 - 3) // 2 + 1
        proj_input_dim = freq * config.subsampling_conv_channels[-1]
        self.input_proj_linear = nn.Linear(
            proj_input_dim, config.hidden_size, bias=False
        )

    def __call__(self, audio_mel: mx.array, mask: mx.array):
        # audio_mel: [B, T, F_in]
        # Add channel dim: [B, T, F, 1]
        x = mx.expand_dims(audio_mel, -1)

        x, mask = self.layer0(x, mask)  # [B, T1, F1, C1]
        x, mask = self.layer1(x, mask)  # [B, T2, F2, C2]

        # Flatten F*C -> [B, T, F*C]
        B, T, F, C = x.shape
        x = x.reshape(B, T, F * C)

        # Project to hidden_size
        x = self.input_proj_linear(x)

        return x, mask


class ConformerFeedForward(nn.Module):
    """Macaron-style FFW with residual scaling."""

    def __init__(self, config: AudioConfig):
        super().__init__()
        self.gradient_clipping = config.gradient_clipping
        self.residual_weight = config.residual_weight

        self.pre_layer_norm = AudioRMSNorm(config.hidden_size)
        self.ffw_layer_1 = ClippableLinear(
            config.hidden_size, config.hidden_size * 4, bias=False
        )
        self.ffw_layer_2 = ClippableLinear(
            config.hidden_size * 4, config.hidden_size, bias=False
        )
        self.post_layer_norm = AudioRMSNorm(config.hidden_size)

    def __call__(self, x: mx.array) -> mx.array:
        residual = x
        x = mx.clip(x, -self.gradient_clipping, self.gradient_clipping)
        x = self.pre_layer_norm(x)
        x = self.ffw_layer_1(x)
        x = nn.silu(x)
        x = self.ffw_layer_2(x)
        x = mx.clip(x, -self.gradient_clipping, self.gradient_clipping)
        x = self.post_layer_norm(x)
        return residual + x * self.residual_weight


class AudioRelativePositionEmbedding(nn.Module):
    """Sinusoidal relative position embedding for chunked attention."""

    def __init__(self, config: AudioConfig):
        super().__init__()
        self.num_heads = config.num_attention_heads
        self.channels = config.hidden_size
        self.head_dim = self.channels // self.num_heads
        self.max_backward = max(0, config.attention_context_left - 1)
        self.max_forward = config.attention_context_right

        self.pos_proj = nn.Linear(
            self.channels, self.num_heads * self.head_dim, bias=False
        )

        min_timescale = 1.0
        max_timescale = 10000.0
        num_timescales = self.channels // 2
        log_timescale_increment = math.log(max_timescale / min_timescale) / max(
            num_timescales - 1, 1
        )
        inv_timescales = min_timescale * mx.exp(
            mx.arange(num_timescales) * -log_timescale_increment
        )
        self._inv_timescales = inv_timescales.reshape(1, 1, num_timescales)

    def _get_timing_signal(self, position: mx.array, dtype: mx.Dtype) -> mx.array:
        position = position.astype(mx.float32)[..., None]
        scaled_time = position * self._inv_timescales
        signal = mx.concatenate([mx.sin(scaled_time), mx.cos(scaled_time)], axis=-1)
        return signal.astype(dtype)

    def _relative_shift(
        self,
        term_bd,
        batch_size,
        num_heads,
        num_blocks,
        block_size,
        context_size,
        max_span_plus_1,
    ):
        pad_amount = (context_size + 1) - max_span_plus_1
        term_bd = mx.pad(term_bd, [(0, 0), (0, 0), (0, 0), (0, 0), (0, pad_amount)])
        term_bd = term_bd.reshape(
            batch_size, num_heads, num_blocks, block_size * (context_size + 1)
        )
        term_bd = term_bd[:, :, :, : block_size * context_size]
        term_bd = term_bd.reshape(
            batch_size, num_heads, num_blocks, block_size, context_size
        )
        return term_bd

    def __call__(self, queries: mx.array, keys: mx.array) -> mx.array:
        B, U, W, N, H = queries.shape
        C = keys.shape[2]

        pos_indices = mx.arange(self.max_backward, -self.max_forward - 1, -1)[None]
        max_span_plus_1 = pos_indices.shape[1]

        sin_emb = self._get_timing_signal(pos_indices, queries.dtype)
        sin_emb = self.pos_proj(sin_emb.astype(self.pos_proj.weight.dtype))
        sin_emb = sin_emb.reshape(max_span_plus_1, self.num_heads, self.head_dim)
        sin_emb = sin_emb.astype(queries.dtype)

        queries_p = queries.transpose(0, 3, 1, 2, 4)
        keys_p = keys.transpose(0, 3, 1, 4, 2)
        term_ac = queries_p @ keys_p

        sin_emb_t = sin_emb.transpose(1, 2, 0)
        q_reshaped = queries_p.reshape(B, N, U * W, H)
        term_bd = (q_reshaped @ sin_emb_t).reshape(B, N, U, W, max_span_plus_1)

        term_bd = self._relative_shift(term_bd, B, N, U, W, C, max_span_plus_1)

        return term_ac + term_bd


class AudioAttention(nn.Module):
    """Chunked local attention with relative position embeddings and logit softcapping.

    Attribute names match checkpoint keys: self_attn.{q_proj,k_proj,v_proj,post,relative_k_proj,...}
    """

    def __init__(self, config: AudioConfig):
        super().__init__()
        self.num_heads = config.num_attention_heads
        self.hidden_size = config.hidden_size
        self.head_dim = self.hidden_size // self.num_heads

        self.chunk_size = config.attention_chunk_size
        self.max_future_horizon = config.attention_context_right
        self.max_past_horizon = max(0, config.attention_context_left - 1)
        self.context_size = (
            self.chunk_size + self.max_past_horizon + self.max_future_horizon
        )
        self.invalid_logits_value = config.attention_invalid_logits_value
        self.softcap = config.attention_logit_cap

        self.relative_k_proj = nn.Linear(
            self.hidden_size, self.num_heads * self.head_dim, bias=False
        )
        self.per_dim_scale = mx.zeros((self.head_dim,))

        self.q_proj = ClippableLinear(
            self.hidden_size, self.num_heads * self.head_dim, bias=False
        )
        self.k_proj = ClippableLinear(
            self.hidden_size, self.num_heads * self.head_dim, bias=False
        )
        self.v_proj = ClippableLinear(
            self.hidden_size, self.num_heads * self.head_dim, bias=False
        )
        self.post = ClippableLinear(self.hidden_size, self.hidden_size, bias=False)

        self.q_scale = (self.head_dim**-0.5) / math.log(2)
        self.k_scale = math.log(1 + math.e) / math.log(2)

        # Relative position embedding internals (reusing AudioRelativePositionEmbedding logic)
        self._rel_pos = AudioRelativePositionEmbedding.__new__(
            AudioRelativePositionEmbedding
        )
        self._rel_pos.num_heads = self.num_heads
        self._rel_pos.channels = self.hidden_size
        self._rel_pos.head_dim = self.head_dim
        self._rel_pos.max_backward = self.max_past_horizon
        self._rel_pos.max_forward = self.max_future_horizon
        min_timescale = 1.0
        max_timescale = 10000.0
        num_timescales = self.hidden_size // 2
        log_timescale_increment = math.log(max_timescale / min_timescale) / max(
            num_timescales - 1, 1
        )
        inv_timescales = min_timescale * mx.exp(
            mx.arange(num_timescales) * -log_timescale_increment
        )
        self._rel_pos._inv_timescales = inv_timescales.reshape(1, 1, num_timescales)
        # Override pos_proj to use our relative_k_proj
        self._rel_pos.pos_proj = self.relative_k_proj

    def _pad_dim1(self, x: mx.array, pad_left: int, pad_right: int) -> mx.array:
        pads = [(0, 0)] * x.ndim
        pads[1] = (pad_left, pad_right)
        return mx.pad(x, pads)

    def _convert_to_block(self, x: mx.array) -> mx.array:
        """[B, T, ...] -> [B, num_blocks, chunk_size, ...]"""
        B, T = x.shape[0], x.shape[1]
        rest = x.shape[2:]
        num_blocks = (T + self.chunk_size - 1) // self.chunk_size
        pad_len = num_blocks * self.chunk_size - T
        if pad_len > 0:
            x = self._pad_dim1(x, 0, pad_len)
        return x.reshape(B, num_blocks, self.chunk_size, *rest)

    def _extract_block_context(self, x: mx.array) -> mx.array:
        """[B, T, ...] -> [B, num_blocks, context_size, ...]"""
        pad_left = self.max_past_horizon
        pad_right = self.max_future_horizon + self.chunk_size - 1
        x = self._pad_dim1(x, pad_left, pad_right)
        T_padded = x.shape[1]
        num_blocks = (T_padded - self.context_size) // self.chunk_size + 1
        starts = mx.arange(num_blocks) * self.chunk_size
        offsets = mx.arange(self.context_size)
        indices = starts[:, None] + offsets[None, :]
        return x[:, indices]

    def __call__(
        self, hidden_states: mx.array, mask: mx.array, causal_valid_mask: mx.array
    ) -> mx.array:
        B, T = hidden_states.shape[:2]
        qkv_shape = (B, T, self.num_heads, self.head_dim)

        q = self.q_proj(hidden_states).astype(mx.float32).reshape(qkv_shape)
        k = self.k_proj(hidden_states).astype(mx.float32).reshape(qkv_shape)
        v = self.v_proj(hidden_states).astype(mx.float32).reshape(qkv_shape)

        per_dim_scale = nn.softplus(self.per_dim_scale)
        q = q * (self.q_scale * per_dim_scale)
        k = k * self.k_scale

        query_blocks = self._convert_to_block(q)
        key_blocks = self._extract_block_context(k)
        value_blocks = self._extract_block_context(v)
        U = query_blocks.shape[1]

        valid_mask = ~mask
        extracted_valid = self._extract_block_context(valid_mask)
        condition = (
            extracted_valid[:, None, :, None, :]
            & causal_valid_mask[None, None, None, :, :]
        )

        logits = self._rel_pos(query_blocks, key_blocks)
        logits = mx.tanh(logits / self.softcap) * self.softcap
        logits = mx.where(condition, logits, self.invalid_logits_value)

        probs = mx.softmax(logits, axis=-1)
        context = mx.einsum("bnuwc,bucnh->buwnh", probs, value_blocks)
        context = context.reshape(B, U * self.chunk_size, self.num_heads, self.head_dim)
        context = context[:, :T]

        # Reshape [B, T, N, H] -> [B, T, D] and post-project
        B_out, T_out, N, H = context.shape
        context = context.reshape(B_out, T_out, N * H)
        return self.post(context)


class ConformerLightConv1d(nn.Module):
    """Light convolution: norm -> linear(2x) -> GLU -> depthwise_conv1d(causal) -> norm -> SiLU -> linear + residual."""

    def __init__(self, config: AudioConfig):
        super().__init__()
        self.gradient_clipping = config.gradient_clipping
        self.causal_padding = config.conv_kernel_size - 1

        self.pre_layer_norm = AudioRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.linear_start = ClippableLinear(
            config.hidden_size, config.hidden_size * 2, bias=False
        )

        self.depthwise_conv1d = nn.Conv1d(
            in_channels=config.hidden_size,
            out_channels=config.hidden_size,
            kernel_size=config.conv_kernel_size,
            stride=1,
            padding=0,
            bias=False,
        )
        self.depthwise_conv1d.weight = mx.zeros(
            (config.hidden_size, config.conv_kernel_size, 1)
        )

        self.conv_norm = AudioRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.linear_end = ClippableLinear(
            config.hidden_size, config.hidden_size, bias=False
        )

    def __call__(self, x: mx.array) -> mx.array:
        residual = x

        x = self.pre_layer_norm(x)
        x = self.linear_start(x)

        # GLU: split in half along last dim and gate
        x1, x2 = mx.split(x, 2, axis=-1)
        x = x1 * mx.sigmoid(x2)

        # Causal padding for Conv1d
        x = mx.pad(x, [(0, 0), (self.causal_padding, 0), (0, 0)])

        # Depthwise conv1d using conv_general for groups support
        x = mx.conv_general(
            x,
            self.depthwise_conv1d.weight,
            stride=1,
            padding=0,
            groups=x.shape[-1],
        )

        x = mx.clip(x, -self.gradient_clipping, self.gradient_clipping)
        x = self.conv_norm(x)
        x = nn.silu(x)
        x = self.linear_end(x)

        return x + residual


class ConformerBlock(nn.Module):
    """Macaron-style Conformer block.

    Attribute names match checkpoint: layers.X.{feed_forward1,self_attn,lconv1d,feed_forward2,norm_*}
    """

    def __init__(self, config: AudioConfig):
        super().__init__()
        self.gradient_clipping = config.gradient_clipping
        self.feed_forward1 = ConformerFeedForward(config)
        self.self_attn = AudioAttention(config)
        self.lconv1d = ConformerLightConv1d(config)
        self.feed_forward2 = ConformerFeedForward(config)
        self.norm_pre_attn = AudioRMSNorm(config.hidden_size)
        self.norm_post_attn = AudioRMSNorm(config.hidden_size)
        self.norm_out = AudioRMSNorm(config.hidden_size)

    def __call__(
        self, x: mx.array, mask: mx.array, causal_valid_mask: mx.array
    ) -> mx.array:
        x = self.feed_forward1(x)

        # Attention with pre/post norm and residual
        residual = x
        x = mx.clip(x, -self.gradient_clipping, self.gradient_clipping)
        x = self.norm_pre_attn(x)
        x = self.self_attn(x, mask, causal_valid_mask)
        x = mx.clip(x, -self.gradient_clipping, self.gradient_clipping)
        x = residual + self.norm_post_attn(x)

        # Zero out invalid positions before lconv1d
        validity_mask = (~mask)[:, :, None].astype(x.dtype)
        x = x * validity_mask

        x = self.lconv1d(x)
        x = self.feed_forward2(x)
        x = mx.clip(x, -self.gradient_clipping, self.gradient_clipping)
        return self.norm_out(x)


class AudioEncoder(nn.Module):
    """Gemma 4 audio encoder based on the Universal Speech Model architecture."""

    def __init__(self, config: AudioConfig):
        super().__init__()
        self.config = config

        self.subsample_conv_projection = SubSampleConvProjection(config)
        self.layers = [ConformerBlock(config) for _ in range(config.num_hidden_layers)]

        if config.output_proj_dims is not None:
            self.output_proj = nn.Linear(
                config.hidden_size, config.output_proj_dims, bias=True
            )
        else:
            self.output_proj = None

    def _build_causal_valid_mask(self) -> mx.array:
        """Build the local causal+validity mask for chunked attention."""
        chunk_size = self.config.attention_chunk_size
        max_future_horizon = self.config.attention_context_right
        max_past_horizon = max(0, self.config.attention_context_left - 1)
        upper_diagonal = max_past_horizon + max_future_horizon
        context_size = chunk_size + max_past_horizon + max_future_horizon

        lower_causal = mx.tril(mx.ones((context_size, chunk_size))).T
        upper_causal = mx.tril(mx.ones((chunk_size, context_size)), k=upper_diagonal)
        mask = (lower_causal * upper_causal).astype(mx.bool_)
        return mask

    def __call__(self, audio_mel: mx.array, audio_mel_mask: mx.array) -> tuple:
        audio_encodings, current_mask = self.subsample_conv_projection(
            audio_mel, audio_mel_mask
        )

        causal_valid_mask = self._build_causal_valid_mask()

        for block in self.layers:
            audio_encodings = block(audio_encodings, current_mask, causal_valid_mask)

        if self.output_proj is not None:
            audio_encodings = self.output_proj(audio_encodings)

        if current_mask.shape[1] != audio_encodings.shape[1]:
            target_len = audio_encodings.shape[1]
            current_mask = current_mask[:, :target_len]

        audio_encodings = mx.where(current_mask[:, :, None], 0.0, audio_encodings)

        return audio_encodings, current_mask
