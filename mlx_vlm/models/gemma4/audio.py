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
    """RMSNorm with scale_shift=0.0 (weight used directly, no +1 offset)."""

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.weight = mx.ones((dim,))
        self.eps = eps

    def __call__(self, x: mx.array) -> mx.array:
        return mx.fast.rms_norm(x, self.weight, self.eps)


class SSCPConvBlock(nn.Module):
    """Conv2d -> LayerNorm(channels) -> ReLU with manual asymmetric padding."""

    def __init__(self, config: AudioConfig, idx: int, input_freq_dim: int):
        super().__init__()
        in_channels = 1 if idx == 0 else config.sscp_conv_channel_size[idx - 1]
        out_channels = config.sscp_conv_channel_size[idx]
        kernel_t, kernel_f = config.sscp_conv_kernel_size[idx]
        stride_t, stride_f = config.sscp_conv_stride_size[idx]
        self.time_stride = stride_t

        # Compute asymmetric padding
        if (
            config.sscp_conv_time_pad_top is not None
            and config.sscp_conv_time_pad_bottom is not None
        ):
            pad_t_top = config.sscp_conv_time_pad_top
            pad_t_bottom = config.sscp_conv_time_pad_bottom
        elif config.sscp_conv_padding_type == "semicausal":
            pad_t_top = kernel_t // 2
            pad_t_bottom = 0 if config.streaming else kernel_t // 2
        else:  # reverse_causal
            pad_t_top = 0
            pad_t_bottom = 0 if config.streaming else kernel_t - 1

        pad_f_left = 1
        pad_f_right = 1
        self.padding = (pad_t_top, pad_t_bottom, pad_f_left, pad_f_right)

        # MLX Conv2d: input [B, H, W, C], weight [C_out, kH, kW, C_in]
        self.conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=(kernel_t, kernel_f),
            stride=(stride_t, stride_f),
            padding=0,
            bias=False,
        )

        # LayerNorm over channel dim (last dim in MLX channel-last format)
        self.norm = nn.LayerNorm(out_channels, eps=config.sscp_conv_eps, bias=False)

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

    def __init__(self, config: AudioConfig):
        super().__init__()

        # Pre-calculate output frequency dims for each conv block
        current_f = config.input_feat_size
        f_out_dims = []

        for i in range(2):
            kernel_t, kernel_f = config.sscp_conv_kernel_size[i]

            if (
                config.sscp_conv_time_pad_top is not None
                and config.sscp_conv_time_pad_bottom is not None
            ):
                pass
            elif config.sscp_conv_padding_type == "semicausal":
                pass
            else:
                pass

            _, stride_f = config.sscp_conv_stride_size[i]
            pad_f_left, pad_f_right = 1, 1

            f_in_padded = current_f + pad_f_left + pad_f_right
            f_out = (f_in_padded - kernel_f) // stride_f + 1
            f_out_dims.append(f_out)
            current_f = f_out

        self.conv_0 = SSCPConvBlock(
            config, idx=0, input_freq_dim=config.input_feat_size
        )
        self.conv_1 = SSCPConvBlock(config, idx=1, input_freq_dim=f_out_dims[0])

        final_c_out = config.sscp_conv_channel_size[-1]
        final_f_out = f_out_dims[-1]
        self.input_proj_linear = nn.Linear(
            final_c_out * final_f_out, config.hidden_size, bias=False
        )

    def __call__(self, audio_mel: mx.array, mask: mx.array):
        # audio_mel: [B, T, F_in]
        # Add channel dim: [B, T, F, 1]
        x = mx.expand_dims(audio_mel, -1)

        x, mask = self.conv_0(x, mask)  # [B, T1, F1, C1]
        x, mask = self.conv_1(x, mask)  # [B, T2, F2, C2]

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
        self.residual_weight = config.conf_residual_weight

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
        self.num_heads = config.conf_num_attention_heads
        self.channels = config.hidden_size
        self.head_dim = self.channels // self.num_heads
        self.max_backward = max(0, config.conf_attention_context_left - 1)
        self.max_forward = config.conf_attention_context_right

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
        # position: [1, F_span]
        position = position.astype(mx.float32)[..., None]  # [1, F_span, 1]
        scaled_time = position * self._inv_timescales  # [1, F_span, channels//2]
        signal = mx.concatenate([mx.sin(scaled_time), mx.cos(scaled_time)], axis=-1)
        return signal.astype(dtype)  # [1, F_span, channels]

    def _relative_shift(
        self,
        term_bd: mx.array,
        batch_size: int,
        num_heads: int,
        num_blocks: int,
        block_size: int,
        context_size: int,
        max_span_plus_1: int,
    ) -> mx.array:
        # term_bd: [B, N, U, W, F_span] -> [B, N, U, W, C]
        pad_amount = (context_size + 1) - max_span_plus_1
        term_bd = mx.pad(term_bd, [(0, 0), (0, 0), (0, 0), (0, 0), (0, pad_amount)])
        # [B, N, U, W*(C+1)]
        term_bd = term_bd.reshape(
            batch_size,
            num_heads,
            num_blocks,
            block_size * (context_size + 1),
        )
        # Slice to [B, N, U, W*C]
        term_bd = term_bd[:, :, :, : block_size * context_size]
        # [B, N, U, W, C]
        term_bd = term_bd.reshape(
            batch_size, num_heads, num_blocks, block_size, context_size
        )
        return term_bd

    def __call__(self, queries: mx.array, keys: mx.array) -> mx.array:
        # queries: [B, U, W, N, H], keys: [B, U, C, N, H]
        B, U, W, N, H = queries.shape
        C = keys.shape[2]

        # Position indices: [max_backward, ..., 0, -1, ..., -max_forward]
        pos_indices = mx.arange(self.max_backward, -self.max_forward - 1, -1)[
            None
        ]  # [1, F_span]
        max_span_plus_1 = pos_indices.shape[1]

        # Sinusoidal embeddings -> project
        sin_emb = self._get_timing_signal(
            pos_indices, queries.dtype
        )  # [1, F_span, channels]
        sin_emb = self.pos_proj(
            sin_emb.astype(self.pos_proj.weight.dtype)
        )  # [1, F_span, N*H]
        sin_emb = sin_emb.reshape(
            max_span_plus_1, self.num_heads, self.head_dim
        )  # [F_span, N, H]
        sin_emb = sin_emb.astype(queries.dtype)

        # term_ac: content-content (Q @ K^T)
        queries_p = queries.transpose(0, 3, 1, 2, 4)  # [B, N, U, W, H]
        keys_p = keys.transpose(0, 3, 1, 4, 2)  # [B, N, U, H, C]
        term_ac = queries_p @ keys_p  # [B, N, U, W, C]

        # term_bd: content-position (Q @ pos_emb^T)
        sin_emb_t = sin_emb.transpose(1, 2, 0)  # [N, H, F_span]
        q_reshaped = queries_p.reshape(B, N, U * W, H)  # [B, N, U*W, H]
        term_bd = (q_reshaped @ sin_emb_t).reshape(
            B, N, U, W, max_span_plus_1
        )  # [B, N, U, W, F_span]

        # Relative shift
        term_bd = self._relative_shift(term_bd, B, N, U, W, C, max_span_plus_1)

        return term_ac + term_bd  # [B, N, U, W, C]


class AudioAttention(nn.Module):
    """Chunked local attention with relative position embeddings and logit softcapping."""

    def __init__(self, config: AudioConfig):
        super().__init__()
        self.num_heads = config.conf_num_attention_heads
        self.hidden_size = config.hidden_size
        self.head_dim = self.hidden_size // self.num_heads

        self.chunk_size = config.conf_attention_chunk_size
        self.max_future_horizon = config.conf_attention_context_right
        self.max_past_horizon = max(0, config.conf_attention_context_left - 1)
        self.context_size = (
            self.chunk_size + self.max_past_horizon + self.max_future_horizon
        )
        self.invalid_logits_value = config.conf_attention_invalid_logits_value
        self.softcap = config.conf_attention_logit_cap

        self.relative_position_embedding = AudioRelativePositionEmbedding(config)
        self.per_dim_scale = mx.zeros((self.head_dim,))
        self.per_dim_key_scale = mx.ones((self.head_dim,))

        self.q_proj = ClippableLinear(
            self.hidden_size, self.num_heads * self.head_dim, bias=False
        )
        self.k_proj = ClippableLinear(
            self.hidden_size, self.num_heads * self.head_dim, bias=False
        )
        self.v_proj = ClippableLinear(
            self.hidden_size, self.num_heads * self.head_dim, bias=False
        )

        self.q_scale = (self.head_dim**-0.5) / math.log(2)
        self.k_scale = 1.0 / math.log(2)

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

        # Gather overlapping windows via index array
        starts = mx.arange(num_blocks) * self.chunk_size
        offsets = mx.arange(self.context_size)
        indices = starts[:, None] + offsets[None, :]  # [num_blocks, context_size]

        return x[:, indices]

    def __call__(
        self,
        hidden_states: mx.array,
        mask: mx.array,
        causal_valid_mask: mx.array,
    ) -> mx.array:
        # hidden_states: [B, T, D], mask: [B, T] (True=invalid)
        B, T = hidden_states.shape[:2]
        qkv_shape = (B, T, self.num_heads, self.head_dim)

        # Project and cast to float32 for attention computation
        q = self.q_proj(hidden_states).astype(mx.float32).reshape(qkv_shape)
        k = self.k_proj(hidden_states).astype(mx.float32).reshape(qkv_shape)
        v = self.v_proj(hidden_states).astype(mx.float32).reshape(qkv_shape)

        # Per-dimension scaling
        per_dim_scale = nn.softplus(self.per_dim_scale)
        per_dim_key_scale = nn.softplus(self.per_dim_key_scale)

        q = q * (self.q_scale * per_dim_scale)
        k = k * (self.k_scale * per_dim_key_scale)

        # Chunk queries into non-overlapping blocks, keys/values into overlapping context
        query_blocks = self._convert_to_block(q)  # [B, U, W, N, H]
        key_blocks = self._extract_block_context(k)  # [B, U, C, N, H]
        value_blocks = self._extract_block_context(v)  # [B, U, C, N, H]
        U = query_blocks.shape[1]

        # Build combined validity mask
        valid_mask = ~mask  # True = valid
        extracted_valid = self._extract_block_context(valid_mask)  # [B, U, C]

        # [B, U, C] -> [B, 1, U, 1, C] & [W, C] -> [1, 1, 1, W, C]
        condition = (
            extracted_valid[:, None, :, None, :]
            & causal_valid_mask[None, None, None, :, :]
        )  # [B, 1, U, W, C]

        # Compute logits via relative position embedding
        logits = self.relative_position_embedding(
            query_blocks, key_blocks
        )  # [B, N, U, W, C]

        # Softcapping with tanh
        logits = mx.tanh(logits / self.softcap) * self.softcap

        # Apply mask
        logits = mx.where(condition, logits, self.invalid_logits_value)

        # Softmax and weighted sum
        probs = mx.softmax(logits, axis=-1)  # [B, N, U, W, C]

        # [B, N, U, W, C] x [B, U, C, N, H] -> [B, U, W, N, H]
        context = mx.einsum("bnuwc,bucnh->buwnh", probs, value_blocks)

        # Reshape back: [B, U*W, N, H] and trim to original length
        context = context.reshape(B, U * self.chunk_size, self.num_heads, self.head_dim)
        context = context[:, :T]

        return context  # [B, T, N, H]


class ConformerAttention(nn.Module):
    """Wraps AudioAttention with pre/post norm, post projection, and residual."""

    def __init__(self, config: AudioConfig):
        super().__init__()
        self.gradient_clipping = config.gradient_clipping
        self.pre_attn_norm = AudioRMSNorm(config.hidden_size)
        self.attn = AudioAttention(config)
        self.post = ClippableLinear(config.hidden_size, config.hidden_size, bias=False)
        self.post_norm = AudioRMSNorm(config.hidden_size)

    def __call__(
        self, x: mx.array, mask: mx.array, causal_valid_mask: mx.array
    ) -> mx.array:
        residual = x
        x = mx.clip(x, -self.gradient_clipping, self.gradient_clipping)
        x = self.pre_attn_norm(x)

        # Attention output: [B, T, N, H]
        x = self.attn(x, mask, causal_valid_mask)

        # Reshape: [B, T, N, H] -> [B, T, D]
        B, T, N, H = x.shape
        x = x.reshape(B, T, N * H)

        x = self.post(x)
        x = mx.clip(x, -self.gradient_clipping, self.gradient_clipping)
        return residual + self.post_norm(x)


class ConformerLightConv1d(nn.Module):
    """Light convolution: norm -> linear(2x) -> GLU -> depthwise_conv1d(causal) -> norm -> SiLU -> linear + residual."""

    def __init__(self, config: AudioConfig):
        super().__init__()
        self.gradient_clipping = config.gradient_clipping
        self.causal_padding = config.conf_conv_kernel_size - 1

        self.pre_layer_norm = AudioRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.linear_start = ClippableLinear(
            config.hidden_size, config.hidden_size * 2, bias=False
        )

        # Depthwise Conv1d: groups=hidden_size
        # MLX Conv1d input: [B, L, C], weight: [C_out, K, C_in/groups]
        self.depthwise_conv1d = nn.Conv1d(
            in_channels=config.hidden_size,
            out_channels=config.hidden_size,
            kernel_size=config.conf_conv_kernel_size,
            stride=1,
            padding=0,  # Manual causal padding
            bias=False,
        )
        # Set groups after init since MLX Conv1d might not accept it in constructor
        # Actually MLX Conv1d does support groups - let me reconstruct properly
        self.depthwise_conv1d = nn.Conv1d(
            in_channels=config.hidden_size,
            out_channels=config.hidden_size,
            kernel_size=config.conf_conv_kernel_size,
            stride=1,
            padding=0,
            bias=False,
        )
        # Manually set the weight shape for depthwise (groups=hidden_size)
        # Weight shape: [out_channels, kernel_size, in_channels/groups] = [D, K, 1]
        self.depthwise_conv1d.weight = mx.zeros(
            (config.hidden_size, config.conf_conv_kernel_size, 1)
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

        # Causal padding for Conv1d: pad left of time dim only
        # MLX Conv1d input: [B, L, C]
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
    """Macaron-style Conformer block: FFW -> Attention -> LConv1d -> FFW -> Clamp -> RMSNorm."""

    def __init__(self, config: AudioConfig):
        super().__init__()
        self.gradient_clipping = config.gradient_clipping
        self.ffw_layer_start = ConformerFeedForward(config)
        self.attention = ConformerAttention(config)
        self.lconv1d = ConformerLightConv1d(config)
        self.ffw_layer_end = ConformerFeedForward(config)
        self.norm = AudioRMSNorm(config.hidden_size)

    def __call__(
        self, x: mx.array, mask: mx.array, causal_valid_mask: mx.array
    ) -> mx.array:
        x = self.ffw_layer_start(x)
        x = self.attention(x, mask, causal_valid_mask)

        # Zero out invalid positions before lconv1d
        validity_mask = (~mask)[:, :, None].astype(x.dtype)  # [B, T, 1]
        x = x * validity_mask

        x = self.lconv1d(x)
        x = self.ffw_layer_end(x)
        x = mx.clip(x, -self.gradient_clipping, self.gradient_clipping)
        return self.norm(x)


class AudioEncoder(nn.Module):
    """Gemma 4 audio encoder based on the Universal Speech Model architecture."""

    def __init__(self, config: AudioConfig):
        super().__init__()
        self.config = config

        self.subsample_conv_projection = SubSampleConvProjection(config)
        self.conformer = [
            ConformerBlock(config) for _ in range(config.conf_num_hidden_layers)
        ]

        if config.output_proj_dims is not None:
            self.output_proj = nn.Linear(
                config.hidden_size, config.output_proj_dims, bias=True
            )
        else:
            self.output_proj = None

        self.reduction_factor = config.conf_reduction_factor

    def _build_causal_valid_mask(self) -> mx.array:
        """Build the local causal+validity mask for chunked attention."""
        chunk_size = self.config.conf_attention_chunk_size
        max_future_horizon = self.config.conf_attention_context_right
        max_past_horizon = max(0, self.config.conf_attention_context_left - 1)
        upper_diagonal = max_past_horizon + max_future_horizon
        context_size = chunk_size + max_past_horizon + max_future_horizon

        # Lower causal: tril(ones(context_size, chunk_size)).T -> [chunk_size, context_size]
        lower_causal = mx.tril(mx.ones((context_size, chunk_size))).T

        # Upper causal: tril(ones(chunk_size, context_size), k=upper_diagonal)
        upper_causal = mx.tril(mx.ones((chunk_size, context_size)), k=upper_diagonal)

        # Combined mask (element-wise AND via multiplication of binary masks)
        mask = (lower_causal * upper_causal).astype(mx.bool_)
        return mask  # [chunk_size, context_size]

    def __call__(self, audio_mel: mx.array, audio_mel_mask: mx.array) -> tuple:
        """
        Args:
            audio_mel: [B, T, mel_bins] mel spectrogram features
            audio_mel_mask: [B, T] boolean mask (True = padding/invalid)

        Returns:
            audio_encodings: [B, T_out, output_dims]
            mask: [B, T_out] boolean mask
        """
        # Subsample via SSCP
        audio_encodings, current_mask = self.subsample_conv_projection(
            audio_mel, audio_mel_mask
        )

        # Build causal validity mask for chunked attention
        causal_valid_mask = self._build_causal_valid_mask()

        # Run conformer blocks
        for block in self.conformer:
            audio_encodings = block(audio_encodings, current_mask, causal_valid_mask)

        # Optional reduction factor (stride-based downsampling)
        if self.reduction_factor > 1:
            audio_encodings = audio_encodings[:, :: self.reduction_factor]
            current_mask = current_mask[:, :: self.reduction_factor]

        # Output projection
        if self.output_proj is not None:
            audio_encodings = self.output_proj(audio_encodings)

        # Ensure mask length matches
        if current_mask.shape[1] != audio_encodings.shape[1]:
            target_len = audio_encodings.shape[1]
            current_mask = current_mask[:, :target_len]

        # Zero out padding positions
        audio_encodings = mx.where(current_mask[:, :, None], 0.0, audio_encodings)

        return audio_encodings, current_mask
