"""
Cascades Conformer audio encoder for Phi-4 Multimodal, ported to MLX.

Architecture:
- NeMo Conv Subsampling (time_reduction=8, depthwise separable)
- 24 Conformer blocks (attention_dim=1024, 16 heads)
- T5 relative attention bias
- Absolute positional encoding
- Mean-variance normalization embedding
"""

import math
from typing import Optional

import mlx.core as mx
import mlx.nn as nn
import numpy as np

from .config import AudioConfig

# =============================================================================
# Basic Modules
# =============================================================================


class Swish(nn.Module):
    def __call__(self, x: mx.array) -> mx.array:
        return x * mx.sigmoid(x)


def get_activation(name: str = "relu"):
    name = name.lower()
    if name == "relu":
        return nn.ReLU()
    if name == "gelu":
        return nn.GELU()
    if name == "swish":
        return Swish()
    if name == "sigmoid":
        return nn.Sigmoid()
    return lambda x: x


class GLU(nn.Module):
    """Gated Linear Unit: splits input in half, gates with activation."""

    def __init__(self, dim: int = -1, act_name: str = "sigmoid"):
        super().__init__()
        self.dim = dim
        self.act = get_activation(act_name)

    def __call__(self, x: mx.array) -> mx.array:
        half = x.shape[self.dim] // 2
        if self.dim == -1:
            half_x = x[..., :half]
            gate = x[..., half:]
        else:
            half_x = mx.take(x, mx.arange(half), axis=self.dim)
            gate = mx.take(x, mx.arange(half, x.shape[self.dim]), axis=self.dim)
        return half_x * self.act(gate)


class GLULinear(nn.Module):
    """Linear + GLU."""

    def __init__(self, input_dim, output_dim, glu_type="sigmoid", bias=True):
        super().__init__()
        self.linear = nn.Linear(input_dim, output_dim * 2, bias=bias)
        self.glu = GLU(-1, glu_type)

    def __call__(self, x: mx.array) -> mx.array:
        return self.glu(self.linear(x))


class GLUPointWiseConv(nn.Module):
    """GLU with pointwise Conv1D for the Conformer conv module.

    In the checkpoint:
    - ext_pw_conv_1d: Conv1d(input_dim, output_dim*2, kernel_size=1)
    - b1, b2: bias parameters (1, output_dim, 1)

    We implement this with a Linear (equivalent to Conv1d with kernel=1).
    """

    def __init__(self, input_dim, output_dim, glu_type="sigmoid", bias_in_glu=True):
        super().__init__()
        self.output_dim = output_dim
        self.bias_in_glu = bias_in_glu
        # Conv1d(in, out*2, k=1) is equivalent to Linear(in, out*2)
        self.ext_pw_conv_1d = nn.Linear(input_dim, output_dim * 2, bias=True)
        self.glu_act = get_activation(glu_type)
        if bias_in_glu:
            self.b1 = mx.zeros((output_dim,))
            self.b2 = mx.zeros((output_dim,))

    def __call__(self, x: mx.array) -> mx.array:
        # x: (B, T, C) - already in channel-last format
        x = self.ext_pw_conv_1d(x)
        x1 = x[..., : self.output_dim]
        x2 = x[..., self.output_dim :]
        if self.bias_in_glu:
            x = (x1 + self.b1) * self.glu_act(x2 + self.b2)
        else:
            x = x1 * self.glu_act(x2)
        return x


# =============================================================================
# Feed Forward Module
# =============================================================================


class FeedForward(nn.Module):
    """Feed Forward module with GLU.

    Architecture: LayerNorm -> GLULinear(d_model, d_inner) -> Linear(d_inner, d_model)
    """

    def __init__(self, d_model, d_inner, activation="sigmoid", bias_in_glu=True):
        super().__init__()
        self.layer_norm = nn.LayerNorm(d_model)
        # net.0 = GLULinear(d_model, d_inner) - has .linear submodule
        # net.2 = Linear(d_inner, d_model)
        self.net_0 = GLULinear(d_model, d_inner, activation, bias=bias_in_glu)
        self.net_2 = nn.Linear(d_inner, d_model, bias=True)

    def __call__(self, x: mx.array) -> mx.array:
        x = self.layer_norm(x)
        x = self.net_0(x)
        x = self.net_2(x)
        return x


# =============================================================================
# Depthwise Separable Conv1D
# =============================================================================


class DepthWiseSeparableConv1d(nn.Module):
    """Depthwise separable Conv1D.

    - dw_conv: depthwise Conv1d (groups=input_dim)
    - pw_conv: pointwise Conv1d (1x1)

    MLX Conv1d weight shape: (out_channels, kernel_size, in_channels)
    For depthwise: groups=input_dim, so each filter operates on 1 channel.
    """

    def __init__(
        self, input_dim, out_channel, kernel_size, depthwise_multiplier=1, padding=0
    ):
        super().__init__()
        self.padding = padding
        self.dw_conv = nn.Conv1d(
            input_dim,
            input_dim * depthwise_multiplier,
            kernel_size,
            stride=1,
            padding=padding,
            groups=input_dim,
        )
        if out_channel != 0:
            self.pw_conv = nn.Conv1d(
                input_dim * depthwise_multiplier,
                out_channel,
                kernel_size=1,
                stride=1,
                padding=0,
            )
        self.out_channel = out_channel

    def __call__(self, x: mx.array) -> mx.array:
        # x: (B, T, C)
        x = self.dw_conv(x)
        if self.out_channel != 0:
            x = self.pw_conv(x)
        return x


# =============================================================================
# Conv Module (Conformer)
# =============================================================================


class ConvModule(nn.Module):
    """Conformer convolution module.

    Architecture:
    1. LayerNorm
    2. GLU pointwise conv (input_dim -> input_dim, via 2*input_dim with GLU)
    3. Depthwise separable Conv1D
    4. LayerNorm (when cnn_layer_norm=True)
    5. Swish activation
    6. Pointwise Conv1D (ext_pw_conv_1d)
    """

    def __init__(
        self,
        input_dim,
        ext_pw_out_channel,
        depthwise_seperable_out_channel,
        ext_pw_kernel_size,
        kernel_size,
        depthwise_multiplier,
        causal=False,
        batch_norm=False,
        cnn_layer_norm=True,
        activation="relu",
        glu_type="sigmoid",
        bias_in_glu=True,
    ):
        super().__init__()
        self.layer_norm = nn.LayerNorm(input_dim)
        self.input_dim = input_dim
        self.ext_pw_out_channel = ext_pw_out_channel
        self.causal = causal
        self.kernel_size = kernel_size

        if ext_pw_out_channel != 0:
            self.glu = GLUPointWiseConv(
                input_dim, ext_pw_out_channel, glu_type, bias_in_glu
            )

        if causal:
            padding = kernel_size - 1
        else:
            padding = (kernel_size - 1) // 2

        self.dw_sep_conv_1d = DepthWiseSeparableConv1d(
            input_dim,
            depthwise_seperable_out_channel,
            kernel_size,
            depthwise_multiplier,
            padding=padding,
        )

        self.act = get_activation(activation)

        if ext_pw_out_channel != 0:
            # Final pointwise conv: Linear equivalent for k=1
            self.ext_pw_conv_1d = nn.Linear(input_dim, ext_pw_out_channel, bias=True)

    def __call__(self, x: mx.array) -> mx.array:
        x = self.layer_norm(x)

        if self.ext_pw_out_channel != 0:
            x = self.glu(x)

        # Depthwise separable conv
        x = self.dw_sep_conv_1d(x)

        # Trim causal padding
        if self.causal and self.kernel_size > 1:
            x = x[:, : -(self.kernel_size - 1), :]

        x = self.act(x)

        # Final pointwise conv
        if self.ext_pw_out_channel != 0:
            x = self.ext_pw_conv_1d(x)

        return x


# =============================================================================
# Multi-Head Attention
# =============================================================================


class MultiHeadedAttention(nn.Module):
    """Multi-Head Attention with optional T5 relative bias."""

    def __init__(self, n_head, n_feat):
        super().__init__()
        self.d_k = n_feat // n_head
        self.h = n_head
        self.scale = self.d_k**-0.5

        self.linear_q = nn.Linear(n_feat, n_feat, bias=True)
        self.linear_k = nn.Linear(n_feat, n_feat, bias=True)
        self.linear_v = nn.Linear(n_feat, n_feat, bias=True)
        self.linear_out = nn.Linear(n_feat, n_feat, bias=True)

    def __call__(
        self,
        query: mx.array,
        key: mx.array,
        value: mx.array,
        mask: Optional[mx.array] = None,
        relative_attention_bias: Optional[mx.array] = None,
    ) -> mx.array:
        B = query.shape[0]

        q = self.linear_q(query).reshape(B, -1, self.h, self.d_k).transpose(0, 2, 1, 3)
        k = self.linear_k(key).reshape(B, -1, self.h, self.d_k).transpose(0, 2, 1, 3)
        v = self.linear_v(value).reshape(B, -1, self.h, self.d_k).transpose(0, 2, 1, 3)

        # Scaled dot-product attention
        scores = (q * self.scale) @ k.transpose(0, 1, 3, 2)

        if relative_attention_bias is not None:
            scores = scores + relative_attention_bias

        if mask is not None:
            scores = mx.where(mask, scores, mx.array(float("-inf")))

        attn = mx.softmax(scores, axis=-1)
        if mask is not None:
            attn = mx.where(mask, attn, mx.array(0.0))

        x = attn @ v  # (B, H, T, d_k)
        x = x.transpose(0, 2, 1, 3).reshape(B, -1, self.h * self.d_k)

        return self.linear_out(x)


# =============================================================================
# T5 Relative Attention Bias
# =============================================================================


class T5RelativeAttentionLogitBias(nn.Module):
    """T5-style relative attention bias (asymmetric, no bucketing)."""

    def __init__(self, num_heads, max_distance=1000):
        super().__init__()
        self.num_heads = num_heads
        self.max_distance = max_distance
        # Asymmetric: 2 * max_distance buckets
        self.num_buckets = max_distance * 2
        self.bias_values = nn.Embedding(self.num_buckets, num_heads)

    def __call__(self, x: mx.array) -> mx.array:
        maxpos = x.shape[1]
        context_position = mx.arange(maxpos)[:, None]
        memory_position = mx.arange(maxpos)[None, :]
        relative_position = memory_position - context_position

        # Clip to [-max_distance, max_distance-1]
        relative_position = mx.clip(
            relative_position, -self.max_distance, self.max_distance - 1
        )

        # Asymmetric: shift by num_buckets // 2
        bias_idx = relative_position + self.num_buckets // 2

        t5_bias = self.bias_values(bias_idx)  # (L, L, H)
        t5_bias = t5_bias.transpose(2, 0, 1)[None, :, :, :]  # (1, H, L, L)

        return t5_bias


# =============================================================================
# Positional Encoding
# =============================================================================


class AbsolutePositionalEncoding(nn.Module):
    """Sinusoidal absolute positional encoding."""

    def __init__(self, d_model, max_len=5000):
        super().__init__()
        self.d_model = d_model
        self.xscale = math.sqrt(d_model)
        self._build_pe(max_len)

    def _build_pe(self, max_len):
        pe = np.zeros((max_len, self.d_model), dtype=np.float32)
        position = np.arange(0, max_len, dtype=np.float32)[:, None]
        div_term = np.exp(
            np.arange(0, self.d_model, 2, dtype=np.float32)
            * -(math.log(10000.0) / self.d_model)
        )
        pe[:, 0::2] = np.sin(position * div_term)
        pe[:, 1::2] = np.cos(position * div_term)
        self.pe = mx.array(pe[None, :, :])  # (1, max_len, d_model)

    def __call__(self, x: mx.array) -> mx.array:
        T = x.shape[1]
        if T > self.pe.shape[1]:
            self._build_pe(T)
        return x * self.xscale + self.pe[:, :T]


# =============================================================================
# Mean/Variance Normalization
# =============================================================================


class MeanVarianceNormLayer(nn.Module):
    """Global mean/variance normalization for input features."""

    def __init__(self, input_size):
        super().__init__()
        self.global_mean = mx.zeros((input_size,))
        self.global_invstd = mx.ones((input_size,))

    def __call__(self, x: mx.array) -> mx.array:
        return (x - self.global_mean) * self.global_invstd


# =============================================================================
# NeMo Conv Subsampling
# =============================================================================


class DWPWConvPair(nn.Module):
    """A pair of depthwise + pointwise Conv2d layers."""

    def __init__(self, channels, kernel_size, stride, padding):
        super().__init__()
        self.dw = nn.Conv2d(
            channels,
            channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            groups=channels,
        )
        self.pw = nn.Conv2d(
            channels,
            channels,
            kernel_size=1,
            stride=1,
            padding=0,
        )

    def __call__(self, x):
        x = self.dw(x)
        x = nn.relu(self.pw(x))
        return x


class NemoConvSubsampling(nn.Module):
    """NeMo-style convolutional subsampling (dw_striding).

    For time_reduction=8:
    - Layer 0: Conv2d(1, conv_ch, 3x3, stride=2)  -> T/2, F/2
    - Layer 1: ReLU
    - DW/PW pair 0: DW Conv2d + PW Conv2d -> T/4, F/4
    - DW/PW pair 1: DW Conv2d + PW Conv2d -> T/8, F/8
    - out: Linear(conv_ch * (F/8), feat_out)

    MLX Conv2d expects: (B, H, W, C_in) and weight (C_out, kH, kW, C_in)
    """

    def __init__(
        self, feat_in, feat_out, time_reduction=8, conv_channels=1024, causal=False
    ):
        super().__init__()
        self.time_reduction = time_reduction
        self.causal = causal
        sampling_num = int(math.log(time_reduction, 2))
        kernel_size = 3
        stride = 2
        padding = (kernel_size - 1) // 2

        # Layer 0: Standard Conv2d(1 -> conv_channels)
        self.conv_0 = nn.Conv2d(
            1, conv_channels, kernel_size=kernel_size, stride=stride, padding=padding
        )

        # Subsequent layers: DW Conv2d + PW Conv2d pairs
        self.dw_pw_layers = [
            DWPWConvPair(conv_channels, kernel_size, stride, padding)
            for _ in range(sampling_num - 1)
        ]

        # Calculate output frequency dimension after all subsampling
        freq_out = feat_in
        for _ in range(sampling_num):
            freq_out = (freq_out + 2 * padding - kernel_size) // stride + 1

        self.out = nn.Linear(conv_channels * freq_out, feat_out, bias=True)
        self.conv2d_subsampling = True

    def __call__(self, x: mx.array, mask: Optional[mx.array] = None):
        """
        Args:
            x: (B, T, feat_in) mel spectrogram
            mask: (B, T) validity mask
        Returns:
            x: (B, T', feat_out) subsampled features
            mask: (B, 1, T') subsampled mask
        """
        B, T, F = x.shape
        # (B, T, F) -> (B, T, F, 1) for MLX Conv2d (channel last)
        x = x[:, :, :, None]

        # Layer 0: Conv2d + ReLU
        x = nn.relu(self.conv_0(x))

        # DW + PW layers
        for pair in self.dw_pw_layers:
            x = pair(x)

        # x: (B, T', F', C) -> match PyTorch order (B, T', C*F')
        # PyTorch Conv2d outputs (B, C, T, F) -> transpose to (B, T, C, F) -> flatten to (B, T, C*F)
        # MLX Conv2d outputs (B, T, F, C) -> transpose to (B, T, C, F) -> flatten to (B, T, C*F)
        B, T_out, F_out, C = x.shape
        x = x.transpose(0, 1, 3, 2).reshape(B, T_out, C * F_out)

        # Linear projection
        x = self.out(x)

        # Update mask
        if mask is not None:
            feature_lens = mask.sum(axis=1)
            padding_length = mx.ceil(feature_lens / self.time_reduction).astype(
                mx.int32
            )
            max_audio_length = T_out
            indices = mx.arange(max_audio_length)[None, :]  # (1, T_out)
            pad_mask = indices < padding_length[:, None]  # (B, T_out)
            mask = pad_mask[:, None, :]  # (B, 1, T_out)

        return x, mask


# =============================================================================
# Conformer Encoder Layer
# =============================================================================


class ConformerEncoderLayer(nn.Module):
    """Single Conformer block.

    Forward: x += 0.5*FFN_in(x)
             x += Attn(LN(x))
             x += Conv(x)
             x += 0.5*FFN_out(x)
             x = LN(x)
    """

    def __init__(self, config: AudioConfig):
        super().__init__()
        d_model = config.attention_dim
        d_ffn = config.linear_units

        self.feed_forward_in = FeedForward(
            d_model, d_ffn, config.activation, config.bias_in_glu
        )

        self.self_attn = MultiHeadedAttention(config.attention_heads, d_model)

        self.conv = ConvModule(
            d_model,
            config.ext_pw_out_channel,
            config.depthwise_seperable_out_channel,
            config.ext_pw_kernel_size,
            config.kernel_size,
            config.depthwise_multiplier,
            causal=config.causal,
            batch_norm=config.batch_norm,
            cnn_layer_norm=config.cnn_layer_norm,
            activation=config.conv_activation,
            glu_type=config.conv_glu_type,
            bias_in_glu=config.bias_in_glu,
        )

        self.feed_forward_out = FeedForward(
            d_model, d_ffn, config.activation, config.bias_in_glu
        )

        self.layer_norm_att = nn.LayerNorm(d_model)
        self.layer_norm = nn.LayerNorm(d_model)

    def __call__(
        self,
        x: mx.array,
        mask: Optional[mx.array] = None,
        relative_attention_bias: Optional[mx.array] = None,
    ) -> mx.array:
        x = x + 0.5 * self.feed_forward_in(x)
        norm_x = self.layer_norm_att(x)
        x = x + self.self_attn(
            norm_x,
            norm_x,
            norm_x,
            mask=mask,
            relative_attention_bias=relative_attention_bias,
        )
        x = x + self.conv(x)
        x = x + 0.5 * self.feed_forward_out(x)
        return self.layer_norm(x)


# =============================================================================
# Conformer Encoder
# =============================================================================


class ConformerEncoder(nn.Module):
    """Cascades Conformer encoder.

    Pipeline:
    1. MeanVarianceNormLayer (global normalization)
    2. NemoConvSubsampling (time reduction)
    3. AbsolutePositionalEncoding
    4. T5RelativeAttentionLogitBias
    5. 24x ConformerEncoderLayer
    """

    def __init__(self, config: AudioConfig):
        super().__init__()
        self.config = config
        d = config.attention_dim

        self.encoder_embedding = MeanVarianceNormLayer(config.input_size)

        self.embed = NemoConvSubsampling(
            feat_in=config.input_size,
            feat_out=d,
            time_reduction=config.time_reduction,
            conv_channels=config.conv_channels,
            causal=config.causal,
        )

        self.relative_attention_bias_layer = T5RelativeAttentionLogitBias(
            num_heads=config.attention_heads,
            max_distance=config.t5_bias_max_distance,
        )

        self.encoders = [
            ConformerEncoderLayer(config) for _ in range(config.num_blocks)
        ]

    def __call__(
        self,
        xs_pad: mx.array,
        masks: Optional[mx.array] = None,
    ):
        """
        Args:
            xs_pad: (B, T, 80) mel spectrogram features
            masks: (B, T) validity mask

        Returns:
            output: (B, T', attention_dim) encoded features
            masks: (B, 1, T') validity mask after subsampling
        """
        # 1. Global mean-variance normalization
        xs_pad = self.encoder_embedding(xs_pad)

        # 2. Conv subsampling
        input_tensor, masks = self.embed(xs_pad, masks)

        # 3. Handle long sequences by chunking (max_seq_len for pos encoding)
        max_seq_len = 500
        seq_len = input_tensor.shape[1]
        unfolded = False

        if seq_len > max_seq_len:
            unfolded = True
            ori_bz = input_tensor.shape[0]
            # Pad to multiple of max_seq_len
            if seq_len % max_seq_len > 0:
                chunk_pad_size = max_seq_len - (seq_len % max_seq_len)
            else:
                chunk_pad_size = 0

            if chunk_pad_size > 0:
                pad = mx.zeros(
                    (input_tensor.shape[0], chunk_pad_size, input_tensor.shape[2])
                )
                input_tensor = mx.concatenate([input_tensor, pad], axis=1)

            # Unfold: (B, T_padded, D) -> (B * num_chunks, max_seq_len, D)
            B, T_padded, D = input_tensor.shape
            num_chunks = T_padded // max_seq_len
            input_tensor = input_tensor.reshape(B * num_chunks, max_seq_len, D)

        # 4. Compute T5 relative attention bias
        relative_attention_bias = self.relative_attention_bias_layer(input_tensor)

        # 5. Run through conformer blocks
        for layer in self.encoders:
            input_tensor = layer(
                input_tensor,
                mask=None,  # Simplified: no streaming mask for inference
                relative_attention_bias=relative_attention_bias,
            )

        # 6. Refold if we unfolded
        if unfolded:
            D = input_tensor.shape[-1]
            input_tensor = input_tensor.reshape(ori_bz, -1, D)
            if chunk_pad_size > 0:
                input_tensor = input_tensor[:, :-chunk_pad_size, :]

        return input_tensor, masks

    def sanitize(self, weights):
        """Sanitize checkpoint weights for the audio encoder.

        Key transformations:
        - Conv2d weights: PyTorch (out, in, kH, kW) -> MLX (out, kH, kW, in)
        - Conv1d weights for actual convolutions: PyTorch (out, in, kW) -> MLX (out, kW, in)
        - Conv1d k=1 weights mapped to Linear: (out, in, 1) -> (out, in)
        - GLUPointWiseConv b1/b2: (1, C, 1) -> (C,)
        - Map sequential conv indices to named layers
        """
        sanitized = {}

        # Keys whose Conv1d(k=1) weights should become Linear weights
        linear_conv_keys = {"glu.ext_pw_conv_1d", "conv.ext_pw_conv_1d"}

        for k, v in weights.items():
            new_key = k

            # Map embed.conv.N sequential indices to named layers
            # Conv sequential: 0=Conv2d, 1=ReLU, 2=DW, 3=PW, 4=ReLU, 5=DW, 6=PW, 7=ReLU
            if "embed.conv." in k:
                parts = k.split("embed.conv.")
                rest = parts[1]
                idx_str = rest.split(".")[0]
                param = rest.split(".", 1)[1]
                idx = int(idx_str)

                if idx == 0:
                    new_key = parts[0] + "embed.conv_0." + param
                elif idx == 1:
                    continue  # ReLU
                elif idx == 2:
                    new_key = parts[0] + "embed.dw_pw_layers.0.dw." + param
                elif idx == 3:
                    new_key = parts[0] + "embed.dw_pw_layers.0.pw." + param
                elif idx == 4:
                    continue  # ReLU
                elif idx == 5:
                    new_key = parts[0] + "embed.dw_pw_layers.1.dw." + param
                elif idx == 6:
                    new_key = parts[0] + "embed.dw_pw_layers.1.pw." + param
                elif idx == 7:
                    continue  # ReLU

            # Map feed_forward net indices
            if ".net.0.linear." in new_key:
                new_key = new_key.replace(".net.0.linear.", ".net_0.linear.")
            elif ".net.2." in new_key:
                new_key = new_key.replace(".net.2.", ".net_2.")

            # --- Weight shape transformations ---

            # GLUPointWiseConv b1/b2: (1, C, 1) -> (C,)
            if ("glu.b1" in new_key or "glu.b2" in new_key) and v.ndim == 3:
                v = v.reshape(-1)
                sanitized[new_key] = v
                continue

            # Conv2d weight: PyTorch (out, in, kH, kW) -> MLX (out, kH, kW, in)
            if v.ndim == 4 and "weight" in new_key:
                v = v.transpose(0, 2, 3, 1)
                sanitized[new_key] = v
                continue

            # Conv1d weights (3D)
            if v.ndim == 3 and "weight" in new_key:
                is_k1_to_linear = any(lk in new_key for lk in linear_conv_keys)

                if is_k1_to_linear:
                    # Conv1d(in, out, k=1): (out, in, 1) -> Linear (out, in)
                    v = v[:, :, 0]  # Remove kernel dimension
                else:
                    # Regular Conv1d: PyTorch (out, in, kW) -> MLX (out, kW, in)
                    v = v.transpose(0, 2, 1)

                sanitized[new_key] = v
                continue

            sanitized[new_key] = v

        return sanitized


# =============================================================================
# Audio Projection
# =============================================================================


class AudioProjection(nn.Module):
    """Projects audio features to LM hidden size.

    Two branches: 'speech' and 'vision', each:
    Linear(audio_dim, hidden_size) -> GELU -> Linear(hidden_size, hidden_size)
    """

    def __init__(self, audio_dim, hidden_size):
        super().__init__()
        self.speech = AudioProjectionBranch(audio_dim, hidden_size)
        self.vision = AudioProjectionBranch(audio_dim, hidden_size)

    def __call__(self, x: mx.array, mode: str = "speech") -> mx.array:
        if mode == "speech":
            return self.speech(x)
        elif mode == "vision":
            return self.vision(x)
        else:
            raise ValueError(f"Unknown projection mode: {mode}")


class AudioProjectionBranch(nn.Module):
    """Single branch of audio projection: Linear -> GELU -> Linear."""

    def __init__(self, audio_dim, hidden_size):
        super().__init__()
        # Matches checkpoint: audio_projection.speech.0 and .2
        self.proj_0 = nn.Linear(audio_dim, hidden_size, bias=True)
        self.proj_2 = nn.Linear(hidden_size, hidden_size, bias=True)

    def __call__(self, x: mx.array) -> mx.array:
        x = self.proj_0(x)
        x = nn.gelu(x)
        x = self.proj_2(x)
        return x
