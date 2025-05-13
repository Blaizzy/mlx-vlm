import math
from dataclasses import dataclass
from typing import Tuple, Union

import mlx.core as mx
import mlx.nn as nn


@dataclass
class AudioConfig:
    input_feat_size: int = 80
    hidden_size: int = 1536
    conf_attention_chunk_size: int = 12
    conf_attention_context_left: int = 13
    conf_attention_context_right: int = 0
    conf_attention_invalid_logits_value: float = -1e9
    conf_attention_logit_cap: float = 50.0
    conf_num_attention_heads: int = 8
    conf_num_hidden_layers: int = 12
    conf_conv_kernel_size: int = 5
    conf_positional_bias_size: int = 256
    conf_reduction_factor: int = 4
    conf_residual_weight: float = 0.5
    sscp_conv_channel_size: tuple[int, int] = (128, 32)
    sscp_conv_group_norm_eps: float = 1e-3
    sscp_conv_kernel_size: tuple[tuple[int, int], tuple[int, int]] = ((3, 3), (3, 3))
    sscp_conv_stride_size: tuple[tuple[int, int], tuple[int, int]] = ((2, 2), (2, 2))


class Gemma3p5RMSNorm(nn.Module):
    def __init__(
        self,
        dim: int,
        *args,
        eps: float = 1e-6,
        scale_shift: float = 1.0,
        with_scale: bool = True,
        **kwargs,
    ):
        super().__init__()
        self.eps = eps
        self.scale_shift = scale_shift
        self.with_scale = with_scale

        if self.with_scale:
            self.weight = mx.ones(dim)
        else:
            self.weight = mx.array(1.0)

    def extra_repr(self):
        return f"{tuple(self.weight.shape)}, eps={self.eps}"

    def __call__(self, x: mx.array):
        x, original_dtype = self._guard_against_excess_precision(x)

        scale = self.weight
        if self.scale_shift != 0.0:
            scale += self.scale_shift

        mean_squared = x.pow(2).mean(-1, keepdim=True)
        root_mean_squared = x * mx.rsqrt(mean_squared + self.eps)
        # Llama does x.to(float16) * w whilst Gemma2 is (x * w).to(float16)
        # See https://github.com/huggingface/transformers/pull/29402
        scaled = root_mean_squared * scale.float()
        return scaled.type(original_dtype)

    def _guard_against_excess_precision(self, x: mx.array) -> tuple[mx.array, mx.Dtype]:
        # TODO(ryanmullins): Implement Torch equivalent to jax.lax.reduce_precision
        return x.float(), x.dtype


class Gemma3p5AudioRelativePositionEmbedding(nn.Module):

    def __init__(self, config: AudioConfig, *args, **kwargs):
        super().__init__()
        self.config = config

        self.num_heads = self.config.conf_num_attention_heads
        self.hidden_size = self.config.hidden_size
        self.units_per_head = self.hidden_size // self.num_heads
        self.max_backward = self.config.conf_attention_context_left
        self.max_forward = self.config.conf_attention_context_right

        self.pos_proj = nn.Linear(
            self.hidden_size, self.num_heads * self.units_per_head, bias=False
        )

    def _get_timing_signal_1d_pos(
        self, position: mx.array, channels: int, dtype: mx.Dtype
    ) -> mx.array:
        assert position.ndim == 2
        position = mx.expand_dims(position.float(), axis=-1)

        min_timescale = 1.0
        max_timescale = 1.0e4
        num_timescales = channels // 2
        log_timescale_increment = math.log(
            float(max_timescale) / float(min_timescale)
        ) / max(num_timescales - 1, 1)
        inv_timescales = min_timescale * mx.exp(
            mx.arange(num_timescales) * -log_timescale_increment
        )
        inv_timescales = mx.expand_dims(
            mx.expand_dims(inv_timescales.float(), axis=0), axis=0
        ).to(device=position.device)

        scaled_time = position * inv_timescales

        timing_signal = mx.concatenate(
            [mx.sin(scaled_time), mx.cos(scaled_time)], dim=-1
        )
        timing_signal_padding = (0, np.mod(channels, 2), 0, 0, 0, 0)
        timing_signal = mx.pad(timing_signal, timing_signal_padding)

        return timing_signal.type(dtype)

    def __call__(self, queries: mx.array, keys: mx.array) -> mx.array:
        b, u, w = queries.shape[:3]
        _, _, c = keys.shape[:3]
        n = self.num_heads
        h = self.units_per_head
        l = self.max_backward
        r = self.max_forward
        lr = l + r
        assert c == w + lr

        pos = mx.expand_dims(mx.arange(l, -r - 1, -1), axis=0)
        assert pos.shape == (1, lr + 1)

        sin_emb = self._get_timing_signal_1d_pos(
            pos, self.hidden_size, dtype=queries.dtype
        )

        # Note: In JAX code, sl.Dense modifies sin_emb with a jax.numpy.einsum("...d,dnh->...nh") but this has been
        # converted to nn.Linear to align with the Transformers style guide.
        batch_dims = sin_emb.shape[:-1]
        sin_emb_projected: mx.array = self.pos_proj(sin_emb)
        sin_emb = sin_emb_projected.reshape(*batch_dims, n, h)
        sin_emb = sin_emb.squeeze(0)

        # Note: In JAX code, sl.TransformerXLRelativePositionEmbedding computes term_ac with
        # torch.einsum("BuwNH,BucNH->BNuwc", ...), this converts that operation to a matmul to comply with the
        # Transformers style guide.
        queries_p = queries.permute(0, 3, 1, 2, 4)
        keys_p_t = keys.permute(0, 3, 1, 4, 2)
        term_ac = mx.matmul(queries_p, keys_p_t)

        # Note: In JAX code, sl.TransformerXLRelativePositionEmbedding computes term_bd with
        # torch.einsum("BuwNH,FNH->BNuwF", ...), this converts that operation to a matmul to comply with the
        # Transformers style guide.
        queries_p_bd = queries.permute(0, 3, 1, 2, 4)
        sin_emb_p_t = sin_emb.permute(1, 2, 0)
        term_bd = mx.matmul(queries_p_bd, sin_emb_p_t)
        # Perform relative shift in order to get [B, N, U, W, C]
        # Pads the input to [B, N, U, W, C + 1]
        term_bd_pad = (0, c - lr, 0, 0, 0, 0, 0, 0, 0, 0)
        term_bd = mx.pad(term_bd, term_bd_pad)
        term_bd = term_bd.reshape((b, n, u, w * (c + 1)))
        term_bd = term_bd[:, :, :, : w * c]
        # Reshapes to [B, N, U, W, C]. Note the output last dim is 1-smaller
        # than the input, which "pushses" one element off to the next row for each
        # row. The accumulated effect is row_i is right-shifted i steps (i>=0).
        term_bd = term_bd.reshape((b, n, u, w, c))

        return term_ac + term_bd


class AudioAttention(nn.Module):
    def __init__(self, config: AudioConfig, *args, **kwargs):
        super().__init__()
        self.config = config

        self.num_heads = self.config.conf_num_attention_heads
        self.hidden_size = self.config.hidden_size
        self.units_per_head = self.hidden_size // self.num_heads
        self.chunk_size = self.config.conf_attention_chunk_size
        self.max_past_horizon = self.config.conf_attention_context_left
        self.max_future_horizon = self.config.conf_attention_context_right
        self.attention_invalid_logits_value = (
            self.config.conf_attention_invalid_logits_value
        )
        self.attention_logits_soft_cap = self.config.conf_attention_logit_cap

        if self.chunk_size < 1:
            raise ValueError(f"Expected {self.chunk_size=} >= 1.")
        if self.max_past_horizon < 1:
            raise ValueError(f"Expected {self.max_past_horizon=} >= 1.")
        if self.max_future_horizon < 0:
            raise ValueError(f"Expected {self.max_future_horizon=} >= 0.")
        if self.max_future_horizon == 0 and self.max_past_horizon == 0:
            raise ValueError("max_horizon and max_future_horizon cannot both be 0.")
        if self.attention_logits_soft_cap < 0.0:
            raise ValueError(
                f"{self.attention_logits_soft_cap=} should be None or non-negative."
            )

        self.relative_position_embedding = Gemma3p5AudioRelativePositionEmbedding(
            config
        )
        self.per_dim_scale = mx.zeros((self.units_per_head,))
        self.qkv_linear = nn.Linear(
            self.hidden_size, 3 * self.num_heads * self.units_per_head, bias=False
        )

    def _pad_dim1(
        self,
        x: mx.array,
        dim10_val: int,
        dim11_val: int,
        padding_val: Union[bool, float] = 0.0,
    ) -> mx.array:
        padding_tuple = [0] * x.ndim * 2
        dim_idx_from_end = x.ndim - 2
        start_idx_for_dim = 2 * dim_idx_from_end
        padding_tuple[start_idx_for_dim] = dim10_val
        padding_tuple[start_idx_for_dim + 1] = dim11_val
        padding_tuple = tuple(padding_tuple)
        x = mx.pad(x, padding_tuple, mode="constant", constant_value=padding_val)
        return x

    def _convert_to_block(
        self, x: mx.array, padding_val: Union[bool, float] = 0.0
    ) -> mx.array:
        shape = x.shape
        b, t = shape[:2]
        num_blocks = (t + self.block_size - 1) // self.block_size

        if (padding_len := num_blocks * self.block_size - t) > 0:
            x = self._pad_dim1(x, 0, padding_len, padding_val)

        permute_dims = (b, num_blocks, self.block_size) + shape[2:]
        x = x.permute(permute_dims).contiguous()
        return x

    def _extract_block_context(
        self, x: mx.array, padding_val: Union[bool, float] = 0.0
    ) -> mx.array:
        x = self._pad_dim1(
            x,
            self.max_past_horizon,
            self.max_future_horizon + self.block_size + 1,
            padding_val,
        )

        outer_dims = x.shape[:1]
        inner_dims = x.shape[2:]

        target_dim = x.shape[1]
        frame_len = self.block_size + self.max_past_horizon + self.max_future_horizon
        frame_step = self.block_size

        output_size = target_dim - frame_len + 1
        num_frames = (output_size + frame_step - 1) // frame_step

        if not num_frames:
            return mx.zeros(
                outer_dims + (0, frame_len) + inner_dims, dtype=x.dtype, device=x.device
            )

        subframe_factor = math.gcd(frame_len, frame_step)
        padding_left = 0
        padding_right = 0

        if subframe_factor > 1:
            padding_right += -target_dim % frame_len

        x = self._pad_dim1(x, padding_left, padding_right, padding_val)

        if subframe_factor > 1:
            x = x.reshape(outer_dims + (-1, subframe_factor) + inner_dims)
            frame_len //= subframe_factor
            frame_step //= subframe_factor

        x = x.unfold(dimension=1, size=frame_len, step=frame_step)
        permute_dims = (
            tuple(range(len(outer_dims)))
            + (len(outer_dims),)
            + (x.ndim - 1,)
            + tuple(range(len(outer_dims) + 1, x.ndim - 1))
        )
        x = x.permute(*permute_dims).contiguous()
        return x

    def __call__(self, x: mx.array, mask: mx.array) -> mx.array:
        batch_dims = x.shape[:-1]
        qkv_projected: mx.Tensor = self.qkv_linear(x)
        qkv = qkv_projected.reshape(*batch_dims, 3, self.num_heads, self.units_per_head)

        q = mx.take_along_axis(qkv, indices=mx.array([0]), axis=2).astype(mx.float32)
        k = mx.take_along_axis(qkv, indices=mx.array([1]), axis=2).astype(mx.float32)
        v = mx.take_along_axis(qkv, indices=mx.array([2]), axis=2).astype(mx.float32)

        q_scale = 1 / math.sqrt(self.units_per_head)
        r_softplus_0 = (
            1.442695041  # Ported from JAX Sequence Layers; 1.0 / jax.nn.softplus(0.0)
        )
        q_scale = mx.array(q_scale * r_softplus_0, dtype=mx.float32)
        q = q * q_scale * nn.softplus(self.per_dim_scale)

        batch_size, q_time = q.shape[:2]
        context_size = self.block_size + self.max_past_horizon + self.max_future_horizon

        k_blocks = self._extract_block_context(k)
        q_blocks = self._convert_to_block(q)
        num_query_blocks = q_blocks.shape[1]
        v_blocks = self._extract_block_context(v)

        valid_mask_blocks: mx.array = self._extract_block_context(
            mask, padding_val=False
        )
        valid_mask_blocks = mx.expand_dims(valid_mask_blocks, axis=1)
        valid_mask_blocks = mx.expand_dims(valid_mask_blocks, axis=-2)
        lower_causal_mask = mx.tril(
            mx.ones((context_size, self.block_size), dtype=mx.bool_),
            diagonal=0,
        ).T
        upper_causal_mask = mx.tril(
            mx.ones((self.block_size, context_size), dtype=mx.bool_),
            diagonal=self.max_past_horizon + self.max_future_horizon,
        )
        local_causal_valid_mask = mx.ones(
            (self.block_size, context_size), dtype=mx.bool_
        )
        local_causal_valid_mask = (
            local_causal_valid_mask * lower_causal_mask * upper_causal_mask
        )
        valid_mask_blocks = mx.logical_and(valid_mask_blocks, local_causal_valid_mask)

        # Embed queries and keys
        logits = self.relative_position_embedding(q_blocks, k_blocks)

        # Apply attention logit softcap
        softcap = mx.array(self.attention_logits_soft_cap, dtype=mx.float32)
        logits = logits / softcap
        logits = mx.tanh(logits)
        logits = logits * softcap

        logits = mx.where(
            valid_mask_blocks, logits, self.attention_invalid_logits_value
        )
        probabilities = mx.softmax(logits, dim=-1, dtype=mx.float32)
        # Note: In JAX code, sl.LocalDotProductSelfAttention computes context_vectors with
        # torch.einsum("BNuwc,BucNH->BuwNH", ...), this implementation converts from einsums to matmul operations to
        # comply with the Transformers style guide.
        b_dim, n_dim, u_dim, w_dim, c_dim = probabilities.shape
        h_dim = v_blocks.shape[-1]
        prob_bun = probabilities.transpose(0, 2, 1, 3, 4).reshape(-1, w_dim, c_dim)
        v_bun = v_blocks.transpose(0, 1, 3, 2, 4).reshape(-1, c_dim, h_dim)
        result_bmm = mx.matmul(prob_bun, v_bun)
        context_vectors = result_bmm.reshape(b_dim, u_dim, n_dim, w_dim, h_dim).transpose(
            0, 1, 3, 2, 4
        )
        context_vectors = context_vectors.reshape(
            (
                batch_size,
                num_query_blocks * self.chunk_size,
                self.num_heads,
                self.units_per_head,
            )
        )
        context_vectors = context_vectors[:, :q_time]

        return context_vectors, mask


class Gemma3p5AudioSSCPConvBlock(nn.Module):
    def __init__(self, config: AudioConfig, idx: int, *args, **kwargs):
        super().__init__()
        self.config = config

        self.out_channels = self.config.sscp_conv_channel_size[idx]
        self.kernel_size = self.config.sscp_conv_kernel_size[idx]
        self.stride = self.config.sscp_conv_stride_size[idx]

        # input_channels is equal to either the out_channels from the prior
        # Conv2d or 1 if this is the first Conv2d.
        if idx > 0:
            self.in_channels = self.config.sscp_conv_channel_size[idx - 1]
        else:
            self.in_channels = 1

        self.conv = nn.Conv2d(
            in_channels=self.in_channels,
            out_channels=self.out_channels,
            kernel_size=self.kernel_size,
            stride=self.stride,
            padding=(0, 0),
            bias=False,
        )
        self.norm = nn.GroupNorm(
            num_groups=1,
            dims=self.out_channels,
            eps=self.config.sscp_conv_group_norm_eps,
            pytorch_compatible=True,
        )

    def __call__(self, x: mx.array) -> mx.array:
        x = self.conv(x)
        x = self.norm(x)
        return mx.relu(x)


class Gemma3p5AudioSubSampleConvProjection(nn.Module):

    def __init__(self, config: AudioConfig, *args, **kwargs):
        super().__init__()
        self.config = config

        self.input_proj_in_shape = (
            self.config.sscp_conv_channel_size[1],
            self.config.sscp_conv_channel_size[1],
        )
        self.input_proj_in_features = np.prod(self.input_proj_in_shape)

        self.conv_0 = Gemma3p5AudioSSCPConvBlock(config, 0)
        self.conv_1 = Gemma3p5AudioSSCPConvBlock(config, 1)
        self.input_proj_linear = nn.Linear(
            self.input_proj_in_features, self.config.hidden_size, bias=False
        )

    def __call__(self, x: mx.array) -> mx.array:
        x = mx.expand_dims(x, axis=-1)
        x = self.conv_0(x)
        x = self.conv_1(x)
        batch_dims = x.shape[: -len(self.input_proj_in_shape)]
        x_flat = x.reshape(-1, self.input_proj_in_features)
        x_proj: mx.array = self.input_proj_linear(x_flat)
        return x_proj.reshape(*batch_dims, self.config.hidden_size)


class Gemma3p5AudioConformerAttention(nn.Module):
    def __init__(self, config: AudioConfig, *args, **kwargs):
        super().__init__()
        self.config = config

        unit_per_head = self.config.hidden_size // self.config.conf_num_attention_heads
        self.post_in_shape = (self.config.conf_num_attention_heads, unit_per_head)
        self.post_in_features = np.prod(self.post_in_shape)

        self.pre_attn_norm = Gemma3p5RMSNorm(self.config.hidden_size)
        self.attn = AudioAttention(config)
        self.post = nn.Linear(
            self.post_in_features, self.config.hidden_size, bias=False
        )
        self.post_norm = Gemma3p5RMSNorm(self.config.hidden_size)

    def __call__(self, x: mx.array, mask: mx.array) -> mx.array:
        x = self.pre_attn_norm(x)
        x = self.attn(x, mask)

        batch_dims = x.shape[: -len(self.post_in_shape)]
        x_flat = x.reshape(-1, self.post_in_features)
        output_flat: mx.array = self.post(x_flat)
        x = output_flat.reshape(*batch_dims, self.config.hidden_size)

        return self.post_norm(x)


class Gemma3p5AudioConformerFeedForward(nn.Module):
    def __init__(self, config: AudioConfig, *args, **kwargs):
        super().__init__()
        self.config = config

        self.pre_layer_norm = Gemma3p5RMSNorm(self.config.hidden_size)
        self.ffw_layer_1 = nn.Linear(
            self.config.hidden_size, self.config.hidden_size * 4, bias=False
        )
        self.ffw_layer_2 = nn.Linear(
            self.config.hidden_size * 4, self.config.hidden_size, bias=False
        )
        self.post_layer_norm = Gemma3p5RMSNorm(self.config.hidden_size)
        self.post_layer_scale = mx.array(self.config.conf_residual_weight)

    def __call__(self, x: mx.array) -> Tuple[mx.array, mx.array]:
        x = self.pre_layer_norm(x)
        x = self.ffw_layer_1(x)
        x = self.ffw_layer_2(x)
        x = self.pre_layer_norm(x)
        x *= self.post_layer_scale
        return x


class Gemma3p5AudioConformerLightConv1d(nn.Module):
    def __init__(self, config: AudioConfig, *args, **kwargs):
        super().__init__()
        self.config = config

        self.pre_layer_norm = Gemma3p5RMSNorm(self.config.hidden_size)
        self.linear_start = nn.Linear(
            self.config.hidden_size, self.config.hidden_size * 2, bias=False
        )
        self.depthwise_conv1d = nn.Conv1d(
            in_channels=self.config.hidden_size,
            out_channels=self.config.hidden_size,
            kernel_size=self.config.conf_conv_kernel_size,
            stride=1,
            padding=0,  # Manual causal padding
            groups=self.config.hidden_size,  # Depthwise
            bias=False,
        )
        self.conv_norm = Gemma3p5RMSNorm(self.config.hidden_size)
        self.linear_end = nn.Linear(
            self.config.hidden_size, self.config.hidden_size, bias=False
        )

    def __call__(self, x: mx.array) -> mx.array:
        x = self.pre_layer_norm(x)
        x = self.linear_start(x)
        x = nn.glu(x, axis=-1)
        x = self.depthwise_conv1d(x)
        x = self.conv_norm(x)
        x = nn.silu(x)
        return self.linear_end(x)


class Gemma3p5AudioConformerBlock(nn.Module):

    def __init__(self, config: AudioConfig, *args, **kwargs):
        super().__init__()
        self.config = config

        self.ffw_layer_start = Gemma3p5AudioConformerFeedForward(self.config)
        self.attention = Gemma3p5AudioConformerAttention(self.config)
        self.lconv1d = Gemma3p5AudioConformerLightConv1d(self.config)
        self.ffw_layer_end = Gemma3p5AudioConformerFeedForward(self.config)
        self.norm = Gemma3p5RMSNorm(self.config.hidden_size)

    def __call__(self, x: mx.array, mask: mx.array) -> mx.array:
        x = self.ffw_layer_start(x)
        x = self.attention(x, mask)
        x = self.lconv1d(x)
        x = self.ffw_layer_end(x)
        x = self.norm(x)
        return x


class AudioModel(nn.Module):
    def __init__(self, config: AudioConfig, *args, **kwargs):
        super().__init__()
        self.config = config

        self.subsample_conv_projection = Gemma3p5AudioSubSampleConvProjection(config)
        self.conformer = [
            Gemma3p5AudioConformerBlock(config)
            for _ in range(config.conf_num_hidden_layers)
        ]


    def __call__(self, x: mx.array, mask: mx.array) -> Tuple[mx.array, mx.array]:
        x = self.subsample_conv_projection(x)
        for block in self.conformer:
            x = block(x, mask)

        if self.config.conf_reduction_factor > 1:
            x = x[:, :: self.config.conf_reduction_factor]
            mask = mask[:, :: self.config.conf_reduction_factor]

        expanded_mask = mask.expand_dims(-1)
        fill_value = mx.array(0.0, dtype=x.dtype)
        x = mx.where(expanded_mask, fill_value, x)

        return x, mask
