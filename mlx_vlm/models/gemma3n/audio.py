import math
from typing import Optional, Tuple, Union

import mlx.core as mx
import mlx.nn as nn

from ..base import check_array_shape
from .config import AudioConfig, ModelConfig
from .language import Gemma3nRMSNorm


def convert_torch_to_mlx_pad_width(padding, input_shape):
    """Convert PyTorch padding to MLX pad_width format"""
    ndim = len(input_shape)

    # Initialize with no padding for all dimensions
    pad_width = [(0, 0)] * ndim

    # Set padding only for the dimensions that exist in the input
    # PyTorch p2d format: (left, right, top, bottom, front, back, ...)
    # For 2D tensor with padding (12, 11, 0, 0):
    # - Last dim gets (left=12, right=11)
    # - Second to last dim gets (top=0, bottom=0)

    if ndim >= 1 and len(padding) >= 2:
        # Last dimension
        pad_width[-1] = (padding[0], padding[1])
    if ndim >= 2 and len(padding) >= 4:
        # Second to last dimension
        pad_width[-2] = (padding[2], padding[3])
    if ndim >= 3 and len(padding) >= 6:
        # Third to last dimension
        pad_width[-3] = (padding[4], padding[5])
    if ndim >= 4 and len(padding) >= 8:
        # Fourth to last dimension
        pad_width[-4] = (padding[6], padding[7])

    return pad_width


class Gemma3nAudioRelativePositionEmbedding(nn.Module):

    def __init__(self, config: AudioConfig, *args, **kwargs):
        super().__init__()
        self.config = config

        self.num_heads = self.config.conf_num_attention_heads
        self.channels = self.config.hidden_size
        self.head_dim = self.channels // self.num_heads
        self.max_backward = (
            self.config.conf_attention_context_left - 1
            if self.config.conf_attention_context_left > 0
            else 0
        )
        self.max_forward = self.config.conf_attention_context_right

        self.pos_proj = nn.Linear(
            self.channels, self.num_heads * self.head_dim, bias=False
        )

        min_timescale = 1.0
        max_timescale = 1.0e4
        num_timescales = self.channels // 2
        log_timescale_increment = math.log(
            float(max_timescale) / float(min_timescale)
        ) / max(num_timescales - 1, 1)
        inv_timescales = min_timescale * mx.exp(
            mx.arange(num_timescales) * -log_timescale_increment
        )

        self._inv_timescales = mx.array(inv_timescales)[None, None, ...]

    def _get_timing_signal_1d_pos(self, position: mx.array, dtype) -> mx.array:
        assert position.ndim == 2
        position = mx.expand_dims(position.astype(mx.float32), axis=-1)

        scaled_time = position * self._inv_timescales
        timing_signal = mx.concatenate(
            [mx.sin(scaled_time), mx.cos(scaled_time)], axis=-1
        )
        return timing_signal.astype(dtype)

    def _relative_shift(
        self,
        term_bd_before_shift: mx.array,
        batch_size: int,
        num_heads: int,
        num_query_blocks: int,
        query_block_size: int,
        key_context_size: int,
        max_span_plus_1: int,
    ) -> mx.array:
        pad_amount_last_dim = (key_context_size + 1) - max_span_plus_1

        # We only pad the last dimension on the right.
        padding_tuple = (0, pad_amount_last_dim)

        term_bd_padded = mx.pad(
            term_bd_before_shift,
            convert_torch_to_mlx_pad_width(padding_tuple, term_bd_before_shift.shape),
        )
        # Shape after pad: [B, N, U, W, C+1]
        # Reshape for slicing (emulating JAX's behavior)
        # [B, N, U, W * (C+1)]
        term_bd_reshaped = term_bd_padded.reshape(
            (
                batch_size,
                num_heads,
                num_query_blocks,
                query_block_size * (key_context_size + 1),
            )
        )

        # Slice to effective [B, N, U, W * C]
        term_bd_sliced = term_bd_reshaped[
            :, :, :, : query_block_size * key_context_size
        ]

        # Reshape back to [B, N, U, W, C]
        term_bd_shifted = term_bd_sliced.reshape(
            (
                batch_size,
                num_heads,
                num_query_blocks,
                query_block_size,
                key_context_size,
            )
        )
        return term_bd_shifted

    def __call__(self, queries: mx.array, keys: mx.array) -> mx.array:
        # queries: [B, U, W, N, H] (batch, num_query_blocks, query_block_size, num_heads, head_dim)
        # keys:    [B, U, C, N, H] (batch, num_query_blocks, key_context_size, num_heads, head_dim)
        # C = W + L + R (key_context_size)
        # F_span = L + R + 1 (max_span + 1)

        batch_size, num_query_blocks, query_block_size, num_heads, head_dim = (
            queries.shape
        )
        _, _, key_context_size, _, _ = keys.shape

        # Relative positions for sinusoidal embeddings: [L, L-1, ..., -R]
        # Length is L+R+1 = self.max_span + 1
        pos_indices = mx.expand_dims(
            mx.arange(self.max_backward, -self.max_forward - 1, -1), axis=0
        )  # Shape [1, F_span]

        max_span_plus_1 = pos_indices.shape[1]  # F_span

        sin_emb_timing_signal = self._get_timing_signal_1d_pos(
            pos_indices, dtype=queries.dtype
        )  # Shape [1, F_span, self.channels]

        # Project sinusoidal embeddings: [1, F_span, self.channels] -> [1, F_span, N*H]
        projected_sin_emb = self.pos_proj(sin_emb_timing_signal)
        # Reshape to [1, F_span, N, H] then squeeze to [F_span, N, H]
        sin_emb = projected_sin_emb.reshape(
            1, max_span_plus_1, self.num_heads, self.head_dim
        ).squeeze(
            0
        )  # Shape [F, N, H]

        # term_ac: Query-Key content interaction
        # queries: [B, U, W, N, H] -> transpose to [B, N, U, W, H] for matmul
        # keys:    [B, U, C, N, H] -> transpose to [B, N, U, H, C] for matmul
        queries_p = queries.transpose(0, 3, 1, 2, 4)  # [B, N, U, W, H]
        keys_p_t = keys.transpose(0, 3, 1, 4, 2)  # [B, N, U, H, C]
        term_ac = mx.matmul(queries_p, keys_p_t)  # [B, N, U, W, C]

        # term_bd: Query-Position interaction
        # Original einsum: term_bd_unshifed = mx.einsum('buwnh,fnh->bnuwf', queries, sin_emb)
        # queries shape: [B, U, W, N, H]
        # sin_emb shape: [F, N, H]
        # Target output shape: [B, N, U, W, F]

        # Transpose queries to [B, N, U, W, H] for easier broadcasting with sin_emb
        q_transposed = queries.transpose(0, 3, 1, 2, 4)

        # Permute sin_emb to [N, H, F] to prepare for matmul
        # sin_emb original is [F, N, H]
        s_transposed = sin_emb.transpose(1, 2, 0)  # Shape: [N, H, F]

        # Reshape queries for matmul: [B, N, U*W, H]
        q_reshaped = q_transposed.reshape(
            batch_size, num_heads, num_query_blocks * query_block_size, head_dim
        )

        # Perform matmul: [B, N, U*W, H] @ [N, H, F]
        # s_permuted ([N, H, F]) will be broadcast to [B, N, H, F]
        # Result: [B, N, U*W, F]
        term_bd_unshifed_matmul = mx.matmul(q_reshaped, s_transposed)

        # Reshape to target [B, N, U, W, F]
        term_bd_unshifed = term_bd_unshifed_matmul.reshape(
            batch_size,
            num_heads,
            num_query_blocks,
            query_block_size,
            max_span_plus_1,
        )

        # Apply relative shift to term_bd_unshifed
        term_bd_shifted = self._relative_shift(
            term_bd_unshifed,
            batch_size,
            num_heads,
            num_query_blocks,
            query_block_size,
            key_context_size,
            max_span_plus_1,
        )  # Shape [B, N, U, W, C]

        return term_ac + term_bd_shifted


class Gemma3nAudioAttention(nn.Module):
    def __init__(self, config: AudioConfig, *args, **kwargs):
        super().__init__()
        self.config = config

        self.num_heads = self.config.conf_num_attention_heads
        self.hidden_size = self.config.hidden_size
        self.head_dim = self.hidden_size // self.num_heads

        self.chunk_size = self.config.conf_attention_chunk_size
        self.max_future_horizon = self.config.conf_attention_context_right
        self.max_past_horizon = max(0, self.config.conf_attention_context_left - 1)
        self.attention_invalid_logits_value = (
            self.config.conf_attention_invalid_logits_value
        )
        self.attention_logits_soft_cap = self.config.conf_attention_logit_cap
        self.context_size = (
            self.chunk_size + self.max_past_horizon + self.max_future_horizon
        )

        self.relative_position_embedding = Gemma3nAudioRelativePositionEmbedding(config)
        self.per_dim_scale = mx.zeros((self.head_dim,))

        self.q_proj = nn.Linear(
            self.hidden_size, self.num_heads * self.head_dim, bias=False
        )
        self.k_proj = nn.Linear(
            self.hidden_size, self.num_heads * self.head_dim, bias=False
        )
        self.v_proj = nn.Linear(
            self.hidden_size, self.num_heads * self.head_dim, bias=False
        )

        q_scale = self.head_dim**-0.5
        # Fix: Implement softplus manually since nn.softplus doesn't exist in MLX
        # softplus(x) = log(1 + exp(x))
        r_softplus_0 = 1.0 / mx.log(2.0)
        self._q_scale = q_scale * r_softplus_0

        lower_causal_mask = mx.tril(
            mx.ones((self.context_size, self.chunk_size), dtype=mx.bool_),
            k=0,
        ).T
        upper_causal_mask = mx.tril(
            mx.ones((self.chunk_size, self.context_size), dtype=mx.bool_),
            k=self.max_past_horizon + self.max_future_horizon,
        )
        local_causal_valid_mask = mx.ones(
            (self.chunk_size, self.context_size), dtype=mx.bool_
        )
        local_causal_valid_mask = (
            local_causal_valid_mask * lower_causal_mask * upper_causal_mask
        )
        self._local_causal_valid_mask = local_causal_valid_mask

        self._softcap = mx.array(self.attention_logits_soft_cap, dtype=mx.float32)

    def _pad_dim1(
        self,
        x: mx.array,
        dim10_val: int,
        dim11_val: int,
    ) -> mx.array:
        padding_tuple = [0] * x.ndim * 2
        dim_idx_from_end = x.ndim - 2
        start_idx_for_dim = 2 * dim_idx_from_end
        padding_tuple[start_idx_for_dim] = dim10_val
        padding_tuple[start_idx_for_dim + 1] = dim11_val

        return mx.pad(x, convert_torch_to_mlx_pad_width(tuple(padding_tuple), x.shape))

    def _convert_to_block(
        self, x: mx.array, padding_val: Union[bool, float] = 0.0
    ) -> mx.array:
        shape = x.shape
        b, t = shape[:2]
        num_blocks = (t + self.chunk_size - 1) // self.chunk_size

        if (padding_len := num_blocks * self.chunk_size - t) > 0:
            x = self._pad_dim1(x, 0, padding_len)

        permute_dims = (b, num_blocks, self.chunk_size) + shape[2:]
        return x.reshape(permute_dims)

    def unfold_mlx(self, x, dimension, size, step):
        # Get the shape and determine the number of windows
        shape = x.shape
        dim_size = shape[dimension]
        num_windows = (dim_size - size) // step + 1

        # Create indices for each window
        windows = []
        for i in range(num_windows):
            start_idx = i * step
            end_idx = start_idx + size

            # Create slice objects for all dimensions
            slices = [slice(None)] * len(shape)
            slices[dimension] = slice(start_idx, end_idx)

            windows.append(x[tuple(slices)])

        # Stack along a new dimension
        return mx.stack(windows, axis=dimension + 1)

    def _extract_block_context(self, x: mx.array) -> mx.array:
        pad_left = self.max_past_horizon

        pad_right = self.max_future_horizon + self.chunk_size - 1
        x = self._pad_dim1(x, pad_left, pad_right)

        frame_len = self.context_size
        frame_step = self.chunk_size
        # Create windows using sliding window approach for MLX
        # x shape: (batch, time, ...)
        batch_size = x.shape[0]
        time_dim = x.shape[1]
        other_dims = x.shape[2:]

        x_unfolded = self.unfold_mlx(x, 1, frame_len, frame_step)

        if x.ndim > 2 and x_unfolded.ndim > 3:
            x_unfolded = x_unfolded.transpose(0, 2, 1, 3, 4)

        return x_unfolded

    def __call__(self, x: mx.array, mask: mx.array) -> mx.array:
        query_states = self.q_proj(x).reshape(
            *x.shape[:-1], self.num_heads, self.head_dim
        )
        key_states = self.k_proj(x).reshape(
            *x.shape[:-1], self.num_heads, self.head_dim
        )
        value_states = self.v_proj(x).reshape(
            *x.shape[:-1], self.num_heads, self.head_dim
        )

        per_dim_scale_sp = mx.logaddexp(self.per_dim_scale, 0.0)

        broadcast_shape = (1, 1, 1, self.head_dim)
        per_dim_scale_sp_broadcast = per_dim_scale_sp.reshape(broadcast_shape)
        query_states = query_states * self._q_scale * per_dim_scale_sp_broadcast

        batch_size, q_time = query_states.shape[:2]

        query_blocks = self._convert_to_block(query_states)
        key_blocks = self._extract_block_context(key_states)
        value_blocks = self._extract_block_context(value_states)
        num_query_blocks = query_blocks.shape[1]

        # 1. Create a mask indicating originally valid positions.
        original_valid_mask = ~mask  # True for valid, False for padded

        # 2. Extract blocks from this validity mask.
        extracted_valid_mask_blocks = self._extract_block_context(
            original_valid_mask
        ).transpose(0, 2, 1)

        # If subframe_factor was used in _extract_block_context for a [B, T] input mask,
        # the shape might be [B, U, C/SF, SF]. Reshape to [B, U, C].
        # batch_size and num_query_blocks are known from query_blocks.
        # self.context_size is C.
        if (
            extracted_valid_mask_blocks.ndim == 4
            and extracted_valid_mask_blocks.shape[0] == batch_size
            and extracted_valid_mask_blocks.shape[1] == num_query_blocks
            and extracted_valid_mask_blocks.shape[2]
            * extracted_valid_mask_blocks.shape[3]
            == self.context_size
        ):
            extracted_valid_mask_blocks = extracted_valid_mask_blocks.reshape(
                batch_size, num_query_blocks, self.context_size
            )
        # After potential reshape, ensure it's [B, U, C] if it was from a [B,T] mask.
        # This assertion might be too strict if _extract_block_context handles higher-rank inputs differently,
        # but for the mask case, this should hold.
        if extracted_valid_mask_blocks.shape != (
            batch_size,
            num_query_blocks,
            self.context_size,
        ):
            raise ValueError(
                "Shape of extracted_valid_mask_blocks"
                f" {extracted_valid_mask_blocks.shape} is not ({batch_size},"
                f" {num_query_blocks}, {self.context_size}) after potential reshape."
            )

        # 3. Expand dimensions for broadcasting with logits and causal mask.
        # Target shape for broadcasting with logits [B,N,U,W,C]
        # extracted_valid_mask_blocks to [B, 1, U, 1, C]
        condition_from_input_validity = mx.expand_dims(
            extracted_valid_mask_blocks, axis=1
        )
        condition_from_input_validity = mx.expand_dims(
            condition_from_input_validity, axis=-2
        )

        # self.local_causal_valid_mask is [W, C], True where allowed by local window.
        # Expand to [1, 1, 1, W, C]
        condition_from_causality = self._local_causal_valid_mask[None, None, None, ...]

        # 4. Combine the two conditions.
        # final_condition will be True where a key is *both* originally valid *and* causally accessible.
        # Broadcasts to [B, 1, U, W, C]
        final_condition_for_where = mx.logical_and(
            condition_from_input_validity,
            condition_from_causality,  # Ensure same device
        )

        # Embed queries and keys
        logits = self.relative_position_embedding(query_blocks, key_blocks)

        # Apply attention logit softcap
        # Ensure softcap is on the same device as logits
        logits = logits / self._softcap
        logits = nn.tanh(logits)
        logits = logits * self._softcap

        # Apply the combined mask.
        # final_condition_for_where will broadcast with logits [B,N,U,W,C]
        logits = mx.where(
            final_condition_for_where, logits, self.attention_invalid_logits_value
        )
        probabilities = mx.softmax(logits.astype(mx.float32), axis=-1).astype(
            value_blocks.dtype
        )

        # context_vectors is adapted from jax.numpy.einsum("BNuwc,BucNH->BuwNH", ...)
        b_dim, n_dim, u_dim, w_dim, c_dim = probabilities.shape
        h_dim = value_blocks.shape[-1]
        prob_bun = probabilities.transpose(0, 2, 1, 3, 4).reshape(-1, w_dim, c_dim)
        v_bun = value_blocks.transpose(0, 1, 3, 2, 4).reshape(-1, c_dim, h_dim)
        result_bmm = mx.matmul(prob_bun, v_bun)
        context_vectors = result_bmm.reshape(
            b_dim, u_dim, n_dim, w_dim, h_dim
        ).transpose(0, 1, 3, 2, 4)
        context_vectors = context_vectors.reshape(
            (
                batch_size,
                num_query_blocks * self.chunk_size,
                self.num_heads,
                self.head_dim,
            )
        )
        context_vectors = context_vectors[:, :q_time]

        return context_vectors


class Gemma3nCumulativeGroupNorm(nn.Module):
    """Applies Group Normalization cumulatively over the time dimension.

    This layer normalizes the input by calculating the mean and variance
    cumulatively over the time dimension (dim 1). The statistics are computed
    over all feature dimensions (specified by `feature_dims` and `num_channels`)
    for elements marked as valid by the optional `mask`.

    If a `mask` is provided (True for valid, False for invalid/padded),
    invalid time steps do not contribute to the statistics calculation, and
    their corresponding output values are zeroed out.

    Scale and bias, if enabled, are applied per-channel (last dimension).
    This behavior is similar to JAX's `GroupNormalization` with `num_groups=1`
    and `cumulative=True`.
    """

    def __init__(
        self,
        num_channels: int,  # Number of channels (size of the last dimension)
        feature_dims: Tuple[
            int
        ],  # Sizes of non-channel feature dimensions, e.g., (H, W) for input [B,T,H,W,C]
        eps: float = 1e-3,
        use_scale: bool = True,
        use_bias: bool = False,
    ):
        super().__init__()
        self.num_channels = num_channels
        self.feature_dims = tuple(feature_dims)
        self.eps = eps
        self.use_scale = use_scale
        self.use_bias = use_bias

        if self.use_scale:
            # Scale parameter depends only on the channel dimension
            self.weight = mx.ones(num_channels)
        else:
            self.weight = None

        if self.use_bias:
            # Bias parameter depends only on the channel dimension
            self.bias = mx.zeros(num_channels)
        else:
            self.bias = None

        # Axes for normalization: all dimensions except Batch (0) and Time (1).
        # For input [B, T, *feature_dims, C], these are dims from 2 onwards.
        self.reduction_axes = tuple(range(2, 2 + len(self.feature_dims) + 1))

    def __call__(self, x: mx.array, mask: Optional[mx.array] = None) -> mx.array:
        """Applies cumulative group norm, optionally using a mask.

        Args:
          x: Input tensor, shape [B, T, *feature_dims, C].
          mask: Optional boolean mask, shape [B, T]. True indicates a valid
            (non-padded) time step. If None, all time steps are considered valid.

        Returns:
          Normalized tensor with the same shape as x.
        """
        expected_input_suffix = self.feature_dims + (self.num_channels,)
        if x.shape[2:] != expected_input_suffix:
            raise ValueError(
                f"Input tensor shape suffix {x.shape[2:]} does not match expected"
                f" suffix (feature_dims + num_channels) {expected_input_suffix}"
            )

        if mask is not None:
            if mask.shape != x.shape[:2]:
                raise ValueError(
                    f"Mask shape {mask.shape} must match input Batch/Time dimensions {x.shape[:2]}"
                )
            if mask.dtype != mx.bool:
                raise TypeError("Mask must be a boolean tensor.")

        input_dtype = x.dtype
        # Calculations are performed in float32 for numerical stability.
        calc_dtype = mx.float32
        x_calc = x.astype(calc_dtype)

        # Prepare a broadcastable mask (`mask_calc`).
        # If no mask is provided, treat all elements as valid
        # (mask_calc is all ones).
        # Otherwise, expand the [B, T] mask to [B, T, 1, ..., 1] for broadcasting.
        if mask is not None:
            mask_suffix_shape = (1,) * len(expected_input_suffix)
            mask_calc = mask.reshape(mask.shape + mask_suffix_shape).astype(calc_dtype)
        else:
            mask_calc = mx.ones_like(x_calc).astype(calc_dtype)

        # Mask the input for sum calculation: only valid elements contribute.
        x_masked_for_sum = x_calc * mask_calc

        # Cumulative Statistics Calculation
        # 1. Sum of values over reduction axes at each time step.
        sum_values_at_t = mx.sum(
            x_masked_for_sum, axis=self.reduction_axes, keepdims=True
        )
        # 2. Cumulative sum of values over time.
        cum_sum_values = mx.cumsum(sum_values_at_t, axis=1)

        # 3. Count of valid elements in the normalization group at each time step.
        #    (A "group" here consists of all features at a given Batch, Time).
        elements_in_group_at_t = mx.sum(
            mask_calc, axis=self.reduction_axes, keepdims=True
        )
        # 4. Cumulative count of valid elements over time.
        cum_count_elements = mx.cumsum(elements_in_group_at_t, axis=1)
        # Avoid division by zero if all preceding elements were masked.
        safe_cum_count_elements = mx.clip(cum_count_elements, 1, None)

        # 5. Cumulative mean.
        cum_mean = cum_sum_values / safe_cum_count_elements

        # 6. Sum of squared differences from the cumulative mean.
        #    Only sum for valid elements: (x_calc - cum_mean)^2 * mask_calc.
        #    Using x_calc here for the difference, as cum_mean already accounts for masking.
        squared_diff_from_mean = (x_calc - cum_mean) ** 2
        sum_sq_diff_at_t = mx.sum(
            squared_diff_from_mean * mask_calc,
            axis=self.reduction_axes,
            keepdims=True,
        )
        # 7. Cumulative sum of squared differences over time.
        cum_sum_sq_diff = mx.cumsum(sum_sq_diff_at_t, axis=1)

        # 8. Cumulative variance.
        cum_variance = cum_sum_sq_diff / safe_cum_count_elements

        # Normalize the input using the calculated cumulative statistics:
        # (x - E[x]) / sqrt(Var[x] + eps)
        normalized_x = (x_calc - cum_mean) * mx.rsqrt(cum_variance + self.eps)

        # Apply affine transformation (scale and bias) if enabled.
        # Scale and bias are applied per-channel (last dimension).
        if self.use_scale and self.weight is not None:
            scale = self.weight.astype(calc_dtype)
            # Reshape for broadcasting: [C] -> [1, ..., 1, C]
            scale_view_shape = [1] * (x.ndim - 1) + [self.num_channels]
            normalized_x = normalized_x * scale.reshape(scale_view_shape)

        if self.use_bias and self.bias is not None:
            bias = self.bias.astype(calc_dtype)
            bias_view_shape = [1] * (x.ndim - 1) + [self.num_channels]
            normalized_x = normalized_x + bias.reshape(bias_view_shape)

        # Zero out outputs for time steps that were originally masked (where mask_calc is 0).
        # This ensures padded/invalid positions in the input result in zero output.
        final_output = normalized_x * mask_calc

        return final_output.astype(input_dtype)


class Gemma3nAudioSSCPConvBlock(nn.Module):
    def __init__(
        self,
        idx: int,
        input_freq_dim: int,
        config: AudioConfig,
        manual_padding: Tuple[int, int, int, int] = (0, 0, 0, 0),
        *args,
        **kwargs,
    ):
        super().__init__()
        self.config = config
        self.manual_padding = manual_padding

        # in_channels is 1 for the first block, or C_out from previous block's conv
        in_channels = 1 if idx == 0 else self.config.sscp_conv_channel_size[idx - 1]
        out_channels = self.config.sscp_conv_channel_size[idx]
        kernel_h, kernel_w = self.config.sscp_conv_kernel_size[idx]
        stride_h, stride_w = self.config.sscp_conv_stride_size[idx]

        self.conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=(
                kernel_h,
                kernel_w,
            ),  # Kernel (kH, kW) operates on (Time, Freq_dim)
            stride=(stride_h, stride_w),
            padding=(0, 0),  # Manual padding is used
            bias=False,
        )

        # Calculate output frequency dimension (f_out_conv) after this convolution.
        # input_freq_dim is the unpadded width (feature dimension).
        # self.manual_padding is (pad_F_left, pad_F_right, pad_T_top, pad_T_bottom)
        f_in_padded = (
            input_freq_dim
            + self.manual_padding[0]  # pad_F_left
            + self.manual_padding[1]  # pad_F_right
        )
        f_out_conv = (f_in_padded - kernel_w) // stride_w + 1

        self.norm = Gemma3nCumulativeGroupNorm(
            num_channels=out_channels,  # Channels of the conv output
            feature_dims=(f_out_conv,),  # The frequency dimension size after conv
            eps=self.config.sscp_conv_eps,
            use_scale=True,
            use_bias=False,
        )

    def __call__(self, x: mx.array) -> mx.array:
        # Input audio_encodings is [B, C_in, T_in, F_in] (e.g., C_in=1)
        # manual_padding is (pad_F_left, pad_F_right, pad_T_top, pad_T_bottom)
        # F.pad applies to last two dims: F_in then T_in

        audio_encodings_padded = mx.pad(
            x, convert_torch_to_mlx_pad_width(self.manual_padding, x.shape)
        )

        # Expected padded shape for F_in, k_w=3, pad_F=(1,1) -> F_padded = F_in+2
        # Expected padded shape for T_in, k_h=3, pad_T=(0,2) -> T_padded = T_in+2
        audio_encodings_conv = self.conv(audio_encodings_padded.transpose(0, 2, 3, 1))
        # Expected conv output shape: [B, C_out, T_out, F_out]
        # Input to norm is [B, T_out, F_out, C_out]
        x_normed = self.norm(audio_encodings_conv)
        # Output of norm is [B, T_out, F_out, C_out], permute back to [B, C_out, T_out, F_out]
        audio_encodings_normed = x_normed.transpose(0, 3, 1, 2)
        return nn.relu(audio_encodings_normed)


class Gemma3nAudioSubSampleConvProjection(nn.Module):

    def __init__(self, config: AudioConfig, *args, **kwargs):
        super().__init__()
        self.config = config

        current_f_for_block_input = (
            config.input_feat_size
        )  # Start with original feature dim
        calculated_block_padding = []
        calculated_f_out_dims = []  # Tracking frequency dimension output sizes

        for i in range(2):  # Assuming 2 conv layers as per sscp_conv_... arrays
            kernel_h, kernel_w = config.sscp_conv_kernel_size[i]
            stride_h, stride_w = config.sscp_conv_stride_size[i]
            # Assuming dilation rate of 1 for frequency dimension as it's not in config
            # effective_kernel_w = (kernel_w - 1) * dilation_w + 1 # Not needed if hardcoding freq padding

            # Padding for Time (Height for Conv2d) - REVERSE_CAUSAL like
            # JAX 'reverse_causal' padding is (0, kernel_size - 1)
            pad_t_top = 0
            pad_t_bottom = kernel_h - 1

            # Frequency Padding (Width for Conv2d)
            # Based on JAX effective padding (1,1) for F_in=10, K_w=3, S_w=2
            # and the successful test configuration.
            # If kernel/stride/input_freq for frequency changes, this might need re-evaluation
            # to match generic JAX 'SAME' behavior if it differs.
            pad_f_left = 1
            pad_f_right = 1

            manual_padding_tuple = (
                pad_f_left,
                pad_f_right,
                pad_t_top,
                pad_t_bottom,
            )
            calculated_block_padding.append(manual_padding_tuple)

            # Calculate output frequency dimension after this convolution
            # This uses the actual padding applied and kernel/stride.
            f_in_padded = current_f_for_block_input + pad_f_left + pad_f_right
            f_out_after_conv = (
                f_in_padded - kernel_w
            ) // stride_w + 1  # Assuming dilation_w = 1
            calculated_f_out_dims.append(f_out_after_conv)
            current_f_for_block_input = f_out_after_conv

        self.conv_0 = Gemma3nAudioSSCPConvBlock(
            idx=0,
            input_freq_dim=config.input_feat_size,  # Pass original feature dim
            config=config,
            manual_padding=calculated_block_padding[0],
        )
        self.conv_1 = Gemma3nAudioSSCPConvBlock(
            idx=1,
            input_freq_dim=calculated_f_out_dims[0],  # Output freq dim from conv_0
            config=config,
            manual_padding=calculated_block_padding[1],
        )
        final_c_out = config.sscp_conv_channel_size[-1]
        final_f_out = calculated_f_out_dims[-1]  # Final frequency dimension
        self.input_proj_in_features = final_c_out * final_f_out
        self.input_proj_linear = nn.Linear(
            self.input_proj_in_features, self.config.hidden_size, bias=False
        )

    def __call__(self, x: mx.array) -> mx.array:
        # audio_encodings is [B, T, F_in]
        # Reshape to [B, 1, T, F_in] (Batch, Channels=1, Height=Time, Width=F_in)
        audio_encodings_reshaped = mx.expand_dims(x, 1)
        x = self.conv_0(audio_encodings_reshaped)
        x = self.conv_1(x)
        # x from conv_1 is [B, C_out_1, T_out_1, F_out_1]
        b, c_out, t_out, f_out = x.shape
        # Permute to [B, T_out_1, F_out_1, C_out_1] then flatten F_out_1 and C_out_1
        x_transposed = x.transpose(0, 2, 3, 1)
        output_flattened = x_transposed.reshape(b, t_out, f_out * c_out)
        output = self.input_proj_linear(output_flattened)
        return output


class Gemma3nAudioConformerAttention(nn.Module):
    def __init__(self, config: AudioConfig, *args, **kwargs):
        super().__init__()
        self.config = config

        head_dim = self.config.hidden_size // self.config.conf_num_attention_heads
        self.post_in_shape = (self.config.conf_num_attention_heads, head_dim)
        self.post_in_features = self.config.hidden_size

        self._gradient_clipping = mx.array(self.config.gradient_clipping)

        self.pre_attn_norm = Gemma3nRMSNorm(self.config.hidden_size)
        self.attn = Gemma3nAudioAttention(config)
        self.post = nn.Linear(
            self.post_in_features, self.config.hidden_size, bias=False
        )
        self.post_norm = Gemma3nRMSNorm(self.config.hidden_size)

    def __call__(self, x: mx.array, mask: mx.array) -> mx.array:
        audio_encodings_input_to_attn = x
        x = mx.clip(x, -self._gradient_clipping, self._gradient_clipping)
        audio_encodings_norm = self.pre_attn_norm(x)
        # Output of self.attn is [B, T, NumHeads, HeadDim]
        audio_encodings_attn_out = self.attn(audio_encodings_norm, mask)

        # Reshape from [B, T, NumHeads, HeadDim] to [B, T, NumHeads * HeadDim]
        # NumHeads * HeadDim = hidden_size
        b, t, num_heads, head_dim = audio_encodings_attn_out.shape
        audio_encodings_reshaped = audio_encodings_attn_out.reshape(
            b, t, num_heads * head_dim
        )

        x = self.post(audio_encodings_reshaped)
        x = mx.clip(x, -self._gradient_clipping, self._gradient_clipping)
        return audio_encodings_input_to_attn + self.post_norm(x)


class Gemma3nAudioConformerFeedForward(nn.Module):
    def __init__(self, config: AudioConfig, *args, **kwargs):
        super().__init__()
        self.config = config

        self._gradient_clipping = mx.array(self.config.gradient_clipping)

        self.pre_layer_norm = Gemma3nRMSNorm(self.config.hidden_size)
        self.ffw_layer_1 = nn.Linear(
            self.config.hidden_size, self.config.hidden_size * 4, bias=False
        )
        self.ffw_layer_2 = nn.Linear(
            self.config.hidden_size * 4, self.config.hidden_size, bias=False
        )
        self.post_layer_norm = Gemma3nRMSNorm(self.config.hidden_size)
        self._post_layer_scale = mx.array(self.config.conf_residual_weight)

    def __call__(self, x: mx.array) -> mx.array:
        residual = x
        x = mx.clip(x, -self._gradient_clipping, self._gradient_clipping)
        x = self.pre_layer_norm(x)
        x: mx.array = self.ffw_layer_1(x)  # jax.numpy.einsum("...a,ab->...b")
        x = nn.silu(x)  # Add SiLU (Swish) activation
        x: mx.array = self.ffw_layer_2(x)  # jax.numpy.einsum("...a,ab->...b")
        x = mx.clip(x, -self._gradient_clipping, self._gradient_clipping)
        x = self.post_layer_norm(x)
        return residual + (x * self._post_layer_scale)


class Gemma3nAudioConformerLightConv1d(nn.Module):
    def __init__(self, config: AudioConfig, *args, **kwargs):
        super().__init__()
        self.config = config

        self.pre_layer_norm = Gemma3nRMSNorm(
            self.config.hidden_size, eps=self.config.rms_norm_eps
        )
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
        self._gradient_clipping = mx.array(self.config.gradient_clipping)
        self.conv_norm = Gemma3nRMSNorm(
            self.config.hidden_size, eps=self.config.rms_norm_eps
        )
        self.linear_end = nn.Linear(
            self.config.hidden_size, self.config.hidden_size, bias=False
        )

        self.causal_padding = self.config.conf_conv_kernel_size - 1

    def __call__(self, audio_encodings: mx.array) -> mx.array:
        audio_encodings_residual = audio_encodings  # Save for residual connection

        audio_encodings = self.pre_layer_norm(audio_encodings)
        audio_encodings = self.linear_start(audio_encodings)
        audio_encodings = nn.glu(audio_encodings, axis=-1)
        # Permute for Conv1d: [B, T, D] -> [B, D, T]
        audio_encodings_transposed = audio_encodings.transpose(0, 2, 1)
        # Apply manual causal padding
        audio_encodings_transposed_padded = mx.pad(
            audio_encodings_transposed,
            convert_torch_to_mlx_pad_width(
                (self.causal_padding, 0), audio_encodings_transposed.shape
            ),
        )
        audio_encodings = self.depthwise_conv1d(
            audio_encodings_transposed_padded.transpose(0, 2, 1)
        )
        audio_encodings = mx.clip(
            audio_encodings, -self._gradient_clipping, self._gradient_clipping
        )
        audio_encodings = self.conv_norm(audio_encodings)
        audio_encodings = nn.silu(audio_encodings)
        audio_encodings = self.linear_end(audio_encodings)
        output = audio_encodings + audio_encodings_residual
        return output


class Gemma3nAudioConformerBlock(nn.Module):

    def __init__(self, config: AudioConfig, *args, **kwargs):
        super().__init__()
        self.config = config

        self.ffw_layer_start = Gemma3nAudioConformerFeedForward(self.config)
        self.attention = Gemma3nAudioConformerAttention(self.config)
        self.lconv1d = Gemma3nAudioConformerLightConv1d(self.config)
        self.ffw_layer_end = Gemma3nAudioConformerFeedForward(self.config)
        self._gradient_clipping = mx.array(self.config.gradient_clipping)
        self.norm = Gemma3nRMSNorm(self.config.hidden_size)

    def __call__(self, audio_encodings: mx.array, audio_mel_mask: mx.array) -> mx.array:
        audio_encodings = self.ffw_layer_start(audio_encodings)
        audio_encodings = self.attention(audio_encodings, audio_mel_mask)
        validity_mask_for_lconv = ~audio_mel_mask  # True for valid
        audio_encodings_for_lconv_input = audio_encodings * mx.expand_dims(
            validity_mask_for_lconv, -1
        ).astype(audio_encodings.dtype)
        audio_encodings = self.lconv1d(audio_encodings_for_lconv_input)

        audio_encodings = self.ffw_layer_end(audio_encodings)
        audio_encodings = mx.clip(
            audio_encodings, -self._gradient_clipping, self._gradient_clipping
        )
        output = self.norm(audio_encodings)
        return output


class AudioModel(nn.Module):
    def __init__(self, config: AudioConfig, *args, **kwargs):
        super().__init__()
        self.config = config

        self.subsample_conv_projection = Gemma3nAudioSubSampleConvProjection(config)
        self.conformer = [
            Gemma3nAudioConformerBlock(config)
            for _ in range(config.conf_num_hidden_layers)
        ]

    def __call__(
        self, audio_mel: mx.array, audio_mel_mask: mx.array
    ) -> Tuple[mx.array, mx.array]:
        audio_encodings = self.subsample_conv_projection(
            audio_mel
        )  # audio_encodings: [B, T_sub, D]

        # Subsample the input audio_mel_mask to match the time dimension of audio_encodings (T_sub)
        t_sub = audio_encodings.shape[1]

        time_stride_product = 1
        for stride_pair_idx in range(len(self.config.sscp_conv_stride_size)):
            time_stride_product *= self.config.sscp_conv_stride_size[stride_pair_idx][0]

        # Create indices for gathering from the original mask.
        # These indices map to original time steps corresponding to the start of each
        # receptive field in the subsampled output.
        indices = mx.arange(t_sub) * time_stride_product
        indices = mx.clip(
            indices, None, a_max=audio_mel_mask.shape[1] - 1
        )  # Ensure indices are valid

        # Expand indices for batch compatibility if B > 1 and indices is 1D.
        if audio_mel_mask.ndim > 1 and indices.ndim == 1:
            indices = indices[None, :]
            indices = mx.broadcast_to(
                indices, (audio_mel_mask.shape[0], indices.shape[1])
            )  # [B, T_sub]
        elif (
            audio_mel_mask.ndim == indices.ndim
            and audio_mel_mask.shape[0] == 1
            and indices.shape[0] != 1
            and t_sub == indices.shape[0]
        ):
            # Handle case where B=1 but indices became [T_sub] instead of [1, T_sub]
            indices = indices[None, :]

        current_mask = mx.take_along_axis(audio_mel_mask, indices, axis=1)  # [B, T_sub]

        # Fallback: Ensure mask length matches feature length after gather.
        if current_mask.shape[1] != t_sub:
            print(
                "Warning: Subsampled mask length %s mismatch with feature length %s after gather. Adjusting.",
                current_mask.shape[1],
                t_sub,
            )
            if current_mask.shape[1] > t_sub:
                current_mask = current_mask[:, :t_sub]
            else:  # current_mask.shape[1] < t_sub
                padding_needed = t_sub - current_mask.shape[1]
                current_mask = mx.pad(
                    current_mask,
                    convert_torch_to_mlx_pad_width(
                        (0, padding_needed), current_mask.shape
                    ),
                )

        for i, block in enumerate(self.conformer):
            audio_encodings = block(
                audio_encodings, current_mask
            )  # Pass the processed mask

        if self.config.conf_reduction_factor > 1:
            audio_encodings = audio_encodings[:, :: self.config.conf_reduction_factor]
            # Reduce the mask as well
            current_mask = current_mask[:, :: self.config.conf_reduction_factor]

        # Final masking of audio_encodings based on the final current_mask
        # Ensure current_mask length matches the finally reduced audio_encodings length
        if current_mask.shape[1] != audio_encodings.shape[1]:
            target_len = audio_encodings.shape[1]
            mask_current_len = current_mask.shape[1]
            if target_len > mask_current_len:
                padding_needed = target_len - mask_current_len
                current_mask = mx.pad(
                    current_mask,
                    convert_torch_to_mlx_pad_width(
                        (0, padding_needed), current_mask.shape
                    ),
                )
            elif mask_current_len > target_len:  # mask is longer
                current_mask = current_mask[:, :target_len]

        audio_encodings = mx.where(current_mask[..., None], 0.0, audio_encodings)
        return audio_encodings, current_mask

    def sanitize(self, weights):
        sanitized_weights = {}
        for k, v in weights.items():
            if "conv.weight" in k:
                if check_array_shape(v):
                    sanitized_weights[k] = v
                else:
                    sanitized_weights[k] = v.transpose(0, 2, 3, 1)
            elif "conv1d.weight" in k:
                if check_array_shape(v):
                    sanitized_weights[k] = v
                else:
                    sanitized_weights[k] = v.transpose(0, 2, 1)
            else:
                sanitized_weights[k] = v

        return sanitized_weights
