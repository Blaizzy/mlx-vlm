import math
from typing import Any, Dict, Literal, Optional, Tuple, Union

import backoff
import mlx.core as mx
import mlx.nn as nn
import numpy as np

# Special token id for audio
_AUDIO_SPECIAL_TOKEN_ID = 200011  # '<|endoftext11|>'


@backoff.on_exception(backoff.expo, Exception, max_tries=10)
def np_loadtxt_with_retry(filepath):
    """np.loadtxt with retry
    Args:
        filepath: str
            file path to the numpy array.
    """
    result = np.loadtxt(filepath, dtype="f")
    return result


def check_array_shape(arr):
    shape = arr.shape

    # Check if the shape has 4 dimensions
    if len(shape) != 4:
        return False

    out_channels, kH, KW, _ = shape

    # Check if out_channels is the largest, and kH and KW are the same
    if (out_channels >= kH) and (out_channels >= KW) and (kH == KW):
        return True
    else:
        return False


class T5RelativeAttentionLogitBias(nn.Module):
    def __init__(self, num_heads, num_buckets=-1, max_distance=1000, symmetric=False):
        super().__init__()
        self.num_heads = num_heads
        self.num_buckets = num_buckets
        self.max_distance = max_distance
        self.symmetric = symmetric
        self._skip_bucketing = self.num_buckets < 0
        if self._skip_bucketing:
            self.num_buckets = max_distance
        else:
            raise NotImplementedError(
                "T5 attention bias with bucketed positions is not yet tested"
            )
        if not self.symmetric:
            self.num_buckets *= 2
        self.bias_values = nn.Embedding(self.num_buckets, self.num_heads)

    def forward(self, x):
        # instantiate bias compatible with shape of x
        maxpos = x.size(1)
        context_position = mx.arange(maxpos, dtype=mx.int64)[:, None]
        memory_position = mx.arange(maxpos, dtype=mx.int64)[None, :]
        relative_position = memory_position - context_position
        # clipping to a maximum distance using ops that play well with ONNX export
        relative_position = relative_position.masked_fill(
            relative_position < -self.max_distance, -self.max_distance
        )
        relative_position = relative_position.masked_fill(
            relative_position > self.max_distance - 1, self.max_distance - 1
        )

        # mapping from relative position to index in the bias parameter
        if self._skip_bucketing:
            bias_idx = relative_position
        else:
            bias_idx = self._bucket_relative_position(relative_position)
        if self.symmetric:
            bias_idx = bias_idx.abs()
        else:
            bias_idx += self.num_buckets // 2

        t5_rel_att_bias = self.bias_values(bias_idx)  # [L, L, H]
        t5_rel_att_bias = t5_rel_att_bias.permute(2, 0, 1).unsqueeze(0)  # [1, H, L, L]

        return t5_rel_att_bias

    def _bucket_relative_position(self, relative_position):
        # this is a placeholder (isn't tested, likely buggy) using HuggingFace implem as a reference
        # this also needs to be extended to support asymmetric +/- ve positions
        relative_buckets = 0
        if not self.causal:
            num_buckets //= 2
            relative_buckets += (relative_position > 0).to(mx.int64) * num_buckets
            relative_position = mx.abs(relative_position)
        else:
            relative_position = -mx.min(
                relative_position, mx.zeros_like(relative_position)
            )
        # now relative_position is in the range [0, inf)

        # half of the buckets are for exact increments in positions
        max_exact = num_buckets // 2
        is_small = relative_position < max_exact

        # The other half of the buckets are for logarithmically bigger bins in positions up to max_distance
        relative_position_if_large = max_exact + (
            mx.log(relative_position.float() / max_exact)
            / math.log(self.max_distance / max_exact)
            * (num_buckets - max_exact)
        ).to(mx.int64)
        relative_position_if_large = mx.min(
            relative_position_if_large,
            mx.full_like(relative_position_if_large, num_buckets - 1),
        )

        relative_buckets += mx.where(
            is_small, relative_position, relative_position_if_large
        )
        return relative_buckets


class CausalConv1D(nn.Conv1d):
    """
    A causal version of nn.Conv1d where each step would have limited access to locations on its right or left
    All arguments are the same as nn.Conv1d except padding.
    If padding is set None, then paddings are set automatically to make it a causal convolution where each location would not see any steps on its right.
    If padding is set as a list (size of 2), then padding[0] would be used as left padding and padding[1] as right padding.
    It would make it possible to control the number of steps to be accessible on the right and left.
    This mode is not supported when stride > 1. padding[0]+padding[1] should be equal to (kernel_size - 1).
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: Union[str, int] = 0,
        dilation: int = 1,
        groups: int = 1,
        bias: bool = True,
        padding_mode: str = "zeros",
        device=None,
        dtype=None,
    ) -> None:
        self.cache_drop_size = None
        if padding is None:
            self._left_padding = kernel_size - 1
            self._right_padding = stride - 1
        else:
            if stride != 1 and padding != kernel_size - 1:
                raise ValueError("No striding allowed for non-symmetric convolutions!")
            if isinstance(padding, int):
                self._left_padding = padding
                self._right_padding = padding
            elif (
                isinstance(padding, list)
                and len(padding) == 2
                and padding[0] + padding[1] == kernel_size - 1
            ):
                self._left_padding = padding[0]
                self._right_padding = padding[1]
            else:
                raise ValueError(f"Invalid padding param: {padding}!")

        self._max_cache_len = self._left_padding

        super().__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=0,
            dilation=dilation,
            groups=groups,
            bias=bias,
        )

    def update_cache(self, x, cache=None):
        if cache is None:
            new_x = mx.pad(x, pad=(self._left_padding, self._right_padding))
            next_cache = cache
        else:
            new_x = mx.pad(x, pad=(0, self._right_padding))
            new_x = mx.cat([cache, new_x], dim=-1)
            if self.cache_drop_size > 0:
                next_cache = new_x[:, :, : -self.cache_drop_size]
            else:
                next_cache = new_x
            next_cache = next_cache[:, :, -cache.size(-1) :]
        return new_x, next_cache

    def __call__(self, x, cache=None):
        x, cache = self.update_cache(x, cache=cache)
        x = super().__call__(x)
        if cache is None:
            return x
        else:
            return x, cache


class CausalConv2D(nn.Conv2d):
    """
    A causal version of nn.Conv2d where each location in the 2D matrix would have no access to locations on its right or down
    All arguments are the same as nn.Conv2d except padding which should be set as None
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: Union[str, int] = 0,
        dilation: int = 1,
        groups: int = 1,
        bias: bool = True,
    ) -> None:
        if padding is not None:
            raise ValueError("Argument padding should be set to None for CausalConv2D.")
        self._left_padding = kernel_size - 1
        self._right_padding = stride - 1

        padding = 0
        super().__init__(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            dilation,
            groups,
            bias,
        )

    def __call__(
        self,
        x,
    ):
        if self.training:
            x = mx.pad(
                x,
                (self._left_padding, self._right_padding),
                (self._left_padding, self._right_padding),
            )
        else:
            x = mx.pad(
                x,
                (self._left_padding, self._right_padding, 0, 0),
            )
        x = super().__call__(x)
        return x


def calc_length(lengths, all_paddings, kernel_size, stride, ceil_mode, repeat_num=1):
    """Calculates the output length of a Tensor passed through a convolution or max pooling layer"""
    add_pad: float = all_paddings - kernel_size
    one: float = 1.0
    for i in range(repeat_num):
        lengths = (lengths + add_pad) / stride + one
        if ceil_mode:
            lengths = mx.ceil(mx.array(lengths))
        else:
            lengths = mx.floor(mx.array(lengths))
    return lengths.astype(mx.int32)


class NemoConvSubsampling(nn.Module):

    def __init__(
        self,
        feat_in,
        feat_out,
        subsampling_factor=4,
        subsampling="dw_striding",
        conv_channels=256,
        subsampling_conv_chunking_factor=1,
        activation=nn.ReLU(),
        is_causal=False,
    ):
        super().__init__()
        self._subsampling = subsampling
        self._conv_channels = conv_channels
        self._feat_in = feat_in
        self._feat_out = feat_out

        if subsampling_factor % 2 != 0:
            raise ValueError("Sampling factor should be a multiply of 2!")
        self._sampling_num = int(math.log(subsampling_factor, 2))
        self.subsampling_factor = subsampling_factor
        self.is_causal = is_causal
        self.subsampling_causal_cond = subsampling in (
            "dw_striding",
            "striding",
            "striding_conv1d",
        )

        if (
            subsampling_conv_chunking_factor != -1
            and subsampling_conv_chunking_factor != 1
            and subsampling_conv_chunking_factor % 2 != 0
        ):
            raise ValueError(
                "subsampling_conv_chunking_factor should be -1, 1, or a power of 2"
            )
        self.subsampling_conv_chunking_factor = subsampling_conv_chunking_factor

        in_channels = 1
        layers = []

        if subsampling == "dw_striding":
            self._stride = 2
            self._kernel_size = 3
            self._ceil_mode = False

            if self.is_causal:
                self._left_padding = self._kernel_size - 1
                self._right_padding = self._stride - 1
                self._max_cache_len = subsampling_factor + 1
            else:
                self._left_padding = (self._kernel_size - 1) // 2
                self._right_padding = (self._kernel_size - 1) // 2
                self._max_cache_len = 0

            # Layer 1
            if self.is_causal:
                layers.append(
                    CausalConv2D(
                        in_channels=in_channels,
                        out_channels=conv_channels,
                        kernel_size=self._kernel_size,
                        stride=self._stride,
                        padding=None,
                    )
                )
            else:
                layers.append(
                    nn.Conv2d(
                        in_channels=in_channels,
                        out_channels=conv_channels,
                        kernel_size=self._kernel_size,
                        stride=self._stride,
                        padding=self._left_padding,
                    )
                )
            in_channels = conv_channels
            layers.append(activation)

            for i in range(self._sampling_num - 1):
                if self.is_causal:
                    layers.append(
                        CausalConv2D(
                            in_channels=in_channels,
                            out_channels=in_channels,
                            kernel_size=self._kernel_size,
                            stride=self._stride,
                            padding=None,
                            groups=in_channels,
                        )
                    )
                else:
                    layers.append(
                        nn.Conv2d(
                            in_channels=in_channels,
                            out_channels=in_channels,
                            kernel_size=self._kernel_size,
                            stride=self._stride,
                            padding=self._left_padding,
                            groups=in_channels,
                        )
                    )

                layers.append(
                    nn.Conv2d(
                        in_channels=in_channels,
                        out_channels=conv_channels,
                        kernel_size=1,
                        stride=1,
                        padding=0,
                        groups=1,
                    )
                )
                layers.append(activation)
                in_channels = conv_channels

        elif subsampling == "striding":
            self._stride = 2
            self._kernel_size = 3
            self._ceil_mode = False

            if self.is_causal:
                self._left_padding = self._kernel_size - 1
                self._right_padding = self._stride - 1
                self._max_cache_len = subsampling_factor + 1
            else:
                self._left_padding = (self._kernel_size - 1) // 2
                self._right_padding = (self._kernel_size - 1) // 2
                self._max_cache_len = 0

            for i in range(self._sampling_num):
                if self.is_causal:
                    layers.append(
                        CausalConv2D(
                            in_channels=in_channels,
                            out_channels=conv_channels,
                            kernel_size=self._kernel_size,
                            stride=self._stride,
                            padding=None,
                        )
                    )
                else:
                    layers.append(
                        nn.Conv2d(
                            in_channels=in_channels,
                            out_channels=conv_channels,
                            kernel_size=self._kernel_size,
                            stride=self._stride,
                            padding=self._left_padding,
                        )
                    )
                layers.append(activation)
                in_channels = conv_channels

        elif subsampling == "striding_conv1d":
            in_channels = feat_in

            self._stride = 2
            self._kernel_size = 5
            self._ceil_mode = False

            if self.is_causal:
                self._left_padding = self._kernel_size - 1
                self._right_padding = self._stride - 1
                self._max_cache_len = subsampling_factor + 1
            else:
                self._left_padding = (self._kernel_size - 1) // 2
                self._right_padding = (self._kernel_size - 1) // 2
                self._max_cache_len = 0

            for i in range(self._sampling_num):
                if self.is_causal:
                    layers.append(
                        CausalConv1D(
                            in_channels=in_channels,
                            out_channels=(
                                feat_out
                                if self._sampling_num == i + 1
                                else conv_channels
                            ),
                            kernel_size=self._kernel_size,
                            stride=self._stride,
                            padding=None,
                        )
                    )
                else:
                    layers.append(
                        nn.Conv1d(
                            in_channels=in_channels,
                            out_channels=(
                                feat_out
                                if self._sampling_num == i + 1
                                else conv_channels
                            ),
                            kernel_size=self._kernel_size,
                            stride=self._stride,
                            padding=self._left_padding,
                        )
                    )
                layers.append(activation)
                in_channels = conv_channels

        elif subsampling == "dw_striding_conv1d":
            in_channels = feat_in

            self._stride = 2
            self._kernel_size = 5
            self._ceil_mode = False

            self._left_padding = (self._kernel_size - 1) // 2
            self._right_padding = (self._kernel_size - 1) // 2

            # Layer 1
            layers.extend(
                [
                    nn.Conv1d(
                        in_channels=in_channels,
                        out_channels=in_channels,
                        kernel_size=self._kernel_size,
                        stride=self._stride,
                        padding=self._left_padding,
                        groups=in_channels,
                    ),
                    nn.Conv1d(
                        in_channels=in_channels,
                        out_channels=(
                            feat_out if self._sampling_num == 1 else conv_channels
                        ),
                        kernel_size=1,
                        stride=1,
                        padding=0,
                        groups=1,
                    ),
                ]
            )
            in_channels = conv_channels
            layers.append(activation)

            for i in range(self._sampling_num - 1):
                layers.extend(
                    [
                        nn.Conv1d(
                            in_channels=in_channels,
                            out_channels=in_channels,
                            kernel_size=self._kernel_size,
                            stride=self._stride,
                            padding=self._left_padding,
                            groups=in_channels,
                        ),
                        nn.Conv1d(
                            in_channels=in_channels,
                            out_channels=(
                                feat_out
                                if self._sampling_num == i + 2
                                else conv_channels
                            ),
                            kernel_size=1,
                            stride=1,
                            padding=0,
                            groups=1,
                        ),
                    ]
                )
                layers.append(activation)
                in_channels = conv_channels

        else:
            raise ValueError(f"Not valid sub-sampling: {subsampling}!")

        if subsampling in ["dw_striding", "striding"]:
            in_length = mx.array(feat_in, dtype=mx.float32)
            out_length = calc_length(
                lengths=in_length,
                all_paddings=self._left_padding + self._right_padding,
                kernel_size=self._kernel_size,
                stride=self._stride,
                ceil_mode=self._ceil_mode,
                repeat_num=self._sampling_num,
            )
            self.out = nn.Linear(conv_channels * int(out_length), feat_out)
            self.conv2d_subsampling = True
        elif subsampling in ["striding_conv1d", "dw_striding_conv1d"]:
            self.out = None
            self.conv2d_subsampling = False
        else:
            raise ValueError(f"Not valid sub-sampling: {subsampling}!")

        self.conv = layers

    def get_sampling_frames(self):
        return [1, self.subsampling_factor]

    def get_streaming_cache_size(self):
        return [0, self.subsampling_factor + 1]

    def __call__(self, x, mask):
        """
        Forward method for NeMo subsampling.
        Args:
            x[Batch, Time, Filters]: torch.Tensor
                input tensor
            x_mask: torch.Tensor
                input mask
        Returns:
            x: torch.Tensor
                Resulting tensor from subsampling (B, T // time_reduction_factor, feat_out)
            pad_mask: torch.Tensor
                tensor of padded hidden state sequences (B, 1, T // time_reduction_factor)
        """
        # Unsqueeze Channel Axis
        if self.conv2d_subsampling:
            x = mx.expand_dims(x, axis=1)
        # Transpose to Channel First mode
        else:
            x = x.transpose(0, 2, 1)

        # split inputs if chunking_factor is set
        if self.subsampling_conv_chunking_factor != -1 and self.conv2d_subsampling:
            if self.subsampling_conv_chunking_factor == 1:
                # if subsampling_conv_chunking_factor is 1, we split only if needed
                # avoiding a bug / feature limiting indexing of tensors to 2**31
                # see https://github.com/pytorch/pytorch/issues/80020
                x_ceil = 2**31 / self._conv_channels * self._stride * self._stride
                if mx.numel(x) > x_ceil:
                    need_to_split = True
                else:
                    need_to_split = False
            else:
                # if subsampling_conv_chunking_factor > 1 we always split
                need_to_split = True

            if need_to_split:
                x, success = self.conv_split_by_batch(x)
                if not success:  # if unable to split by batch, try by channel
                    if self._subsampling == "dw_striding":
                        x = self.conv_split_by_channel(x)
                    else:
                        for conv in self.conv:
                            x = conv(x)  # try anyway
            else:
                for conv in self.conv:
                    x = conv(x)
        else:
            for conv in self.conv:
                x = conv(x)

        # Flatten Channel and Frequency Axes
        if self.conv2d_subsampling:
            b, c, t, f = x.size()
            x = self.out(x.transpose(0, 2, 1).reshape(b, t, -1))
        # Transpose to Channel Last mode
        else:
            x = x.transpose(0, 2, 1)

        if mask is None:
            return x, None

        max_audio_length = x.shape[1]
        feature_lens = mask.sum(1)
        padding_length = mx.ceil(feature_lens / self.subsampling_factor)
        if self.is_causal and self.subsampling_causal_cond:
            feature_lens_remainder = feature_lens % self.subsampling_factor
            padding_length[feature_lens_remainder != 1] += 1
        pad_mask = mx.arange(0, max_audio_length, device=x.device).expand(
            padding_length.size(0), -1
        ) < padding_length.unsqueeze(1)
        return x, pad_mask.unsqueeze(1)

    def conv_split_by_batch(self, x):
        """Tries to split input by batch, run conv and concat results"""
        b, _, _, _ = x.size()
        if b == 1:  # can't split if batch size is 1
            return x, False

        if self.subsampling_conv_chunking_factor > 1:
            cf = self.subsampling_conv_chunking_factor
        else:
            # avoiding a bug / feature limiting indexing of tensors to 2**31
            # see https://github.com/pytorch/pytorch/issues/80020
            x_ceil = 2**31 / self._conv_channels * self._stride * self._stride
            p = math.ceil(math.log(mx.numel(x) / x_ceil, 2))
            cf = 2**p

        new_batch_size = b // cf
        if new_batch_size == 0:  # input is too big
            return x, False

        out_chunks = []
        for chunk in mx.split(x, new_batch_size, 0):
            for conv in self.conv:
                chunk = conv(chunk)
            out_chunks.append(chunk)

        return mx.cat(out_chunks, 0), True

    def conv_split_by_channel(self, x):
        """For dw convs, tries to split input by time, run conv and concat results"""
        x = self.conv[0](x)  # full conv2D
        x = self.conv[1](x)  # activation

        for i in range(self._sampling_num - 1):
            _, c, t, _ = x.size()

            if self.subsampling_conv_chunking_factor > 1:
                cf = self.subsampling_conv_chunking_factor
            else:
                # avoiding a bug / feature limiting indexing of tensors to 2**31
                # see https://github.com/pytorch/pytorch/issues/80020
                p = math.ceil(math.log(mx.numel(x) / 2**31, 2))
                cf = 2**p

            new_c = int(c // cf)
            if new_c == 0:
                new_c = 1

            new_t = int(t // cf)
            if new_t == 0:
                new_t = 1

            x = self.channel_chunked_conv(
                self.conv[i * 3 + 2], new_c, x
            )  # conv2D, depthwise

            # splitting pointwise convs by time
            x = mx.cat(
                [self.conv[i * 3 + 3](chunk) for chunk in mx.split(x, new_t, 2)], 2
            )  # conv2D, pointwise
            x = self.conv[i * 3 + 4](x)  # activation
        return x

    def channel_chunked_conv(self, conv, chunk_size, x):
        """Performs channel chunked convolution"""

        ind = 0
        out_chunks = []
        for chunk in mx.split(x, chunk_size, 1):
            step = chunk.size()[1]

            if self.is_causal:
                chunk = mx.pad(
                    chunk,
                    (
                        self._kernel_size - 1,
                        self._stride - 1,
                        self._kernel_size - 1,
                        self._stride - 1,
                    ),
                )
                # MLX's conv2d doesn't accept bias as a keyword argument
                # Apply bias separately after convolution
                ch_out = mx.conv2d(
                    chunk,
                    conv.weight[ind : ind + step, :, :, :],
                    stride=self._stride,
                    padding=0,
                    groups=step,
                )
                # Add bias manually
                bias_view = conv.bias[ind : ind + step].reshape(step, 1, 1)
                ch_out = ch_out + bias_view
            else:
                # MLX's conv2d doesn't accept bias as a keyword argument
                # Apply bias separately after convolution
                ch_out = mx.conv2d(
                    chunk,
                    conv.weight[ind : ind + step, :, :, :],
                    stride=self._stride,
                    padding=self._left_padding,
                    groups=step,
                )
                # Add bias manually
                bias_view = conv.bias[ind : ind + step].reshape(step, 1, 1)
                ch_out = ch_out + bias_view
            out_chunks.append(ch_out)
            ind += step

        return mx.cat(out_chunks, 1)

    def change_subsampling_conv_chunking_factor(
        self, subsampling_conv_chunking_factor: int
    ):
        if (
            subsampling_conv_chunking_factor != -1
            and subsampling_conv_chunking_factor != 1
            and subsampling_conv_chunking_factor % 2 != 0
        ):
            raise ValueError(
                "subsampling_conv_chunking_factor should be -1, 1, or a power of 2"
            )
        self.subsampling_conv_chunking_factor = subsampling_conv_chunking_factor


class DepthWiseSeperableConv1d(nn.Module):

    def __init__(
        self,
        input_dim,
        depthwise_seperable_out_channel,
        kernel_size,
        depthwise_multiplier,
        padding=0,
    ):
        super().__init__()

        self.dw_conv = nn.Conv1d(
            input_dim,
            input_dim * depthwise_multiplier,
            kernel_size,
            1,
            padding=padding,
            groups=input_dim,
        )

        if depthwise_seperable_out_channel != 0:
            self.pw_conv = nn.Conv1d(
                input_dim * depthwise_multiplier,
                depthwise_seperable_out_channel,
                1,
                1,
                0,
            )
        else:
            self.pw_conv = nn.Identity()
        self.depthwise_seperable_out_channel = depthwise_seperable_out_channel

    def __call__(self, x):
        """
        Args:
            x: torch.Tensor
                input tensor
        """
        x = self.dw_conv(x)
        if self.depthwise_seperable_out_channel != 0:
            x = self.pw_conv(x)
        return x


class BatchNorm1d(nn.Module):
    def __init__(self, num_features, eps=1e-5, momentum=0.1):
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum

        # Learnable parameters
        self.weight = mx.ones((num_features,))
        self.bias = mx.zeros((num_features,))

        # Running statistics
        self.running_mean = mx.zeros((num_features,))
        self.running_var = mx.ones((num_features,))

    def __call__(self, x):
        # x shape: (batch_size, num_features, seq_len)
        # Compute statistics along batch and sequence dimensions
        mean = mx.mean(x, axis=(0, 2), keepdims=True)
        var = mx.var(x, axis=(0, 2), keepdims=True)

        # Update running statistics
        self.running_mean = (
            1 - self.momentum
        ) * self.running_mean + self.momentum * mean.squeeze()
        self.running_var = (
            1 - self.momentum
        ) * self.running_var + self.momentum * var.squeeze()

        # Normalize
        x_norm = (x - mean) / mx.sqrt(var + self.eps)

        # Apply learnable parameters
        return self.weight.reshape(1, -1, 1) * x_norm + self.bias.reshape(1, -1, 1)


# TODO: Improve using GLU class
class GLUPointWiseConv(nn.Module):
    def __init__(
        self,
        input_dim,
        output_dim,
        kernel_size,
        glu_type="sigmoid",
        bias_in_glu=True,
        causal=False,
    ):
        super().__init__()

        self.glu_type = glu_type
        self.output_dim = output_dim
        self.bias_in_glu = bias_in_glu
        if causal:
            self.ext_pw_conv_1d = nn.Conv1d(
                input_dim, output_dim * 2, kernel_size, 1, padding=(kernel_size - 1)
            )
        else:
            self.ext_pw_conv_1d = nn.Conv1d(
                input_dim,
                output_dim * 2,
                kernel_size,
                1,
                padding=(kernel_size - 1) // 2,
            )

        if glu_type == "sigmoid":
            self.glu_act = nn.Sigmoid()
        elif glu_type == "relu":
            self.glu_act = nn.ReLU()
        elif glu_type == "gelu":
            self.glu_act = nn.GELU()
        elif glu_type == "swish":
            self.glu_act = nn.SiLU()
        else:
            raise ValueError(f"Unsupported activation type {self.glu_act}")

        if bias_in_glu:
            self.b1 = mx.zeros((1, output_dim, 1))
            self.b2 = mx.zeros((1, output_dim, 1))

    def __call__(self, x):
        # to be consistent with GLULinear, we assume the input always has the #channel (#dim) in the last dimension of the tensor, so need to switch the dimension first for 1D-Conv case
        x = x.transpose((0, 2, 1))
        x = self.ext_pw_conv_1d(x)
        if self.glu_type == "bilinear":
            if self.bias_in_glu:
                x = (x[:, 0 : self.output_dim, :] + self.b1) * (
                    x[:, self.output_dim : self.output_dim * 2, :] + self.b2
                )
            else:
                x = (x[:, 0 : self.output_dim, :]) * (
                    x[:, self.output_dim : self.output_dim * 2, :]
                )
        else:
            if self.bias_in_glu:
                x = (x[:, 0 : self.output_dim, :] + self.b1) * self.glu_act(
                    x[:, self.output_dim : self.output_dim * 2, :] + self.b2
                )
            else:
                x = (x[:, 0 : self.output_dim, :]) * self.glu_act(
                    x[:, self.output_dim : self.output_dim * 2, :]
                )

        x = x.transpose((0, 2, 1))
        return x


class ConformerConvModule(nn.Module):

    def __init__(
        self,
        input_dim,
        ext_pw_out_channel,
        depthwise_seperable_out_channel,
        ext_pw_kernel_size,
        kernel_size,
        depthwise_multiplier,
        dropout_rate,
        causal=False,
        batch_norm=False,
        chunk_se=0,
        chunk_size=18,
        activation="relu",
        glu_type="sigmoid",
        bias_in_glu=True,
        linear_glu_in_convm=False,
        export=False,
    ):
        super().__init__()
        self.layer_norm = nn.LayerNorm(input_dim)
        self.input_dim = input_dim
        self.ext_pw_out_channel = ext_pw_out_channel
        self.ext_pw_kernel_size = ext_pw_kernel_size
        self.depthwise_seperable_out_channel = depthwise_seperable_out_channel
        self.glu_type = glu_type
        self.bias_in_glu = bias_in_glu
        self.linear_glu_in_convm = linear_glu_in_convm
        self.causal = causal

        self._add_ext_pw_layer()

        self.batch_norm = batch_norm
        self.kernel_size = kernel_size

        if batch_norm:

            self.bn_layer = BatchNorm1d(input_dim)

        if activation == "relu":
            self.act = nn.ReLU()
        elif activation == "gelu":
            self.act = nn.GELU()
        elif activation == "swish":
            self.act = nn.SiLU()
        else:
            raise ValueError(f"Activation function {activation} not supported")

        self.dropout = nn.Dropout(dropout_rate)
        self.export = export

        if causal:
            if export:  # Inference only.
                padding = (
                    0  # A cache is concatenated to the left. No padding in the kernel.
                )
            else:
                # Training only. Padding will be added symmetrically on both sides.
                # After convolution, clip off kernel_size-1 points on the right.
                padding = kernel_size - 1
        else:
            padding = (kernel_size - 1) // 2

        self.dw_sep_conv_1d = DepthWiseSeperableConv1d(
            input_dim,
            depthwise_seperable_out_channel,
            kernel_size,
            depthwise_multiplier,
            padding=padding,
        )

        if depthwise_seperable_out_channel != 0:
            if input_dim != depthwise_seperable_out_channel:
                self.ln2 = nn.Linear(depthwise_seperable_out_channel, input_dim)
        else:
            if depthwise_multiplier != 1:
                self.ln2 = nn.Linear(input_dim * depthwise_multiplier, input_dim)

    def _add_ext_pw_layer(self):
        """
        This function is an extension of __init__ function
        and dedicated to the convolution module creation
        of the conformer.
        """
        self.ln1 = self.glu = self.bn_layer = self.ext_pw_conv_1d = (
            nn.Identity()
        )  # jit hacks.
        self.squeeze_excitation = nn.Identity()  # jit.
        self.apply_ln1 = self.fix_len1 = False  # jit.

        if self.ext_pw_out_channel != 0:
            if self.causal:
                self.ext_pw_conv_1d = nn.Conv1d(
                    self.input_dim,
                    self.ext_pw_out_channel,
                    self.ext_pw_kernel_size,
                    1,
                    padding=(self.ext_pw_kernel_size - 1),
                )
                if self.ext_pw_kernel_size > 1:
                    self.fix_len1 = True
                else:
                    self.fix_len1 = False
            else:
                self.ext_pw_conv_1d = nn.Conv1d(
                    self.input_dim,
                    self.ext_pw_out_channel,
                    self.ext_pw_kernel_size,
                    1,
                    padding=(self.ext_pw_kernel_size - 1) // 2,
                )
                self.fix_len1 = False

            if self.linear_glu_in_convm:
                self.glu = GLULinear(
                    self.input_dim,
                    self.ext_pw_out_channel,
                    self.glu_type,
                    self.bias_in_glu,
                )
            else:
                self.glu = GLUPointWiseConv(
                    self.input_dim,
                    self.ext_pw_out_channel,
                    self.ext_pw_kernel_size,
                    self.glu_type,
                    self.bias_in_glu,
                    self.causal,
                )

            if self.input_dim != self.ext_pw_out_channel:
                self.apply_ln1 = True
                self.ln1 = nn.Linear(self.ext_pw_out_channel, self.input_dim)
            else:
                self.apply_ln1 = False
        else:
            self.pw_conv_simplify_w = mx.ones((3,))
            self.pw_conv_simplify_b = mx.zeros((3,))

    def __call__(self, x):
        """ConvModule Forward.
        Args:
            x: torch.Tensor
                input tensor.
        """
        x = self.layer_norm(x)

        if self.ext_pw_out_channel != 0:
            x = self.glu(x)
            if self.causal and self.ext_pw_kernel_size > 1:
                x = x[:, : -(self.ext_pw_kernel_size - 1), :]
            if self.apply_ln1:
                x = self.ln1(x)
        else:
            x_0 = x * self.pw_conv_simplify_w[0] + self.pw_conv_simplify_b[0]
            x_1 = x * self.pw_conv_simplify_w[1] + self.pw_conv_simplify_b[1]
            x = x_0 + x_1

        x = x.transpose((0, 2, 1))

        x = self.dw_sep_conv_1d(x)
        if self.causal and self.kernel_size > 1:
            x = x[:, :, : -(self.kernel_size - 1)]
        if hasattr(self, "ln2"):
            x = x.transpose((0, 2, 1))
            x = self.ln2(x)
            x = x.transpose((0, 2, 1))
        if self.batch_norm:
            x = self.bn_layer(x)
        x = self.act(x)

        if self.ext_pw_out_channel != 0:
            x = self.ext_pw_conv_1d(x)
            if self.fix_len1:
                x = x[:, :, : -(self.ext_pw_kernel_size - 1)]

            if self.apply_ln1:
                x = x.transpose((0, 2, 1))
                x = self.ln1(x)
                x = x.transpose((0, 2, 1))

            x = x.transpose((0, 2, 1))
        else:
            x = x.unsqueeze(1).transpose((0, 1, 3, 2))
            x = x * self.pw_conv_simplify_w[2] + self.pw_conv_simplify_b[2]
            x = x.squeeze(1)

        x = self.dropout(x)
        return x


class GLU(nn.Module):
    """Implement Gated Linear Unit (GLU) module"""

    def __init__(self, dim: int = -1, act_name: str = "sigmoid") -> None:
        super().__init__()
        self.dim = dim
        self.act_name = act_name.lower()

        if self.act_name == "relu":
            self.act_fn = nn.ReLU()
        elif self.act_name == "gelu":
            self.act_fn = nn.GELU()
        elif self.act_name == "swish":
            self.act_fn = nn.SiLU()
        elif self.act_name == "sigmoid":
            self.act_fn = nn.Sigmoid()
        else:
            self.act_fn = nn.Identity()

    def __call__(self, x: mx.array) -> mx.array:
        half_x, gate = mx.split(x, 2, axis=self.dim)
        return half_x * self.act_fn(gate)


class GLULinear(nn.Module):
    """Linear + GLU module
    Args:
        input_dim: int
            input size
        output_dim: int
            output size.
        glu_type:
            activation function name used in glu module.
            default "sigmoid" (swish function).
        bias_in_glu: bool, optional
            If True, the addtive bias is added. Default False.
    """

    def __init__(
        self,
        input_dim,
        output_dim,
        glu_type="sigmoid",
        bias_in_glu=True,
    ):
        super().__init__()
        self.linear = nn.Linear(input_dim, output_dim * 2, bias_in_glu)
        self.glu_act = GLU(-1, glu_type)

    def __call__(self, x):
        """GLULinear forward
        Args:
            x: torch.Tensor
                inpute tensor.
        """
        x = self.linear(x)
        return self.glu_act(x)


class FeedForward(nn.Module):
    """Feed Forward module for Conformer."""

    def __init__(
        self,
        d_model,
        d_inner,
        dropout_rate,
        activation="sigmoid",
        bias_in_glu=True,
    ):
        super().__init__()
        self.layer_norm = nn.LayerNorm(d_model)
        module = GLULinear(d_model, d_inner, bias_in_glu=True)
        self.net = [
            module,
            nn.Dropout(dropout_rate),
            nn.Linear(d_inner, d_model),
            nn.Dropout(dropout_rate),
        ]

    def __call__(self, x):
        # Layer normalization
        x = self.layer_norm(x)
        for layer in self.net:
            x = layer(x)
        return x


class ConformerAttention(nn.Module):
    """Multi-headed attention module for Conformer."""

    def __init__(
        self,
        n_head,
        n_feat,
        dropout_rate,
        attention_inner_dim=-1,
        glu_type="swish",
        bias_in_glu=True,
        use_pt_scaled_dot_product_attention=False,
        n_value=-1,
        group_size: int = 1,
    ):
        super().__init__()

        if n_value == -1:
            n_value = n_feat
        if attention_inner_dim == -1:
            attention_inner_dim = n_feat
        assert attention_inner_dim % n_head == 0

        # We assume d_v always equals d_k
        self.d_k = attention_inner_dim // n_head
        self.scale = self.d_k**-0.5
        self.h = n_head
        assert n_head % group_size == 0, "group_size must divide n_head"
        self.g = group_size
        self.h_k = n_head // group_size

        self.linear_q = nn.Linear(n_feat, attention_inner_dim)
        self.linear_k = nn.Linear(n_feat, attention_inner_dim // group_size)
        self.linear_v = nn.Linear(n_feat, attention_inner_dim // group_size)
        self.linear_out = nn.Linear(attention_inner_dim // group_size, n_feat)
        self.dropout = dropout_rate

    def __call__(self, query, key, value, mask=None):
        batch_size, seq_len, _ = query.shape
        q = (
            self.linear_q(query)
            .reshape((batch_size, seq_len, self.heads, -1))
            .transpose((0, 2, 1, 3))
        )
        k = (
            self.linear_k(key)
            .reshape((batch_size, seq_len, self.heads, -1))
            .transpose((0, 2, 1, 3))
        )
        v = (
            self.linear_v(value)
            .reshape((batch_size, seq_len, self.heads, -1))
            .transpose((0, 2, 1, 3))
        )

        # Compute attention scores
        attention_scores = mx.matmul(q, mx.transpose(k, (0, 1, 3, 2))) * self.scale

        # Apply mask if provided
        if mask is not None:
            # Expand mask for broadcasting with attention scores
            mask = mx.expand_dims(mx.expand_dims(mask, 1), 1)
            attention_scores = mx.where(mask, attention_scores, mx.array(float("-inf")))

        # Apply softmax to get attention weights
        attention_weights = nn.softmax(attention_scores, axis=-1)

        # Apply dropout to attention weights during training
        if self.dropout > 0:
            attention_weights = nn.Dropout(self.dropout)(attention_weights)

        # Apply attention weights to values
        context = mx.matamul(attention_weights, v)

        # Transpose and reshape back
        context = context.transpose(0, 2, 1, 3).reshape(batch_size, seq_len, -1)

        # Final projection
        output = self.linear_out(context)

        return output


class ConformerEncoderLayer(nn.Module):
    """A single Conformer block."""

    def __init__(
        self,
        d_model=512,
        ext_pw_out_channel=0,
        depthwise_seperable_out_channel=256,
        depthwise_multiplier=1,
        n_head=4,
        d_ffn=2048,
        ext_pw_kernel_size=1,
        kernel_size=3,
        dropout_rate=0.1,
        causal=False,
        batch_norm=False,
        activation="relu",
        chunk_se=0,
        chunk_size=18,
        conv_activation="relu",
        conv_glu_type="sigmoid",
        bias_in_glu=True,
        linear_glu_in_convm=False,
        attention_innner_dim=-1,
        attention_glu_type="swish",
        activation_checkpointing="",
        export=False,
        use_pt_scaled_dot_product_attention=False,
        attn_group_sizes: int = 1,
    ):
        super().__init__()

        self.feed_forward_in = FeedForward(
            d_model=d_model,
            d_inner=d_ffn,
            dropout_rate=dropout_rate,
            activation=activation,
            bias_in_glu=bias_in_glu,
        )

        self.self_attn = ConformerAttention(
            n_head,
            d_model,
            dropout_rate,
            attention_innner_dim,
            attention_glu_type,
            bias_in_glu,
            use_pt_scaled_dot_product_attention=use_pt_scaled_dot_product_attention,
            group_size=attn_group_sizes,
        )
        self.conv = ConformerConvModule(
            d_model,
            ext_pw_out_channel,
            depthwise_seperable_out_channel,
            ext_pw_kernel_size,
            kernel_size,
            depthwise_multiplier,
            dropout_rate,
            causal,
            batch_norm,
            chunk_se,
            chunk_size,
            conv_activation,
            conv_glu_type,
            bias_in_glu,
            linear_glu_in_convm,
            export=export,
        )
        self.feed_forward_out = FeedForward(
            d_model=d_model,
            d_inner=d_ffn,
            dropout_rate=dropout_rate,
            activation=activation,
            bias_in_glu=bias_in_glu,
        )

        self.layer_norm_att = nn.LayerNorm(d_model)
        self.layer_norm = nn.LayerNorm(d_model)

    def __call__(self, x, mask=None):
        x = x + 0.5 * self.feed_forward_in(x)
        norm_x = self.layer_norm_att(x)

        x = x + self.self_attn(norm_x, mask)
        x = x + self.conv(x)
        x = x + 0.5 * self.feed_forward_out(x)

        out = self.layer_norm(x)
        return out


class MeanVarianceNormLayer(nn.Module):
    """Mean/variance normalization layer.
    Will substract mean and multiply input by inverted standard deviation.
    Typically used as a very first layer in a model.
    Args:
        input_size: int
            layer input size.
    """

    def __init__(self, input_size):
        super().__init__()
        self.input_size = input_size
        self.global_mean = mx.array(mx.zeros(input_size))
        self.global_invstd = mx.array(mx.ones(input_size))

    def forward(self, input_: mx.array) -> mx.array:
        """MeanVarianceNormLayer Forward
        Args:
            input_: torch.Tensor
                input tensor.
        """
        return (input_ - self.global_mean) * self.global_invstd

    def load_mean_invstd(self, mean_file, invstd_file, cuside_features=False):
        """Load feature mean and variance used for normalization.
        Args:
            mean_file: str
                path to the feature mean statistics file.
            invstd_file: str
                path to the features inverted standard deviation
                 statistics file.
            cuside_features: bool
                Boolean that indicates CUSIDE is being used.
                The statistics of CUSIDE features are copied
                from the normal features
        """
        self.global_mean.data = mx.array(np_loadtxt_with_retry(mean_file))
        self.global_invstd.data = mx.array(np_loadtxt_with_retry(invstd_file))

        if cuside_features:
            self.global_mean.data = mx.cat(
                (self.global_mean.data, self.global_mean.data), 0
            )
            self.global_invstd.data = mx.cat(
                (self.global_invstd.data, self.global_invstd.data), 0
            )


class ConformerEncoder(nn.Module):
    """Conformer encoder for audio processing."""

    def __init__(
        self,
        input_size,
        chunk_size,
        left_chunk,
        num_lang=None,
        attention_dim=256,
        attention_heads=4,
        linear_units=2048,
        num_blocks=6,
        dropout_rate=0.1,
        input_layer="nemo_conv",
        causal=True,
        batch_norm=False,
        cnn_out=-1,
        cnn_layer_norm=False,
        ext_pw_out_channel=0,
        ext_pw_kernel_size=1,
        depthwise_seperable_out_channel=256,
        depthwise_multiplier=1,
        chunk_se=0,
        kernel_size=3,
        activation="relu",
        conv_activation="relu",
        conv_glu_type="sigmoid",
        bias_in_glu=True,
        linear_glu_in_convm=False,
        attention_glu_type="swish",
        export=False,
        extra_layer_output_idx=-1,
        extra_multi_layer_output_idxs=[],
        activation_checkpointing="",
        relative_attention_bias_args=None,
        time_reduction=4,
        use_pt_scaled_dot_product_attention=False,
        nemo_conv_settings=None,
        conv2d_extra_padding: Literal["feat", "feat_time", "none", True] = "none",
        replication_pad_for_subsample_embedding=False,
        attention_group_size=1,
        encoder_embedding_config=None,
    ):
        super().__init__()

        self.input_size = input_size
        self.input_layer = input_layer
        self.chunk_size = chunk_size
        self.left_chunk = left_chunk
        self.attention_dim = attention_dim
        self.num_heads = attention_heads
        self.attention_group_size = attention_group_size
        self.time_reduction = time_reduction
        self.nemo_conv_settings = nemo_conv_settings
        self.encoder_embedding_config = encoder_embedding_config

        if self.input_layer == "nemo_conv":
            default_nemo_conv_settings = {
                "subsampling": "dw_striding",
                "subsampling_factor": self.time_reduction,
                "feat_in": input_size,
                "feat_out": attention_dim,
                "conv_channels": 256,
                "subsampling_conv_chunking_factor": 1,
                "activation": nn.ReLU(),
                "is_causal": False,
            }
            # Override any of the defaults with the incoming, user settings
            if nemo_conv_settings:
                default_nemo_conv_settings.update(nemo_conv_settings)
                for i in ["subsampling_factor", "feat_in", "feat_out"]:
                    assert (
                        i not in nemo_conv_settings
                    ), "{i} should be specified outside of the NeMo dictionary"

            self.embed = NemoConvSubsampling(
                **default_nemo_conv_settings,
            )
        else:
            raise ValueError("unknown input_layer: " + input_layer)

        # Positional encoding - using sinusoidal positional embedding
        # In a complete implementation, we would use a proper positional encoding here

        # Conformer blocks
        self.encoders = [
            ConformerEncoderLayer(
                d_model=attention_dim,
                ext_pw_out_channel=ext_pw_out_channel,
                depthwise_seperable_out_channel=depthwise_seperable_out_channel,
                depthwise_multiplier=depthwise_multiplier,
                n_head=attention_heads,
                d_ffn=linear_units,
                ext_pw_kernel_size=ext_pw_kernel_size,
                kernel_size=kernel_size,
                dropout_rate=dropout_rate,
                causal=causal,
                batch_norm=batch_norm,
                activation=activation,
                chunk_se=chunk_se,
                chunk_size=chunk_size,
                conv_activation=conv_activation,
                conv_glu_type=conv_glu_type,
                bias_in_glu=bias_in_glu,
                linear_glu_in_convm=linear_glu_in_convm,
                attention_glu_type=attention_glu_type,
                # activation_checkpointing=attn_checkpointing(activation_checkpointing, i),
                export=export,
                use_pt_scaled_dot_product_attention=use_pt_scaled_dot_product_attention,
                attn_group_sizes=attention_group_size,
            )
            for _ in range(num_blocks)
        ]

        if not hasattr(self, "encoder_embedding"):
            self.encoder_embedding = MeanVarianceNormLayer(
                self.encoder_embedding_config["input_size"]
            )

        self.relative_attention_bias_type = (
            relative_attention_bias_args.get("type")
            if relative_attention_bias_args
            else None
        )
        if self.relative_attention_bias_type == "t5":
            assert (
                self.num_heads % self.attention_group_size == 0
            ), "attention_group_size must divide n_head"
            self.relative_attention_bias_layer = T5RelativeAttentionLogitBias(
                self.num_heads // self.attention_group_size,
                max_distance=relative_attention_bias_args.get(
                    "t5_bias_max_distance", 1000
                ),
                symmetric=relative_attention_bias_args.get("t5_bias_symmetric", False),
            )
        else:
            raise NotImplementedError

    def post_init(self, init_model_config=None):
        """Initialize model weights from a pretrained model."""

        pretrained_speech_encoder_path = init_model_config.get(
            "pretrained_speech_encoder_path", None
        )
        if pretrained_speech_encoder_path:
            import torch

            model_state = torch.load(pretrained_speech_encoder_path, map_location="cpu")
            encoder_state_dict = {}
            for k, v in model_state.items():
                if "encoder." in k:
                    tmp_k = k.replace("encoder.", "")
                    encoder_state_dict[tmp_k] = v

            if hasattr(self, "encoder_embedding"):
                del self.encoder_embedding
            self.load_weights(list(encoder_state_dict.items()))

        if not hasattr(self, "encoder_embedding"):
            self.encoder_embedding = MeanVarianceNormLayer(
                self.encoder_embedding_config["input_size"]
            )

        mean_file = init_model_config.get("mean_file", None)
        invstd_file = init_model_config.get("invstd_file", None)
        if mean_file is not None and invstd_file is not None:
            self.encoder_embedding.load_mean_invstd(mean_file, invstd_file)

    def __call__(self, x, mask=None):
        """
        Args:
            x: Input tensor (batch_size, time_length, feat_dim)
            mask: Mask tensor for valid data (batch_size, time_length)

        Returns:
            x: Encoded features (batch_size, new_length, attention_dim)
            mask: Mask for the encoded features
        """

        # Handle input layer based on its type
        if len(self.embed) > 1:  # conv2d case
            # Add channel dimension: (batch_size, time_length, feat_dim) -> (batch_size, 1, time_length, feat_dim)
            x = mx.expand_dims(x, axis=1)

            # Apply convolutional layers
            for layer in self.embed[:2]:
                x = layer(x)

            # For each conv layer, the sequence length is reduced
            if mask is not None:
                # Downsample the mask to match the reduced sequence length
                mask = mx.reshape(mask, (mask.shape[0], -1, 2))[:, :, 0]

            for layer in self.embed[2:4]:
                x = layer(x)

            # Downsample mask again
            if mask is not None:
                mask = mx.reshape(mask, (mask.shape[0], -1, 2))[:, :, 0]

            for layer in self.embed[4:]:
                x = layer(x)

            # Downsample mask once more
            if mask is not None:
                mask = mx.reshape(mask, (mask.shape[0], -1, 2))[:, :, 0]

            # Rearrange from (batch_size, channels, time, freq) -> (batch_size, time, channels*freq)
            batch_size, channels, time, freq = x.shape
            x = mx.transpose(x, (0, 2, 1, 3))
            x = mx.reshape(x, (batch_size, time, channels * freq))

        else:  # Linear case
            x = self.embed[0](x)

        # Apply Conformer blocks
        for encoder in self.encoders:
            x = encoder(x, mask)

        # Final normalization
        x = self.norm(x)

        return x, mask


class AudioModel(nn.Module):
    """Audio embedding."""

    def __init__(self, config, **kwargs):
        super().__init__()
        self.config = config

        # Get hidden size for text LM
        hidden_size = config.n_embd if hasattr(config, "n_embd") else config.hidden_size

        # Dropout setting
        self.drop = None
        if hasattr(config, "embd_pdrop") or hasattr(config, "embed_pdrop"):
            embd_drop = (
                config.embd_pdrop
                if hasattr(config, "embd_pdrop")
                else config.embed_pdrop
            )
            self.embd_drop = embd_drop  # Store dropout rate

        # Configure audio processor
        if (
            isinstance(config.audio_processor, dict)
            and config.audio_processor.get("name", None) == "cascades"
        ):
            encoder_config = config.audio_processor.get("config", None)
            assert encoder_config is not None

            # Create encoder
            self.encoder = ConformerEncoder(**encoder_config)

            audio_dim_out = encoder_config["attention_dim"]
            n_mels = encoder_config["input_size"]
        else:
            raise NotImplementedError("Audio processor not implemented")

        self.audio_dim_out = audio_dim_out
        self.audio_dim_in = n_mels

        # Configuration
        self.freeze_audio_processor = kwargs.get("freeze_audio_processor", False)
        self.downsample_rate = kwargs.get("downsample_rate", 1)

        # Projection layer
        projection_cls = kwargs.get("projection_cls", "linear")
        if projection_cls == "linear":
            self.audio_projection = nn.Linear(audio_dim_out, hidden_size)
        elif projection_cls == "mlp":
            # Follow llava-v1.5's implementation
            dim_projection = hidden_size
            depth = 2
            self.linear_downsample_rate = self.downsample_rate

            # Create projection for speech mode
            layers_for_speech = [
                nn.Linear(audio_dim_out * self.linear_downsample_rate, dim_projection)
            ]
            for _ in range(1, depth):
                layers_for_speech.extend(
                    [nn.GELU(), nn.Linear(dim_projection, dim_projection)]
                )

            audio_projection_for_speech = layers_for_speech

            # Create projection for vision mode
            layers_for_vision = [
                nn.Linear(audio_dim_out * self.linear_downsample_rate, dim_projection)
            ]
            for _ in range(1, depth):
                layers_for_vision.extend(
                    [nn.GELU(), nn.Linear(dim_projection, dim_projection)]
                )

            audio_projection_for_vision = layers_for_vision

            # Store as a dictionary
            self.audio_projection = {
                "speech": audio_projection_for_speech,
                "vision": audio_projection_for_vision,
            }
        else:
            raise NotImplementedError(f"projection_cls = {projection_cls}")

        self.vocab_size = config.vocab_size
        self.input_embeds = None
        self.audio_embed_sizes = None

    def post_init(self, audio_config):
        """Initialize the audio encoder with pretrained weights."""
        if audio_config.get("name", None) == "cascades":
            init_model_config = audio_config.get("init_model", {})
            self.encoder.post_init(init_model_config)

            # Remove init_model to save memory
            if "init_model" in audio_config:
                audio_config.pop("init_model")

    def set_audio_embeds(self, input_embeds):
        self.input_embeds = input_embeds

    def set_audio_embed_sizes(self, audio_embed_sizes):
        self.audio_embed_sizes = audio_embed_sizes

    def get_audio_features(
        self, input_embeds, audio_attention_mask, audio_projection_mode="speech"
    ):
        """Process audio inputs through the encoder and projection layers."""

        # Apply encoder with or without gradient based on freeze setting
        if self.freeze_audio_processor:
            # In MLX, we would implement a mechanism to stop gradient flow
            audio_features, masks = self.encoder(input_embeds, audio_attention_mask)
        else:
            audio_features, masks = self.encoder(input_embeds, audio_attention_mask)

        # Apply projection based on its type
        if isinstance(self.audio_projection, dict):
            # Sequential projection for the specified mode
            projection_layers = self.audio_projection[audio_projection_mode]

            # Apply the layers in sequence
            audio_set_tensor = audio_features
            for layer in projection_layers:
                audio_set_tensor = layer(audio_set_tensor)
        else:
            # Single linear projection
            audio_set_tensor = self.audio_projection(audio_features)

        return audio_set_tensor

    def __call__(
        self,
        input_ids,
        input_embeds,
        audio_embed_sizes=None,
        audio_attention_mask=None,
        audio_projection_mode="speech",
        **kwargs,
    ):
        """
        Forward pass for audio embeddings.

        Args:
            input_ids: Input text ids (B, U)
            input_embeds: Audio features (B, T, D)  B: num audios in a sequence
        """
        # Use cached inputs if available
        if self.input_embeds is not None:
            input_embeds = self.input_embeds.copy()
            self.input_embeds = None

        if self.audio_embed_sizes is not None:
            audio_embed_sizes = self.audio_embed_sizes.copy()
            self.audio_embed_sizes = None

        # Reshape input_ids if needed
        input_shape = input_ids.shape
        input_ids = mx.reshape(input_ids, (-1, input_shape[-1]))

        # Find positions of audio token IDs
        positions = mx.array(np.nonzero(input_ids == _AUDIO_SPECIAL_TOKEN_ID)[0])

        # Determine target device and dtype from projection layer
        if isinstance(self.audio_projection, dict):
            target_dtype = mx.float32  # Default dtype
        else:
            target_dtype = mx.float32

        # Convert input_embeds to target dtype if available
        if input_embeds is not None:
            input_embeds = input_embeds.astype(target_dtype)

        # Process audio if audio tokens are present
        if len(positions) > 0:
            audio_set_tensor = self.get_audio_features(
                input_embeds, audio_attention_mask, audio_projection_mode
            )
        else:
            # Create dummy audio tensor for training
            if True:  # Equivalent to self.training in PyTorch
                audio_embeds = mx.zeros((1, 500, self.audio_dim_in), dtype=target_dtype)
                audio_attention_mask = mx.ones((1, 500), dtype=mx.int32)
                audio_set_tensor = self.get_audio_features(
                    audio_embeds, audio_attention_mask, audio_projection_mode
                )

        # Get token embeddings
        hidden_states = kwargs["wte"](input_ids)

        if len(positions) > 0:
            # Validate that we have correct number of positions
            assert audio_embed_sizes.sum().item() == len(
                positions
            ), f"Number of encoder outputs ({audio_embed_sizes.sum().item()}) must match number of audio tokens ({len(positions)})"

            # Create a list of audio features based on sizes
            merged_audio_set_tensor = []
            start_idx = 0
            for i in range(len(audio_embed_sizes)):
                size = audio_embed_sizes[i]
                merged_audio_set_tensor.append(audio_set_tensor[i, :size])
                start_idx += size

            # Concatenate all features
            merged_audio_set_tensor = mx.concatenate(merged_audio_set_tensor, axis=0)
            merged_audio_set_tensor = merged_audio_set_tensor.astype(
                hidden_states.dtype
            )

            # Create a new hidden_states with audio embeddings inserted
            for i, pos in enumerate(positions):
                batch_idx, seq_idx = pos
                # Update the tensor at the specified position
                hidden_states = mx.indexed_update(
                    hidden_states,
                    ((batch_idx, seq_idx),),
                    mx.expand_dims(merged_audio_set_tensor[i], axis=0),
                )
        else:
            # For training with no audio tokens, add a small contribution to maintain gradient flow
            if True:  # Equivalent to self.training
                hidden_states = hidden_states + (0 * audio_set_tensor[:, 0]).sum()

        # Apply dropout if configured
        if self.drop is not None:
            hidden_states = nn.Dropout(self.embd_drop)(hidden_states)

        return hidden_states

    def sanitize(self, weights):
        sanitized_weights = {}
        for k, v in weights.items():
            if (
                "embed.conv" in k
                and k.endswith("weight")
                and any(f"embed.conv.{i}" in k for i in range(9))
            ):
                # Check if the weight tensor is already in the correct format
                if check_array_shape(v):
                    sanitized_weights[k] = v
                else:
                    # PyTorch conv2d weight tensors have shape:
                    #   [out_channels, in_channels, kH, KW]
                    # MLX conv2d expects the weight be of shape:
                    #   [out_channels, kH, KW, in_channels]
                    sanitized_weights[k] = v.transpose(0, 2, 3, 1)
            elif (
                "ext_pw_conv_1d" in k
                or "depthwise_seperable_conv_1d" in k
                or "dw_sep_conv_1d" in k
            ) and k.endswith("weight"):
                # Check if the weight tensor is already in the correct format
                if check_array_shape(v):
                    sanitized_weights[k] = v
                else:
                    sanitized_weights[k] = v.transpose(0, 2, 1)
            else:
                sanitized_weights[k] = v

        return sanitized_weights
