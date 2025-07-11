import inspect
from collections.abc import Sequence
from dataclasses import dataclass
from math import sqrt
from typing import Dict, List, Optional, Tuple, Type

import mlx.core as mx
import mlx.nn as nn

from mlx_vlm.models.gemma3n.config import VisionConfig

from ..base import check_array_shape
from ..kernels import bicubic_interpolate, nearest_interpolate


# https://github.com/huggingface/new-model-addition-timm-gemma3p5-non-fork/blob/mobilenet-gemma3n-rw/timm/models/mobilenetv5.py#L24
class MobileNetV5MultiScaleFusionAdapter(nn.Module):
    """Multi-layer fusion token adapter.
    Attributes:
      out_filters: The number of output filters.
      output_resolution: The output resolution.
      activation: The activation function.
      expansion_ratio: The expansion ratio.
      upsampling_interpolation: The upsampling interpolation.
      use_layer_scale: Whether to use layer scale.
      layer_scale_init_value: The initial value of the layer scale.
      skip_projection: Whether to skip the projection.
      name: The name of the module.
      upsize: The upsampling fn.
      downsize: The downsampling fn.
    """

    def __init__(
        self,
        in_chs: List[int],
        out_chs: int,
        output_resolution: int,
        expansion_ratio: float = 2.0,
        interpolation_mode: str = "nearest",
        use_layer_scale: bool = False,
        layer_scale_init_value: float = 1e-5,
        noskip: bool = True,
    ):
        super().__init__()
        self.in_channels = sum(in_chs) if isinstance(in_chs, Sequence) else in_chs
        self.out_channels = out_chs
        self.output_resolution = to_2tuple(output_resolution)
        self.expansion_ratio = expansion_ratio
        self.interpolation_mode = interpolation_mode
        self.use_layer_scale = use_layer_scale
        self.layer_scale_init_value = layer_scale_init_value
        self.noskip = noskip

        norm_layer = RMSNormAct2d
        self.ffn = UniversalInvertedResidual(
            in_chs=self.in_channels,
            out_chs=self.out_channels,
            dw_kernel_size_mid=0,
            exp_ratio=self.expansion_ratio,
            norm_layer=norm_layer,
            noskip=self.noskip,
            layer_scale_init_value=(
                self.layer_scale_init_value if self.use_layer_scale else None
            ),
        )

        self.norm = norm_layer(self.out_channels, eps=1e-6, apply_act=False)

    def __call__(self, inputs: list[mx.array]) -> mx.array:
        inputs = [i.transpose(0, 3, 1, 2) for i in inputs]
        high_resolution = inputs[0].shape[
            -2:
        ]  # Assuming the first input is the highest resolution.
        resized_inputs = []

        for _, img in enumerate(inputs):
            if any([r < hr for r, hr in zip(img.shape[-2:], high_resolution)]):
                img = nearest_interpolate(img, size=high_resolution)

            resized_inputs.append(img)

        channel_cat_imgs = mx.concatenate(
            resized_inputs, axis=1
        )  # Cat on channel dim, must equal self.in_channels
        img = self.ffn(channel_cat_imgs.swapaxes(1, 3)).swapaxes(1, 3)

        if any([ro != rh for ro, rh in zip(high_resolution, self.output_resolution)]):
            if (
                high_resolution[0] % self.output_resolution[0] != 0
                or high_resolution[1] % self.output_resolution[1] != 0
            ):
                img = bicubic_interpolate(img, self.output_resolution)
            else:
                h_strides = high_resolution[0] // self.output_resolution[0]
                w_strides = high_resolution[1] // self.output_resolution[1]

                img = nn.AvgPool2d(
                    kernel_size=(h_strides, w_strides),
                    stride=(h_strides, w_strides),
                )(img.swapaxes(1, 3))

            img = self.norm(img) if self.noskip else img

        return img


# https://github.com/huggingface/new-model-addition-timm-gemma3p5-non-fork/blob/mobilenet-gemma3n-rw/timm/layers/layer_scale.py#L22
class LayerScale2d(nn.Module):
    def __init__(self, dim: int, init_values: float = 1e-5, inplace: bool = False):
        super().__init__()
        self.inplace = inplace
        self.gamma = init_values * mx.ones((dim,))

    def __call__(self, x: mx.array) -> mx.array:
        return x.mul_(self.gamma) if self.inplace else x * self.gamma


def rms_norm2d(
    x: mx.array,
    normalized_shape: List[int],
    weight: Optional[mx.array] = None,
    eps: float = 1e-5,
):
    assert len(normalized_shape) == 1
    dtype = x.dtype
    v = mx.power(x, 2)
    v = mx.mean(v, axis=1, keepdims=True)
    x = x * mx.rsqrt(v + eps)
    if weight is not None:
        x = x.astype(dtype) * weight.reshape(1, -1, 1, 1)
    return x


# https://github.com/huggingface/new-model-addition-timm-gemma3p5-non-fork/blob/mobilenet-gemma3n-rw/timm/layers/norm_act.py#L504
class RMSNormAct2d(nn.RMSNorm):
    def __init__(
        self,
        num_channels,
        eps=1e-6,
        apply_act: bool = True,
    ):
        super().__init__(dims=num_channels, eps=eps)
        self.normalized_shape = [num_channels]
        self.drop = nn.Identity()
        self.act = nn.GELU() if apply_act else nn.Identity()

    def __call__(self, x: mx.array) -> mx.array:

        x = x.transpose(0, 3, 1, 2)  # Convert from NHWC to NCHW
        x = rms_norm2d(x, self.normalized_shape, self.weight, self.eps)
        x = self.drop(x)
        x = self.act(x)
        x = x.transpose(0, 2, 3, 1)  # Convert back to NHWC
        return x


# https://github.com/huggingface/new-model-addition-timm-gemma3p5-non-fork/blob/mobilenet-gemma3n-rw/timm/models/_efficientnet_blocks.py#L310
class UniversalInvertedResidual(nn.Module):
    def __init__(
        self,
        in_chs: int,
        out_chs: int,
        dw_kernel_size_start: int = 0,
        dw_kernel_size_mid: int = 3,
        dw_kernel_size_end: int = 0,
        stride: int = 1,
        dilation: int = 1,
        group_size: int = 1,
        pad_type: str = "",
        noskip: bool = False,
        exp_ratio: float = 1.0,
        norm_layer=RMSNormAct2d,
        conv_kwargs: Optional[Dict] = None,
        drop_path_rate: float = 0.0,
        layer_scale_init_value: Optional[float] = 1e-5,
    ):
        super().__init__()
        self.has_skip = (in_chs == out_chs and stride == 1) and not noskip
        if stride > 1:
            assert dw_kernel_size_start or dw_kernel_size_mid or dw_kernel_size_end

        if dw_kernel_size_start:
            dw_start_stride = stride if not dw_kernel_size_mid else 1
            dw_start_groups = num_groups(group_size, in_chs)
            self.dw_start = ConvNormAct(
                nn.Conv2d,
                in_chs,
                in_chs,
                kernel_size=dw_kernel_size_start,
                stride=dw_start_stride,
                padding=(dw_kernel_size_start - 1) // 2,
                dilation=dilation,
                groups=dw_start_groups,
                bias=False,
                apply_act=False,
                eps=1e-05,
            )
        else:
            self.dw_start = nn.Identity()

        mid_chs = make_divisible(in_chs * exp_ratio)
        self.pw_exp = ConvNormAct(
            nn.Conv2d,
            in_chs,
            mid_chs,
            kernel_size=1,
            stride=1,
            padding=0,
            groups=1,
            bias=False,
            eps=1e-05,
        )

        if dw_kernel_size_mid:
            dw_mid_groups = num_groups(group_size, mid_chs)
            self.dw_mid = ConvNormAct(
                Conv2dSame,
                mid_chs,
                mid_chs,
                kernel_size=dw_kernel_size_mid,
                stride=stride,
                padding=0,
                dilation=dilation,
                groups=dw_mid_groups,
                bias=False,
                eps=1e-05,
            )
        else:
            self.dw_mid = nn.Identity()

        self.pw_proj = ConvNormAct(
            nn.Conv2d,
            mid_chs,
            out_chs,
            kernel_size=1,
            stride=1,
            padding=0,
            groups=1,
            bias=False,
            apply_act=False,
            eps=1e-05,
        )
        if layer_scale_init_value is not None:
            self.layer_scale = LayerScale2d(out_chs, layer_scale_init_value)
        else:
            self.layer_scale = nn.Identity()

    def __call__(self, x: mx.array) -> mx.array:
        shortcut = x
        x = self.dw_start(x)
        x = self.pw_exp(x)
        x = self.dw_mid(x)
        x = self.pw_proj(x)
        x = self.layer_scale(x)
        if self.has_skip:
            x = x + shortcut
        return x


# https://github.com/huggingface/new-model-addition-timm-gemma3p5-non-fork/blob/mobilenet-gemma3n-rw/timm/layers/conv_bn_act.py#L15
class ConvNormAct(nn.Module):
    def __init__(
        self,
        conv_cls,
        in_chs: int,
        out_chs: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 0,
        dilation: int = 1,
        groups: int = 1,
        bias: bool = False,
        apply_act: bool = True,
        eps: float = 1e-6,
    ):
        super().__init__()
        self.out_chs = out_chs
        self.conv = conv_cls(
            in_chs,
            out_chs,
            kernel_size,
            stride,
            padding,
            (dilation, dilation),
            groups,
            bias,
        )
        self.bn = RMSNormAct2d(out_chs, eps=eps, apply_act=apply_act)

    def __call__(self, x: mx.array) -> mx.array:
        c = self.conv(x)
        r = self.bn(c)
        return r


def pad_same(
    x,
    kernel_size: List[int],
    stride: List[int],
    dilation: List[int] = (1, 1),
    value: float = 0,
):
    """
    Input should be in MLX format
    """
    ih, iw = x.shape[1:3]
    pad_h = get_same_padding(ih, kernel_size[0], stride[0], dilation[0])
    pad_w = get_same_padding(iw, kernel_size[1], stride[1], dilation[1])

    # MLX pad format: [(low, high), (low, high), ...] for each axis
    # Padding order is reversed compared to PyTorch F.pad
    pad_widths = [
        (0, 0),  # No padding for batch dimension
        (pad_h // 2, pad_h - pad_h // 2),  # Height padding
        (pad_w // 2, pad_w - pad_w // 2),  # Width padding
        (0, 0),  # No padding for channel dimension
    ]

    x = mx.pad(x, pad_widths, constant_values=value)
    return x


def get_padding_value(padding, kernel_size, **kwargs) -> Tuple[Tuple, bool]:
    dynamic = False
    if isinstance(padding, str):
        # for any string padding, the padding will be calculated for you, one of three ways
        padding = padding.lower()
        if padding == "same":
            # TF compatible 'SAME' padding, has a performance and GPU memory allocation impact
            if is_static_pad(kernel_size, **kwargs):
                # static case, no extra overhead
                padding = get_padding(kernel_size, **kwargs)
            else:
                # dynamic 'SAME' padding, has runtime/GPU memory overhead
                padding = 0
                dynamic = True
        elif padding == "valid":
            # 'VALID' padding, same as padding=0
            padding = 0
        else:
            # Default to PyTorch style 'same'-ish symmetric padding
            padding = get_padding(kernel_size, **kwargs)
    return padding, dynamic


def get_same_padding(
    input_size: int, kernel_size: int, stride: int, dilation: int = 1
) -> int:
    """Calculate padding needed for 'same' output size."""
    effective_kernel_size = dilation * (kernel_size - 1) + 1
    output_size = (input_size + stride - 1) // stride
    total_padding = max(
        0, (output_size - 1) * stride + effective_kernel_size - input_size
    )
    return total_padding


def get_padding(kernel_size, stride=1, dilation=1, **_):
    """Get symmetric padding for given kernel size."""
    if isinstance(kernel_size, int):
        kernel_size = [kernel_size, kernel_size]
    if isinstance(stride, int):
        stride = [stride, stride]
    if isinstance(dilation, int):
        dilation = [dilation, dilation]

    padding = []
    for k, d in zip(kernel_size, dilation):
        effective_k = d * (k - 1) + 1
        pad_total = effective_k - 1
        padding.append(pad_total // 2)
    return tuple(padding)


def is_static_pad(kernel_size, stride=1, dilation=1, **_):
    """Check if padding can be calculated statically."""
    if isinstance(kernel_size, int):
        kernel_size = [kernel_size, kernel_size]
    if isinstance(stride, int):
        stride = [stride, stride]
    if isinstance(dilation, int):
        dilation = [dilation, dilation]

    # Static padding is possible when stride is 1 for all dimensions
    return all(s == 1 for s in stride)


class Conv2dSame(nn.Conv2d):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.kernel_size = self.weight.shape[1:3]

    def __call__(self, x: mx.array) -> mx.array:
        x = pad_same(x, self.kernel_size, self.stride, self.dilation)
        y = mx.conv2d(
            x, self.weight, self.stride, self.padding, self.dilation, self.groups
        )
        if "bias" in self:
            y = y + self.bias
        return y


# https://github.com/huggingface/new-model-addition-timm-gemma3p5-non-fork/blob/mobilenet-gemma3n-rw/timm/models/_efficientnet_blocks.py#L629
class EdgeResidual(nn.Module):
    def __init__(
        self,
        in_chs: int,
        out_chs: int,
        exp_kernel_size: int = 3,
        stride: int = 1,
        dilation: int = 1,
        group_size: int = 0,
        pad_type: str = "",
        force_in_chs: int = 0,
        noskip: bool = False,
        expand_ratio: float = 1.0,
        pw_kernel_size: int = 1,
        norm_layer=RMSNormAct2d,
    ):
        super().__init__()

        if force_in_chs > 0:
            mid_chs = make_divisible(force_in_chs * expand_ratio)
        else:
            mid_chs = make_divisible(in_chs * expand_ratio)

        groups = num_groups(group_size, mid_chs)

        self.has_skip = (in_chs == out_chs and stride == 1) and not noskip

        self.conv_exp = Conv2dSame(
            in_chs,
            mid_chs,
            kernel_size=exp_kernel_size,
            stride=stride,
            padding=0,
            dilation=(dilation, dilation),
            groups=groups,
            bias=False,
        )

        self.bn1 = norm_layer(mid_chs, eps=1e-05) if norm_layer else nn.Identity()

        # Point-wise linear projection
        padding_pwl = (pw_kernel_size - 1) // 2
        self.conv_pwl = nn.Conv2d(
            mid_chs,
            out_chs,
            kernel_size=pw_kernel_size,
            padding=padding_pwl,
            bias=False,
        )

        self.bn2 = (
            norm_layer(out_chs, eps=1e-05, apply_act=False)
            if norm_layer
            else nn.Identity()
        )

    def __call__(self, x: mx.array) -> mx.array:
        shortcut = x
        x = self.conv_exp(x)
        x = self.bn1(x)
        x = self.conv_pwl(x)
        x = self.bn2(x)
        if self.has_skip:
            x = x + shortcut
        return x


# https://github.com/huggingface/new-model-addition-timm-gemma3p5-non-fork/blob/mobilenet-gemma3n-rw/timm/models/_efficientnet_blocks.py#L449
class MobileAttention(nn.Module):
    def __init__(
        self,
        in_chs: int,
        out_chs: int,
        stride: int = 1,
        dw_kernel_size: int = 3,
        dilation: int = 1,
        group_size: int = 1,
        pad_type: str = "",
        num_heads: int = 8,
        key_dim: int = 64,
        value_dim: int = 64,
        use_multi_query: bool = True,
        query_strides: Tuple[int, int] = (1, 1),
        kv_stride: int = 1,
        cpe_dw_kernel_size: int = 3,
        noskip: bool = False,
        act_layer=nn.GELU,
        aa_layer=None,
        drop_path_rate: float = 0.0,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        layer_scale_init_value: Optional[float] = 1e-5,
        use_bias: bool = False,
    ):
        super().__init__()
        self.has_skip = (stride == 1 and in_chs == out_chs) and not noskip
        self.query_strides = to_2tuple(query_strides)
        self.kv_stride = kv_stride
        self.has_query_stride = any([s > 1 for s in self.query_strides])

        # Normalization layer
        self.norm = RMSNormAct2d(
            in_chs,
            eps=1e-05,
            apply_act=False,
        )
        # Determine number of heads if not provided
        if num_heads is None:
            assert in_chs % key_dim == 0
            num_heads = in_chs // key_dim

        # Attention layer
        if use_multi_query:
            self.attn = MultiQueryAttention2d(
                in_chs,
                dim_out=out_chs,
                num_heads=num_heads,
                key_dim=key_dim,
                value_dim=value_dim,
                query_strides=query_strides,
                kv_stride=kv_stride,
                dilation=dilation,
                padding=pad_type,
                dw_kernel_size=dw_kernel_size,
                attn_drop=attn_drop,
                proj_drop=proj_drop,
            )
        else:
            raise NotImplementedError("attention not implemented")

        # Layer scaling
        if layer_scale_init_value is not None:
            self.layer_scale = LayerScale2d(out_chs, layer_scale_init_value)
        else:
            self.layer_scale = nn.Identity()

        # Drop path for residual connection
        self.drop_path = nn.Identity()

    def __call__(self, x: mx.array) -> mx.array:
        shortcut = x
        x = self.norm(x)
        x = self.attn(x)
        x = self.layer_scale(x)

        # Apply skip connection if available
        if self.has_skip:
            x = self.drop_path(x) + shortcut
        return x


def create_conv2d(
    in_channels,
    out_channels,
    kernel_size,
    stride=1,
    dilation=1,
    depthwise=False,
    bias=False,
    **kwargs,
):
    """Helper function to create a 2D convolution with common parameters"""
    if depthwise:
        # Depthwise convolution
        return nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=(kernel_size - 1) // 2 * dilation,
            dilation=dilation,
            groups=in_channels,
            bias=bias,
        )
    else:
        # Regular convolution
        return nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=(kernel_size - 1) // 2 * dilation,
            dilation=dilation,
            bias=bias,
        )


def to_2tuple(x):
    """Convert input to 2-tuple"""
    if isinstance(x, tuple):
        return x
    return (x, x)


class NamedSequential(nn.Module):
    def __init__(self):
        super().__init__()
        self._order = []

    def add_module(self, name, module):
        setattr(self, name, module)
        self._order.append(name)

    def __call__(self, x):
        for name in self._order:
            x = getattr(self, name)(x)
        return x


# https://github.com/huggingface/new-model-addition-timm-gemma3p5-non-fork/blob/mobilenet-gemma3n-rw/timm/layers/attention2d.py#L82
class MultiQueryAttention2d(nn.Module):
    def __init__(
        self,
        dim: int,
        dim_out: Optional[int] = None,
        num_heads: int = 8,
        key_dim: int = 64,
        value_dim: int = 64,
        query_strides: Tuple[int, int] = (1, 1),
        kv_stride: int = 1,
        dilation: int = 1,
        padding: str = "",
        dw_kernel_size: int = 3,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
    ):
        super().__init__()
        dim_out = dim_out or dim
        self.num_heads = num_heads
        self.query_strides = to_2tuple(query_strides)
        self.kv_stride = kv_stride
        self.fused_attn = True
        self.key_dim = key_dim
        self.value_dim = value_dim
        head_dim = key_dim
        self.scale = head_dim**-0.5

        self.query = NamedSequential()
        self.query.add_module(
            "proj",
            create_conv2d(
                dim,
                self.num_heads * self.key_dim,
                kernel_size=1,
            ),
        )
        self.key = NamedSequential()
        if kv_stride > 1:
            self.key.add_module(
                "down_conv",
                create_conv2d(
                    dim,
                    dim,
                    kernel_size=dw_kernel_size,
                    stride=kv_stride,
                    dilation=dilation,
                    padding=padding,
                    depthwise=True,
                ),
            )
            self.key.add_module("norm", RMSNormAct2d(dim, eps=1e-6, apply_act=False))
        self.key.add_module(
            "proj", create_conv2d(dim, key_dim, kernel_size=1, bias=False)
        )

        self.value = NamedSequential()
        if kv_stride > 1:
            self.value.add_module(
                "down_conv",
                create_conv2d(
                    dim,
                    dim,
                    kernel_size=dw_kernel_size,
                    stride=kv_stride,
                    dilation=dilation,
                    padding=padding,
                    depthwise=True,
                ),
            )
            self.value.add_module("norm", RMSNormAct2d(dim, eps=1e-6, apply_act=False))
        self.value.add_module(
            "proj", create_conv2d(dim, value_dim, kernel_size=1, bias=False)
        )

        # Attention dropout
        self.attn_drop = nn.Dropout(attn_drop) if attn_drop > 0 else nn.Identity()

        # Output projection
        self.output = NamedSequential()
        self.output.add_module(
            "proj",
            create_conv2d(
                value_dim * num_heads,
                dim_out,
                kernel_size=1,
                stride=1,
                bias=False,
            ),
        )
        self.proj_drop = nn.Dropout(proj_drop) if proj_drop > 0 else nn.Identity()

    def _reshape_input(self, t: mx.array):
        """
        Input shape MLX: [B, H, W, C]
        Input shape PyTorch: [B, C, H, W]

        PyTorch Reshape: [B, C, H, W] -> [B, C, -1] -> [B, -1, C] -> [B, 1, -1, C] -> SDPA
        MLX Reshape: [B, H, W, C] -> [B, -1, C] -> [B, 1, -1, C] -> SDPA
        """
        s = t.shape
        t = t.reshape(s[0], -1, s[3])[:, None, :, :]

        return t

    def _reshape_projected_query(self, t: mx.array, num_heads: int, key_dim: int):
        """
        Input shape MLX: [B, H, W, C] where C = num_heads * key_dim
        """
        B, H, W, C = t.shape
        # t = t.reshape(B, H, W, num_heads, key_dim)
        t = t.reshape(B, H * W, num_heads, key_dim)
        return t.transpose(0, 2, 1, 3)

    def _reshape_output(self, t: mx.array, num_heads: int, h_px: int, w_px: int):
        """
        Input shape: [B, NH, L, D] where L = h_px * w_px
        Output shape MLX: [B, H, W, C] where C = NH * D
        """
        B, NH, L, D = t.shape
        # First transpose to [B, L, NH, D]
        t = t.transpose(0, 2, 1, 3)
        # Then reshape to [B, H, W, NH*D]
        t = t.reshape(B, h_px, w_px, NH * D)
        return t

    def __call__(self, x: mx.array, attn_mask: Optional[mx.array] = None) -> mx.array:
        B, H, W, C = x.shape
        q = self.query(x)
        q = self._reshape_projected_query(q, self.num_heads, self.key_dim)

        k = self.key(x)
        k = self._reshape_input(k)

        v = self.value(x)
        v = self._reshape_input(v)

        if self.fused_attn:
            o = mx.fast.scaled_dot_product_attention(
                q,
                k,
                v,
                scale=1.0 / sqrt(q.shape[-1]),
            )
        else:
            raise NotImplementedError("unfused attention not implemented")

        o = self._reshape_output(
            o, self.num_heads, H // self.query_strides[0], W // self.query_strides[1]
        )
        x = self.output(o)
        return x


def num_groups(group_size: Optional[int], channels: int) -> int:
    if not group_size:  # 0 or None
        return 1  # normal conv with 1 group
    else:
        # NOTE group_size == 1 -> depthwise conv
        assert channels % group_size == 0
        return channels // group_size


def make_divisible(v, divisor: int = 8, min_value=None, round_limit: float = 0.9):
    min_value = min_value or divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < round_limit * v:
        new_v += divisor
    return new_v


@dataclass(frozen=True)
class EdgeResidualConfig:
    kernel_size: int = 3
    filters: int = 32
    strides: int = 1
    expand_ratio: float = 4.0
    is_multiscale: bool = False


def _er(kernel_size, filters, strides=1, expand_ratio=4.0, is_multiscale=False):
    return EdgeResidualConfig(
        kernel_size=kernel_size,
        filters=filters,
        strides=strides,
        expand_ratio=expand_ratio,
        is_multiscale=is_multiscale,
    )


@dataclass(frozen=True)
class UniversalInvertedResidualConfig:
    start_dw_kernel_size: int = 0  # Zero size means no conv
    mid_dw_kernel_size: int = 0  # Zero size means no conv
    filters: int = 32
    strides: int = 1
    expand_ratio: float = 4.0
    is_multiscale: bool = False


def _uir(
    start_dw_kernel_size,
    mid_dw_kernel_size,
    filters,
    strides=1,
    expand_ratio=4.0,
    is_multiscale=False,
):
    return UniversalInvertedResidualConfig(
        start_dw_kernel_size=start_dw_kernel_size,
        mid_dw_kernel_size=mid_dw_kernel_size,
        filters=filters,
        strides=strides,
        expand_ratio=expand_ratio,
        is_multiscale=is_multiscale,
    )


@dataclass(frozen=True)
class MultiQueryAttentionBlockConfig:
    num_heads: int = 8
    kv_dim: int = 16
    kv_strides: int = 1
    mmqa_avg_pool_kv: bool = False
    mmqa_dropout: float = 0.0
    mmqa_dw_kernel_size: int = 3
    is_multiscale: bool = False


def _mmqa(
    num_heads,
    kv_dim,
    kv_strides,
    mmqa_avg_pool_kv=False,
    is_multiscale=False,
):
    conf = MultiQueryAttentionBlockConfig(
        num_heads=num_heads,
        kv_dim=kv_dim,
        kv_strides=kv_strides,
        mmqa_avg_pool_kv=mmqa_avg_pool_kv,
        is_multiscale=is_multiscale,
    )
    return conf


# https://github.com/huggingface/new-model-addition-timm-gemma3p5-non-fork/blob/mobilenet-gemma3n-rw/timm/models/mobilenetv5.py#L596
def gemma3n_mobilenet_def():
    return [
        # Stage 1: Edge Residuals
        [_er(3, 128, 2)] + [_er(3, 128, 1)] * 2,
        # Stage 2: Universal Inverted Residuals
        [_uir(3, 5, 256, 2, 6.0)] + [_uir(k, 0, 256) for k in [5, 3, 5, 3]],
        # Stage 3: Universal Inverted Residuals with Multi-Query Attention
        [_uir(5, 5, 640, 2, 6.0)]
        + [_uir(5, 0, 640)] * 7
        + [_uir(0, 0, 640, 1, 1.0)]
        + [_mmqa(12, 64, 2), _uir(0, 0, 640, 1, 2.0)] * 13
        + [_mmqa(12, 64, 2), _uir(0, 0, 640, 1, 2.0, is_multiscale=True)],
        # Stage 4: Universal Inverted Residuals with Multi-Query Attention
        [_uir(5, 5, 1280, 2, 6.0)]
        + [_mmqa(16, 96, 1), _uir(0, 0, 1280, 1, 2.0)] * 18
        + [_mmqa(16, 96, 1), _uir(0, 0, 1280, 1, 2.0, is_multiscale=True)],
    ]


class VisionTower(nn.Module):
    def __init__(self, config: VisionConfig):
        super().__init__()
        self.conv_stem = ConvNormAct(
            Conv2dSame,
            in_chs=3,
            out_chs=64,
            kernel_size=3,
            stride=2,
            padding=0,
            eps=1e-05,
            bias=True,
        )
        msfa_indices = (3, 4)
        msfa_output_resolution = (16, 16)

        (num_features, self.blocks) = self.build()
        self.num_features = self.head_hidden_size = (
            num_features  # output of msfa is output of forward_features()
        )
        self.msfa_indices = msfa_indices
        self.msfa_output_resolution = msfa_output_resolution

        self.msfa = MobileNetV5MultiScaleFusionAdapter(
            in_chs=[1920],
            out_chs=2048,
            output_resolution=self.msfa_output_resolution,
        )

    def build(self):
        blocks = []
        in_chs = self.conv_stem.out_chs
        for stage, block_config in enumerate(gemma3n_mobilenet_def()):
            block_group = []
            for config in block_config:
                match config:
                    case EdgeResidualConfig(
                        kernel_size, filters, strides, expand_ratio, is_multiscale
                    ):
                        x = EdgeResidual(
                            exp_kernel_size=kernel_size,
                            in_chs=in_chs,
                            out_chs=filters,
                            stride=strides,
                            expand_ratio=expand_ratio,
                        )
                        in_chs = filters  # in_chs of next is out_chs of prev
                        block_group.append(x)
                    case UniversalInvertedResidualConfig(
                        start_dw_kernel_size,
                        mid_dw_kernel_size,
                        filters,
                        strides,
                        expand_ratio,
                        is_multiscale,
                    ):
                        x = UniversalInvertedResidual(
                            in_chs=in_chs,
                            out_chs=filters,
                            dw_kernel_size_start=start_dw_kernel_size,
                            dw_kernel_size_mid=mid_dw_kernel_size,
                            stride=strides,
                            exp_ratio=expand_ratio,
                        )
                        in_chs = filters
                        block_group.append(x)
                    case MultiQueryAttentionBlockConfig(
                        num_heads,
                        kv_dim,
                        kv_strides,
                        mmqa_avg_pool_kv,
                        is_multiscale,
                    ):
                        x = MobileAttention(
                            in_chs=in_chs,
                            out_chs=in_chs,
                            stride=1,
                            num_heads=num_heads,
                            key_dim=kv_dim,
                            value_dim=kv_dim,
                            kv_stride=kv_strides,
                            act_layer=None,
                        )
                        block_group.append(x)
                    case _:
                        continue
            blocks.append(block_group)
        return (in_chs, blocks)

    def __call__(
        self, x: mx.array, output_hidden_states: Optional[bool] = None
    ) -> mx.array:
        feat_idx = 0
        x = x.transpose(0, 2, 3, 1)  # Convert from NCHW to NHWC
        x = self.conv_stem(x)
        intermediates = []

        if feat_idx in self.msfa_indices:
            intermediates.append(x)

        # MBV5 is constructed of 4 stages, each stage is a group of blocks.
        for block_group in self.blocks:
            feat_idx += 1
            for block in block_group:
                x = block(x)

            if feat_idx in self.msfa_indices:
                intermediates.append(x)

        x = self.msfa(intermediates)
        return x


class VisionModel(nn.Module):
    def __init__(self, config: VisionConfig):
        super().__init__()
        self.model_type = config.model_type
        if self.model_type not in ["gemma3", "gemma3_vision", "gemma3n_vision"]:
            raise ValueError(f"Unsupported model type: {self.model_type}")

        self.timm_model = VisionTower(config)

    def __call__(
        self, x: mx.array, output_hidden_states: Optional[bool] = None
    ) -> mx.array:
        return self.timm_model(x, output_hidden_states)

    def sanitize(self, weights):
        sanitized_weights = {}
        skip_transpose = False
        _, H, _, C = weights["vision_tower.timm_model.blocks.0.0.conv_exp.weight"].shape
        if C > H:
            skip_transpose = True

        for k, v in weights.items():
            # PyTorch conv2d weight: [out_channels, in_channels, kH, kW]
            # MLX conv2d weight: [out_channels, kH, KW, in_channels]
            if ("conv" in k and "weight" in k) or ("attn" and "proj.weight") in k:
                if len(v.shape) == 4 and not skip_transpose:
                    v = v.transpose(0, 2, 3, 1)
            sanitized_weights[k] = v

        return sanitized_weights
