import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Callable, Optional, OrderedDict, Tuple, Union

import mlx.core as mx
import mlx.nn as nn
import numpy as np


@dataclass
class AudioConfig:
    input_feat_size: int = 80
    hidden_size: int = 1536
    conf_attention_chunk_size: int = 12
    conf_attention_context_left: int = 13
    conf_attention_context_right: int = 0
    conf_attention_logit_cap: float = 50.0
    conf_num_attention_heads: int = 8
    conf_num_hidden_layers: int = 12
    conf_conv_kernel_size: int = 5
    conf_positional_bias_size: int = 256
    conf_reduction_factor: int = 4
    conf_residual_weight: float = 0.5
    sscp_conv_channel_size: tuple[int, int] = (128, 32)
    sscp_conv_kernel_size: tuple[tuple[int, int], tuple[int, int]] = ((3, 3), (3, 3))
    sscp_conv_stride_size: tuple[tuple[int, int], tuple[int, int]] = ((2, 2), (2, 2))


# (x: mx.array, mask: mx.BoolArray (no BoolArray in mlx))
type SLSequence = Tuple[mx.array, mx.array]


class SequenceLayer(nn.Module):
    layers: Callable[[SLSequence], SLSequence]

    def __call__(self, x: SLSequence) -> SLSequence:
        return self.layers(x)


class SequenceLayerConv2d(SequenceLayer):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel: tuple[int, int],
        stride: tuple[int, int],
        *args,
        padding: tuple[int, int] = (0, 0),
        use_bias: bool = False,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel = kernel
        self.stride = stride

        self.padding = padding
        self.use_bias = use_bias

        self.conv = nn.Conv2d(
            in_channels=self.in_channels,
            out_channels=self.out_channels,
            kernel_size=self.kernel,
            stride=self.stride,
            padding=self.padding,
            bias=self.use_bias,
        )

    def __call__(self, x: SLSequence) -> SLSequence:
        y, mask = x
        y = self.conv(y)
        return y, mask


class SequenceLayerEinsum(SequenceLayer):

    def __init__(self, shape: Sequence[int], equation: str, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.shape = shape
        self.equation = equation
        self.weight = mx.empty(self.shape)

    def forward(self, x: SLSequence) -> SLSequence:
        y, mask = x
        y = mx.einsum(self.equation, y, self.weight)
        return y, mask


class SequenceLayerDense(SequenceLayerEinsum):
    def __init__(self, shape: tuple[int, int], *args, **kwargs):
        super().__init__(*args, shape=shape, equation="...a,ab->...b", **kwargs)


class SequenceLayerDenseShaped(SequenceLayer):
    def __init__(
        self,
        *args,
        input_shape: Sequence[int] = (),
        output_shape: Sequence[int] = (),
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.input_shape = tuple(input_shape)
        self.input_dims = "".join(
            chr(ord("a") + i) for i in range(len(self.input_shape))
        )
        self.input_weight_shape = self.input_shape or (1,)
        self.input_weight_dims = self.input_dims or "I"
        self.output_shape = tuple(output_shape)
        self.output_dims = "".join(
            chr(ord("a") + i + len(self.input_shape))
            for i in range(len(self.output_shape))
        )
        self.output_weight_shape = self.output_shape or (1,)
        self.output_weight_dims = self.output_dims or "O"
        self.equation = f"BT{self.input_dims},{self.input_weight_dims}{self.output_weight_dims}->BT{self.output_dims}"

        weight_shape = self.input_weight_shape + self.output_weight_shape
        self.weight = mx.empty(weight_shape)

    def __call__(self, x: SLSequence) -> SLSequence:
        y, mask = x
        y = mx.einsum(self.equation, y, self.weight)
        return y, mask


class SequenceLayerDepthwiseConv1D(SequenceLayer):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        num_groups: int,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.conv = nn.Conv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=1,
            padding=0,  # Manual causal padding
            groups=num_groups,  # Depthwise
            bias=False,
        )

    def forward(self, x: SLSequence) -> SLSequence:
        y, mask = x
        y = self.conv(y)
        return y, mask


class SequenceLayerExpandDims(SequenceLayer):
    def __init__(self, dims: Union[int, Sequence[int]], *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.dims = (dims,) if isinstance(dims, int) else dims

    def _normalize_dims(
        self,
        x_ndims: int,
    ) -> Sequence[int]:
        dims = [d + x_ndims if d < 0 else d for d in self.dims]
        dims = sorted(dims)
        for d in dims:
            if d < 0 or d > x_ndims:
                raise ValueError(f"Received invalid dim for expansion: {d}")
        return dims

    def __call__(self, x: SLSequence) -> SLSequence:
        y, mask = x
        y_dims = self._normalize_dims(y.ndim)
        for d in y_dims:
            y = mx.expand_dims(y, axis=d)
        return y, mask


class SequenceLayerGatedLinearUnit(SequenceLayer):
    def __call__(self, x: SLSequence) -> SLSequence:
        x, mask = x
        feature, gate = mx.split(x, 2, dim=-1)
        gate = mx.sigmoid(gate)
        x = feature * gate
        return x, mask


class SequenceLayerGroupNorm(SequenceLayer):
    def __init__(
        self, num_groups: int, num_channels: int, *args, eps: float = 1e-3, **kwargs
    ):
        super().__init__(*args, **kwargs)

        self.num_groups = num_groups
        self.num_channels = num_channels
        self.eps = eps

        self.norm = nn.GroupNorm(
            num_groups=self.num_groups, num_channels=self.num_channels, eps=self.eps
        )

    def __call__(self, x: SLSequence) -> SLSequence:
        y, mask = x
        y = self.norm(y)
        return y, mask


class SequenceLayerLocalDotProductSelfAttention(SequenceLayer):
    def __init__(
        self,
        num_heads: int,
        hidden_size: int,
        block_size: int,
        max_past_horizon: int,
        relative_position_embedding: nn.Module,
        *args,
        max_future_horizon: int = 0,
        attention_invalid_logits_value: float = -1.0e9,
        attention_logits_soft_cap: float = 50.0,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.num_heads = num_heads
        self.hidden_size = hidden_size
        self.units_per_head = hidden_size // self.num_heads
        self.block_size = block_size
        self.max_past_horizon = max_past_horizon
        self.max_future_horizon = max_future_horizon
        self.attention_invalid_logits_value = attention_invalid_logits_value
        self.attention_logits_soft_cap = attention_logits_soft_cap

        if self.block_size < 1:
            raise ValueError(f"Expected {self.block_size=} >= 1.")
        if self.max_past_horizon < 1:
            raise ValueError(f"Expected {self.max_past_horizon=} >= 1.")
        if self.max_future_horizon < 0:
            raise ValueError(f"Expected {self.max_future_horizon=} >= 0.")
        if self.max_future_horizon == 0 and self.max_past_horizon == 0:
            raise ValueError("max_horizon and max_future_horizon cannot both be 0.")
        if self.attention_logits_soft_cap < 0.0:
            raise ValueError(f"{self.attention_logits_soft_cap=} should be None or non-negative.")

        self.relative_position_embedding = relative_position_embedding

        self.per_dim_scale = mx.empty((self.units_per_head,))

        self.qkv_proj = SequenceLayerEinsum(
            shape=(self.hidden_size, 3, self.num_heads, self.units_per_head),
            equation="...a,abcd->...bcd",
        )

    def _pad_dim1(
        self, x: mx.array, dim10_val: int, dim11_val: int, padding_val: Union[bool, float] = 0.0
    ) -> mx.array: 
        padding_tuple = [0] * x.ndim * 2
        dim_idx_from_end = x.ndim - 2
        start_idx_for_dim = 2 * dim_idx_from_end
        padding_tuple[start_idx_for_dim] = dim10_val
        padding_tuple[start_idx_for_dim + 1] = dim11_val
        padding_tuple = tuple(padding_tuple)
        x = mx.pad(x, padding_tuple, mode="constant", constant_value=padding_val)
        return x

    def _convert_to_block(self, x: mx.array, padding_val: Union[bool, float] = 0.0) -> mx.array:
        shape = x.shape
        b, t = shape[:2]
        num_blocks = (t + self.block_size - 1) // self.block_size

        if (padding_len := num_blocks * self.block_size - t) > 0:
            x = self._pad_dim1(x, 0, padding_len, padding_val)

        permute_dims = (b, num_blocks, self.block_size) + shape[2:]
        x = x.permute(permute_dims).contiguous()
        return x

    def _extract_block_context(self, x: mx.array, padding_val: Union[bool, float] = 0.0) -> mx.array:
        x = self._pad_dim1(x, self.max_past_horizon, self.max_future_horizon + self.block_size + 1, padding_val)

        outer_dims = x.shape[:1]
        inner_dims = x.shape[2:]

        target_dim = x.shape[1]
        frame_len = self.block_size + self.max_past_horizon + self.max_future_horizon
        frame_step = self.block_size

        output_size = target_dim - frame_len + 1
        num_frames = (output_size + frame_step - 1) // frame_step

        if not num_frames:
            return mx.zeros(outer_dims + (0, frame_len) + inner_dims, dtype=x.dtype, device=x.device)

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

    def forward(self, x: SLSequence) -> SLSequence:
        y, mask = x

        qkv: mx.Tensor = self.qkv_proj(y)
        
        q = None
        k = None
        v = None
        # TODO: mx.select doesn't exist

        #q = torch.select(qkv, dim=2, index=0).float()
        #k = torch.select(qkv, dim=2, index=1).float()
        #v = torch.select(qkv, dim=2, index=2).float()

        q_scale = 1 / math.sqrt(self.units_per_head)
        r_softplus_0 = 1.442695041  # Ported from JAX Sequence Layers; 1.0 / jax.nn.softplus(0.0)
        q_scale = mx.array(q_scale * r_softplus_0, dtype=mx.float32)
        q = q * q_scale * nn.softplus(self.per_dim_scale)

        batch_size, q_time = q.shape[:2]
        context_size = self.block_size + self.max_past_horizon + self.max_future_horizon

        k_blocks = self._extract_block_context(k)
        q_blocks = self._convert_to_block(q)
        num_query_blocks = q_blocks.shape[1]
        v_blocks = self._extract_block_context(v)

        valid_mask_blocks: mx.array = self._extract_block_context(mask, padding_val=False)
        valid_mask_blocks = valid_mask_blocks.unsqueeze(1).unsqueeze(-2)
        lower_causal_mask = mx.tril(
            mx.ones((context_size, self.block_size), dtype=mx.bool_),
            diagonal=0,
        ).T
        upper_causal_mask = mx.tril(
            mx.ones((self.block_size, context_size), dtype=mx.bool_),
            diagonal=self.max_past_horizon + self.max_future_horizon,
        )
        local_causal_valid_mask = mx.ones((self.block_size, context_size), dtype=mx.bool_)
        local_causal_valid_mask = local_causal_valid_mask * lower_causal_mask * upper_causal_mask
        valid_mask_blocks = mx.logical_and(valid_mask_blocks, local_causal_valid_mask)

        # Embed queries and keys
        logits = self.relative_position_embedding(q_blocks, k_blocks)

        # Apply attention logit softcap
        softcap = mx.array(self.attention_logits_soft_cap, dtype=mx.float32)
        logits = logits / softcap
        logits = mx.tanh(logits)
        logits = logits * softcap

        logits = mx.where(valid_mask_blocks, logits, self.attention_invalid_logits_value)
        probabilities = mx.softmax(logits, dim=-1, dtype=mx.float32)
        context_vectors = mx.einsum("BNuwc,BucNH->BuwNH", probabilities, v_blocks)
        context_vectors = context_vectors.reshape(
            (batch_size, num_query_blocks * self.block_size, self.num_heads, self.units_per_head)
        )
        context_vectors = context_vectors[:, :q_time]

        return context_vectors, mask


class SequenceLayerMaskInvalid(SequenceLayer):
    def __call__(self, x: SLSequence) -> SLSequence:
        y, mask = x
        if mask.dtype != mx.bool_:
             mask = mask.astype(mx.bool_)
        expanded_mask = mask.expand_dims(-1)
        fill_value = mx.array(0.0, dtype=y.dtype)
        y_masked = mx.where(expanded_mask, fill_value, y)
        return y_masked, mask

class SequenceLayerRelu(SequenceLayer):
    def __call__(self, x: SLSequence) -> SLSequence:
        x, mask = x
        x = nn.relu(x)
        return x, mask

class SequenceLayerResidual(SequenceLayer):
    def __init__(
        self,
        layers: nn.Sequential,
        *args,
        shortcut_layers: Optional[nn.Sequential] = None,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.layers = layers
        self.shortcut_layers = shortcut_layers

    def residual_function(self, x: SLSequence, shortcut_x: SLSequence) -> SLSequence:
        y = x[0] + shortcut_x[0]
        mask = x[1] | shortcut_x[1]
        return y, mask

    def __call__(self, x: SLSequence) -> SLSequence:
        y: SLSequence = self.layers(x)
        if self.shortcut_layers is not None:
            shortcut_y: SLSequence = self.shortcut_layers(x)
            y = self.residual_function(y, shortcut_y)
        return y


class SequenceLayerRMSNorm(SequenceLayer):
    def __init__(
        self, shape: Sequence[int], *args, dim: int = -1, eps: float = 1e-6, **kwargs
    ):
        super().__init__(*args, **kwargs)

        self.shape = shape
        self.dim = dim
        self.eps = eps
        self.scale = mx.ones(self.shape)

    def forward(self, x: SLSequence) -> SLSequence:
        y, mask = x
        y_dtype = y.dtype
        y = y.float()
        mean_squared = y.pow(2).mean(dim=self.dim, keepdim=True)
        root_mean_squared = y * mx.rsqrt(mean_squared + self.eps)
        scaled = root_mean_squared * self.scale.float()
        return scaled.type(y_dtype), mask


class SequenceLayerScale(SequenceLayer):
    def __init__(self, factor: float, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.factor = factor

    def forward(self, x: SLSequence) -> SLSequence:
        x, mask = x
        x = x * self.factor
        return x, mask


class SequenceLayerSwish(SequenceLayer):
    def forward(self, x: SLSequence) -> SLSequence:
        x, mask = x
        x = mx.silu(x)
        return x, mask


class SequenceLayerTransformerXLRelativePositionEmbedding(nn.Module):
    def __init__(
        self,
        num_heads: int,
        hidden_size: int,
        max_backward: int,
        max_forward: int,
        position_bias_dim: int,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.num_heads = num_heads
        self.hidden_size = hidden_size
        self.units_per_head = self.hidden_size // self.num_heads
        self.max_backward = max_backward
        self.max_forward = max_forward
        self.position_bias_dim = position_bias_dim

        self.pos_proj = SequenceLayerEinsum(
            shape=(self.hidden_size, self.num_heads, self.units_per_head),
            equation="...d,dnh->...nh",
        )

    def _get_timing_signal_1d_pos(self, position: mx.array, channels: int, dtype: mx.dtype) -> mx.array:
        assert position.ndim == 2
        position = position.float().unsqueeze(-1)

        min_timescale = 1.0
        max_timescale = 1.0e4
        num_timescales = channels // 2
        log_timescale_increment = math.log(float(max_timescale) / float(min_timescale)) / max(num_timescales - 1, 1)
        inv_timescales = min_timescale * mx.exp(mx.arange(num_timescales) * -log_timescale_increment)
        inv_timescales = inv_timescales.float().unsqueeze(0).unsqueeze(0).to(device=position.device)

        scaled_time = position * inv_timescales

        timing_signal = mx.concatenate([mx.sin(scaled_time), mx.cos(scaled_time)], dim=-1)
        timing_signal_padding = (0, np.mod(channels, 2), 0, 0, 0, 0)
        timing_signal = mx.pad(timing_signal, timing_signal_padding)

        return timing_signal.type(dtype)

    def forward(self, queries: mx.array, keys: mx.array) -> mx.array:
        b, u, w = queries.shape[:3]
        _, _, c = keys.shape[:3]
        n = self.num_heads
        l = self.max_backward
        r = self.max_forward
        lr = l + r
        assert c == w + lr

        pos = mx.arange(l, -r - 1, -1).unsqueeze(0)
        assert pos.shape == (1, lr + 1)

        sin_emb = self._get_timing_signal_1d_pos(pos, self.position_bias_dim, dtype=queries.dtype)
        sin_emb: mx.array = self.pos_proj((sin_emb, None))[0]
        sin_emb = sin_emb.squeeze(0)

        term_ac = mx.einsum("BuwNH,BucNH->BNuwc", queries, keys)
        term_bd = mx.einsum("BuwNH,FNH->BNuwF", queries, sin_emb)

        # Perform relative shift in order to get [B, N, U, W, C]
        # Pads the input to [B, N, U, W, C + 1]
        term_bd_pad = (0, c - lr, 0, 0, 0, 0, 0, 0, 0, 0)
        term_bd = nn.functional.pad(term_bd, term_bd_pad)
        term_bd = term_bd.reshape((b, n, u, w * (c + 1)))
        term_bd = term_bd[:, :, :, : w * c]
        # Reshapes to [B, N, U, W, C]. Note the output last dim is 1-smaller
        # than the input, which "pushses" one element off to the next row for each
        # row. The accumulated effect is row_i is right-shifted i steps (i>=0).
        term_bd = term_bd.reshape((b, n, u, w, c))
        return term_ac + term_bd


class Gemma3p5AudioSSCPConvBlock(SequenceLayer):
    def __init__(self, config: AudioConfig, idx: int, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.config = config

        self.out_channels = self.config.sscp_conv_channel_size[idx]
        self.kernel_size = self.config.sscp_conv_kernel_size[idx]
        self.stride = self.config.sscp_conv_stride_size[idx]

        # input_channels is equal to either the out_channels from the prior
        # Conv2d or 1 if this is the first Conv2d.
        if idx > 0:
            self.input_channels = self.config.sscp_conv_channel_size[idx - 1]
        else:
            self.input_channels = 1

        self.layers = nn.Sequential(
            OrderedDict(
                [
                    (
                        "conv2d",
                        SequenceLayerConv2d(
                            in_channels=self.input_channels,
                            out_channels=self.out_channels,
                            kernel=self.kernel_size,
                            stride=self.stride,
                        ),
                    ),
                    (
                        "norm",
                        SequenceLayerGroupNorm(
                            num_groups=1, num_channels=self.out_channels
                        ),
                    ),
                    ("relu", SequenceLayerRelu()),
                ]
            )
        )


class Gemma3p5AudioSubSampleConvProjection(SequenceLayer):
    def __init__(self, config: AudioConfig, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.config = config

        self.layers = nn.Sequential(
            OrderedDict(
                [
                    ("expand", SequenceLayerExpandDims(dims=-1)),
                    ("conv_0", Gemma3p5AudioSSCPConvBlock(config, 0)),
                    ("conv_1", Gemma3p5AudioSSCPConvBlock(config, 1)),
                    (
                        "input_proj",
                        SequenceLayerDenseShaped(
                            input_shape=(
                                self.config.sscp_conv_channel_size[1],
                                self.config.sscp_conv_channel_size[1],
                            ),
                            output_shape=(self.config.hidden_size,),
                        ),
                    ),
                ]
            )
        )


class Gemma3p5AudioConformerAttention(SequenceLayer):
    def __init__(self, config: AudioConfig, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.config = config

        self.layers = SequenceLayerResidual(
            layers=nn.Sequential(
                OrderedDict(
                    [
                        (
                            "pre_attn_norm",
                            SequenceLayerRMSNorm(shape=(self.config.hidden_size,)),
                        ),
                        
                ("attn", SequenceLayerLocalDotProductSelfAttention(
                    num_heads=self.config.conf_num_attention_heads,
                    hidden_size=self.config.hidden_size,
                    block_size=self.config.conf_attention_chunk_size,
                    attention_logits_soft_cap=self.config.conf_attention_logit_cap,
                    max_past_horizon=self.config.conf_attention_context_left,
                    max_future_horizon=self.config.conf_attention_context_right,
                    relative_position_embedding=SequenceLayerTransformerXLRelativePositionEmbedding(
                        num_heads=self.config.conf_num_attention_heads,
                        hidden_size=self.config.hidden_size,
                        max_backward=self.config.conf_attention_context_left,
                        max_forward=self.config.conf_attention_context_right,
                        position_bias_dim=self.config.hidden_size,
                    ),
                )),

                        (
                            "post_attn_dense",
                            SequenceLayerDenseShaped(
                                input_shape=(
                                    self.config.conf_num_attention_heads,
                                    self.config.hidden_size
                                    // self.config.conf_num_attention_heads,
                                ),
                                output_shape=(self.config.hidden_size,),
                            ),
                        ),
                        (
                            "post_attn_norm",
                            SequenceLayerRMSNorm(shape=(self.config.hidden_size,)),
                        ),
                    ]
                )
            )
        )


class Gemma3p5AudioConformerFeedForward(SequenceLayer):
    def __init__(self, config: AudioConfig, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.config = config

        self.layers = SequenceLayerResidual(
            layers=nn.Sequential(
                OrderedDict(
                    [
                        (
                            "pre_layer_norm",
                            SequenceLayerRMSNorm(shape=(self.config.hidden_size,)),
                        ),
                        (
                            "ffw_layer_1",
                            SequenceLayerDense(
                                shape=(
                                    self.config.hidden_size,
                                    self.config.hidden_size * 4,
                                )
                            ),
                        ),
                        (
                            "ffw_layer_2",
                            SequenceLayerDense(
                                shape=(
                                    self.config.hidden_size * 4,
                                    self.config.hidden_size,
                                )
                            ),
                        ),
                        (
                            "post_layer_norm",
                            SequenceLayerRMSNorm(shape=(self.config.hidden_size,)),
                        ),
                        (
                            "post_layer_scale",
                            SequenceLayerScale(factor=config.conf_residual_weight),
                        ),
                    ]
                )
            )
        )


class Gemma3p5AudioConformerLightConv1d(SequenceLayer):
    def __init__(self, config: AudioConfig, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.config = config

        self.layers = SequenceLayerResidual(
            layers=nn.Sequential(OrderedDict([
                ("pre_layer_norm", SequenceLayerRMSNorm(shape=(self.config.hidden_size, ))),
                ("linear_start", SequenceLayerDense(shape=(self.config.hidden_size, self.config.hidden_size * 2))),
                ("glu", SequenceLayerGatedLinearUnit()),
                ("depthwise_conv1d", SequenceLayerDepthwiseConv1D(
                    in_channels=self.config.hidden_size,
                    out_channels=self.config.hidden_size,
                    kernel_size=self.config.conf_conv_kernel_size,
                    num_groups=self.config.hidden_size,
                )),
                ("conv_norm", SequenceLayerRMSNorm(shape=(self.config.hidden_size, ))),
                ("conv_activation", SequenceLayerSwish()),
                ("linear_end", SequenceLayerDense(shape=(self.config.hidden_size, self.config.hidden_size))),
            ]))
        )




class Gemma3p5AudioConformerBlock(SequenceLayer):
    def __init__(self, config: AudioConfig, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.config = config

        self.layers = nn.Sequential(
            OrderedDict(
                [
                    ("ffw_layer_start", Gemma3p5AudioConformerFeedForward(self.config)),
                    ("attention", Gemma3p5AudioConformerAttention(self.config)),
                    ("lconv1d", Gemma3p5AudioConformerLightConv1d(self.config)),
                    ("ffw_layer_end", Gemma3p5AudioConformerFeedForward(self.config)),
                    ("norm", SequenceLayerRMSNorm(shape=(self.config.hidden_size,))),
                ]
            )
        )


class Gemma3p5AudioUniformReducer(nn.Module):
    def __init__(self, config: AudioConfig, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.config = config
        self.reduction_factor = config.conf_reduction_factor

    def __call__(self, x: mx.array, mask: mx.array) -> mx.array:
        if self.reduction_factor > 1:
            x = x[:, :: self.reduction_factor]
            mask = mask[:, :: self.reduction_factor]
        return x, mask


class AudioModel(nn.Module):
    def __init__(self, config: AudioConfig, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.config = config

        self.subsample_conv_projection = Gemma3p5AudioSubSampleConvProjection(config)
        self.conformer_blocks = [
            Gemma3p5AudioConformerBlock(config)
            for _ in range(config.conf_num_hidden_layers)
        ]
        self.uniform_reducer = Gemma3p5AudioUniformReducer(config)

        self.layers = nn.Sequential(OrderedDict([
            ("subsample_conv_projection", Gemma3p5AudioSubSampleConvProjection(config)),
            ("conformer", nn.Sequential(OrderedDict([
                (f"block_{i}", Gemma3p5AudioConformerBlock(config))
                for i in range(config.conf_num_hidden_layers)
            ]))),
            ("reducer", Gemma3p5AudioUniformReducer(config)),
            ("mask_invalid", SequenceLayerMaskInvalid()),
        ]))

    def __call__(self, x: mx.array) -> mx.array:
        raise NotImplementedError()
