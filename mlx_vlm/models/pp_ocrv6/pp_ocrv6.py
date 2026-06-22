import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import mlx.core as mx
import mlx.nn as nn
import numpy as np
from PIL import Image

from .config import ModelConfig


@dataclass
class ModelOutput:
    last_hidden_state: mx.array
    hidden_states: Optional[tuple[mx.array, ...]] = None

    @property
    def logits(self):
        return self.last_hidden_state


def _pair(value):
    if isinstance(value, (list, tuple)):
        return tuple(value)
    return (value, value)


def _act(name):
    if name is None:
        return lambda x: x
    if name == "relu":
        return nn.relu
    if name == "gelu":
        return nn.gelu
    if name == "silu" or name == "swish":
        return nn.silu
    if name == "hardswish":
        return nn.Hardswish()
    raise ValueError(f"Unsupported activation: {name}")


def _hard_sigmoid(x):
    return mx.clip(0.2 * x + 0.5, 0.0, 1.0)


def _to_nhwc(x):
    if x.ndim == 4 and x.shape[1] in (1, 3):
        return x.transpose(0, 2, 3, 1)
    return x


def _to_nchw(x):
    return x.transpose(0, 3, 1, 2)


def _same_pad_2x2(x):
    return mx.pad(x, [(0, 0), (0, 1), (0, 1), (0, 0)])


def _upsample(x, scale):
    if scale == 1:
        return x
    return mx.repeat(mx.repeat(x, scale, axis=1), scale, axis=2)


class ConvBNLayer(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        groups=1,
        activation="relu",
        bias=False,
        padding=None,
    ):
        super().__init__()
        kernel_size = _pair(kernel_size)
        stride = _pair(stride)
        if padding is None:
            padding = (kernel_size[0] // 2, kernel_size[1] // 2)
        self.convolution = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            groups=groups,
            bias=bias,
        )
        self.normalization = nn.BatchNorm(out_channels)
        self.activation = _act(activation)

    def __call__(self, x):
        return self.activation(self.normalization(self.convolution(x)))


class HeadConvBNLayer(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=1,
        activation="relu",
        bias=False,
        transpose=False,
    ):
        super().__init__()
        if transpose:
            self.convolution = nn.ConvTranspose2d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                stride=stride,
                bias=True,
            )
        else:
            self.convolution = nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                bias=bias,
            )
        self.norm = nn.BatchNorm(out_channels)
        self.activation = _act(activation)

    def __call__(self, x):
        return self.activation(self.norm(self.convolution(x)))


class SqueezeExcitation(nn.Module):
    def __init__(self, channels, reduction=4, names_for_backbone=False):
        super().__init__()
        if names_for_backbone:
            self.convolutions = [
                nn.Conv2d(channels, channels // reduction, kernel_size=1),
                nn.ReLU(),
                nn.Conv2d(channels // reduction, channels, kernel_size=1),
            ]
        else:
            self.conv1 = nn.Conv2d(channels, channels // reduction, kernel_size=1)
            self.conv2 = nn.Conv2d(channels // reduction, channels, kernel_size=1)
        self.names_for_backbone = names_for_backbone

    def __call__(self, x):
        residual = x
        y = mx.mean(x, axis=(1, 2), keepdims=True)
        if self.names_for_backbone:
            y = self.convolutions[0](y)
            y = nn.relu(y)
            y = self.convolutions[2](y)
        else:
            y = nn.relu(self.conv1(y))
            y = self.conv2(y)
        return residual * _hard_sigmoid(y)


class LargeStem(nn.Module):
    def __init__(self, config):
        super().__init__()
        s = config.stem_channels
        self.stem1 = ConvBNLayer(
            s[0], s[1], 3, stride=config.stem_strides[0], activation=config.hidden_act
        )
        self.stem2a = ConvBNLayer(
            s[1],
            s[1] // 2,
            2,
            stride=config.stem_strides[1],
            activation=config.hidden_act,
            padding=0,
        )
        self.stem2b = ConvBNLayer(
            s[1] // 2,
            s[1],
            2,
            stride=config.stem_strides[2],
            activation=config.hidden_act,
            padding=0,
        )
        self.stem3 = ConvBNLayer(
            s[1] * 2,
            s[1],
            3,
            stride=config.stem_strides[3],
            activation=config.hidden_act,
        )
        self.stem4 = ConvBNLayer(
            s[1],
            s[2],
            1,
            stride=config.stem_strides[4],
            activation=config.hidden_act,
            padding=0,
        )
        self.pool = nn.MaxPool2d(kernel_size=2, stride=1)

    def __call__(self, x):
        x = self.stem1(x)
        padded = _same_pad_2x2(x)
        x2 = self.stem2b(_same_pad_2x2(self.stem2a(padded)))
        x1 = self.pool(padded)
        return self.stem4(self.stem3(mx.concatenate([x1, x2], axis=-1)))


class SmallStem(nn.Module):
    def __init__(self, config):
        super().__init__()
        s = config.stem_channels
        self.conv1 = ConvBNLayer(s[0], s[1], 3, stride=2, activation=None)
        self.conv2 = ConvBNLayer(s[1], s[2], 3, stride=2, activation=None)

    def __call__(self, x):
        return self.conv2(nn.gelu(self.conv1(x)))


class LCNetV4Layer(nn.Module):
    def __init__(self, in_channels, out_channels, stride, kernel_size, use_se, config):
        super().__init__()
        stride = _pair(stride)
        self.has_residual = in_channels == out_channels and stride == (1, 1)
        self.use_rep_dw = self.has_residual
        if self.use_rep_dw:
            self.token_conv = nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=kernel_size // 2,
                groups=in_channels,
            )
        else:
            self.token_conv = ConvBNLayer(
                in_channels,
                in_channels,
                kernel_size,
                stride=stride,
                groups=in_channels,
                activation=None,
            )
        self.token_squeeze_excitation = (
            SqueezeExcitation(in_channels, config.reduction, names_for_backbone=True)
            if use_se
            else None
        )
        self.channel_conv1 = ConvBNLayer(
            in_channels, in_channels * 2, 1, activation=None, padding=0
        )
        self.channel_conv2 = ConvBNLayer(
            in_channels * 2, out_channels, 1, activation=None, padding=0
        )

    def __call__(self, x):
        x = self.token_conv(x)
        if self.token_squeeze_excitation is not None:
            x = self.token_squeeze_excitation(x)
        residual = x
        x = self.channel_conv1(x)
        x = nn.gelu(x)
        x = self.channel_conv2(x)
        return residual + x if self.has_residual else x


class LCNetV4Stage(nn.Module):
    def __init__(self, config, stage_index):
        super().__init__()
        self.blocks = [
            LCNetV4Layer(in_c, out_c, stride, kernel, use_se, config)
            for kernel, in_c, out_c, stride, use_se in config.block_configs[stage_index]
        ]

    def __call__(self, x):
        for block in self.blocks:
            x = block(x)
        return x


class LCNetV4Encoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.convolution = (
            LargeStem(config) if config.stem_type == "large" else SmallStem(config)
        )
        self.blocks = [
            LCNetV4Stage(config, i) for i in range(len(config.block_configs))
        ]

    def __call__(self, x):
        hidden_states = []
        x = self.convolution(x)
        hidden_states.append(x)
        for block in self.blocks:
            x = block(x)
            hidden_states.append(x)
        return tuple(hidden_states)


class LCNetV4Backbone(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.encoder = LCNetV4Encoder(config)
        self.out_indices = config.out_indices or [1, 2, 3, 4]

    def __call__(self, pixel_values):
        hidden_states = self.encoder(_to_nhwc(pixel_values))
        return tuple(hidden_states[i] for i in self.out_indices), hidden_states


class SmallDetDepthwiseSeparableConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, reduction):
        super().__init__()
        self.depthwise_convolution = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
            groups=in_channels,
        )
        self.pointwise_convolution = nn.Conv2d(
            out_channels, out_channels // 4, kernel_size=1, bias=False
        )
        self.squeeze_excitation_module = SqueezeExcitation(out_channels // 4, reduction)

    def __call__(self, x):
        x = self.pointwise_convolution(self.depthwise_convolution(x))
        return x + self.squeeze_excitation_module(x)


class ResidualSELayer(nn.Module):
    def __init__(self, in_channels, out_channels, reduction):
        super().__init__()
        self.in_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.squeeze_excitation_block = SqueezeExcitation(out_channels, reduction)

    def __call__(self, x):
        x = self.in_conv(x)
        return x + self.squeeze_excitation_block(x)


class SmallDetNeck(nn.Module):
    def __init__(self, config):
        super().__init__()
        channels = (
            config.layer_list_out_channels or config.backbone_config.stage_out_channels
        )
        self.insert_conv = [
            ResidualSELayer(c, config.neck_out_channels, config.reduction)
            for c in channels
        ]
        self.input_conv = [
            SmallDetDepthwiseSeparableConv(
                config.neck_out_channels,
                config.neck_out_channels,
                config.dilated_kernel_size,
                config.reduction,
            )
            for _ in channels
        ]

    def __call__(self, feature_maps):
        fused = [conv(feature) for conv, feature in zip(self.insert_conv, feature_maps)]
        for i in range(2, -1, -1):
            fused[i] = fused[i] + _upsample(fused[i + 1], 2)
        features = [conv(feature) for conv, feature in zip(self.input_conv, fused)]
        upsampled = [
            features[0],
            _upsample(features[1], 2),
            _upsample(features[2], 4),
            _upsample(features[3], 8),
        ]
        return mx.concatenate(upsampled[::-1], axis=-1)


class MediumIntraclassBlock(nn.Module):
    def __init__(self, config, in_channels):
        super().__init__()
        cfg = config.intraclass_block_config
        reduced = in_channels // config.reduce_factor
        self.conv_reduce_channel = nn.Conv2d(
            in_channels, reduced, *cfg["reduce_channel"]
        )
        for name in [
            "vertical_long_to_small_conv_longratio",
            "vertical_long_to_small_conv_midratio",
            "vertical_long_to_small_conv_shortratio",
            "horizontal_small_to_long_conv_longratio",
            "horizontal_small_to_long_conv_midratio",
            "horizontal_small_to_long_conv_shortratio",
            "symmetric_conv_long_longratio",
            "symmetric_conv_long_midratio",
            "symmetric_conv_long_shortratio",
        ]:
            setattr(self, name, nn.Conv2d(reduced, reduced, *cfg[name]))
        self.conv_final = HeadConvBNLayer(
            reduced,
            in_channels,
            kernel_size=cfg["return_channel"][0],
            stride=cfg["return_channel"][1],
            padding=cfg["return_channel"][2],
            bias=True,
        )

    def __call__(self, x):
        residual = x
        x = self.conv_reduce_channel(x)
        x = (
            self.symmetric_conv_long_longratio(x)
            + self.vertical_long_to_small_conv_longratio(x)
            + self.horizontal_small_to_long_conv_longratio(x)
        )
        x = (
            self.symmetric_conv_long_midratio(x)
            + self.vertical_long_to_small_conv_midratio(x)
            + self.horizontal_small_to_long_conv_midratio(x)
        )
        x = (
            self.symmetric_conv_long_shortratio(x)
            + self.vertical_long_to_small_conv_shortratio(x)
            + self.horizontal_small_to_long_conv_shortratio(x)
        )
        return residual + self.conv_final(x)


class MediumDetNeck(nn.Module):
    def __init__(self, config):
        super().__init__()
        out = config.neck_out_channels
        quarter = out // 4
        channels = config.backbone_config.stage_out_channels
        self.scale_factor_list = config.scale_factor_list
        self.input_channel_adjustment_convolution = [
            nn.Conv2d(c, out, kernel_size=1, bias=False) for c in channels
        ]
        self.input_feature_projection_convolution = [
            nn.Conv2d(out, quarter, kernel_size=9, padding=4) for _ in channels
        ]
        self.path_aggregation_head_convolution = [
            nn.Conv2d(quarter, quarter, kernel_size=3, stride=2, padding=1, bias=False)
            for _ in channels[1:]
        ]
        self.path_aggregation_lateral_convolution = [
            nn.Conv2d(quarter, quarter, kernel_size=9, padding=4) for _ in channels
        ]
        self.intraclass_blocks = [
            MediumIntraclassBlock(config, quarter)
            for _ in range(config.intraclass_block_number)
        ]

    def __call__(self, feature_maps):
        adjusted = [
            conv(x)
            for conv, x in zip(self.input_channel_adjustment_convolution, feature_maps)
        ]
        top_down = [None] * 4
        top_down[3] = adjusted[3]
        for i in range(2, -1, -1):
            top_down[i] = adjusted[i] + _upsample(top_down[i + 1], 2)
        projected = [
            conv(top_down[i] if i < 3 else adjusted[-1])
            for i, conv in enumerate(self.input_feature_projection_convolution)
        ]
        bottom_up = [None] * 4
        bottom_up[0] = projected[0]
        for i in range(1, 4):
            bottom_up[i] = projected[i] + self.path_aggregation_head_convolution[i - 1](
                bottom_up[i - 1]
            )
        lateral = [
            self.path_aggregation_lateral_convolution[i](
                projected[0] if i == 0 else bottom_up[i]
            )
            for i in range(4)
        ]
        refined = [block(x) for block, x in zip(self.intraclass_blocks, lateral)]
        upsampled = [
            _upsample(x, scale) for x, scale in zip(refined, self.scale_factor_list)
        ]
        return mx.concatenate(upsampled[::-1], axis=-1)


class DetHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        in_channels = config.neck_out_channels
        self.conv_down = HeadConvBNLayer(
            in_channels,
            in_channels // 4,
            config.kernel_list[0],
            padding=config.kernel_list[0] // 2,
        )
        self.conv_up = HeadConvBNLayer(
            in_channels // 4,
            in_channels // 4,
            config.kernel_list[1],
            stride=2,
            transpose=True,
        )
        self.conv_final = nn.ConvTranspose2d(
            in_channels // 4, 1, kernel_size=config.kernel_list[2], stride=2
        )

    def __call__(self, x):
        return nn.sigmoid(self.conv_final(self.conv_up(self.conv_down(x))))


class TinyRecHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        in_channels = config.backbone_config.block_configs[-1][-1][2]
        self.conv1 = nn.Conv1d(
            in_channels,
            in_channels,
            kernel_size=5,
            padding=2,
            groups=in_channels,
            bias=False,
        )
        self.norm1 = nn.BatchNorm(in_channels)
        self.conv2 = nn.Conv1d(in_channels, in_channels, kernel_size=1, bias=False)
        self.norm2 = nn.BatchNorm(in_channels)
        self.fc1 = nn.Linear(in_channels, config.hidden_size)
        self.fc2 = nn.Linear(config.hidden_size, config.head_out_channels)
        self.act_fn = nn.Hardswish()

    def __call__(self, x):
        x = x.squeeze(1)
        x = self.act_fn(self.norm1(self.conv1(x)))
        x = self.act_fn(self.norm2(self.conv2(x)))
        return nn.softmax(self.fc2(self.fc1(x)), axis=-1)


class RecAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.num_heads = config.num_attention_heads
        self.head_dim = config.hidden_size // config.num_attention_heads
        self.scale = self.head_dim**-0.5
        self.qkv = nn.Linear(
            config.hidden_size, 3 * config.hidden_size, bias=config.qkv_bias
        )
        self.projection = nn.Linear(config.hidden_size, config.hidden_size)

    def __call__(self, x):
        b, length, dim = x.shape
        qkv = (
            self.qkv(x)
            .reshape(b, length, 3, self.num_heads, self.head_dim)
            .transpose(2, 0, 3, 1, 4)
        )
        q, k, v = qkv[0], qkv[1], qkv[2]
        out = mx.fast.scaled_dot_product_attention(q, k, v, scale=self.scale)
        return self.projection(out.transpose(0, 2, 1, 3).reshape(b, length, dim))


class RecMLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        hidden = int(config.hidden_size * config.mlp_ratio)
        self.fc1 = nn.Linear(config.hidden_size, hidden)
        self.fc2 = nn.Linear(hidden, config.hidden_size)
        self.activation = _act(config.hidden_act)

    def __call__(self, x):
        return self.fc2(self.activation(self.fc1(x)))


class RecBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.layer_norm1 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.self_attn = RecAttention(config)
        self.layer_norm2 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.mlp = RecMLP(config)

    def __call__(self, x):
        x = x + self.self_attn(self.layer_norm1(x))
        return x + self.mlp(self.layer_norm2(x))


class SmallRecEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        in_channels = config.backbone_config.block_configs[-1][-1][2]
        self.conv_block = [
            ConvBNLayer(
                in_channels,
                config.hidden_size,
                (1, 1),
                activation=config.hidden_act,
                padding=0,
            ),
            ConvBNLayer(
                in_channels,
                config.hidden_size,
                (1, 1),
                activation=config.hidden_act,
                padding=0,
            ),
            ConvBNLayer(
                config.hidden_size,
                config.hidden_size,
                config.conv_kernel_size,
                activation=config.hidden_act,
                groups=config.hidden_size,
            ),
        ]
        self.svtr_block = [RecBlock(config) for _ in range(config.depth)]
        self.norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

    def __call__(self, x):
        residual = self.conv_block[0](x)
        x = self.conv_block[1](x)
        x = x + self.conv_block[2](x)
        b, h, w, c = x.shape
        x = x.reshape(b, h * w, c)
        for block in self.svtr_block:
            x = block(x)
        x = self.norm(x).reshape(b, h, w, c)
        x = x + residual
        return x.squeeze(1)


class SmallRecHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.encoder = SmallRecEncoder(config)
        self.head = nn.Linear(config.hidden_size, config.head_out_channels)

    def __call__(self, x):
        return nn.softmax(self.head(self.encoder(x)), axis=-1)


class PPOCRV6Body(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.backbone = LCNetV4Backbone(config.backbone_config)
        if config.is_detection:
            self.neck = (
                MediumDetNeck(config)
                if config.model_type == "pp_ocrv6_medium_det"
                else SmallDetNeck(config)
            )

    def __call__(self, pixel_values):
        feature_maps, hidden_states = self.backbone(pixel_values)
        if hasattr(self, "neck"):
            return self.neck(feature_maps), hidden_states
        return feature_maps[-1], hidden_states


class Model(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.model = PPOCRV6Body(config)
        if config.is_detection:
            self.head = DetHead(config)
        elif config.model_type == "pp_ocrv6_tiny_rec":
            self.head = TinyRecHead(config)
        else:
            self.head = SmallRecHead(config)

    def __call__(self, pixel_values, **kwargs):
        hidden, hidden_states = self.model(pixel_values)
        if self.config.is_recognition:
            hidden = nn.AvgPool2d(kernel_size=(3, 2), stride=(3, 2))(hidden)
        out = self.head(hidden)
        if self.config.is_detection:
            out = _to_nchw(out)
        return ModelOutput(last_hidden_state=out, hidden_states=hidden_states)

    def sanitize(self, weights):
        new_weights = {}
        for key, value in weights.items():
            if "num_batches_tracked" in key:
                continue
            if value.ndim == 4:
                if key.endswith(("conv_up.convolution.weight", "conv_final.weight")):
                    value = value.transpose(1, 2, 3, 0)
                else:
                    value = value.transpose(0, 2, 3, 1)
            elif value.ndim == 3 and key.startswith("head.conv"):
                value = value.transpose(0, 2, 1)
            new_weights[key] = value
        return new_weights


class ImageProcessor:
    model_input_names = ["pixel_values"]

    def __init__(self, **kwargs):
        self.config = kwargs
        self.character_list = kwargs.get("character_list", [])
        processor_type = kwargs.get("image_processor_type") or ""
        self.is_detection = "Det" in processor_type
        self.size = kwargs.get("size")
        if self.size is None and not self.is_detection:
            self.size = {"height": 48, "width": 320}
        self.pad_size = kwargs.get("pad_size", self.size)
        self.image_mean = kwargs.get("image_mean", [0.485, 0.456, 0.406])
        self.image_std = kwargs.get("image_std", [0.229, 0.224, 0.225])
        self.rescale_factor = kwargs.get("rescale_factor", 1 / 255)
        self.do_pad = kwargs.get("do_pad", False)
        self.max_image_width = kwargs.get("max_image_width", 3200)
        self.limit_side_len = kwargs.get("limit_side_len", 736)
        self.limit_type = kwargs.get("limit_type", "min")
        self.image_mode = kwargs.get("image_mode", "BGR")

    @classmethod
    def from_pretrained(cls, path, **kwargs):
        path = Path(path)
        with open(path / "preprocessor_config.json", encoding="utf-8") as f:
            data = json.load(f)
        data.update(kwargs)
        return cls(**data)

    def save_pretrained(self, save_directory, **kwargs):
        save_directory = Path(save_directory)
        save_directory.mkdir(parents=True, exist_ok=True)
        with open(
            save_directory / "preprocessor_config.json", "w", encoding="utf-8"
        ) as f:
            json.dump(self.config, f, ensure_ascii=False, indent=2)

    def _to_array(self, image):
        if isinstance(image, Image.Image):
            image = image.convert("RGB")
            arr = np.array(image)
        else:
            arr = np.array(image)
            if arr.ndim == 2:
                arr = np.stack([arr, arr, arr], axis=-1)
        if self.image_mode == "BGR":
            arr = arr[..., ::-1]
        return arr

    def _resize_rec(self, arr):
        h, w = arr.shape[:2]
        target_h = int(self.size["height"])
        default_w = int(self.size["width"])
        ratio = max(w / h, default_w / target_h)
        target_w = min(
            self.max_image_width, max(default_w, math.ceil(target_h * ratio))
        )
        return np.array(
            Image.fromarray(arr).resize((target_w, target_h), Image.Resampling.BILINEAR)
        )

    def _resize_det(self, arr):
        h, w = arr.shape[:2]
        if self.limit_type == "max":
            scale = (
                self.limit_side_len / max(h, w)
                if max(h, w) > self.limit_side_len
                else 1.0
            )
        elif self.limit_type == "min":
            scale = (
                self.limit_side_len / min(h, w)
                if min(h, w) < self.limit_side_len
                else 1.0
            )
        else:
            scale = self.limit_side_len / max(h, w)
        new_h = max(32, int(round(h * scale / 32) * 32))
        new_w = max(32, int(round(w * scale / 32) * 32))
        if (
            self.config.get("max_side_limit")
            and max(new_h, new_w) > self.config["max_side_limit"]
        ):
            scale = self.config["max_side_limit"] / max(new_h, new_w)
            new_h = max(32, int(round(new_h * scale / 32) * 32))
            new_w = max(32, int(round(new_w * scale / 32) * 32))
        return np.array(
            Image.fromarray(arr).resize((new_w, new_h), Image.Resampling.BILINEAR)
        )

    def preprocess(self, images, return_tensors="mlx"):
        if not isinstance(images, (list, tuple)):
            images = [images]
        is_rec = not self.is_detection
        arrays = []
        target_sizes = []
        for image in images:
            arr = self._to_array(image)
            target_sizes.append(arr.shape[:2])
            arr = self._resize_rec(arr) if is_rec else self._resize_det(arr)
            arr = arr.astype(np.float32) * self.rescale_factor
            arr = (arr - np.array(self.image_mean, dtype=np.float32)) / np.array(
                self.image_std, dtype=np.float32
            )
            if is_rec and self.do_pad and arr.shape[1] < self.pad_size["width"]:
                padded = np.zeros(
                    (arr.shape[0], self.pad_size["width"], arr.shape[2]),
                    dtype=arr.dtype,
                )
                padded[:, : arr.shape[1], :] = arr
                arr = padded
            arrays.append(arr.transpose(2, 0, 1))
        pixel_values = mx.array(np.stack(arrays))
        result = {"pixel_values": pixel_values}
        if self.is_detection:
            result["target_sizes"] = target_sizes
        return result

    __call__ = preprocess

    def post_process_text_recognition(self, predictions):
        logits = predictions.last_hidden_state
        probs = mx.max(logits, axis=-1)
        ids = mx.argmax(logits, axis=-1)
        results = []
        for row_ids, row_probs in zip(ids.tolist(), probs.tolist()):
            chars = []
            scores = []
            prev = None
            for idx, score in zip(row_ids, row_probs):
                if idx != 0 and idx != prev and idx < len(self.character_list):
                    chars.append(self.character_list[idx])
                    scores.append(score)
                prev = idx
            results.append(
                {
                    "text": "".join(chars),
                    "score": float(np.mean(scores)) if scores else 0.0,
                }
            )
        return results

    def _unclip(self, contour_box, unclip_ratio):
        import cv2

        polygon = contour_box.reshape(-1, 2).astype(np.float32)
        perimeter = cv2.arcLength(polygon, True)
        if perimeter == 0:
            return np.array([polygon])
        offset = cv2.contourArea(polygon) * unclip_ratio / perimeter
        x, y = polygon[:, 0], polygon[:, 1]
        ccw = (x @ np.roll(y, -1) - y @ np.roll(x, -1)) > 0
        edges = np.roll(polygon, -1, axis=0) - polygon
        directions = edges / np.maximum(
            np.linalg.norm(edges, axis=1, keepdims=True), 1e-6
        )
        normals = (
            np.stack([directions[:, 1], -directions[:, 0]], axis=1)
            if ccw
            else np.stack([-directions[:, 1], directions[:, 0]], axis=1)
        )
        shifted = polygon + offset * normals
        prev_shifted = np.roll(shifted, 1, axis=0)
        prev_dirs = np.roll(directions, 1, axis=0)
        cross = prev_dirs[:, 0] * directions[:, 1] - prev_dirs[:, 1] * directions[:, 0]
        parallel = np.abs(cross) < 1e-6
        cross = np.where(parallel, 1.0, cross)
        vec = shifted - prev_shifted
        t = (vec[:, 0] * directions[:, 1] - vec[:, 1] * directions[:, 0]) / cross
        vertices = prev_shifted + prev_dirs * t[:, None]
        if np.any(parallel):
            vertices[parallel] = polygon[parallel] + 0.5 * offset * (
                np.roll(normals, 1, axis=0)[parallel] + normals[parallel]
            )
        return np.array([vertices.astype(np.float32)])

    def _get_mini_boxes(self, contour):
        import cv2

        bounding_box = cv2.minAreaRect(contour)
        points = sorted(cv2.boxPoints(bounding_box), key=lambda x: x[0])
        if points[1][1] > points[0][1]:
            index_1, index_4 = 0, 1
        else:
            index_1, index_4 = 1, 0
        if points[3][1] > points[2][1]:
            index_2, index_3 = 2, 3
        else:
            index_2, index_3 = 3, 2
        return [
            points[index_1],
            points[index_2],
            points[index_3],
            points[index_4],
        ], min(bounding_box[1])

    def _get_box_score(self, bitmap, box):
        import cv2

        height, width = bitmap.shape[:2]
        box = box.copy()
        xmin = max(0, min(math.floor(box[:, 0].min()), width - 1))
        xmax = max(0, min(math.ceil(box[:, 0].max()), width - 1))
        ymin = max(0, min(math.floor(box[:, 1].min()), height - 1))
        ymax = max(0, min(math.ceil(box[:, 1].max()), height - 1))
        mask = np.zeros((ymax - ymin + 1, xmax - xmin + 1), dtype=np.uint8)
        box[:, 0] -= xmin
        box[:, 1] -= ymin
        cv2.fillPoly(mask, box.reshape(1, -1, 2).astype(np.int32), 1)
        return cv2.mean(bitmap[ymin : ymax + 1, xmin : xmax + 1], mask)[0]

    def _boxes_from_bitmap(
        self,
        prediction,
        bitmap,
        dest_width,
        dest_height,
        box_threshold,
        unclip_ratio,
        min_size,
        max_candidates,
    ):
        import cv2

        height, width = bitmap.shape
        width_scale = dest_width / width
        height_scale = dest_height / height
        contours = cv2.findContours(
            (bitmap * 255).astype(np.uint8), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE
        )
        contours = contours[1] if len(contours) == 3 else contours[0]
        boxes, scores = [], []
        for contour in contours[:max_candidates]:
            points, short_side = self._get_mini_boxes(contour)
            if short_side < min_size:
                continue
            points = np.array(points)
            score = self._get_box_score(prediction, points.reshape(-1, 2))
            if score < box_threshold:
                continue
            box, short_side = self._get_mini_boxes(
                self._unclip(points, unclip_ratio).reshape(-1, 1, 2)
            )
            if short_side < min_size + 2:
                continue
            box = np.array(box)
            box[:, 0] = np.clip(np.round(box[:, 0] * width_scale), 0, dest_width)
            box[:, 1] = np.clip(np.round(box[:, 1] * height_scale), 0, dest_height)
            boxes.append(box.astype(np.int16))
            scores.append(float(score))
        return np.array(boxes, dtype=np.int16), scores

    def post_process_object_detection(
        self,
        predictions,
        threshold=0.3,
        target_sizes=None,
        box_threshold=0.6,
        max_candidates=1000,
        min_size=3,
        unclip_ratio=1.5,
    ):
        if target_sizes is None:
            raise ValueError(
                "target_sizes must be provided for post_process_object_detection"
            )
        logits = np.array(predictions.last_hidden_state)
        results = []
        for prediction, (src_height, src_width) in zip(logits, target_sizes):
            prediction = prediction[0]
            boxes, scores = self._boxes_from_bitmap(
                prediction,
                prediction > threshold,
                src_width,
                src_height,
                box_threshold,
                unclip_ratio,
                min_size,
                max_candidates,
            )
            results.append(
                {
                    "boxes": boxes,
                    "scores": scores,
                    "labels": np.zeros(len(scores), dtype=np.int64),
                }
            )
        return results
