"""PP-DocLayoutV3 hybrid encoder.

AIFI (single level) + top-down FPN + bottom-up PAN over the HGNetV2
features, plus the mask-feature path feeding the mask-enhanced query
init. NHWC MLX; mirrors transformers' PP-DocLayoutV3 implementation.
"""

from typing import Optional, Tuple

import mlx.core as mx
import mlx.nn as nn

from .config import LayoutConfig


def _resolve_activation(name: Optional[str]):
    if name is None:
        return None
    name = name.lower()
    if name == "relu":
        return nn.ReLU()
    if name == "silu":
        return nn.SiLU()
    if name == "gelu":
        return nn.GELU()
    raise ValueError(f"Unsupported activation {name!r}")


class Proj(nn.Module):
    """1x1 conv + BN (+ optional activation); covers encoder_input_proj,
    decoder_input_proj and lateral/downsample convs via key renames."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 1,
        stride: int = 1,
        activation: Optional[str] = "silu",
        eps: float = 1e-5,
    ) -> None:
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=(kernel_size - 1) // 2,
            bias=False,
        )
        self.bn = nn.BatchNorm(out_channels, eps=eps)
        self.activation = _resolve_activation(activation)

    def __call__(self, x: mx.array) -> mx.array:
        x = self.conv(x)
        x = self.bn(x)
        if self.activation is not None:
            x = self.activation(x)
        return x


class RepVgg(nn.Module):
    """Dual-branch (3x3 + 1x1) RepVGG block, branches kept separate."""

    def __init__(self, channels: int, activation: Optional[str], eps: float) -> None:
        super().__init__()
        self.conv1 = Proj(channels, channels, 3, 1, None, eps)
        self.conv2 = Proj(channels, channels, 1, 1, None, eps)
        self.activation = _resolve_activation(activation)

    def __call__(self, x: mx.array) -> mx.array:
        y = self.conv1(x) + self.conv2(x)
        if self.activation is not None:
            y = self.activation(y)
        return y


class CSPRep(nn.Module):
    """CSP block: two 1x1 branches, one through RepVgg blocks, summed."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        num_blocks: int,
        hidden_expansion: float,
        activation: Optional[str],
        eps: float,
    ) -> None:
        super().__init__()
        hidden = int(out_channels * hidden_expansion)
        self.conv1 = Proj(in_channels, hidden, 1, 1, activation, eps)
        self.conv2 = Proj(in_channels, hidden, 1, 1, activation, eps)
        self.bottlenecks = [RepVgg(hidden, activation, eps) for _ in range(num_blocks)]
        # Collapses to identity when hidden == out (the case in this model).
        self.conv3 = (
            Proj(hidden, out_channels, 1, 1, activation, eps)
            if hidden != out_channels
            else None
        )

    def __call__(self, x: mx.array) -> mx.array:
        a = self.conv1(x)
        for b in self.bottlenecks:
            a = b(a)
        s = a + self.conv2(x)
        return self.conv3(s) if self.conv3 is not None else s


class SinePE(nn.Module):
    """2D sinusoidal position embedding, [sin_h|cos_h|sin_w|cos_w]."""

    def __init__(self, embed_dim: int = 256, temperature: float = 10000.0) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.temperature = temperature

    def __call__(self, height: int, width: int) -> mx.array:
        pos_dim = self.embed_dim // 4
        omega = mx.arange(pos_dim, dtype=mx.float32) / pos_dim
        omega = 1.0 / (self.temperature**omega)
        gh = mx.arange(height, dtype=mx.float32)
        gw = mx.arange(width, dtype=mx.float32)
        gh, gw = mx.meshgrid(gh, gw, indexing="ij")
        eh = gh.flatten()[:, None] * omega[None, :]
        ew = gw.flatten()[:, None] * omega[None, :]
        pe = mx.concatenate([mx.sin(eh), mx.cos(eh), mx.sin(ew), mx.cos(ew)], axis=1)
        return pe[None, :, :]


class AIFISelfAttn(nn.Module):
    """MHSA with position embedding added to q,k (not v)."""

    def __init__(self, d: int, n_heads: int) -> None:
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = d // n_heads
        self.scale = self.head_dim**-0.5
        self.q_proj = nn.Linear(d, d)
        self.k_proj = nn.Linear(d, d)
        self.v_proj = nn.Linear(d, d)
        self.out_proj = nn.Linear(d, d)

    def __call__(self, x: mx.array, pos: Optional[mx.array]) -> mx.array:
        B, N, D = x.shape
        qk = x + pos if pos is not None else x
        q = (
            self.q_proj(qk)
            .reshape(B, N, self.n_heads, self.head_dim)
            .transpose(0, 2, 1, 3)
        )
        k = (
            self.k_proj(qk)
            .reshape(B, N, self.n_heads, self.head_dim)
            .transpose(0, 2, 1, 3)
        )
        v = (
            self.v_proj(x)
            .reshape(B, N, self.n_heads, self.head_dim)
            .transpose(0, 2, 1, 3)
        )
        out = mx.fast.scaled_dot_product_attention(q, k, v, scale=self.scale)
        return self.out_proj(out.transpose(0, 2, 1, 3).reshape(B, N, D))


class AIFILayer(nn.Module):
    """Post-norm transformer encoder layer (gelu FFN)."""

    def __init__(self, config: LayoutConfig) -> None:
        super().__init__()
        d = config.encoder_hidden_dim
        self.self_attn = AIFISelfAttn(d, config.encoder_attention_heads)
        self.self_attn_layer_norm = nn.LayerNorm(d, eps=config.layer_norm_eps)
        self.fc1 = nn.Linear(d, config.encoder_ffn_dim)
        self.fc2 = nn.Linear(config.encoder_ffn_dim, d)
        self.final_layer_norm = nn.LayerNorm(d, eps=config.layer_norm_eps)
        self.activation = _resolve_activation(config.encoder_activation_function)

    def __call__(self, x: mx.array, pos: Optional[mx.array]) -> mx.array:
        r = x
        x = self.self_attn(x, pos)
        x = self.self_attn_layer_norm(r + x)
        r = x
        x = self.fc2(self.activation(self.fc1(x)))
        return self.final_layer_norm(r + x)


class AIFI(nn.Module):
    def __init__(self, config: LayoutConfig) -> None:
        super().__init__()
        self.position_embedding = SinePE(
            config.encoder_hidden_dim, config.positional_encoding_temperature
        )
        self.layers = [AIFILayer(config) for _ in range(config.encoder_layers)]

    def __call__(self, x: mx.array) -> mx.array:
        B, H, W, D = x.shape
        seq = x.reshape(B, H * W, D)
        pos = self.position_embedding(H, W).astype(seq.dtype)
        for layer in self.layers:
            seq = layer(seq, pos)
        return seq.reshape(B, H, W, D)


def upsample_nearest2x(x: mx.array) -> mx.array:
    x = mx.repeat(x, 2, axis=1)
    return mx.repeat(x, 2, axis=2)


def upsample_bilinear2x(x: mx.array) -> mx.array:
    """Exact bilinear x2 (align_corners=False) via separable 1D filters."""

    def _up1d(v: mx.array, axis: int) -> mx.array:
        # y[2i] = .75*x[i] + .25*x[i-1]; y[2i+1] = .75*x[i] + .25*x[i+1]
        n = v.shape[axis]
        left = mx.concatenate([mx.take(v, mx.array([0]), axis=axis), v], axis=axis)
        left = mx.take(left, mx.arange(n), axis=axis)
        right = mx.concatenate([v, mx.take(v, mx.array([-1]), axis=axis)], axis=axis)
        right = mx.take(right, mx.arange(n) + 1, axis=axis)
        even = 0.75 * v + 0.25 * left
        odd = 0.75 * v + 0.25 * right
        stacked = mx.stack([even, odd], axis=axis + 1)
        return stacked.reshape(*v.shape[:axis], v.shape[axis] * 2, *v.shape[axis + 1 :])

    return _up1d(_up1d(x, 1), 2)


class _BilinearUpsample(nn.Module):
    """Param-free x2 bilinear (keys occupy scale_heads.{i}.layers.{odd})."""

    def __call__(self, x: mx.array) -> mx.array:
        return upsample_bilinear2x(x)


class ScaleHead(nn.Module):
    """Conv/upsample chain bringing one level to the base stride.

    Param-free upsamples sit explicitly in ``layers`` so conv keys read
    scale_heads.{i}.layers.{2k}.* exactly like torch.
    """

    def __init__(
        self,
        in_channels: int,
        feature_channels: int,
        n_conv: int,
        needs_upsample: bool,
        eps: float,
    ) -> None:
        super().__init__()
        self.layers = []
        for k in range(n_conv):
            inch = in_channels if k == 0 else feature_channels
            self.layers.append(Proj(inch, feature_channels, 3, 1, "silu", eps))
            if needs_upsample:
                self.layers.append(_BilinearUpsample())

    def __call__(self, x: mx.array) -> mx.array:
        for layer in self.layers:
            x = layer(x)
        return x


class MaskFeatFPN(nn.Module):
    """Fuse pan levels at base stride (sum) + output conv."""

    def __init__(self, config: LayoutConfig, eps: float) -> None:
        super().__init__()
        feat, out = config.mask_feature_channels
        strides = sorted(config.feat_strides)
        base = strides[0]
        import math

        # Pan levels arrive stride [8,16,32]; each head convolves up to the
        # base stride (8), upsampling after every conv except when already
        # at base stride.
        self.scale_heads = []
        for s in strides:
            n_conv = max(1, int(math.log2(s) - math.log2(base)))
            self.scale_heads.append(
                ScaleHead(256, feat, n_conv, needs_upsample=(s != base), eps=eps)
            )
        self.output_conv = Proj(feat, out, 3, 1, "silu", eps)

    def __call__(self, pan_maps: list) -> mx.array:
        # pan_maps are stride [8,16,32]; scale heads target base stride 8.
        out = self.scale_heads[0](pan_maps[0])
        for head, feat in zip(self.scale_heads[1:], pan_maps[1:]):
            out = out + head(feat)
        return self.output_conv(out)


class MaskOutput(nn.Module):
    """mask_feat lateral add (stride 4) + prototype conv."""

    def __init__(self, config: LayoutConfig, eps: float) -> None:
        super().__init__()
        feat = config.mask_feature_channels[1]
        self.base_conv = Proj(feat, feat, 3, 1, "silu", eps)
        self.conv = nn.Conv2d(feat, config.num_prototypes, kernel_size=1, bias=True)

    def __call__(self, x: mx.array) -> mx.array:
        return self.conv(self.base_conv(x))


class HybridEncoder(nn.Module):
    def __init__(self, config: LayoutConfig) -> None:
        super().__init__()
        self.config = config
        d = config.encoder_hidden_dim
        self.aifi = [AIFI(config) for _ in config.encode_proj_layers[:1]]
        self.encode_levels = list(config.encode_proj_layers)
        act = config.activation_function
        eps = config.batch_norm_eps
        n = len(config.encoder_in_channels) - 1
        self.lateral_convs = [Proj(d, d, 1, 1, act, eps) for _ in range(n)]
        self.fpn_blocks = [
            CSPRep(2 * d, d, 3, config.hidden_expansion, act, eps) for _ in range(n)
        ]
        self.downsample_convs = [Proj(d, d, 3, 2, act, eps) for _ in range(n)]
        self.pan_blocks = [
            CSPRep(2 * d, d, 3, config.hidden_expansion, act, eps) for _ in range(n)
        ]
        self.mask_feature_head = MaskFeatFPN(config, eps)
        self.encoder_mask_lateral = Proj(
            config.x4_feat_dim, config.mask_feature_channels[1], 3, 1, "silu", eps
        )
        self.encoder_mask_output = MaskOutput(config, eps)

    def __call__(self, feats: list, x4_feat: mx.array) -> Tuple[list, mx.array]:
        for i, lvl in enumerate(self.encode_levels):
            feats[lvl] = self.aifi[i](feats[lvl])

        fpn = [feats[-1]]
        for idx, (lat, blk) in enumerate(zip(self.lateral_convs, self.fpn_blocks)):
            top = lat(fpn[-1])
            fpn[-1] = top
            up = upsample_nearest2x(top)
            fused = mx.concatenate([up, feats[len(feats) - 2 - idx]], axis=-1)
            fpn.append(blk(fused))
        fpn = fpn[::-1]

        pan = [fpn[0]]
        for idx, (down, blk) in enumerate(zip(self.downsample_convs, self.pan_blocks)):
            down_f = down(pan[-1])
            fused = mx.concatenate([down_f, fpn[idx + 1]], axis=-1)
            pan.append(blk(fused))

        mask_feat = self.mask_feature_head(pan)
        mask_feat = upsample_bilinear2x(mask_feat)
        mask_feat = mask_feat + self.encoder_mask_lateral(x4_feat)
        mask_feat = self.encoder_mask_output(mask_feat)
        return pan, mask_feat
