"""Sapiens2 decode heads (pose / seg / normal / pointmap).

The PyTorch reference nests each sub-block in an nn.Sequential:

  deconv_layers = [ConvT, InstanceNorm, SiLU, ConvT, InstanceNorm, SiLU, ...]
  conv_layers   = [Conv,  InstanceNorm, SiLU, Conv,  InstanceNorm, SiLU, ...]

Only the Conv / ConvT entries carry weights (InstanceNorm is affine=False by
default), so the checkpoint contains `deconv_layers.0.weight`, `.3.weight`,
`.6.weight`, ...  To land those exact keys we register the sub-modules as a
**plain list attribute** on the head (MLX serializes
`self.deconv_layers = [m0, m1, ...]` as `deconv_layers.0.*`, `.1.*`, ...).  The
norm/SiLU positions are filled with parameterless `_Slot` wrappers so indices
stay aligned with the PT Sequential.
"""

from typing import List, Sequence, Tuple

import mlx.core as mx
import mlx.nn as nn

from .config import HeadConfig


def pixel_shuffle(x: mx.array, r: int) -> mx.array:
    """Channels-last PixelShuffle.  (B, H, W, C*r*r) → (B, H*r, W*r, C).

    Matches PyTorch's channel grouping: the channel axis is interpreted as
    (C, r, r) where the inner two dims are sub-pixel row/col offsets.
    """
    B, H, W, C = x.shape
    c_out = C // (r * r)
    x = x.reshape(B, H, W, c_out, r, r)
    x = x.transpose(0, 1, 4, 2, 5, 3)  # (B, H, r, W, r, c_out)
    return x.reshape(B, H * r, W * r, c_out)


class _Slot(nn.Module):
    """No-parameter slot that holds an inner module or callable.  Keeps list
    indices aligned with PT's Sequential where norm/SiLU positions have no
    weights."""

    def __init__(self, inner=None):
        super().__init__()
        if isinstance(inner, nn.Module):
            self.inner = inner
            self._call = inner
        else:
            self._call = inner if inner is not None else (lambda y: y)

    def __call__(self, x):
        return self._call(x)


def _run_list(x, modules):
    for m in modules:
        x = m(x)
    return x


def _silu_slot() -> _Slot:
    return _Slot(nn.silu)


def _instance_norm_slot(channels: int) -> _Slot:
    return _Slot(nn.InstanceNorm(dims=channels, eps=1e-5, affine=False))


def _make_deconv_layers(
    in_channels: int,
    out_channels_seq: Sequence[int],
    kernel_sizes: Sequence[int],
) -> List[nn.Module]:
    layers: List[nn.Module] = []
    cur = in_channels
    for out_ch, k in zip(out_channels_seq, kernel_sizes):
        if k == 4:
            pad, out_pad = 1, 0
        elif k == 3:
            pad, out_pad = 1, 1
        elif k == 2:
            pad, out_pad = 0, 0
        else:
            raise ValueError(f"Unsupported deconv kernel {k}")
        layers.append(nn.ConvTranspose2d(
            in_channels=cur, out_channels=out_ch,
            kernel_size=k, stride=2,
            padding=pad, output_padding=out_pad, bias=False,
        ))
        layers.append(_instance_norm_slot(out_ch))
        layers.append(_silu_slot())
        cur = out_ch
    return layers


def _make_conv_layers(
    in_channels: int,
    out_channels_seq: Sequence[int],
    kernel_sizes: Sequence[int],
    stride: int = 1,
) -> List[nn.Module]:
    layers: List[nn.Module] = []
    cur = in_channels
    for out_ch, k in zip(out_channels_seq, kernel_sizes):
        pad = (k - 1) // 2
        layers.append(nn.Conv2d(
            in_channels=cur, out_channels=out_ch,
            kernel_size=k, stride=stride, padding=pad, bias=True,
        ))
        layers.append(_instance_norm_slot(out_ch))
        layers.append(_silu_slot())
        cur = out_ch
    return layers


class _UpsampleBlock(nn.Module):
    """Single (Conv3x3 → PixelShuffle(2) → InstanceNorm → SiLU) unit.

    PT's state_dict key for this block is `<parent>.<i>.0.weight` — i.e. child
    index 0 (the conv), since the shuffle / norm / silu children have no
    weights.  We store children as a plain list to match the integer indexing.
    """

    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        conv = nn.Conv2d(in_channels=in_ch, out_channels=out_ch * 4,
                         kernel_size=3, stride=1, padding=1, bias=True)
        self.children_list = [
            conv,
            _Slot(lambda y: pixel_shuffle(y, 2)),
            _instance_norm_slot(out_ch),
            _silu_slot(),
        ]

    def __call__(self, x):
        return _run_list(x, self.children_list)


# ──────────────────────────────── Task heads ────────────────────────────────


class PoseHeatmapHead(nn.Module):
    """in → N deconvs (2x each) → M 1x1 convs → conv_pose (1x1, K keypoints)."""

    def __init__(self, cfg: HeadConfig):
        super().__init__()
        cur = cfg.in_channels
        self.deconv_layers = _make_deconv_layers(
            cur, cfg.deconv_out_channels, cfg.deconv_kernel_sizes
        )
        cur = cfg.deconv_out_channels[-1]
        self.conv_layers = _make_conv_layers(
            cur, cfg.conv_out_channels, cfg.conv_kernel_sizes
        )
        cur = cfg.conv_out_channels[-1]
        self.conv_pose = nn.Conv2d(cur, cfg.num_keypoints, kernel_size=1, bias=True)

    def __call__(self, x: mx.array) -> mx.array:
        x = _run_list(x, self.deconv_layers)
        x = _run_list(x, self.conv_layers)
        return self.conv_pose(x)


class SegHead(nn.Module):
    def __init__(self, cfg: HeadConfig):
        super().__init__()
        cur = cfg.in_channels
        self.deconv_layers = _make_deconv_layers(
            cur, cfg.deconv_out_channels, cfg.deconv_kernel_sizes
        )
        cur = cfg.deconv_out_channels[-1]
        self.conv_layers = _make_conv_layers(
            cur, cfg.conv_out_channels, cfg.conv_kernel_sizes
        )
        cur = cfg.conv_out_channels[-1]
        self.conv_seg = nn.Conv2d(cur, cfg.num_classes, kernel_size=1, bias=True)

    def __call__(self, x: mx.array) -> mx.array:
        x = _run_list(x, self.deconv_layers)
        x = _run_list(x, self.conv_layers)
        return self.conv_seg(x)


class _DenseHeadBase(nn.Module):
    """Shared scaffold for NormalHead / PointmapHead.

    input_conv is PT Sequential(Conv3x3, InstanceNorm, SiLU) — only index 0
    carries weights, so we keep the same three-slot layout here.
    """

    def __init__(self, cfg: HeadConfig):
        super().__init__()
        ic = cfg.in_channels
        self.input_conv = [
            nn.Conv2d(ic, ic, kernel_size=3, stride=1, padding=1, bias=True),
            _instance_norm_slot(ic),
            _silu_slot(),
        ]

        ups: List[_UpsampleBlock] = []
        cur = ic
        for out_ch in cfg.upsample_channels:
            ups.append(_UpsampleBlock(cur, out_ch))
            cur = out_ch
        self.upsample_blocks = ups

        if cfg.conv_out_channels:
            self.conv_layers = _make_conv_layers(
                cur, cfg.conv_out_channels, cfg.conv_kernel_sizes, stride=1
            )
            cur = cfg.conv_out_channels[-1]
        else:
            self.conv_layers = []
        self._final_in_channels = cur


class NormalHead(_DenseHeadBase):
    def __init__(self, cfg: HeadConfig):
        super().__init__(cfg)
        self.conv_normal = nn.Conv2d(
            self._final_in_channels, 3, kernel_size=1, bias=True
        )

    def __call__(self, x: mx.array) -> mx.array:
        x_n = _run_list(x, self.input_conv)
        x_n = _run_list(x_n, self.upsample_blocks)
        x_n = _run_list(x_n, self.conv_layers)
        return self.conv_normal(x_n)


class PointmapHead(_DenseHeadBase):
    def __init__(self, cfg: HeadConfig):
        super().__init__(cfg)
        self.conv_pointmap = nn.Conv2d(
            self._final_in_channels, 3, kernel_size=1, bias=True
        )
        # Scale regression branch: Conv2d(stride=2) × N → Flatten + Linear stack
        self.scale_conv_layers = _make_conv_layers(
            cfg.in_channels,
            cfg.scale_conv_out_channels,
            cfg.scale_conv_kernel_sizes,
            stride=2,
        )
        # PT: nn.Sequential([Flatten, Linear, SiLU, Linear, SiLU, Linear])
        # → keys .1 .3 .5 for the three linears.  PT Flatten on a (B, C, H, W)
        # tensor produces [c0h0w0, c0h0w1, ..., c1h0w0, ...] — channels outermost.
        # In MLX the conv output is (B, H, W, C), so we must transpose to
        # channels-first BEFORE flattening to reproduce PT's element ordering;
        # otherwise the first Linear (weights trained against PT's layout)
        # multiplies against a permuted input vector.
        dims = cfg.scale_final_layer
        final: List[nn.Module] = [
            _Slot(lambda y: y.transpose(0, 3, 1, 2).reshape(y.shape[0], -1))
        ]
        for i in range(1, len(dims)):
            final.append(nn.Linear(dims[i - 1], dims[i]))
            if i < len(dims) - 1:
                final.append(_silu_slot())
        self.scale_final_layer = final

    def __call__(self, x: mx.array) -> Tuple[mx.array, mx.array]:
        x_p = _run_list(x, self.input_conv)
        x_p = _run_list(x_p, self.upsample_blocks)
        x_p = _run_list(x_p, self.conv_layers)
        pointmap = self.conv_pointmap(x_p)

        x_s = _run_list(x, self.scale_conv_layers)
        scale = _run_list(x_s, self.scale_final_layer)
        return pointmap, scale


def build_head(cfg: HeadConfig) -> nn.Module:
    if cfg.task == "pose":
        return PoseHeatmapHead(cfg)
    if cfg.task == "seg":
        return SegHead(cfg)
    if cfg.task == "normal":
        return NormalHead(cfg)
    if cfg.task == "pointmap":
        return PointmapHead(cfg)
    raise ValueError(f"Unknown task: {cfg.task}")
