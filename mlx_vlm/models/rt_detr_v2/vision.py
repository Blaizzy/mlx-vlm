"""ResNet-vd backbone and hybrid encoder for RT-DETRv2.

The vision tower for RT-DETRv2 has two parts:

  - A ResNet-50-vd / ResNet-101-vd backbone returning features at three
    strides (8, 16, 32).
  - A hybrid encoder consisting of an AIFI transformer applied to the
    deepest backbone level, a top-down FPN, and a bottom-up PAN. Output
    is three feature maps at strides (8, 16, 32) all with
    `encoder_hidden_dim` channels.

BatchNorms are kept as proper layers (not folded into preceding convs at
conversion time) so the checkpoint remains fine-tunable.
"""

from typing import Dict, List, Optional, Tuple

import mlx.core as mx
import mlx.nn as nn

from .config import ModelConfig, RTDetrResNetConfig, RTDetrV2HybridEncoderConfig

# ─── Backbone primitives ───


class ConvNormLayer(nn.Module):
    """Conv2d (no bias) + BatchNorm + activation (mirrors HF `RTDetrResNetConvLayer`)."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        activation: Optional[str] = "relu",
        eps: float = 1e-5,
    ) -> None:
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=kernel_size // 2,
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


class ShortCut(nn.Module):
    """1x1 Conv2d (no bias) + BatchNorm. Used directly in stride-1 shortcuts
    and wrapped in `AvgPoolShortCut` for vd-style downsampling shortcuts."""

    def __init__(
        self, in_channels: int, out_channels: int, stride: int = 1, eps: float = 1e-5
    ) -> None:
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels, out_channels, kernel_size=1, stride=stride, bias=False
        )
        self.bn = nn.BatchNorm(out_channels, eps=eps)

    def __call__(self, x: mx.array) -> mx.array:
        return self.bn(self.conv(x))


class AvgPoolShortCut(nn.Module):
    """vd downsampling shortcut: AvgPool 2x2 stride 2 then 1x1 conv + BN.

    HF stores this as `Sequential[AvgPool2d, ShortCut]`, so the inner
    ShortCut sits at saved key `.1.` (the AvgPool at `.0.` has no params).
    Attribute is named `proj` since MLX attribute names cannot start with
    a digit; the rename pipeline in `convert.py` maps `.shortcut.1.X` to
    `.shortcut.proj.X`.
    """

    def __init__(self, in_channels: int, out_channels: int, eps: float = 1e-5) -> None:
        super().__init__()
        self.pool = nn.AvgPool2d(kernel_size=2, stride=2, padding=0)
        self.proj = ShortCut(in_channels, out_channels, stride=1, eps=eps)

    def __call__(self, x: mx.array) -> mx.array:
        return self.proj(self.pool(x))


class BottleNeckLayer(nn.Module):
    """ResNet bottleneck (3 conv layers + shortcut + skip).

    Shortcut variants (vd):
      - `stride == 2`: AvgPool 2x2 + 1x1 conv + BN (`AvgPoolShortCut`)
      - `stride == 1` and channels change: plain `ShortCut`
      - `stride == 1` and channels match: identity (no params)
    """

    expansion = 4

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 1,
        downsample_in_bottleneck: bool = False,
        activation: str = "relu",
        eps: float = 1e-5,
    ) -> None:
        super().__init__()
        should_apply_shortcut = (in_channels != out_channels) or (stride != 1)
        reduces_channels = out_channels // self.expansion

        if stride == 2:
            self.shortcut: nn.Module = (
                AvgPoolShortCut(in_channels, out_channels, eps=eps)
                if should_apply_shortcut
                else _Identity()
            )
        else:
            self.shortcut = (
                ShortCut(in_channels, out_channels, stride=stride, eps=eps)
                if should_apply_shortcut
                else _Identity()
            )

        # 1x1 -> 3x3 -> 1x1 expand. The stride-2 sits on the first 1x1 if
        # `downsample_in_bottleneck`, otherwise on the middle 3x3.
        first_stride = stride if downsample_in_bottleneck else 1
        middle_stride = stride if not downsample_in_bottleneck else 1
        self.layer = [
            ConvNormLayer(
                in_channels,
                reduces_channels,
                kernel_size=1,
                stride=first_stride,
                eps=eps,
            ),
            ConvNormLayer(
                reduces_channels,
                reduces_channels,
                kernel_size=3,
                stride=middle_stride,
                eps=eps,
            ),
            ConvNormLayer(
                reduces_channels, out_channels, kernel_size=1, activation=None, eps=eps
            ),
        ]
        self.activation = _resolve_activation(activation)

    def __call__(self, x: mx.array) -> mx.array:
        residual = self.shortcut(x)
        for layer in self.layer:
            x = layer(x)
        x = x + residual
        if self.activation is not None:
            x = self.activation(x)
        return x


class Stage(nn.Module):
    """A ResNet stage: `depth` BottleNeckLayers. First block downsamples / projects."""

    def __init__(
        self,
        config: RTDetrResNetConfig,
        in_channels: int,
        out_channels: int,
        stride: int,
        depth: int,
    ) -> None:
        super().__init__()
        first = BottleNeckLayer(
            in_channels,
            out_channels,
            stride=stride,
            downsample_in_bottleneck=config.downsample_in_bottleneck,
            activation=config.hidden_act,
        )
        rest = [
            BottleNeckLayer(
                out_channels,
                out_channels,
                stride=1,
                downsample_in_bottleneck=config.downsample_in_bottleneck,
                activation=config.hidden_act,
            )
            for _ in range(depth - 1)
        ]
        self.layers = [first, *rest]

    def __call__(self, x: mx.array) -> mx.array:
        for layer in self.layers:
            x = layer(x)
        return x


class Embeddings(nn.Module):
    """Stem: three 3x3 ConvNormLayers + a 3x3 stride-2 MaxPool (output stride 4)."""

    def __init__(self, config: RTDetrResNetConfig) -> None:
        super().__init__()
        emb = config.embedding_size
        self.embedder = [
            ConvNormLayer(
                config.num_channels,
                emb // 2,
                kernel_size=3,
                stride=2,
                activation=config.hidden_act,
            ),
            ConvNormLayer(
                emb // 2,
                emb // 2,
                kernel_size=3,
                stride=1,
                activation=config.hidden_act,
            ),
            ConvNormLayer(
                emb // 2, emb, kernel_size=3, stride=1, activation=config.hidden_act
            ),
        ]
        self.pooler = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

    def __call__(self, x: mx.array) -> mx.array:
        for layer in self.embedder:
            x = layer(x)
        return self.pooler(x)


class Encoder(nn.Module):
    """The four staged-bottleneck stages of the ResNet body."""

    def __init__(self, config: RTDetrResNetConfig) -> None:
        super().__init__()
        depths = config.depths
        hidden_sizes = config.hidden_sizes
        embedding_size = config.embedding_size

        stages: List[Stage] = []
        prev_channels = embedding_size
        for i, (out_c, depth) in enumerate(zip(hidden_sizes, depths)):
            if i == 0:
                stride = 2 if config.downsample_in_first_stage else 1
            else:
                stride = 2
            stages.append(
                Stage(config, prev_channels, out_c, stride=stride, depth=depth)
            )
            prev_channels = out_c

        self.stages = stages

    def __call__(self, x: mx.array) -> Tuple[mx.array, ...]:
        outputs: List[mx.array] = []
        for stage in self.stages:
            x = stage(x)
            outputs.append(x)
        return tuple(outputs)


class Backbone(nn.Module):
    """ResNet-50-vd / ResNet-101-vd backbone.

    Forward returns the features selected by `out_features` (default
    stages 2, 3, 4 at strides 8/16/32). Stage 0 (stride 4) is computed
    but dropped from the output.
    """

    def __init__(self, config: RTDetrResNetConfig) -> None:
        super().__init__()
        self.config = config
        self.embedder = Embeddings(config)
        self.encoder = Encoder(config)
        self._out_stage_indices = [
            int(name.removeprefix("stage")) - 1 for name in config.out_features
        ]

    def __call__(self, pixel_values: mx.array) -> Tuple[mx.array, ...]:
        x = self.embedder(pixel_values)
        all_stages = self.encoder(x)
        return tuple(all_stages[i] for i in self._out_stage_indices)


# ─── Hybrid encoder building blocks ───


class EncoderConvNormLayer(nn.Module):
    """Conv2d + BatchNorm + activation with overridable padding.

    Mirrors HF `RTDetrV2ConvNormLayer` (distinct from the backbone's
    `RTDetrResNetConvLayer`). Field names are `conv` / `bn` to match the
    saved state-dict's `conv` / `norm` (the rename pipeline maps `norm`
    to `bn`).
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: Optional[int] = None,
        activation: Optional[str] = None,
        eps: float = 1e-5,
    ) -> None:
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=(kernel_size - 1) // 2 if padding is None else padding,
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


class RepVggBlock(nn.Module):
    """RepVGG block: 3x3 conv + 1x1 conv branches summed and activated.

    Branches are kept separate (not re-parameterized into a single 3x3)
    so the checkpoint stays trainable.
    """

    def __init__(
        self, hidden_channels: int, activation: Optional[str], eps: float
    ) -> None:
        super().__init__()
        self.conv1 = EncoderConvNormLayer(
            hidden_channels, hidden_channels, kernel_size=3, padding=1, eps=eps
        )
        self.conv2 = EncoderConvNormLayer(
            hidden_channels, hidden_channels, kernel_size=1, padding=0, eps=eps
        )
        self.activation = _resolve_activation(activation)

    def __call__(self, x: mx.array) -> mx.array:
        y = self.conv1(x) + self.conv2(x)
        if self.activation is not None:
            y = self.activation(y)
        return y


class CSPRepLayer(nn.Module):
    """CSPNet block built from RepVGG blocks.

    Two parallel 1x1 branches; one feeds N RepVgg blocks, the other is a
    direct path. Sum is optionally re-projected to `out_channels`; the
    projection collapses to identity when
    `hidden_channels == out_channels`.
    """

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
        hidden_channels = int(out_channels * hidden_expansion)
        self.conv1 = EncoderConvNormLayer(
            in_channels, hidden_channels, kernel_size=1, activation=activation, eps=eps
        )
        self.conv2 = EncoderConvNormLayer(
            in_channels, hidden_channels, kernel_size=1, activation=activation, eps=eps
        )
        self.bottlenecks = [
            RepVggBlock(hidden_channels, activation, eps) for _ in range(num_blocks)
        ]
        if hidden_channels != out_channels:
            self.conv3: nn.Module = EncoderConvNormLayer(
                hidden_channels,
                out_channels,
                kernel_size=1,
                activation=activation,
                eps=eps,
            )
        else:
            self.conv3 = _Identity()

    def __call__(self, x: mx.array) -> mx.array:
        a = self.conv1(x)
        for b in self.bottlenecks:
            a = b(a)
        return self.conv3(a + self.conv2(x))


class SinePositionEmbedding(nn.Module):
    """2D sinusoidal position embedding for AIFI. No learnable params.

    Returns a `(1, H*W, embed_dim)` tensor with `embed_dim` split into
    `[sin(h), cos(h), sin(w), cos(w)]` quarters.
    """

    def __init__(self, embed_dim: int = 256, temperature: float = 10000.0) -> None:
        super().__init__()
        if embed_dim % 4 != 0:
            raise ValueError("embed_dim must be divisible by 4")
        self.embed_dim = embed_dim
        self.temperature = temperature

    def __call__(
        self, height: int, width: int, dtype: mx.Dtype = mx.float32
    ) -> mx.array:
        grid_w = mx.arange(width, dtype=dtype)
        grid_h = mx.arange(height, dtype=dtype)
        gw, gh = mx.meshgrid(grid_w, grid_h)  # default "xy" indexing
        pos_dim = self.embed_dim // 4
        omega = mx.arange(pos_dim, dtype=dtype) / pos_dim
        omega = 1.0 / (self.temperature**omega)
        out_w = gw.flatten()[:, None] * omega[None, :]
        out_h = gh.flatten()[:, None] * omega[None, :]
        pe = mx.concatenate(
            [mx.sin(out_h), mx.cos(out_h), mx.sin(out_w), mx.cos(out_w)], axis=1
        )
        return pe[None, :, :]


class EncoderLayer(nn.Module):
    """One AIFI transformer encoder layer.

    Standard pre/post-norm MHSA + FFN. Position embedding is added to
    q,k but not v in self-attention. Field names match the saved
    state-dict (`fc1` / `fc2` directly with no `.mlp` wrapper,
    `out_proj` rather than `o_proj`).
    """

    def __init__(self, config: RTDetrV2HybridEncoderConfig) -> None:
        super().__init__()
        d = config.encoder_hidden_dim
        self.normalize_before = config.normalize_before
        self.self_attn = _SelfAttention(d, config.encoder_attention_heads)
        self.self_attn_layer_norm = nn.LayerNorm(d, eps=config.layer_norm_eps)
        self.fc1 = nn.Linear(d, config.encoder_ffn_dim)
        self.fc2 = nn.Linear(config.encoder_ffn_dim, d)
        self.final_layer_norm = nn.LayerNorm(d, eps=config.layer_norm_eps)
        activation = _resolve_activation(config.encoder_activation_function)
        if activation is None:
            raise ValueError("encoder_activation_function must be set")
        self.activation = activation

    def __call__(self, x: mx.array, pos_embed: Optional[mx.array]) -> mx.array:
        # Self-attention sub-block
        residual = x
        if self.normalize_before:
            x = self.self_attn_layer_norm(x)
        x = self.self_attn(x, pos_embed)
        x = residual + x
        if not self.normalize_before:
            x = self.self_attn_layer_norm(x)

        # FFN sub-block
        residual = x
        if self.normalize_before:
            x = self.final_layer_norm(x)
        x = self.fc2(self.activation(self.fc1(x)))
        x = residual + x
        if not self.normalize_before:
            x = self.final_layer_norm(x)
        return x


class _SelfAttention(nn.Module):
    """MHSA with position embedding added to q,k (not v)."""

    def __init__(self, d_model: int, n_heads: int) -> None:
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.scale = self.head_dim**-0.5
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)

    def __call__(self, x: mx.array, pos_embed: Optional[mx.array]) -> mx.array:
        B, N, D = x.shape
        qk = x + pos_embed if pos_embed is not None else x
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
        out = out.transpose(0, 2, 1, 3).reshape(B, N, D)
        return self.out_proj(out)


class AIFI(nn.Module):
    """Attention-based Intra-scale Feature Interaction: a stack of EncoderLayers
    applied to a single flattened feature map.

    The HF state-dict prefixes AIFI keys with `encoder.encoder.` (older
    naming); the rename pipeline maps that to `vision.hybrid_encoder.aifi.`.
    """

    def __init__(self, config: RTDetrV2HybridEncoderConfig) -> None:
        super().__init__()
        self.position_embedding = SinePositionEmbedding(
            embed_dim=config.encoder_hidden_dim,
            temperature=config.positional_encoding_temperature,
        )
        self.layers = [EncoderLayer(config) for _ in range(config.encoder_layers)]

    def __call__(self, x: mx.array) -> mx.array:
        """Args: (B, H, W, C) NHWC. Returns: same shape."""
        B, H, W, C = x.shape
        x_flat = x.reshape(B, H * W, C)
        pos = self.position_embedding(H, W, dtype=x.dtype)
        for layer in self.layers:
            x_flat = layer(x_flat, pos)
        return x_flat.reshape(B, H, W, C)


# ─── Hybrid encoder ───


class HybridEncoder(nn.Module):
    """AIFI on the deepest level, then top-down FPN, then bottom-up PAN.

    Output: three feature maps at the original strides, all with
    `encoder_hidden_dim` channels.
    """

    def __init__(self, config: RTDetrV2HybridEncoderConfig) -> None:
        super().__init__()
        self.config = config
        self.encode_proj_layers = config.encode_proj_layers
        num_fpn_stages = len(config.encoder_in_channels) - 1
        num_pan_stages = num_fpn_stages
        d = config.encoder_hidden_dim
        act = config.activation_function
        eps = config.batch_norm_eps

        self.aifi = [AIFI(config) for _ in self.encode_proj_layers]

        self.lateral_convs = [
            EncoderConvNormLayer(d, d, kernel_size=1, activation=act, eps=eps)
            for _ in range(num_fpn_stages)
        ]
        self.fpn_blocks = [
            CSPRepLayer(
                in_channels=d * 2,
                out_channels=d,
                num_blocks=3,
                hidden_expansion=config.hidden_expansion,
                activation=act,
                eps=eps,
            )
            for _ in range(num_fpn_stages)
        ]

        self.downsample_convs = [
            EncoderConvNormLayer(d, d, kernel_size=3, stride=2, activation=act, eps=eps)
            for _ in range(num_pan_stages)
        ]
        self.pan_blocks = [
            CSPRepLayer(
                in_channels=d * 2,
                out_channels=d,
                num_blocks=3,
                hidden_expansion=config.hidden_expansion,
                activation=act,
                eps=eps,
            )
            for _ in range(num_pan_stages)
        ]

    def __call__(self, features: Tuple[mx.array, ...]) -> Tuple[mx.array, ...]:
        # AIFI: apply transformer encoder to each level in encode_proj_layers.
        feats = list(features)
        for i, lvl in enumerate(self.encode_proj_layers):
            feats[lvl] = self.aifi[i](feats[lvl])

        # Top-down FPN.
        fpn = [feats[-1]]
        num_fpn = len(self.lateral_convs)
        for idx in range(num_fpn):
            backbone_feat = feats[num_fpn - idx - 1]
            top_feat = fpn[-1]
            top_feat = self.lateral_convs[idx](top_feat)
            fpn[-1] = top_feat
            top_feat = _upsample_nearest_2x(top_feat)
            fused = mx.concatenate([top_feat, backbone_feat], axis=-1)
            fpn.append(self.fpn_blocks[idx](fused))
        fpn.reverse()

        # Bottom-up PAN.
        pan = [fpn[0]]
        num_pan = len(self.downsample_convs)
        for idx in range(num_pan):
            top = pan[-1]
            up = fpn[idx + 1]
            down = self.downsample_convs[idx](top)
            fused = mx.concatenate([down, up], axis=-1)
            pan.append(self.pan_blocks[idx](fused))
        return tuple(pan)


# ─── Encoder input projection ───


class EncoderInputProj(nn.Module):
    """1x1 Conv + BN, per backbone level, projecting to `encoder_hidden_dim`.

    HF saves this as `Sequential[Conv2d, BatchNorm2d]` so the keys are
    `.{N}.0.X` (conv) and `.{N}.1.X` (BN). The rename pipeline maps
    `.0.` to `.conv.` and `.1.` to `.bn.`.
    """

    def __init__(self, in_channels: int, out_channels: int, eps: float) -> None:
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.bn = nn.BatchNorm(out_channels, eps=eps)

    def __call__(self, x: mx.array) -> mx.array:
        return self.bn(self.conv(x))


# ─── Vision tower ───


class VisionTower(nn.Module):
    """Backbone -> per-level input projection -> hybrid encoder.

    This is the real vision module; `VisionModel` below is the framework
    stub that mlx-vlm's loader looks up by name.
    """

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config
        # `ModelConfig.__post_init__` resolves backbone_config into a
        # `RTDetrResNetConfig` regardless of how it was passed in.
        if not isinstance(config.backbone_config, RTDetrResNetConfig):
            raise TypeError(
                "config.backbone_config must be RTDetrResNetConfig after "
                f"ModelConfig.__post_init__, got {type(config.backbone_config).__name__}"
            )
        self.backbone = Backbone(config.backbone_config)
        self.encoder_input_proj = [
            EncoderInputProj(in_c, config.encoder_hidden_dim, eps=config.batch_norm_eps)
            for in_c in config.encoder_in_channels
        ]
        self.hybrid_encoder = HybridEncoder(config._hybrid_encoder_config)

    def __call__(self, pixel_values: mx.array) -> Tuple[mx.array, ...]:
        c_features = self.backbone(pixel_values)
        proj = tuple(p(c) for p, c in zip(self.encoder_input_proj, c_features))
        return self.hybrid_encoder(proj)


class VisionModel(nn.Module):
    """Framework-compatibility stub.

    mlx-vlm's loader (`mlx_vlm.utils.load_model`) instantiates
    `model_class.VisionModel(model_config.vision_config)` purely to call
    its `sanitize` method. The real vision module lives at
    `Model.vision` (a `VisionTower`); weight sanitization is handled by
    `Model.sanitize` so this stub is a no-op.
    """

    def __init__(self, config=None) -> None:
        super().__init__()

    def __call__(self, *args, **kwargs):
        return None

    @staticmethod
    def sanitize(weights: Dict[str, mx.array]) -> Dict[str, mx.array]:
        return weights


# ─── helpers ───


def _upsample_nearest_2x(x: mx.array) -> mx.array:
    """Nearest-neighbour 2x upsample of an NHWC tensor along H,W."""
    B, H, W, C = x.shape
    x = mx.broadcast_to(x[:, :, None, :, None, :], (B, H, 2, W, 2, C))
    return x.reshape(B, H * 2, W * 2, C)


def _resolve_activation(name: Optional[str]):
    if name is None:
        return None
    if name == "relu":
        return nn.ReLU()
    if name == "silu":
        return nn.SiLU()
    if name == "gelu":
        return nn.GELU()
    raise ValueError(f"Unsupported activation: {name}")


class _Identity(nn.Module):
    def __call__(self, x: mx.array) -> mx.array:
        return x
