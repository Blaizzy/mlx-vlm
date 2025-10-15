from typing import List, Optional, Tuple

import mlx.core as mx
import mlx.nn as nn

from .config import VisionConfig


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


class CallableModuleList(list):
    def __call__(self, x: mx.array):
        for item in self:
            x = item(x)
        return x


class MHSA(nn.Module):
    """Multi-headed Self Attention module.

    Source modified from:
    https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vision_transformer.py
    """

    def __init__(
        self,
        dim: int,
        head_dim: int = 32,
        qkv_bias: bool = False,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
    ) -> None:
        super().__init__()
        assert dim % head_dim == 0, "dim should be divisible by head_dim"
        self.head_dim = head_dim
        self.num_heads = dim // head_dim
        self.scale = head_dim**-0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def __call__(self, x: mx.array) -> mx.array:
        # Source: https://github.com/apple/ml-fastvlm/blob/592b4add3c1c8a518e77d95dc6248e76c1dd591f/llava/model/multimodal_encoder/mobileclip/mci.py#L661
        x = x.transpose(0, 3, 1, 2)
        B, C, H, W = x.shape
        N = H * W
        x = x.flatten(start_axis=2).transpose(0, 2, 1)  # (B, N, C)
        qkv = (
            self.qkv(x)
            .reshape(B, N, 3, self.num_heads, self.head_dim)
            .transpose(2, 0, 3, 1, 4)
        )
        q, k, v = qkv

        x = mx.fast.scaled_dot_product_attention(q, k, v, scale=self.scale, mask=None)
        x = x.transpose(0, 2, 1, 3).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)

        x = x.reshape(B, H, W, C)
        return x


class ConvFFN(nn.Module):
    """Convolutional FFN Module."""

    def __init__(
        self,
        in_channels: int,
        hidden_channels: Optional[int] = None,
        out_channels: Optional[int] = None,
        act_layer: nn.Module = nn.GELU,
    ) -> None:
        super().__init__()
        out_channels = out_channels or in_channels
        hidden_channels = hidden_channels or in_channels
        self.conv = NamedSequential()
        self.conv.add_module(
            "conv",
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=7,
                padding=3,
                groups=in_channels,
                bias=False,
            ),
        )
        self.conv.add_module(
            "bn",
            nn.BatchNorm(num_features=out_channels),
        )
        self.fc1 = nn.Conv2d(in_channels, hidden_channels, kernel_size=1)
        self.act = act_layer()
        self.fc2 = nn.Conv2d(hidden_channels, out_channels, kernel_size=1)

    def __call__(self, x: mx.array) -> mx.array:
        x = self.conv(x)
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        return x


class LayerNormChannel(nn.Module):
    """
    LayerNorm only for Channel Dimension.
    Input: tensor in shape [B, H, W, C]
    """

    def __init__(self, num_features, eps=1e-05) -> None:
        super().__init__()
        self.weight = mx.ones(num_features)
        self.bias = mx.zeros(num_features)
        self.eps = eps

    def __call__(self, x: mx.array) -> mx.array:
        u = x.mean(-1, keepdims=True)
        s = mx.power(x - u, 2).mean(-1, keepdims=True)
        x = (x - u) / mx.sqrt(s + self.eps)
        x = self.weight * x + self.bias
        return x


class AttentionBlock(nn.Module):
    """Implementation of metaformer block with MHSA as token mixer.

    For more details on Metaformer structure, please refer to:
    `MetaFormer Is Actually What You Need for Vision <https://arxiv.org/pdf/2111.11418.pdf>`_
    """

    def __init__(
        self,
        dim: int,
        mlp_ratio: float = 4.0,
        act_layer: nn.Module = nn.GELU,
        norm_layer: nn.Module = nn.BatchNorm,
    ):
        super().__init__()

        self.norm = norm_layer(num_features=dim)
        self.token_mixer = MHSA(dim=dim)

        assert mlp_ratio > 0, "MLP ratio should be greater than 0, found: {}".format(
            mlp_ratio
        )
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.convffn = ConvFFN(
            in_channels=dim,
            hidden_channels=mlp_hidden_dim,
            act_layer=act_layer,
        )

        self.layer_scale_1 = mx.ones((1, 1, dim))
        self.layer_scale_2 = mx.ones((1, 1, dim))

    def __call__(self, x: mx.array) -> mx.array:
        x = x + self.layer_scale_1 * self.token_mixer(self.norm(x))
        x = x + self.layer_scale_2 * self.convffn(x)
        return x


class RepCPE(nn.Module):
    """Implementation of conditional positional encoding.

    For more details refer to paper:
    `Conditional Positional Encodings for Vision Transformers <https://arxiv.org/pdf/2102.10882.pdf>`_
    """

    def __init__(
        self,
        in_channels: int,
        embed_dim: int = 768,
        spatial_shape=(7, 7),
    ) -> None:
        super().__init__()
        if isinstance(spatial_shape, int):
            spatial_shape = tuple([spatial_shape] * 2)
        assert isinstance(spatial_shape, Tuple), (
            f'"spatial_shape" must by a sequence or int, '
            f"get {type(spatial_shape)} instead."
        )
        assert len(spatial_shape) == 2, (
            f'Length of "spatial_shape" should be 2, '
            f"got {len(spatial_shape)} instead."
        )

        self.reparam_conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=embed_dim,
            kernel_size=spatial_shape,
            stride=1,
            padding=int(spatial_shape[0] // 2),
            groups=embed_dim,
            bias=True,
        )

    def __call__(self, x: mx.array) -> mx.array:
        return self.reparam_conv(x)


class ReparamLargeKernelConv(nn.Module):
    """Building Block of RepLKNet

    This class defines overparameterized large kernel conv block
    introduced in `RepLKNet <https://arxiv.org/abs/2203.06717>`_

    Reference: https://github.com/DingXiaoH/RepLKNet-pytorch
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int,
        groups: int,
        activation: nn.Module = nn.GELU(),
    ) -> None:
        super(ReparamLargeKernelConv, self).__init__()
        self.activation = activation
        self.lkb_reparam = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=kernel_size // 2,
            dilation=1,
            groups=groups,
            bias=True,
        )

    def __call__(self, x: mx.array) -> mx.array:
        return self.activation(self.lkb_reparam(x))


class PatchEmbed(nn.Module):
    """Convolutional patch embedding layer."""

    def __init__(
        self,
        patch_size: int,
        stride: int,
        in_channels: int,
        embed_dim: int,
    ) -> None:
        super().__init__()
        self.proj = CallableModuleList()
        self.proj.append(
            ReparamLargeKernelConv(
                in_channels=in_channels,
                out_channels=embed_dim,
                kernel_size=patch_size,
                stride=stride,
                groups=in_channels,
            )
        )
        self.proj.append(
            MobileOneBlock(
                in_channels=embed_dim,
                out_channels=embed_dim,
                kernel_size=1,
                stride=1,
                padding=0,
                groups=1,
                use_se=False,
            )
        )

    def __call__(self, x: mx.array) -> mx.array:
        return self.proj(x)


class RepMixer(nn.Module):
    """Reparameterizable token mixer.

    For more details, please refer to Apple's paper:
    `FastViT: A Fast Hybrid Vision Transformer using Structural Reparameterization <https://arxiv.org/pdf/2303.14189.pdf>`_
    """

    def __init__(
        self,
        dim,
        kernel_size=3,
    ):
        super().__init__()
        self.dim = dim
        self.kernel_size = kernel_size

        self.reparam_conv = nn.Conv2d(
            in_channels=self.dim,
            out_channels=self.dim,
            kernel_size=self.kernel_size,
            stride=1,
            padding=self.kernel_size // 2,
            groups=self.dim,
            bias=True,
        )

    def __call__(self, x: mx.array) -> mx.array:
        return self.reparam_conv(x)


class RepMixerBlock(nn.Module):
    """Implementation of Metaformer block with RepMixer as token mixer.

    For more details on Metaformer structure, please refer to:
    `MetaFormer Is Actually What You Need for Vision <https://arxiv.org/pdf/2111.11418.pdf>`_
    """

    def __init__(
        self,
        dim: int,
        kernel_size: int = 3,
        mlp_ratio: float = 4.0,
        act_layer: nn.Module = nn.GELU,
    ):
        super().__init__()

        self.token_mixer = RepMixer(dim, kernel_size=kernel_size)

        assert mlp_ratio > 0, "MLP ratio should be greater than 0, found: {}".format(
            mlp_ratio
        )
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.convffn = ConvFFN(
            in_channels=dim,
            hidden_channels=mlp_hidden_dim,
            act_layer=act_layer,
        )
        self.layer_scale = mx.ones((1, 1, dim))

    def __call__(self, x: mx.array) -> mx.array:
        x = self.token_mixer(x)
        x = x + self.layer_scale * self.convffn(x)
        return x


def basic_blocks(
    dim: int,
    block_index: int,
    num_blocks: List[int],
    token_mixer_type: str,
    kernel_size: int = 3,
    mlp_ratio: float = 4.0,
    act_layer: nn.Module = nn.GELU,
    norm_layer: nn.Module = nn.BatchNorm,
):
    blocks = CallableModuleList()
    for _ in range(num_blocks[block_index]):
        if token_mixer_type == "repmixer":
            blocks.append(
                RepMixerBlock(
                    dim,
                    kernel_size=kernel_size,
                    mlp_ratio=mlp_ratio,
                    act_layer=act_layer,
                )
            )
        elif token_mixer_type == "attention":
            blocks.append(
                AttentionBlock(
                    dim,
                    mlp_ratio=mlp_ratio,
                    act_layer=act_layer,
                    norm_layer=norm_layer,
                )
            )
        else:
            raise ValueError(
                "Token mixer type: {} not supported".format(token_mixer_type)
            )
    return blocks


def build_fast_vit_network(config: VisionConfig):
    network = []
    for i in range(len(config.layers)):
        spatial_shape = config.pos_embs_shapes[i]
        if spatial_shape is not None:
            position_embeddings = RepCPE(
                in_channels=config.embed_dims[i],
                embed_dim=config.embed_dims[i],
                spatial_shape=spatial_shape,
            )
            network.append(position_embeddings)

        stage = basic_blocks(
            config.embed_dims[i],
            i,
            config.layers,
            token_mixer_type=config.token_mixers[i],
            kernel_size=config.repmixer_kernel_size,
            mlp_ratio=config.mlp_ratios[i],
            norm_layer=LayerNormChannel,
        )
        network.append(stage)

        if i >= len(config.layers) - 1:
            break

        # Patch merging/downsampling between stages.
        if config.downsamples[i] or config.embed_dims[i] != config.embed_dims[i + 1]:
            network.append(
                PatchEmbed(
                    patch_size=config.down_patch_size,
                    stride=config.down_stride,
                    in_channels=config.embed_dims[i],
                    embed_dim=config.embed_dims[i + 1],
                )
            )
    return network


class SEBlock(nn.Module):
    """Squeeze and Excite module.

    MLX implementation of `Squeeze-and-Excitation Networks` -
    https://arxiv.org/pdf/1709.01507.pdf
    """

    def __init__(self, in_channels: int, rd_ratio: float = 0.0625):
        """Construct a Squeeze and Excite Module.

        Args:
            in_channels: Number of input channels.
            rd_ratio: Input channel reduction ratio.
        """
        super().__init__()
        self.reduce = nn.Conv2d(
            in_channels=in_channels,
            out_channels=int(in_channels * rd_ratio),
            kernel_size=1,
            stride=1,
            bias=True,
        )
        self.expand = nn.Conv2d(
            in_channels=int(in_channels * rd_ratio),
            out_channels=in_channels,
            kernel_size=1,
            stride=1,
            bias=True,
        )

    def __call__(self, inputs: mx.array) -> mx.array:
        _, h, w, c = inputs.shape
        x = nn.AvgPool2d(kernel_size=[h, w])(inputs)
        x = self.reduce(x)
        x = nn.layers.relu(x)
        x = self.expand(x)
        x = mx.sigmoid(x)
        x = x.reshape(-1, 1, 1, c)
        return inputs * x


class MobileOneBlock(nn.Module):
    """MobileOne building block.

    This implementation only uses the inference time CNN architecture and uses FastViTHD conventions.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
        dilation: int = 1,
        groups: int = 1,
        use_se: bool = False,
    ):
        super().__init__()
        self.groups = groups
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.kernel_size = kernel_size
        self.in_channels = in_channels
        self.out_channels = out_channels

        # Check if SE-ReLU is requested
        if use_se:
            self.se = SEBlock(out_channels)
        else:
            self.se = nn.Identity()

        self.activation = nn.GELU()
        self.reparam_conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=True,
        )

    def __call__(self, x: mx.array) -> mx.array:
        return self.activation(self.se(self.reparam_conv(x)))


class ConvolutionalStem(nn.Module):
    def __init__(self, config: VisionConfig):
        super().__init__()
        in_channels = 3
        out_channels = config.embed_dims[0]
        self.blocks = CallableModuleList(
            [
                MobileOneBlock(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=3,
                    stride=2,
                    padding=1,
                    groups=1,
                ),
                MobileOneBlock(
                    in_channels=out_channels,
                    out_channels=out_channels,
                    kernel_size=3,
                    stride=2,
                    padding=1,
                    groups=out_channels,
                ),
                MobileOneBlock(
                    in_channels=out_channels,
                    out_channels=out_channels,
                    kernel_size=1,
                    stride=1,
                    padding=0,
                    groups=1,
                ),
            ]
        )

    def __call__(self, x: mx.array) -> mx.array:
        return self.blocks(x)


class FastViTHDModel(nn.Module):
    """
    Based on https://github.com/apple/ml-fastvlm/blob/592b4add3c1c8a518e77d95dc6248e76c1dd591f/llava/model/multimodal_encoder/mobileclip/mci.py
    Hardcoded, for now, for:
    - FastViTHD variant
    - Use inference_mode (i.e., modules contain the convolutional reparameterized versions of the architecture)
    """

    def __init__(self, config: VisionConfig):
        super().__init__()
        if config.pos_embs_shapes is None:
            config.pos_embs_shapes = [None] * len(config.layers)
        self.config = config

        # We follow the nomenclature from mci.py
        self.patch_embed = ConvolutionalStem(config)
        self.network = build_fast_vit_network(config)
        self.conv_exp = MobileOneBlock(
            in_channels=config.embed_dims[-1],
            out_channels=int(config.embed_dims[-1] * config.cls_ratio),
            kernel_size=3,
            stride=1,
            padding=1,
            groups=config.embed_dims[-1],
            use_se=True,
        )
        self.head = nn.Linear(
            int(config.embed_dims[-1] * config.cls_ratio), config.num_classes
        )

    def __call__(
        self,
        x: mx.array,
        output_hidden_states: Optional[bool] = None,
    ):
        x = self.patch_embed(x)

        encoder_states = (x,) if output_hidden_states else None
        for layer in self.network:
            x = layer(x)
            if output_hidden_states:
                encoder_states = encoder_states + (x,)

        x = self.conv_exp(x)
        cls_out = self.head(x)

        return cls_out, x, encoder_states


class GlobalPool2D(nn.Module):
    """This class implements global pooling with linear projection."""

    def __init__(self, in_dim: int, out_dim: int) -> None:
        super().__init__()
        self.proj = mx.zeros((in_dim, out_dim))

    def __call__(self, x: mx.array) -> mx.array:
        assert (
            x.ndim == 4
        ), "Input should be 4-dimensional (Batch x in_dim x in_height x in_width). Got: {}".format(
            x.shape
        )

        # [batch, in_height, in_width, in_dim] --> [batch, in_dim]
        x = x.mean(axis=[1, 2])
        # [batch, in_dim] x [in_dim, out_dim] --> [batch, out_dim]
        x = x @ self.proj
        return x


class VisionModel(nn.Module):
    def __init__(self, config: VisionConfig):
        super().__init__()

        self.model_type = config.model_type
        if self.model_type not in ["llava_qwen2", "fastvlm"]:
            raise ValueError(f"Unsupported model type: {self.model_type}")

        self.vision_model = FastViTHDModel(config)

        # Replace projection head, same as in
        # https://github.com/apple/ml-fastvlm/blob/592b4add3c1c8a518e77d95dc6248e76c1dd591f/llava/model/multimodal_encoder/mobileclip/__init__.py#L49
        if config.projection_dim is not None:
            in_dim = int(config.embed_dims[-1] * config.cls_ratio)
            self.vision_model.head = GlobalPool2D(in_dim, config.projection_dim)

    def __call__(
        self, x: mx.array, output_hidden_states: Optional[bool] = None
    ) -> mx.array:
        return self.vision_model(x, output_hidden_states)

    def sanitize(self, weights):
        # Only transpose during conversion from transformers
        W, C = weights[
            "vision_tower.vision_model.patch_embed.blocks.1.reparam_conv.weight"
        ].shape[-2:]
        skip_transpose = W > C

        def is_conv(k):
            if skip_transpose:
                return False
            if ".reparam_conv.weight" in k:
                return True
            if ".conv.weight" in k:
                return True
            if ".fc1.weight" in k:
                return True
            if ".fc2.weight" in k:
                return True
            if ".lkb_reparam.weight" in k:
                return True
            if ".reduce.weight" in k:
                return True
            if ".expand.weight" in k:
                return True
            return False

        sanitized_weights = {}
        for k, v in weights.items():
            if is_conv(k):
                # PyTorch conv2d weight tensors have shape:
                #   [out_channels, in_channels, kH, KW]
                # MLX conv2d expects the weight be of shape:
                #   [out_channels, kH, KW, in_channels]
                if v.ndim == 4:
                    sanitized_weights[k] = v.transpose(0, 2, 3, 1)
                else:
                    sanitized_weights[k] = v
            elif "layer_scale" in k and not skip_transpose:
                sanitized_weights[k] = v.transpose(1, 2, 0)
            elif "num_batches_tracked" in k:
                # I don't think we need this
                continue
            else:
                sanitized_weights[k] = v
        return sanitized_weights
