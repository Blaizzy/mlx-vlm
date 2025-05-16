import copy
import inspect
from dataclasses import dataclass
from functools import partial
from math import sqrt
from typing import Dict, Optional, Union

import cv2
import mlx.core as mx
import mlx.nn as nn
import numpy as np

from .sam import SAMEncoder


@dataclass
class VisionConfig:
    model_type: str
    num_hidden_layers: int = 24
    hidden_size: int = 1024
    intermediate_size: int = 4096
    num_attention_heads: int = 16
    image_size: int = 384
    patch_size: int = 16
    num_channels: int = 3
    layer_norm_eps: float = 1e-5
    cls: str = None
    params: dict = None

    def __post_init__(self):
        if "high_res_cfg" in self.params:
            self.image_size = self.params["high_res_cfg"]["image_size"]

    @classmethod
    def from_dict(cls, params):
        return cls(
            **{
                k: v
                for k, v in params.items()
                if k in inspect.signature(cls).parameters
            }
        )


@dataclass
class MLPConfig:
    hidden_size: int
    intermediate_size: int


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


class AttentionPoolLatent(nn.Module):
    """Attention pooling w/ latent query"""

    def __init__(
        self,
        in_features: int,
        out_features: int = None,
        embed_dim: int = None,
        num_heads: int = 8,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        qk_norm: bool = False,
        latent_len: int = 1,
        latent_dim: int = None,
        pos_embed: str = "",
        pool_type: str = "token",
        norm_layer: Optional[nn.Module] = None,
        drop: float = 0.0,
    ):
        super().__init__()

        embed_dim = embed_dim or in_features
        out_features = out_features or in_features
        assert embed_dim % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim**-0.5
        self.pool = pool_type

        self.latent_dim = latent_dim or embed_dim
        self.latent_len = latent_len
        self.latent = mx.zeros((self.latent_len, embed_dim))[None, :]

        self.q = nn.Linear(embed_dim, embed_dim, bias=qkv_bias)
        self.kv = nn.Linear(embed_dim, embed_dim * 2, bias=qkv_bias)
        self.q_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.proj_drop = nn.Dropout(drop)

        if pos_embed == "abs":
            spatial_len = self.feat_size
            self.pos_embed = mx.zeros((spatial_len, in_features))
        else:
            self.pos_embed = None

        self.norm = nn.LayerNorm(out_features)
        config = MLPConfig(
            hidden_size=embed_dim, intermediate_size=int(embed_dim * mlp_ratio)
        )
        self.mlp = MLP(config)

    def __call__(self, x: mx.array):
        B, N, C = x.shape

        if self.pos_embed is not None:
            x = x + self.pos_embed.unsqueeze(0).to(x.dtype)

        q_latent = mx.array(self.latent)

        q = (
            self.q(q_latent)
            .reshape(B, self.latent_len, self.num_heads, self.head_dim)
            .transpose(0, 2, 1, 3)
        )

        kv = (
            self.kv(x)
            .reshape(B, N, 2, self.num_heads, self.head_dim)
            .transpose(2, 0, 3, 1, 4)
        )
        k, v = mx.split(kv, 2, axis=0)

        q, k = self.q_norm(q), self.k_norm(k)

        x = mx.fast.scaled_dot_product_attention(
            q, k[0], v[0], scale=(1.0 / sqrt(q.shape[-1])), mask=None
        )

        x = x.transpose(0, 2, 1, 3).reshape(B, self.latent_len, C)
        x = self.proj(x)
        x = self.proj_drop(x)

        x = x + self.mlp(self.norm(x))

        # optional pool if latent seq_len > 1 and pooled output is desired
        if self.pool == "token":
            x = x[:, 0]
        elif self.pool == "avg":
            x = x.mean(1)
        return x


class Attention(nn.Module):
    def __init__(
        self,
        dims: int,
        num_heads: int,
        qkv_bias: bool = False,
    ):
        super().__init__()

        if (dims % num_heads) != 0:
            raise ValueError(
                "The input feature dimensions should be divisible by the "
                f"number of heads ({dims} % {num_heads}) != 0"
            )

        self.num_heads = num_heads = num_heads
        head_dim = dims // num_heads
        self.scale = head_dim**-0.5

        self.qkv = nn.Linear(dims, dims * 3, bias=qkv_bias)
        self.proj = nn.Linear(dims, dims, bias=True)

    def __call__(self, x, mask=None):
        qkv = self.qkv(x)
        queries, keys, values = mx.split(qkv, 3, axis=-1)

        num_heads = self.num_heads
        B, L, D = queries.shape
        _, S, _ = keys.shape
        queries = queries.reshape(B, L, num_heads, -1).transpose(0, 2, 1, 3)
        keys = keys.reshape(B, S, num_heads, -1).transpose(0, 2, 1, 3)
        values = values.reshape(B, S, num_heads, -1).transpose(0, 2, 1, 3)

        output = mx.fast.scaled_dot_product_attention(
            queries, keys, values, scale=self.scale, mask=mask
        )
        output = output.transpose(0, 2, 1, 3).reshape(B, L, -1)

        return self.proj(output)


class FastGELUActivation(nn.Module):
    """
    Applies GELU approximation that is slower than QuickGELU but more accurate. See: https://github.com/hendrycks/GELUs
    """

    def __call__(self, input: mx.array) -> mx.array:
        return (
            0.5
            * input
            * (1.0 + mx.tanh(np.sqrt(2 / np.pi) * (input + 0.044715 * (input**3))))
        ).astype(input.dtype)


class MLP(nn.Module):
    def __init__(self, config: Union[VisionConfig, Dict], bias: bool = True):
        super().__init__()
        self.activation_fn = FastGELUActivation()
        self.fc1 = nn.Linear(config.hidden_size, config.intermediate_size, bias=bias)
        self.fc2 = nn.Linear(config.intermediate_size, config.hidden_size, bias=bias)

    def __call__(self, x: mx.array) -> mx.array:
        x = self.activation_fn(self.fc1(x))
        x = self.fc2(x)
        return x


class EncoderLayer(nn.Module):
    def __init__(self, config: VisionConfig):
        super().__init__()
        self.embed_dim = config.hidden_size
        self.attn = Attention(
            config.hidden_size, config.num_attention_heads, qkv_bias=True
        )
        self.norm1 = nn.LayerNorm(self.embed_dim, eps=config.layer_norm_eps)
        self.mlp = MLP(config)
        self.norm2 = nn.LayerNorm(self.embed_dim, eps=config.layer_norm_eps)

    def __call__(self, x: mx.array, mask: Optional[mx.array] = None) -> mx.array:
        y = self.norm1(x)
        y = self.attn(y, mask)
        x = x + y
        y = self.norm2(x)
        y = self.mlp(y)
        return x + y


class VisionEmbeddings(nn.Module):
    def __init__(self, config: VisionConfig, norm_layer: bool = False):
        super().__init__()
        self.config = config
        self.embed_dim = config.hidden_size
        self.image_size = config.image_size
        self.patch_size = config.patch_size

        self.proj = nn.Conv2d(
            in_channels=config.num_channels,
            out_channels=self.embed_dim,
            kernel_size=self.patch_size,
            stride=self.patch_size,
        )

        self.num_patches = (self.image_size // self.patch_size) ** 2
        self.num_positions = self.num_patches

        self.norm = (
            nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
            if norm_layer
            else nn.Identity()
        )

    def __call__(self, x: mx.array) -> mx.array:
        patch_embeddings = self.proj(x)
        patch_embeddings = mx.flatten(patch_embeddings, start_axis=1, end_axis=2)
        return self.norm(patch_embeddings)


class SigLipVisionModel(nn.Module):
    def __init__(
        self,
        config: VisionConfig,
        ignore_head: bool,
        pre_norm: bool = False,
        no_embed_class: bool = True,
    ):
        super().__init__()
        self.num_prefix_tokens = 1
        self.no_embed_class = False
        self.dynamic_img_size = False
        self.ignore_head = ignore_head
        self.cls_token = None
        self.reg_token = None
        self.patch_embed = VisionEmbeddings(config)
        self.norm_pre = nn.LayerNorm(config.hidden_size) if pre_norm else nn.Identity()
        self.blocks = [EncoderLayer(config) for _ in range(config.num_hidden_layers)]
        self.norm = nn.LayerNorm(config.hidden_size)
        num_patches = self.patch_embed.num_patches
        embed_len = (
            num_patches if no_embed_class else num_patches + self.num_prefix_tokens
        )
        self.pos_embed = mx.random.normal((embed_len, config.hidden_size))[None, :]

        norm_layer = partial(nn.LayerNorm, eps=1e-5)
        self.attn_pool = AttentionPoolLatent(
            config.hidden_size,
            num_heads=config.num_attention_heads,
            norm_layer=norm_layer,
        )

    def __call__(
        self,
        x: mx.array,
        output_hidden_states: Optional[bool] = None,
    ) -> mx.array:
        x = self.patch_embed(x)
        x += self.pos_embed
        x = self.norm_pre(x)

        encoder_states = (x,) if output_hidden_states else None
        for l in self.blocks:
            x = l(x, mask=None)
            if output_hidden_states:
                encoder_states = encoder_states + (x,)

        pooler_output = self.norm(x)

        if not self.ignore_head:
            pooler_output = self.attn_pool(pooler_output)
        return pooler_output, x, encoder_states


class HybridVisionModel(nn.Module):
    def __init__(self, config: VisionConfig, resolution: str, ignore_head: bool = True):
        super().__init__()

        self.model_type = config.model_type
        self.resolution = resolution
        if self.model_type != "vision":
            raise ValueError(f"Unsupported model type: {self.model_type}")

        if resolution == "high":
            self.vision_tower = SAMEncoder()
        else:
            self.vision_tower = SigLipVisionModel(config, ignore_head)

    def __call__(self, x: mx.array) -> mx.array:
        if self.resolution == "high":
            return self.vision_tower(x)
        else:
            return self.vision_tower(x)[0]


# def resize_image(image, size, antialias=True):
#     """
#     Resize an image using scipy.ndimage.zoom with an option for bicubic interpolation.

#     Args:
#         image (numpy.ndarray): The input image array.
#         size (tuple): The target size as (width, height).
#         antialias (bool): True to use bicubic interpolation, False to use nearest neighbor.

#     Returns:
#         numpy.ndarray: The resized image array.
#     """
#     # Ensure the image is an array and remove singleton dimensions
#     image = np.array(image[0])

#     # Calculate zoom factors for the spatial dimensions
#     # Note: size is expected as (width, height) but image.shape gives (height, width)
#     current_height, current_width = image.shape[:2]
#     width_factor = size[0] / current_width
#     height_factor = size[1] / current_height
#     zoom_factors = (height_factor, width_factor)  # Apply zoom to height and width

#     # Choose the interpolation order: 3 for bicubic, 0 for nearest
#     order = 3 if antialias else 0

#     # Apply zoom to the image. Handle both grayscale and color images.
#     if image.ndim == 2:  # Grayscale image
#         resized_image = zoom(image, zoom_factors, order=order)
#     elif image.ndim == 3:  # Color image
#         # Apply zoom separately for each channel
#         resized_channels = [
#             zoom(image[:, :, i], zoom_factors, order=order)
#             for i in range(image.shape[2])
#         ]
#         resized_image = np.stack(resized_channels, axis=2)

#     return resized_image


# TODO: Match the output of scipy.ndimage.zoom
def resize_image(image, size, antialias=True):
    """
    Resize an image with OpenCV.

    Args:
        image (numpy.ndarray): The input image array.  Supports H × W or H × W × C.
                               If you pass in a batch (N × H × W × C) just slice the
                               element you want, e.g. image[0].
        size  (tuple): Target size as (width, height) — exactly the same order that
                       cv2.resize expects.
        antialias (bool):
            * True  → high‑quality interpolation (bicubic for upscaling, area for downscaling)
            * False → nearest‑neighbor (fast, blocky)

    Returns:
        numpy.ndarray: The resized image array.
    """
    img = np.ascontiguousarray(np.asarray(image))
    if img.ndim == 4 and img.shape[0] == 1:  # squeeze stray batch dim
        img = img[0]
    h0, w0 = img.shape[:2]

    # --- work out dsize vs fx/fy ---------------------------------------------
    dsize = None
    fx = fy = 0.0
    if isinstance(size, (int, float)):  # uniform scale
        fx = fy = float(size)
    elif isinstance(size, (tuple, list)) and len(size) == 2:
        a, b = size
        # Heuristic: treat "small" floats as scale factors
        if all(isinstance(x, (int, float)) and x < 10 for x in (a, b)):
            fx, fy = float(a), float(b)  # scale factors
        else:
            dsize = (int(a), int(b))  # absolute pixels
    else:
        raise ValueError("target must be scalar or a 2‑tuple")

    # Guard against zeros after int‑casting
    if dsize:
        if dsize[0] <= 0 or dsize[1] <= 0:
            raise ValueError(f"dsize became {dsize}")
    else:
        if fx <= 0 or fy <= 0:
            raise ValueError(f"fx,fy became {(fx, fy)}")

    # --- choose interpolation -------------------------------------------------
    if antialias:
        # Use Lanczos interpolation for potentially better detail preservation
        interp = cv2.INTER_LANCZOS4
    else:
        interp = cv2.INTER_NEAREST

    # --- call OpenCV ----------------------------------------------------------
    return mx.array(cv2.resize(img, dsize=dsize, fx=fx, fy=fy, interpolation=interp))


class VisionModel(nn.Module):
    def __init__(self, config: VisionConfig, ignore_head: bool = True):
        super().__init__()

        self.model_type = config.model_type
        self.config = config
        if self.model_type != "vision":
            raise ValueError(f"Unsupported model type: {self.model_type}")

        if config.cls == "HybridVisionTower":
            self.high_layer_norm = nn.LayerNorm(
                config.params["high_res_cfg"]["output_dim"]
            )
            self.low_layer_norm = nn.LayerNorm(
                config.params["low_res_cfg"]["output_dim"]
            )

            high_res_cfg = copy.deepcopy(config)
            high_res_cfg.image_size = config.params["high_res_cfg"]["image_size"]
            self.vision_tower_high = HybridVisionModel(
                high_res_cfg, "high", ignore_head
            )

            low_res_cfg = copy.deepcopy(config)
            low_res_cfg.image_size = config.params["low_res_cfg"]["image_size"]

            self.vision_tower_low = HybridVisionModel(low_res_cfg, "low", ignore_head)
            self.low_res_size = config.params["low_res_cfg"]["image_size"]
            self.resize = lambda image: resize_image(
                image, (self.low_res_size, self.low_res_size), antialias=True
            )

        else:
            self.vision_tower = SigLipVisionModel(config, ignore_head)

    def __call__(
        self, x: mx.array, output_hidden_states: Optional[bool] = None
    ) -> mx.array:
        if self.config.cls == "HybridVisionTower":
            high_images = x
            low_images = mx.array(self.resize(np.array(x)))[None, :]

            high_res = self.vision_tower_high(high_images)
            low_res = self.vision_tower_low(low_images)

            return (high_res, low_res)
        else:
            return self.vision_tower(x, output_hidden_states)

    def sanitize(self, weights):
        sanitized_weights = {}
        weight_keys = {
            "neck.0.weight",
            "neck.2.weight",
            "neck_hd.0.weight",
            "neck_hd.2.weight",
            "downsamples.0.weight",
            "downsamples.1.weight",
            "patch_embed.proj.weight",
        }
        for k, v in weights.items():
            if "position_ids" in k:
                # Remove unused position_ids
                continue

            elif ".".join(k.split(".")[-3:]) in weight_keys:
                # PyTorch conv2d weight tensors have shape:
                #   [out_channels, in_channels, kH, KW]
                # MLX conv2d expects the weight be of shape:
                #   [out_channels, kH, KW, in_channels]
                if check_array_shape(v):
                    sanitized_weights[k] = v
                else:
                    sanitized_weights[k] = v.transpose(0, 2, 3, 1)

            else:
                sanitized_weights[k] = v

        return sanitized_weights
