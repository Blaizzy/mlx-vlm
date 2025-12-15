from typing import Optional

import mlx.core as mx
import mlx.nn as nn
import numpy as np

from ..interpolate import resize_bilinear
from .config import VisionConfig


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


class Attention(nn.Module):
    def __init__(
        self,
        dims: int,
        num_heads: int,
        query_input_dims: Optional[int] = None,
        key_input_dims: Optional[int] = None,
        value_input_dims: Optional[int] = None,
        value_dims: Optional[int] = None,
        value_output_dims: Optional[int] = None,
        bias: bool = True,
    ):
        super().__init__()

        if (dims % num_heads) != 0:
            raise ValueError(
                "The input feature dimensions should be divisible by the "
                f"number of heads ({dims} % {num_heads}) != 0"
            )

        query_input_dims = query_input_dims or dims
        key_input_dims = key_input_dims or dims
        value_input_dims = value_input_dims or key_input_dims
        value_dims = value_dims or dims
        value_output_dims = value_output_dims or dims

        self.num_heads = num_heads
        head_dim = dims // num_heads
        self.scale = head_dim**-0.5

        self.q_proj = nn.Linear(query_input_dims, dims, bias=bias)
        self.k_proj = nn.Linear(key_input_dims, dims, bias=bias)
        self.v_proj = nn.Linear(value_input_dims, value_dims, bias=bias)
        self.out_proj = nn.Linear(value_dims, value_output_dims, bias=bias)

    def __call__(self, x, mask=None):
        queries = self.q_proj(x)
        keys = self.k_proj(x)
        values = self.v_proj(x)

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
        return self.out_proj(output)


class MLP(nn.Module):
    def __init__(self, config: VisionConfig):
        super().__init__()
        self.activation_fn = nn.GELU(approx="precise")
        self.fc1 = nn.Linear(config.hidden_size, config.intermediate_size, bias=True)
        self.fc2 = nn.Linear(config.intermediate_size, config.hidden_size, bias=True)

    def __call__(self, x: mx.array) -> mx.array:
        x = self.fc1(x)
        x = self.activation_fn(x)
        x = self.fc2(x)
        return x


class EncoderLayer(nn.Module):
    def __init__(self, config: VisionConfig):
        super().__init__()
        self.embed_dim = config.hidden_size
        self.self_attn = Attention(
            config.hidden_size, config.num_attention_heads, bias=True
        )
        self.layer_norm1 = nn.LayerNorm(self.embed_dim, eps=config.layer_norm_eps)
        self.mlp = MLP(config)
        self.layer_norm2 = nn.LayerNorm(self.embed_dim, eps=config.layer_norm_eps)

    def __call__(self, x: mx.array, mask: Optional[mx.array] = None) -> mx.array:
        r = self.self_attn(self.layer_norm1(x), mask)
        h = x + r
        r = self.mlp(self.layer_norm2(h))
        return h + r


class Encoder(nn.Module):
    def __init__(self, config: VisionConfig):
        super().__init__()
        self.layers = [EncoderLayer(config) for _ in range(config.num_hidden_layers)]

    def __call__(
        self,
        x: mx.array,
        output_hidden_states: Optional[bool] = None,
        mask: Optional[mx.array] = None,
    ) -> mx.array:
        encoder_states = (x,) if output_hidden_states else None
        h = x
        for l in self.layers:
            x = l(x, mask=mask)
            if output_hidden_states:
                encoder_states = encoder_states + (x,)

            h = x

        return (h, encoder_states)


def gaussian_blur_axis(image, sigma, axis):
    """
    Applies a 1D Gaussian blur along the given axis.
    This version works for arrays with any number of dimensions.
    """
    radius = int(3 * sigma)
    if radius < 1:
        return image
    x = mx.arange(-radius, radius + 1)
    kernel = mx.exp(-(x**2) / (2 * sigma**2))
    kernel = kernel / mx.sum(kernel)

    # MLX doesn't have a direct apply_along_axis equivalent,
    # so we'll implement the convolution differently based on the axis

    # Helper function to apply 1D convolution along specific axis
    def conv_1d(array, kernel, axis):
        # Reshape kernel to broadcast along the right dimensions
        kernel_shape = [1] * image.ndim
        kernel_shape[axis] = len(kernel)
        kernel_reshaped = kernel.reshape(kernel_shape)

        # Pad the array
        pad_width = [(0, 0)] * image.ndim
        pad_width[axis] = (radius, radius)
        padded = mx.pad(array, pad_width, mode="edge")

        # Perform convolution via sliding window sum
        result = mx.zeros_like(array)
        slices = [slice(None)] * padded.ndim

        for i in range(2 * radius + 1):
            slices[axis] = slice(i, i + array.shape[axis])
            result = result + padded[tuple(slices)] * kernel_reshaped

        return result

    return conv_1d(image, kernel, axis)


class VisionEmbeddings(nn.Module):
    def __init__(self, config: VisionConfig):
        super().__init__()
        self.config = config
        self.embed_dim = config.hidden_size
        self.image_size = config.image_size
        self.patch_size = config.patch_size

        self.patch_embedding = nn.Conv2d(
            config.num_channels,
            config.hidden_size,
            kernel_size=self.patch_size,
            stride=self.patch_size,
        )

        self.num_patches = (self.image_size // self.patch_size) ** 2
        self.num_positions = self.num_patches
        self.position_embedding = nn.Embedding(self.num_positions, self.embed_dim)

    @staticmethod
    def resize_positional_embeddings(
        positional_embeddings: mx.array,
        spatial_shapes: mx.array,
        max_length: int,
    ) -> mx.array:
        """
        Resize positional embeddings to image-specific size and pad to a fixed size.

        Args:
            positional_embeddings (`torch.Tensor`):
                Position embeddings of shape (height, width, embed_dim)
            spatial_shapes (`torch.LongTensor`):
                Spatial shapes of shape (batch_size, 2) to resize the positional embeddings to
            max_length (`int`):
                Maximum length of the positional embeddings to pad resized positional embeddings to

        Returns:
            `torch.Tensor`: Embeddings of shape (batch_size, max_length, embed_dim)
        """
        batch_size = spatial_shapes.shape[0]
        embed_dim = positional_embeddings.shape[-1]
        source_dtype = positional_embeddings.dtype

        resulted_positional_embeddings = mx.zeros(
            (batch_size, max_length, embed_dim)
        ).astype(source_dtype)

        # (height, width, embed_dim) -> (1, embed_dim, height, width) for interpolation
        positional_embeddings = positional_embeddings.transpose(2, 0, 1).reshape(
            1, embed_dim, -1
        )

        # Upcast to float32 on CPU because antialias is not supported for bfloat16/float16 on CPU
        if positional_embeddings.device.type == "cpu":
            positional_embeddings = positional_embeddings.astype(mx.float32)

        for i in range(batch_size):
            # (1, dim, height, width) -> (1, dim, target_height, target_width)
            height, width = spatial_shapes[i]
            # Then upsample width dimension
            resized_embeddings = resize_bilinear(
                positional_embeddings,
                (height, width),
                align_corners=False,
                antialias=True,
            )

            # (1, dim, target_height, target_width) -> (target_height * target_width, dim)
            resized_embeddings = resized_embeddings.reshape(
                embed_dim, height * width
            ).transpose(0, 1)

            # Cast to original dtype
            resized_embeddings = resized_embeddings.astype(source_dtype)

            resulted_positional_embeddings[i, : height * width] = resized_embeddings
            resulted_positional_embeddings[i, height * width :] = resized_embeddings[0]

        return resulted_positional_embeddings

    def __call__(
        self, x: mx.array, spatial_shapes: Optional[mx.array] = None
    ) -> mx.array:
        batch_size = x.shape[0]
        patch_embeddings = self.patch_embedding(x)
        patch_embeddings = mx.flatten(patch_embeddings, start_axis=1, end_axis=2)
        if spatial_shapes is None:
            position_ids = mx.array(np.arange(self.num_positions)[None, :])
            embeddings = patch_embeddings
            embeddings += self.position_embedding(position_ids)

        else:
            # Get positional resized and padded positional embeddings
            positional_embeddings = self.position_embedding.weight.reshape(
                self.position_embedding_size, self.position_embedding_size, -1
            )

            resized_positional_embeddings = self.resize_positional_embeddings(
                positional_embeddings, spatial_shapes, max_length=x.shape[1]
            )

            # Add positional embeddings to patch embeddings
            embeddings = patch_embeds + resized_positional_embeddings
        return embeddings


class SigLipVisionModel(nn.Module):
    def __init__(self, config: VisionConfig):
        super().__init__()

        self.embeddings = VisionEmbeddings(config)
        self.encoder = Encoder(config)
        self.post_layernorm = nn.LayerNorm(config.hidden_size)

    def __call__(
        self,
        x: mx.array,
        spatial_shapes: mx.array,
        output_hidden_states: Optional[bool] = None,
    ) -> mx.array:
        x = self.embeddings(x, spatial_shapes)
        x = x.astype(self.embeddings.patch_embedding.weight.dtype)
        encoder_outputs = self.encoder(
            x=x, output_hidden_states=output_hidden_states, mask=None
        )
        pooler_output = self.post_layernorm(encoder_outputs[0])
        return pooler_output, x, encoder_outputs[-1]


class VisionModel(nn.Module):
    def __init__(self, config: VisionConfig):
        super().__init__()
        self.model_type = config.model_type
        if self.model_type not in ["siglip_vision_model"]:
            raise ValueError(f"Unsupported model type: {self.model_type}")

        self.vision_model = SigLipVisionModel(config)

    def __call__(
        self,
        x: mx.array,
        spatial_shapes: Optional[mx.array] = None,
        output_hidden_states: Optional[bool] = None,
    ) -> mx.array:
        return self.vision_model(x, spatial_shapes, output_hidden_states)

    def sanitize(self, weights):
        sanitized_weights = {}
        for k, v in weights.items():
            if "position_ids" in k:
                # Remove unused position_ids
                continue
            elif "patch_embedding.weight" in k:
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
