import math
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import mlx.core as mx
import mlx.nn as nn
from mlx_lm.models.base import create_attention_mask, scaled_dot_product_attention
from mlx_lm.models.cache import RotatingKVCache
from PIL import Image
from transformers.image_processing_utils import BaseImageProcessor as ImageProcessor
from transformers.image_processing_utils import get_size_dict
from transformers.image_utils import ChannelDimension, PILImageResampling


@dataclass
class LanguageModelOutput:
    logits: mx.array
    cross_attention_states: Optional[List[mx.array]] = None
    encoder_outputs: Optional[List[mx.array]] = None


def expand2square(pil_img, background_color):
    width, height = pil_img.size
    if width == height:
        return pil_img
    elif width > height:
        result = Image.new(pil_img.mode, (width, width), background_color)
        result.paste(pil_img, (0, (width - height) // 2))
        return result
    else:
        result = Image.new(pil_img.mode, (height, height), background_color)
        result.paste(pil_img, ((height - width) // 2, 0))
        return result


def check_array_shape(arr):
    shape = arr.shape

    # Check if the shape has 4 dimensions
    if len(shape) == 4:
        out_channels, kH, KW, _ = shape
        # Check if out_channels is the largest, and kH and KW are the same
        if (out_channels >= kH) and (out_channels >= KW) and (kH == KW):
            return True
        else:
            return False
    # Check if the shape has 3 dimensions
    elif len(shape) == 3:
        _, kW, out_channels = shape
        # Check if out_channels is the largest
        if kW >= out_channels:
            return True
        else:
            return False
    else:
        return False


class BaseImageProcessor(ImageProcessor):
    def __init__(
        self,
        image_mean=(0.5, 0.5, 0.5),
        image_std=(0.5, 0.5, 0.5),
        size=(384, 384),
        crop_size: Dict[str, int] = None,
        resample=PILImageResampling.BICUBIC,
        rescale_factor=1 / 255,
        data_format=ChannelDimension.FIRST,
    ):
        crop_size = (
            crop_size if crop_size is not None else {"height": 384, "width": 384}
        )
        crop_size = get_size_dict(
            crop_size, default_to_square=True, param_name="crop_size"
        )

        self.image_mean = image_mean
        self.image_std = image_std
        self.size = size
        self.resample = resample
        self.rescale_factor = rescale_factor
        self.data_format = data_format
        self.crop_size = crop_size

    @abstractmethod
    def preprocess(self, images):
        pass


# Add this code to visualize the chunked attention mask
def visualize_attention_mask(mask):
    """Visualize attention mask with symbols for better readability."""
    if mask is None:
        print("No mask")
        return

    seq_len = mask.shape[0]

    print("        ", end="")
    for i in range(seq_len):
        print(f"{i:2d} ", end="")
    print()

    for i in range(seq_len):
        print(f"Token {i:2d}: ", end="")
        for j in range(seq_len):
            if mask[i, j]:
                print(" ■ ", end="")
            else:
                print(" ⬚ ", end="")
        print()


def check_activation_stats(name, tensor):
    """Helper function to check for anomalies and log stats."""

    print(f"--- Activation Stats: {name} ---")
    # Check for NaNs/Infs
    has_nan = mx.isnan(tensor).any()
    has_inf = mx.isinf(tensor).any()
    if has_nan:
        print(f"WARNING: Found NaN in {name}")
    if has_inf:
        print(f"WARNING: Found Inf in {name}")

    # Calculate and print stats (ensure computation happens)
    min_val = mx.min(tensor).item()
    max_val = mx.max(tensor).item()
    mean_val = mx.mean(tensor).item()
    std_val = mx.std(tensor).item()
    print(f"  Shape: {tensor.shape}")
    print(f"  Min: {min_val:.4f}, Max: {max_val:.4f}")
    print(f"  Mean: {mean_val:.4f}, Std: {std_val:.4f}")
    print("-" * (len(name) + 24))


def pixel_shuffle(input_tensor, shuffle_ratio):
    # input_tensor: [batch_size, num_patches, channels]
    batch_size, num_patches, channels = input_tensor.shape
    patch_size = int(math.sqrt(num_patches))

    input_tensor = input_tensor.reshape(batch_size, patch_size, patch_size, -1)
    batch_size, height, width, channels = input_tensor.shape

    reshaped_tensor = input_tensor.reshape(
        batch_size, height, int(width * shuffle_ratio), int(channels / shuffle_ratio)
    )
    reshaped_tensor = reshaped_tensor.transpose(0, 2, 1, 3)

    reshaped_tensor = reshaped_tensor.reshape(
        batch_size,
        int(height * shuffle_ratio),
        int(width * shuffle_ratio),
        int(channels / (shuffle_ratio**2)),
    )
    reshaped_tensor = reshaped_tensor.transpose(0, 2, 1, 3)

    output_tensor = reshaped_tensor.reshape(batch_size, -1, reshaped_tensor.shape[-1])
    return output_tensor


def interpolate(pos_embed, size, mode="cubic", align_corners=False):
    """
    MLX implementation of PyTorch's F.interpolate with bicubic mode

    Args:
        pos_embed: MLX array with shape [B, C, H_src, W_src] or [C, H_src, W_src]
        size: Tuple (H_dst, W_dst) - target size
        align_corners: Boolean - whether to align corners

    Returns:
        Interpolated array with shape [B, C, H_dst, W_dst] or [C, H_dst, W_dst]
    """
    # Handle different input shapes
    input_dim = pos_embed.ndim
    original_shape = pos_embed.shape

    if input_dim == 3:
        # [C, H, W] -> [1, C, H, W]
        pos_embed = pos_embed.reshape(1, *original_shape)

    # Get source dimensions
    h_src, w_src = pos_embed.shape[-2:]
    h_dst, w_dst = size

    # Calculate scale factors
    scale_h = h_dst / h_src
    scale_w = w_dst / w_src

    # Create upsampler
    upsampler = nn.Upsample(
        scale_factor=(scale_h, scale_w), mode=mode, align_corners=align_corners
    )

    # Apply upsampling
    result = upsampler(pos_embed)

    # Return in the original dimension format
    if input_dim == 3:
        return result.reshape(original_shape[0], *size)
    return result
