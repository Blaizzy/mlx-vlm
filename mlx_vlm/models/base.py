from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import mlx.core as mx
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


def create_additive_causal_mask(N: int, offset: int = 0):
    rinds = mx.arange(offset + N)
    linds = mx.arange(offset, offset + N) if offset else rinds
    mask = linds[:, None] < rinds[None]
    return mask * -1e9


def create_attention_mask(h: mx.array, cache: Optional[Any] = None):
    T = h.shape[1]
    if T > 1:
        if cache is not None and cache[0] is not None:
            c = cache[0]
            if isinstance(c, RotatingKVCache):
                offset = min(c.max_size - 1, c.offset)
            else:
                offset = c.offset
        else:
            offset = 0
        mask = create_additive_causal_mask(T, offset)
        mask = mask.astype(h.dtype)
    else:
        mask = None
    return mask


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
