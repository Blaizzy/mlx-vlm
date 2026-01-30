import inspect
import math
from abc import abstractmethod
from dataclasses import dataclass
from typing import Dict, List, Optional

import mlx.core as mx
import mlx.nn as nn
from mlx_lm.models.base import create_attention_mask, scaled_dot_product_attention
from PIL import Image


@dataclass
class LanguageModelOutput:
    logits: mx.array
    hidden_states: Optional[List[mx.array]] = None
    cross_attention_states: Optional[List[mx.array]] = None
    encoder_outputs: Optional[List[mx.array]] = None


@dataclass
class InputEmbeddingsFeatures:
    inputs_embeds: mx.array
    attention_mask_4d: Optional[mx.array] = None
    visual_pos_masks: Optional[mx.array] = None
    deepstack_visual_embeds: Optional[mx.array] = None
    per_layer_inputs: Optional[mx.array] = None
    cross_attention_states: Optional[mx.array] = None
    cross_attention_mask: Optional[mx.array] = None
    full_text_row_masked_out_mask: Optional[mx.array] = None
    decoder_inputs_embeds: Optional[mx.array] = None
    attention_mask: Optional[mx.array] = None  # For encoder-decoder models

    def to_dict(self):
        return {
            "inputs_embeds": self.inputs_embeds,
            "attention_mask_4d": self.attention_mask_4d,
            "visual_pos_masks": self.visual_pos_masks,
            "deepstack_visual_embeds": self.deepstack_visual_embeds,
            "per_layer_inputs": self.per_layer_inputs,
            "cross_attention_states": self.cross_attention_states,
            "cross_attention_mask": self.cross_attention_mask,
            "full_text_row_masked_out_mask": self.full_text_row_masked_out_mask,
            "decoder_inputs_embeds": self.decoder_inputs_embeds,
            "attention_mask": self.attention_mask,
        }


@dataclass
class BaseModelConfig:
    @classmethod
    def from_dict(cls, params):
        return cls(
            **{
                k: v
                for k, v in params.items()
                if k in inspect.signature(cls).parameters
            }
        )

    def to_dict(self):
        return {k: v for k, v in self.__dict__.items() if v is not None}


class BaseImageProcessor:
    """
    Base image processor class. Subclasses should implement preprocess().
    Transformers imports are deferred to __init__ for faster module loading.
    """

    def __init__(
        self,
        image_mean=(0.5, 0.5, 0.5),
        image_std=(0.5, 0.5, 0.5),
        size=(384, 384),
        crop_size: Dict[str, int] = None,
        resample=None,
        rescale_factor=1 / 255,
        data_format=None,
    ):
        from transformers.image_processing_utils import get_size_dict
        from transformers.image_utils import ChannelDimension, PILImageResampling

        if resample is None:
            resample = PILImageResampling.BICUBIC
        if data_format is None:
            data_format = ChannelDimension.FIRST

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

    def rescale(
        self,
        image,
        scale: float,
        input_data_format: str = "channels_first",
    ):
        """Rescale an image by a scale factor."""
        return image * scale

    def normalize(
        self,
        image,
        mean,
        std,
        input_data_format: str = "channels_first",
    ):
        """Normalize an image with mean and std."""
        import numpy as np

        mean = np.array(mean, dtype=image.dtype)
        std = np.array(std, dtype=image.dtype)

        if input_data_format == "channels_first":
            # Image shape: [C, H, W]
            mean = mean[:, None, None]
            std = std[:, None, None]
        else:
            # Image shape: [H, W, C]
            pass  # mean and std are already in correct shape

        return (image - mean) / std

    @abstractmethod
    def preprocess(self, images):
        pass


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


@mx.compile
def chunked_attention(
    queries: mx.array,
    keys: mx.array,
    values: mx.array,
    scale: float,
    chunk_size: int,
) -> mx.array:

    L = queries.shape[2]

    outputs = []
    for i in range(0, L, chunk_size):
        end_idx = min(i + chunk_size, L)
        q_chunk = queries[:, :, i:end_idx, :]  # (B, n_heads, chunk, head_dim)

        chunk_output = mx.fast.scaled_dot_product_attention(
            q_chunk, keys, values, scale=scale
        )

        outputs.append(chunk_output)

    return mx.concatenate(outputs, axis=2)  # (B, n_heads, L, head_dim)


def install_auto_processor_patch(target_model_types, processor_cls):
    """
    Install a composable patch on transformers.AutoProcessor.from_pretrained

    Args:
        target_model_types (Union[str, List[str]]): Model types to intercept.
        processor_cls (type): Processor class exposing `from_pretrained`.

    Returns:
        The previous `AutoProcessor.from_pretrained` for reference.
    """
    from transformers import AutoProcessor as _HF_AutoProcessor

    if isinstance(target_model_types, str):
        target_model_types = [target_model_types]
    target_model_types = {t.lower() for t in target_model_types}

    previous_from_pretrained = _HF_AutoProcessor.from_pretrained

    @classmethod
    def _patched_auto_processor_from_pretrained(
        cls, pretrained_model_name_or_path, **kwargs
    ):
        import json as _json
        from pathlib import Path

        try:
            model_path = Path(pretrained_model_name_or_path)
            is_local = model_path.exists() and model_path.is_dir()

            cfg = {}
            if is_local:
                config_path = model_path / "config.json"
                if config_path.exists():
                    with open(config_path, "r", encoding="utf-8") as f:
                        cfg = _json.load(f)
            else:
                try:
                    from huggingface_hub import hf_hub_download

                    cfg_path = hf_hub_download(
                        pretrained_model_name_or_path, "config.json"
                    )
                    with open(cfg_path, "r", encoding="utf-8") as f:
                        cfg = _json.load(f)
                except Exception:
                    cfg = {}

            model_type = str(cfg.get("model_type", "")).lower()
            if model_type in target_model_types:
                return processor_cls.from_pretrained(
                    pretrained_model_name_or_path, **kwargs
                )
        except Exception:
            # On any failure, fall back to previous behavior
            pass

        # Chain to the prior from_pretrained (which may already be patched)
        return previous_from_pretrained.__func__(
            cls, pretrained_model_name_or_path, **kwargs
        )

    _HF_AutoProcessor.from_pretrained = _patched_auto_processor_from_pretrained
    return previous_from_pretrained
