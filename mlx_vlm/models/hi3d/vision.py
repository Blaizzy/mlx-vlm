"""Image conditioning for Hi3D: DINOv2-L tokens joined with the CLIP-L embedding."""

import mlx.core as mx
import mlx.nn as nn
import numpy as np
from PIL import Image, ImageOps

from ..dinov2.config import ModelConfig as DinoConfig
from ..dinov2.dinov2 import Model as Dino
from ..interpolate import resize_bilinear_nhwc
from ..llava.config import VisionConfig as ClipConfig
from ..llava.vision import VisionModel as Clip

DINO_MEAN = (0.485, 0.456, 0.406)
DINO_STD = (0.229, 0.224, 0.225)
CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD = (0.26862954, 0.26130258, 0.27577711)

# The subject fills ``1 / 1.2`` of the square the conditioner was trained on.
SUBJECT_PADDING = 1.2


def prepare_image(image):
    """Frame an image the way the released conditioner expects.

    Crops to the smallest square around the opaque subject, composites it
    over white and pads the square by :data:`SUBJECT_PADDING`. Opaque images
    keep their whole extent.
    """
    if not isinstance(image, Image.Image):
        image = Image.fromarray(np.asarray(image))
    rgba = ImageOps.exif_transpose(image).convert("RGBA")
    box = rgba.getchannel("A").getbbox()
    if box is not None:
        left, upper, right, lower = box
        side = max(right - left, lower - upper)
        left -= (side - (right - left)) // 2
        upper -= (side - (lower - upper)) // 2
        rgba = rgba.crop((left, upper, left + side, upper + side))
    white = Image.new("RGB", rgba.size, (255, 255, 255))
    white.paste(rgba, (0, 0), rgba)
    side = int(max(white.size) * SUBJECT_PADDING)
    padded = Image.new("RGB", (side, side), (255, 255, 255))
    padded.paste(white, ((side - white.width) // 2, (side - white.height) // 2))
    return padded


def _normalize(pixels, mean, std):
    return (pixels - mx.array(mean)) / mx.array(std)


def _l2_normalize(x):
    norm = mx.linalg.norm(x, axis=-1, keepdims=True)
    return x / mx.maximum(norm, 1e-12)


class Conditioner(nn.Module):
    """Frozen float32 DINOv2 and CLIP encoders."""

    def __init__(self, config):
        super().__init__()
        self.dino_size = config.dino_image_size
        self.clip_size = config.clip_image_size
        self.dino = Dino(DinoConfig.from_dict(config.dino_config))
        self.clip = Clip(ClipConfig.from_dict(config.clip_config))

    def __call__(self, image):
        """Condition tokens ``(1, T, D_dino + D_clip)`` for a prepared image."""
        rgb = image.convert("RGB")
        base = mx.array(np.asarray(rgb, dtype=np.float32) / 255.0)[None]
        dino_in = resize_bilinear_nhwc(base, (self.dino_size, self.dino_size))
        clip_rgb = rgb.resize(
            (self.clip_size, self.clip_size), Image.Resampling.LANCZOS
        )
        clip_in = mx.array(np.asarray(clip_rgb, dtype=np.float32) / 255.0)[None]
        tokens = self.dino(_normalize(dino_in, DINO_MEAN, DINO_STD))
        tokens = _l2_normalize(tokens["last_hidden_state"].astype(mx.float32))
        pooled = self.clip(_normalize(clip_in, CLIP_MEAN, CLIP_STD))[0]
        pooled = _l2_normalize(pooled.astype(mx.float32))
        pooled = mx.broadcast_to(
            pooled[:, None], (1, tokens.shape[1], pooled.shape[-1])
        )
        return mx.concatenate([tokens, pooled], axis=-1)
