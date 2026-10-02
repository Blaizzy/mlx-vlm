"""VGGT-Omega: feed-forward camera and depth reconstruction from a sequence
of images (inference only)."""

from typing import Dict

import mlx.core as mx
import mlx.nn as nn

from .aggregator import Aggregator
from .config import ModelConfig
from .heads import CameraHead, DenseHead, TextAlignmentHead

# Heads loaded in ``config.head_dtype``.
_FLOAT_HEADS = ("camera_head.", "dense_head.", "text_alignment_head.")


class Model(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.aggregator = Aggregator(config)
        self.camera_head = CameraHead(config) if config.enable_camera else None
        self.dense_head = DenseHead(config) if config.enable_depth else None
        self.text_alignment_head = (
            TextAlignmentHead(config) if config.enable_alignment else None
        )

    def __call__(self, images: mx.array) -> Dict[str, mx.array]:
        """images: (S, H, W, 3) or (B, S, H, W, 3) RGB in [0, 1]; H and W
        must be multiples of the patch size. The first frame is the
        reference frame of the predicted cameras.

        Returns a dict of lazy float32 arrays (see README): ``pose_enc``
        (B, S, 9), ``depth`` (B, S, H, W, 1), ``depth_conf`` (B, S, H, W),
        ``camera_and_register_tokens`` (B, S, 17, 2D), ``images``, and the
        ``text_alignment_*`` outputs when that head is enabled.
        """
        if images.ndim == 4:
            images = images[None]
        images = images.astype(mx.float32)
        tokens, layers = self.aggregator(images)

        out = {"camera_and_register_tokens": tokens, "images": images}
        if self.camera_head is not None:
            out["pose_enc"] = self.camera_head(tokens)
        if self.dense_head is not None:
            depth, conf = self.dense_head(layers, images.shape[2:4])
            out["depth"], out["depth_conf"] = depth, conf
        if self.text_alignment_head is not None:
            out.update(self.text_alignment_head(tokens))
        return out

    def sanitize(self, weights: Dict[str, mx.array]) -> Dict[str, mx.array]:
        """Cast the head weights to ``config.head_dtype``. Official ``.pt``
        checkpoints go through ``convert.py`` first."""
        if not self.config.head_dtype:
            return weights
        dtype = getattr(mx, self.config.head_dtype)
        return {
            k: v.astype(dtype) if k.startswith(_FLOAT_HEADS) else v
            for k, v in weights.items()
        }
