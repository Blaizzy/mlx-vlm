"""Downsampling modules for granite4_vision projector."""

import math
from fractions import Fraction

import mlx.core as mx
import mlx.nn as nn

from .qformer import Blip2QFormerModel, QFormerConfig


class InterpolateDownsampler:
    """Area interpolation downsampler.

    Given image features of shape (B, orig_side², C), downsamples to
    (B, new_side², C) by averaging over non-overlapping spatial blocks.
    """

    def __init__(self, config):
        self.orig_image_side = (
            config.vision_config.image_size // config.vision_config.patch_size
        )
        self.new_image_side = int(
            self.orig_image_side * Fraction(config.downsample_rate)
        )

    def __call__(self, image_features: mx.array) -> mx.array:
        B, _, C = image_features.shape
        s = self.orig_image_side
        ns = self.new_image_side
        ratio = s // ns

        # Reshape to spatial grid: (B, s, s, C)
        x = image_features.reshape(B, s, s, C)
        # Partition into non-overlapping blocks: (B, ns, ratio, ns, ratio, C)
        x = x.reshape(B, ns, ratio, ns, ratio, C)
        # Average pool over the block dimensions
        x = mx.mean(x, axis=(2, 4))  # (B, ns, ns, C)
        # Flatten back to sequence: (B, ns², C)
        x = x.reshape(B, ns * ns, C)
        return x


class SpatialOffsetDownsampler:
    """Spatial offset downsampler with stride 2.

    Samples one position from each 2x2 block of the spatial feature grid.
    The offset selects which corner: 0=top-left, 1=top-right, 2=bottom-left,
    3=bottom-right.
    """

    def __init__(self, config, offset: int = 0):
        self.orig_image_side = (
            config.vision_config.image_size // config.vision_config.patch_size
        )
        self.new_image_side = self.orig_image_side // 2
        self.offset = offset
        self.offsets = [(0, 0), (0, 1), (1, 0), (1, 1)]
        self.offset_h, self.offset_w = self.offsets[offset]

    def __call__(self, image_features: mx.array) -> mx.array:
        B, _, C = image_features.shape
        s = self.orig_image_side
        ns = self.new_image_side

        # Reshape to spatial grid: (B, s, s, C)
        x = image_features.reshape(B, s, s, C)
        # Partition into 2x2 blocks: (B, ns, 2, ns, 2, C)
        x = x.reshape(B, ns, 2, ns, 2, C)
        # Select the offset position from each block
        x = x[:, :, self.offset_h, :, self.offset_w, :]  # (B, ns, ns, C)
        # Flatten back to sequence: (B, ns², C)
        x = x.reshape(B, ns * ns, C)
        return x


class WindowQFormerDownsampler(nn.Module):
    """Window-based QFormer downsampler.

    This is the main projector module for granite4_vision. It applies
    windowed cross-attention via a QFormer to downsample vision features
    and project them to the LLM hidden dimension.

    The downsample_rate config string "q/w" encodes the query side (q) and
    window side (w). Image features are split into windows of size w×w, and
    each window is attended to by q×q learned queries plus downsampled
    feature embeddings.
    """

    def __init__(self, config, spatial_offset=None):
        super().__init__()
        llm_hidden_size = config.text_config.hidden_size
        vision_hidden_size = config.vision_config.hidden_size

        # Choose downsampler type
        if spatial_offset is not None:
            self.downsampler = SpatialOffsetDownsampler(config, offset=spatial_offset)
        else:
            self.downsampler = InterpolateDownsampler(config)

        # QFormer config
        qformer_config = QFormerConfig(
            hidden_size=vision_hidden_size,
            num_attention_heads=vision_hidden_size // 64,
            intermediate_size=3072,
            encoder_hidden_size=vision_hidden_size,
        )
        self.qformer = Blip2QFormerModel(qformer_config)

        # Window / query dimensions from downsample_rate "q/w"
        self.image_side = (
            config.vision_config.image_size // config.vision_config.patch_size
        )
        q, w = config.downsample_rate.split("/")
        self.query_side, self.window_side = int(q), int(w)
        self.query_length = self.query_side**2

        # Learnable parameters
        embed_std = 1 / math.sqrt(vision_hidden_size)
        self.norm = nn.LayerNorm(vision_hidden_size, eps=1e-6)
        self.query = (
            mx.random.normal((1, self.query_length, vision_hidden_size)) * embed_std
        )
        self.image_positions = (
            mx.random.normal((1, self.window_side**2, vision_hidden_size)) * embed_std
        )

        # Output projection to LLM hidden size
        self.out_linear = nn.Linear(vision_hidden_size, llm_hidden_size, bias=True)

    def _win(self, x: mx.array, side: int, win: int) -> mx.array:
        """Partition a sequence into spatial windows.

        (B, side*side, C) -> (B*n*n, win*win, C) where n = side // win.
        """
        B, _, C = x.shape
        n = side // win
        # (B, side, side, C)
        x = x.reshape(B, side, side, C)
        # (B, n, win, n, win, C)
        x = x.reshape(B, n, win, n, win, C)
        # Transpose to (B, n, n, win, win, C)
        x = mx.transpose(x, axes=(0, 1, 3, 2, 4, 5))
        # Flatten to (B*n*n, win*win, C)
        x = x.reshape(B * n * n, win * win, C)
        return x

    def _unwin(self, xw: mx.array, n: int, win: int) -> mx.array:
        """Reverse windowing back to a flat sequence.

        (B*n*n, win*win, C) -> (B, (n*win)², C).
        """
        Bnn, _, C = xw.shape
        B = Bnn // (n * n)
        side = n * win
        # (B, n, n, win, win, C)
        xw = xw.reshape(B, n, n, win, win, C)
        # Transpose to (B, n, win, n, win, C)
        xw = mx.transpose(xw, axes=(0, 1, 3, 2, 4, 5))
        # Flatten to (B, side*side, C)
        xw = xw.reshape(B, side * side, C)
        return xw

    def __call__(self, image_features: mx.array) -> mx.array:
        B, HW, C = image_features.shape
        n = self.image_side // self.window_side

        # LayerNorm
        image_features = self.norm(image_features)

        # Window the image features
        enc = self._win(image_features, self.image_side, self.window_side)

        # Downsample
        downsampled = self.downsampler(image_features)

        # Window the downsampled features
        new_side = n * self.query_side
        downsampled_w = self._win(downsampled, new_side, self.query_side)

        # Add learnable embeddings
        query_embeds = self.query + downsampled_w
        encoder_embeds = enc + self.image_positions

        # QFormer cross-attention
        out_w = self.qformer(
            query_embeds=query_embeds, encoder_hidden_states=encoder_embeds
        )

        # Un-window
        out = self._unwin(out_w, n=n, win=self.query_side)

        # Project to LLM hidden size
        return self.out_linear(out)
