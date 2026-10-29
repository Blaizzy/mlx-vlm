# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

import mlx.core as mx
import numpy as np

from ..nemotron_parse.vision import VisionModel as RadioVisionModel


def _resize_weights(source: int, target: int, area: bool = False):
    """PyTorch align_corners=False bilinear antialias (or adaptive average)."""
    if area:
        weights = np.zeros((target, source), dtype=np.float32)
        for i in range(target):
            start = i * source // target
            end = ((i + 1) * source + target - 1) // target
            weights[i, start:end] = 1 / (end - start)
    else:
        scale = source / target
        support = max(scale, 1.0)
        centers = (np.arange(target, dtype=np.float32) + 0.5) * scale - 0.5
        weights = np.maximum(
            0,
            1
            - np.abs(centers[:, None] - np.arange(source, dtype=np.float32)) / support,
        )
        weights /= weights.sum(axis=1, keepdims=True)
    return mx.array(weights)


class VisionModel(RadioVisionModel):
    def _positions(self, rows, cols, content_height, content_width, dtype):
        content_rows = (content_height + 15) // 16
        content_cols = (content_width + 15) // 16
        area = min(content_rows, content_cols) == 1
        grid = self.pos_embed.reshape(128, 128, self.config.hidden_size).astype(
            mx.float32
        )
        grid = mx.einsum(
            "rh,hwc,sw->rsc",
            _resize_weights(128, content_rows, area),
            grid,
            _resize_weights(128, content_cols, area),
        )

        # grid_sample(..., padding_mode='border', align_corners=False).
        def coordinates(count, content, target):
            starts = mx.arange(count, dtype=mx.float32) * 16
            extent = mx.clip(content - starts, 0, 16)
            return mx.clip(
                (mx.minimum(starts + 0.5 * extent, content) / content) * target - 0.5,
                0,
                target - 1,
            )

        y = coordinates(rows, content_height, content_rows)
        x = coordinates(cols, content_width, content_cols)
        y0, x0 = mx.floor(y).astype(mx.int32), mx.floor(x).astype(mx.int32)
        y1, x1 = (
            mx.minimum(y0 + 1, content_rows - 1),
            mx.minimum(x0 + 1, content_cols - 1),
        )
        wy, wx = (y - y0)[:, None, None], (x - x0)[None, :, None]
        top = grid[y0][:, x0] * (1 - wx) + grid[y0][:, x1] * wx
        bottom = grid[y1][:, x0] * (1 - wx) + grid[y1][:, x1] * wx
        return (top * (1 - wy) + bottom * wy).reshape(1, rows * cols, -1).astype(dtype)

    def __call__(self, pixel_values, image_sizes, content_sizes):
        if image_sizes is None or content_sizes is None:
            raise ValueError("Packed inference requires image_sizes and content_sizes")
        sizes, contents = image_sizes.tolist(), content_sizes.tolist()
        if len(sizes) != 1 or len(contents) != 1:
            raise ValueError("Nemotron 3.5 MLX inference supports one page at a time")
        height, width = sizes[0]
        ch, cw = contents[0]
        if height % 16 or width % 64 or not (0 < ch <= height and 0 < cw <= width):
            raise ValueError("Invalid packed page geometry")
        if pixel_values.shape != ((height // 16) * (width // 16), 768):
            raise ValueError("Packed patches do not match image_sizes")
        x = self.patch_embed(pixel_values.astype(self.patch_embed.weight.dtype))[None]
        x = x + self._positions(height // 16, width // 16, ch, cw, x.dtype)
        x = mx.concatenate([self.cls_token[None], x], axis=1)
        for block in self.blocks:
            x = block(x)
        summary = x[:, self.config.summary_idxs].reshape(1, -1)
        return self.neck(x[:, 10:], summary, (height, width))
