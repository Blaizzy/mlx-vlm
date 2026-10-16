from typing import Sequence

import mlx.core as mx
import numpy as np

from ..muse_glimmer.vision import PatchEmbedder as MuseGlimmerPatchEmbedder
from ..muse_glimmer.vision import VisionModel as MuseGlimmerVisionModel
from .config import VisionConfig


def _axis_taps(positions: np.ndarray, size: int, side: int):
    """Bilinear taps of a side-long table at a size-long grid
    (align_corners=False, zero padding)."""
    grid = (positions.astype(np.float32) + 0.5) * (side / size) - 0.5
    low = np.floor(grid).astype(np.int32)
    high = low + 1
    frac = grid - low
    low_weight = (1 - frac) * ((low >= 0) & (low < side))
    high_weight = frac * ((high >= 0) & (high < side))
    return (
        np.clip(low, 0, side - 1),
        np.clip(high, 0, side - 1),
        low_weight,
        high_weight,
    )


class PatchEmbedder(MuseGlimmerPatchEmbedder):
    def _position_embeddings(self, grid: Sequence[Sequence[int]]) -> mx.array:
        side = self.num_grid_per_side
        all_indices, all_weights = [], []
        for temporal, height, width in grid:
            temporal, height, width = int(temporal), int(height), int(width)
            # The reference reads the table with the grid as (t, w, h): the
            # raster index of a patch is decoded with height and width swapped.
            # For a square grid this is the same as Muse Glimmer.
            index = np.arange(height * width)
            rows, cols = index // height, index % height
            r0, r1, rw0, rw1 = _axis_taps(rows, width, side)
            c0, c1, cw0, cw1 = _axis_taps(cols, height, side)
            indices = np.stack(
                [r0 * side + c0, r0 * side + c1, r1 * side + c0, r1 * side + c1]
            )
            weights = np.stack([rw0 * cw0, rw0 * cw1, rw1 * cw0, rw1 * cw1])
            all_indices.append(np.tile(indices, (1, temporal)))
            all_weights.append(np.tile(weights, (1, temporal)))

        indices = mx.array(np.concatenate(all_indices, axis=1), dtype=mx.int32)
        weights = mx.array(np.concatenate(all_weights, axis=1), dtype=mx.float32)
        table = self.position_embedding_table(indices)
        return mx.sum(table * weights[..., None], axis=0)


class VisionModel(MuseGlimmerVisionModel):
    def __init__(self, config: VisionConfig):
        super().__init__(config)
        self.patch_embedder = PatchEmbedder(config)
