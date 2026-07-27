"""Regression test for Mage-VL's 2x2 block-order patch positions.

`_positions_from_grid` is the fallback used when a caller supplies `image_grid_thw` but not
`patch_positions`. The reference Qwen2VL-style processor DOES supply positions, so this path is
never exercised end-to-end — which is exactly why it needs a test. It is also the path a Swift
port must reimplement, since there is no Python processor there.

The expectations below were validated against the real processor's output on
`microsoft/Mage-VL`'s own `examples/dog.jpg`: a (1, 128, 64) grid produced 8192 positions that
matched this function row-for-row, exactly.
"""

import numpy as np
import pytest

from mlx_vlm.models.mage_vl.config import VisionConfig
from mlx_vlm.models.mage_vl.mage_vl import _as_grid_list, _positions_from_grid


def _positions(grid, merge=2):
    cfg = VisionConfig(spatial_merge_size=merge)
    return np.array(_positions_from_grid(grid, cfg))


def test_first_block_is_2x2_row_major():
    """Patches arrive grouped so each consecutive run of merge^2 is one merge block."""
    pos = _positions([(1, 4, 4)])
    expected_head = np.array(
        [
            [0, 0, 0], [0, 0, 1], [0, 1, 0], [0, 1, 1],  # block (0,0)
            [0, 0, 2], [0, 0, 3], [0, 1, 2], [0, 1, 3],  # block (0,1)
        ]
    )
    assert np.array_equal(pos[:8], expected_head)


def test_every_group_of_four_shares_one_merge_block():
    """The merger folds each run of 4 into one token; they must come from the same 2x2 cell."""
    pos = _positions([(1, 8, 6)])
    blocks = pos.reshape(-1, 4, 3)
    cells = np.stack([blocks[:, :, 0], blocks[:, :, 1] // 2, blocks[:, :, 2] // 2], axis=-1)
    assert (cells == cells[:, :1, :]).all(), "a merge group spans more than one 2x2 cell"


def test_covers_the_grid_exactly_once():
    pos = _positions([(2, 8, 6)])
    assert pos.shape == (2 * 8 * 6, 3)
    assert len({tuple(p) for p in pos}) == len(pos), "duplicate positions"
    assert pos[:, 0].max() == 1 and pos[:, 1].max() == 7 and pos[:, 2].max() == 5


def test_temporal_axis_is_outermost():
    """All of frame 0 precedes frame 1 — the rope's t lane depends on this ordering."""
    pos = _positions([(3, 4, 4)])
    per_frame = 4 * 4
    for f in range(3):
        assert (pos[f * per_frame : (f + 1) * per_frame, 0] == f).all()


def test_matches_real_processor_shape_for_dog_example():
    """The dog.jpg case that was verified row-for-row against the reference processor."""
    pos = _positions([(1, 128, 64)])
    assert pos.shape == (8192, 3)
    assert np.array_equal(pos[:4], np.array([[0, 0, 0], [0, 0, 1], [0, 1, 0], [0, 1, 1]]))


@pytest.mark.parametrize(
    "raw", [np.array([[1, 4, 4]]), [[1, 4, 4]], np.array([1, 4, 4])]
)
def test_grid_coercion_accepts_processor_shapes(raw):
    assert _as_grid_list(raw) == [(1, 4, 4)]
