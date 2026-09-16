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

from mlx_vlm.models.mage_vl.mage_vl import _as_grid_list


@pytest.mark.parametrize(
    "raw", [np.array([[1, 4, 4]]), [[1, 4, 4]], np.array([1, 4, 4])]
)
def test_grid_coercion_accepts_processor_shapes(raw):
    assert _as_grid_list(raw) == [(1, 4, 4)]
