import pytest

from mlx_vlm.models.qmv_compat import mlx_widens_qmv_sum, qmv_sum_define


@pytest.mark.parametrize(
    "version, widens",
    [
        ("0.32.3", False),
        ("0.31.9", False),
        ("0.32.4", True),
        ("0.32.4.dev20261015+abc123", True),
        ("0.33.0", True),
        ("1.0.0", True),
        ("unknown", True),
    ],
)
def test_mlx_widens_qmv_sum(version, widens):
    assert mlx_widens_qmv_sum(version) is widens


def test_qmv_sum_define_matches_version():
    assert "static_cast<float>(v)" in qmv_sum_define("0.32.4")
    assert "static_cast" not in qmv_sum_define("0.32.3")
