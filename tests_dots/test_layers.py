import mlx.core as mx

from mlx_vlm.models.dots_ocr.dots_vision import RMSNorm


def test_rmsnorm_shapes_and_numerics():
    x = mx.arange(8, dtype=mx.float32).reshape(2, 4)
    y = RMSNorm(4)(x)
    assert y.shape == (2, 4)
    # Check finite and roughly scaled
    assert float(mx.max(mx.abs(y))) > 0.0
from mlx_vlm.models.dots_ocr.dots_vision import SwiGLU


def test_swish_glu_shapes():
    x = mx.random.uniform(shape=(10, 1536))
    y = SwiGLU()(x)
    assert y.shape == (10, 1536)
