import mlx.core as mx

from mlx_vlm.models.dots_ocr.dots_ocr import DotsOCRConfig
from mlx_vlm.models.dots_ocr.dots_vision import DotsVisionTransformer_MLX


def test_compile_toggle_does_not_break_forward(monkeypatch):
    monkeypatch.setenv("MLX_COMPILE", "1")
    cfg = DotsOCRConfig({"vision_config": {"num_layers": 1}})
    model = DotsVisionTransformer_MLX(cfg)
    x = mx.random.uniform(shape=(1, 3, 224, 224))
    y = model(x, [[1, 16, 16]])
    assert y.shape == (64, 1536)


def test_dtype_toggle_fp16_forward(monkeypatch):
    monkeypatch.setenv("MLX_VISION_DTYPE", "fp16")
    cfg = DotsOCRConfig({"vision_config": {"num_layers": 1}})
    model = DotsVisionTransformer_MLX(cfg)
    x = mx.random.uniform(shape=(1, 3, 224, 224))
    y = model(x, [[1, 16, 16]])
    assert y.shape == (64, 1536)
