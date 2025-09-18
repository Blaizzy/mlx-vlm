import importlib.util
from pathlib import Path

MODULE_PATH = Path(__file__).resolve().parents[1] / "mlx_vlm" / "models" / "dots_ocr" / "dots_ocr.py"

spec = importlib.util.spec_from_file_location("mlx_vlm.models.dots_ocr.dots_ocr", MODULE_PATH)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
DotsOCRConfig = module.DotsOCRConfig


def test_config_default_ok():
    c = DotsOCRConfig({})
    assert c.vision.embed_dim == 1536
    assert c.vision.patch_size == 14
    assert c.vision.merge_size == 2


def test_config_reject_bad_heads():
    try:
        DotsOCRConfig({"vision_config": {"embed_dim": 1536, "num_heads": 7}})
    except AssertionError as e:
        assert "divide" in str(e)
    else:
        assert False, "expected assertion"
