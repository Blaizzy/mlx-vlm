import importlib.util
import pathlib

HERE = pathlib.Path(__file__).resolve().parent
_convert_path = HERE.parent / "convert.py"
_spec = importlib.util.spec_from_file_location(
    "mlx_vlm._convert_module", _convert_path
)
if _spec is None or _spec.loader is None:
    raise ImportError("Unable to load base convert module")
_module = importlib.util.module_from_spec(_spec)
_module.__package__ = "mlx_vlm"
_spec.loader.exec_module(_module)
convert = _module.convert  # type: ignore[attr-defined]

from .convert_dots_ocr import cli_scan, iter_safetensors, list_vision_keys

__all__ = ["convert", "cli_scan", "iter_safetensors", "list_vision_keys"]
