from __future__ import annotations

import importlib.util
import pathlib
import sys
from typing import Any

from .convert_dots_ocr import cli_scan, convert_dir_or_file_to_npz, iter_safetensors, list_vision_keys

__all__ = [
    "convert",
    "cli_scan",
    "convert_dir_or_file_to_npz",
    "iter_safetensors",
    "list_vision_keys",
]


_HERE = pathlib.Path(__file__).resolve().parent
_CONVERT_PATH = _HERE.parent / "convert.py"
_CONVERT_SPEC = importlib.util.spec_from_file_location(
    "mlx_vlm._base_convert", _CONVERT_PATH
)
__BASE_MODULE = None


def _load_base_convert_module():
    global __BASE_MODULE
    if __BASE_MODULE is None:
        if _CONVERT_SPEC is None or _CONVERT_SPEC.loader is None:
            raise ImportError("Unable to load mlx_vlm.convert base module")
        module = importlib.util.module_from_spec(_CONVERT_SPEC)
        module.__package__ = "mlx_vlm"
        sys.modules[_CONVERT_SPEC.name] = module
        _CONVERT_SPEC.loader.exec_module(module)
        __BASE_MODULE = module
    return __BASE_MODULE


def convert(*args: Any, **kwargs: Any):
    module = _load_base_convert_module()
    return module.convert(*args, **kwargs)  # type: ignore[attr-defined]
