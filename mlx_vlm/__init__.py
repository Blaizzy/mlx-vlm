import os

os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "1"
os.environ["TRANSFORMERS_VERBOSITY"] = "error"

from .version import __version__

_LAZY_ATTRS = {
    "convert": ("mlx_vlm.convert", "convert"),
    "BatchResponse": ("mlx_vlm.generate", "BatchResponse"),
    "BatchStats": ("mlx_vlm.generate", "BatchStats"),
    "GenerationResult": ("mlx_vlm.generate", "GenerationResult"),
    "PromptCacheState": ("mlx_vlm.generate", "PromptCacheState"),
    "batch_generate": ("mlx_vlm.generate", "batch_generate"),
    "generate": ("mlx_vlm.generate", "generate"),
    "stream_generate": ("mlx_vlm.generate", "stream_generate"),
    "apply_chat_template": ("mlx_vlm.prompt_utils", "apply_chat_template"),
    "get_message_json": ("mlx_vlm.prompt_utils", "get_message_json"),
    "load": ("mlx_vlm.utils", "load"),
    "prepare_inputs": ("mlx_vlm.utils", "prepare_inputs"),
    "process_image": ("mlx_vlm.utils", "process_image"),
    "VisionFeatureCache": ("mlx_vlm.vision_cache", "VisionFeatureCache"),
}

__all__ = ["__version__", *_LAZY_ATTRS]


def __getattr__(name):
    if name not in _LAZY_ATTRS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    import importlib

    module_name, attr_name = _LAZY_ATTRS[name]
    value = getattr(importlib.import_module(module_name), attr_name)
    globals()[name] = value
    return value
