import os

os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "1"
os.environ["TRANSFORMERS_VERBOSITY"] = "error"

# transformers >= 5.13 tightened AutoTokenizer.register to require a config
# *class* (it reads ``key.__module__``); mlx_lm still registers a custom
# tokenizer by string model_type, which raises at import time. Tolerate string
# keys here — before mlx_lm is imported below — so mlx-vlm works across the
# transformers 5.5–5.13 range without pinning the dependency down.
try:
    from transformers.models.auto import auto_factory as _hf_auto_factory

    _hf_orig_register = _hf_auto_factory._LazyAutoMapping.register

    def _hf_safe_register(self, key, value, exist_ok=False):
        if isinstance(key, str):
            self._extra_content[key] = value
            return
        return _hf_orig_register(self, key, value, exist_ok=exist_ok)

    _hf_auto_factory._LazyAutoMapping.register = _hf_safe_register
except Exception:  # pragma: no cover - never block import on a shim failure
    pass

from .convert import convert
from .generate import (
    BatchResponse,
    BatchStats,
    GenerationResult,
    PromptCacheState,
    batch_generate,
    generate,
    stream_generate,
)
from .prompt_utils import apply_chat_template, get_message_json
from .utils import load, prepare_inputs, process_image
from .version import __version__
from .vision_cache import VisionFeatureCache
