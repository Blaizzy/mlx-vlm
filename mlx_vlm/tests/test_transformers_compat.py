import importlib.util
from pathlib import Path

from transformers import AutoProcessor, AutoTokenizer
from transformers.models.auto.processing_auto import processor_class_from_name
from transformers.models.auto.tokenization_auto import REGISTERED_TOKENIZER_CLASSES


def _load_compat_module():
    module_path = Path(__file__).resolve().parents[1] / "_transformers_compat.py"
    spec = importlib.util.spec_from_file_location("mlx_vlm_transformers_compat", module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_legacy_auto_tokenizer_register_accepts_string_key():
    compat = _load_compat_module()
    compat.install_transformers_legacy_registration_shims()

    class LegacyTokenizer:
        pass

    AutoTokenizer.register("LegacyTokenizer", fast_tokenizer_class=LegacyTokenizer)

    assert REGISTERED_TOKENIZER_CLASSES["LegacyTokenizer"] is LegacyTokenizer


def test_legacy_auto_processor_register_accepts_string_key():
    compat = _load_compat_module()
    compat.install_transformers_legacy_registration_shims()

    class LegacyProcessor:
        pass

    AutoProcessor.register("legacy-processor", LegacyProcessor)

    assert processor_class_from_name("LegacyProcessor") is LegacyProcessor
