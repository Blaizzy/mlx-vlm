# Fix pre-release transformers wheel issues for Gemma4:
# 1. Missing Gemma4ImageProcessorKwargs export (needed by fast processor import)
# 2. Fast image processor has bugs - force slow processor via monkey-patch
try:
    import transformers.models.gemma4.image_processing_gemma4 as _ip

    if not hasattr(_ip, "Gemma4ImageProcessorKwargs"):
        from transformers.image_processing_utils_fast import ImagesKwargs

        class Gemma4ImageProcessorKwargs(ImagesKwargs, total=False):
            max_soft_tokens: int | None
            patch_size: int | None
            pooling_kernel_size: int | None

        _ip.Gemma4ImageProcessorKwargs = Gemma4ImageProcessorKwargs

    # Force slow image processor by removing the fast one from auto-mapping
    try:
        from transformers.models.gemma4 import image_processing_gemma4_fast as _ipf

        if hasattr(_ipf, "Gemma4ImageProcessorFast"):
            _ipf.Gemma4ImageProcessorFast = _ip.Gemma4ImageProcessor
    except (ImportError, AttributeError):
        pass
except (ImportError, AttributeError):
    pass

from .config import AudioConfig, ModelConfig, TextConfig, VisionConfig
from .gemma4 import LanguageModel, Model, VisionModel
