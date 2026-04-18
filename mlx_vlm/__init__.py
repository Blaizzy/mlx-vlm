import os

os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "1"
os.environ["TRANSFORMERS_VERBOSITY"] = "error"


def _install_transformers_compat_shims():
    """Back-fill transformers helpers that remote-code modeling files expect.

    Microsoft's Phi-4-reasoning-vision-15B (and siblings) ship custom
    `modeling_phi4_visionr.py` that decorates methods with
    `transformers.models.siglip2.image_processing_siglip2.filter_out_non_signature_kwargs()`.
    Recent transformers releases moved/removed that helper, which raises
    AttributeError at class-definition time before any of our code runs.
    Add a no-op decorator so the class body loads and the model can be
    served through our own phi4_siglip implementation.
    """
    try:
        from transformers.models.siglip2 import image_processing_siglip2

        if not hasattr(image_processing_siglip2, "filter_out_non_signature_kwargs"):

            def _filter_out_non_signature_kwargs(*_args, **_kwargs):
                def decorator(fn):
                    return fn

                return decorator

            image_processing_siglip2.filter_out_non_signature_kwargs = (
                _filter_out_non_signature_kwargs
            )
    except Exception:
        pass


_install_transformers_compat_shims()

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
