import atexit
import os

os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "1"
os.environ["TRANSFORMERS_VERBOSITY"] = "error"

from ._stream_cleanup import clear_mlx_streams
from .convert import convert
from .generate import (
    AudioGenerationResult,
    BatchResponse,
    BatchStats,
    GenerationResult,
    PromptCacheState,
    batch_generate,
    generate,
    generate_audio,
    save_audio,
    stream_generate,
)
from .prompt_utils import apply_chat_template, get_message_json
from .utils import load, prepare_inputs, process_image
from .version import __version__
from .vision_cache import VisionFeatureCache

# MLX streams are thread-local and must be released before interpreter teardown.
atexit.register(clear_mlx_streams)


def predict(model, processor, input, questions, **kwargs):
    """Predict typed decisions through the shared Python and CLI interface."""
    # Keep the CLI module unloaded so `python -m mlx_vlm.decide` runs normally.
    from .decide import predict as predict_decisions

    return predict_decisions(model, processor, input, questions, **kwargs)
