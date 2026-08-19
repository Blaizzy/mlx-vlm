import atexit
import os

os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "1"
os.environ["TRANSFORMERS_VERBOSITY"] = "error"

from ._stream_cleanup import clear_mlx_streams
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

# MLX streams are thread-local and must be released before interpreter teardown.
atexit.register(clear_mlx_streams)
