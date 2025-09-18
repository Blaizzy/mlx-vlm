import os

from .generate import GenerationResult, generate, stream_generate
from .prompt_utils import apply_chat_template, get_message_json
from .utils import load, prepare_inputs, process_image
from .version import __version__

def convert(*args, **kwargs):
    from . import convert as convert_module

    return convert_module.convert(*args, **kwargs)

os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "1"
