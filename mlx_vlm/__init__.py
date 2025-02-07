from .prompt_utils import apply_chat_template, get_message_json
from .utils import (
    GenerationResult,
    convert,
    generate,
    load,
    prepare_inputs,
    process_image,
    quantize_model,
    stream_generate,
)
from .version import __version__
from .video_generate import VideoFrameExtractor, process_vision_info
