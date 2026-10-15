"""Youtu-Parsing-Omni (``youtu_vita``) model package.

Ported from tencent/Youtu-Parsing-Omni (https://huggingface.co/tencent/Youtu-Parsing-Omni).
Copyright (C) 2026 Tencent. The original code is distributed under the Youtu-Parsing
license (https://huggingface.co/tencent/Youtu-Parsing-Omni/blob/main/LICENSE);
modified / re-implemented here for MLX.
"""

from .config import ModelConfig, OmniConfig, TextConfig
from .language import LanguageModel
from .omni import OmniModel
from .processing_youtu_vita import YoutuVitaImageProcessor, YoutuVitaProcessor
from .youtu_vita import Model
