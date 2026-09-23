from transformers import AutoTokenizer

from ..base import install_auto_processor_patch
from .config import ModelConfig
from .language import LanguageModel
from .qwen3_5_moe_text import Model

# Text-only exports often keep the VLM tokenizer config's processor_class,
# which makes AutoProcessor demand image-processor files the repo lacks.
install_auto_processor_patch("qwen3_5_moe_text", AutoTokenizer)
