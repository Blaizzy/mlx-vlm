from ..qwen3_vl.language import LanguageModel as Qwen3VLLanguageModel
from ..qwen3_vl.language import Qwen3VLDecoderLayer, Qwen3VLModel
from ..qwen3_vl.language import (  # noqa: F401 (backwards compat)
    Qwen3VLSparseMoeBlock as Qwen3VLMoESparseMoeBlock,
)
from .config import ModelConfig, TextConfig

# The base Qwen3VLDecoderLayer now handles MoE (when num_experts > 0),
# so the MoE-specific decoder layer and model are thin wrappers.

Qwen3VLMoEDecoderLayer = Qwen3VLDecoderLayer


class Qwen3VLMoEModel(Qwen3VLModel):
    pass


class LanguageModel(Qwen3VLLanguageModel):
    def __init__(self, args: TextConfig, config: ModelConfig = None):
        super().__init__(args, config)
