from ..qwen3_5.language import LanguageModel as Qwen3_5LanguageModel
from ..qwen3_5.language import Qwen3_5DecoderLayer, Qwen3_5Model
from ..qwen3_5.language import (  # noqa: F401 (backwards compat)
    Qwen3_5SparseMoeBlock as Qwen3_5MoeSparseMoeBlock,
)
from .config import ModelConfig, TextConfig

# The base Qwen3_5DecoderLayer now handles MoE (when num_experts > 0),
# so the MoE-specific decoder layer and model are thin wrappers.

Qwen3_5MoeDecoderLayer = Qwen3_5DecoderLayer


class Qwen3_5MoeModel(Qwen3_5Model):
    pass


class LanguageModel(Qwen3_5LanguageModel):
    def __init__(self, args: TextConfig, config: ModelConfig = None):
        super().__init__(args, config)
