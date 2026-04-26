from ..glm4v.language import Glm4vDecoderLayer, GLM4VModel
from ..glm4v.language import LanguageModel as Glm4vLanguageModel
from ..glm4v.language import MoE, MoEGate  # noqa: F401 (backwards compat)
from .config import ModelConfig, TextConfig

# The base Glm4vDecoderLayer now handles MoE (when n_routed_experts is set),
# so the MoE-specific decoder layer and model are thin wrappers.

Glm4vMoeDecoderLayer = Glm4vDecoderLayer


class Glm4vMoeModel(GLM4VModel):
    pass


class LanguageModel(Glm4vLanguageModel):
    def __init__(self, args: TextConfig, config: ModelConfig = None):
        super().__init__(args, config)
