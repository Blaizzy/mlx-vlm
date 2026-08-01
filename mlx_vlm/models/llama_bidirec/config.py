from dataclasses import dataclass

from ..llama.config import ModelConfig as LlamaModelConfig


@dataclass
class ModelConfig(LlamaModelConfig):
    pass
