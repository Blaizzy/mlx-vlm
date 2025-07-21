import inspect
from dataclasses import dataclass

from ..base import BaseModelConfig


@dataclass
class TextConfig(BaseModelConfig):
    max_position_embeddings: int = 4096


class LanguageModel:
    pass
