import inspect
from dataclasses import dataclass


@dataclass
class TextConfig:
    max_position_embeddings: int = 4096

    @classmethod
    def from_dict(cls, params):
        return cls(
            **{
                k: v
                for k, v in params.items()
                if k in inspect.signature(cls).parameters
            }
        )


class LanguageModel:
    pass
