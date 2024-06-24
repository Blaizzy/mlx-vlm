import inspect
from dataclasses import dataclass


@dataclass
class TextConfig:
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
