from dataclasses import dataclass, field
from typing import Dict

from ..gemma4.config import ModelConfig as Gemma4ModelConfig
from ..gemma4.config import TextConfig, VisionConfig
from ..qwen3_vl.config import _config_kwargs, _maybe_deserialize_config


@dataclass
class ModelConfig(Gemma4ModelConfig):
    decision_config: Dict = field(default_factory=dict)

    @classmethod
    def from_dict(cls, params):
        params = dict(params or {})
        params["text_config"] = _maybe_deserialize_config(
            TextConfig, params.get("text_config"), require_all_fields=False
        )
        params["vision_config"] = _maybe_deserialize_config(
            VisionConfig, params.get("vision_config")
        )
        return cls(**_config_kwargs(cls, params))
