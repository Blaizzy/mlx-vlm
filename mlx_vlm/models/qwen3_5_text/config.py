from dataclasses import dataclass
from typing import Dict, List, Optional, Union

from ..base import BaseModelConfig
from ..qwen3_5.config import (
    TextConfig,
    resolve_qwen_eos_token_id,
    sanitize_quantization_config,
)
from ..qwen3_vl.config import _config_kwargs, _maybe_deserialize_config


@dataclass
class ModelConfig(BaseModelConfig):
    text_config: TextConfig
    model_type: str = "qwen3_5_text"
    eos_token_id: Optional[Union[int, List[int]]] = None
    quantization: Optional[Dict] = None
    quantization_config: Optional[Dict] = None

    def __post_init__(self):
        self.eos_token_id = resolve_qwen_eos_token_id(
            self.eos_token_id, self.text_config
        )
        quantization = self.quantization
        self.quantization = sanitize_quantization_config(quantization)
        if self.quantization_config == quantization:
            self.quantization_config = self.quantization
        else:
            self.quantization_config = sanitize_quantization_config(
                self.quantization_config
            )

    @classmethod
    def from_dict(cls, params):
        params = dict(params)
        text_params = params.get("text_config")
        decoder_only = text_params is None
        if decoder_only:
            text_params = params
        params["text_config"] = _maybe_deserialize_config(
            TextConfig, text_params, require_all_fields=True
        )
        if decoder_only:
            # The decoder fields sit at the top level, so there is no outer
            # eos_token_id distinct from the decoder's. Clear it after the
            # decoder config is built so the shared resolver derives the value
            # and adds the chat stop token.
            params["eos_token_id"] = None
        return cls(**_config_kwargs(cls, params))
