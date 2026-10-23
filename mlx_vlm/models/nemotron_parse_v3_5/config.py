# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

from dataclasses import dataclass, field

from ..base import BaseModelConfig
from ..nemotron_parse.config import TextConfig as MBartConfig
from ..nemotron_parse.config import VisionConfig as RadioConfig


@dataclass
class VisionConfig(RadioConfig):
    hidden_size: int = 1152
    mlp_ratio: float = 4304 / 1152
    num_layers: int = 27
    num_cls_tokens: int = 2
    num_register_tokens: int = 8
    summary_idxs: list[int] = field(default_factory=lambda: [0, 1])


@dataclass
class TextConfig(MBartConfig):
    model_type: str = "nemotron_parse_v3_5"
    vocab_size: int = 72512
    max_position_embeddings: int = 16384
    self_attention_kv_heads: int = 8


@dataclass
class ModelConfig(BaseModelConfig):
    model_type: str = "nemotron_parse_v3_5"
    vision_config: VisionConfig = field(default_factory=VisionConfig)
    text_config: TextConfig = field(default_factory=TextConfig)
    max_sequence_length: int = 16384
    eos_token_id: int = 2
    pad_token_id: int = 1
    tie_word_embeddings: bool = True

    @classmethod
    def from_dict(cls, params):
        params["text_config"] = params.get("text_config") or params.get("decoder", {})
        return cls(
            vision_config=VisionConfig.from_dict(params.get("vision_config", {})),
            text_config=TextConfig.from_dict(params["text_config"]),
            max_sequence_length=params.get("max_sequence_length", 16384),
        )
