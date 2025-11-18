from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional

from ..base import BaseModelConfig
from ..qwen3_vl_moe.config import TextConfig as ThinkerTextConfig
from ..qwen3_vl_moe.config import VisionConfig as ThinkerVisionConfig


class FlexibleConfig:
    """
    Lightweight container that mirrors the behaviour of Hugging Face configs.

    The Hugging Face `configuration_qwen3_omni_moe.py` file exposes several nested
    configs (audio encoder, talker, code2wav, ...).  In MLX we only need access to
    the attributes that are actually consumed by the runtime modules.  To avoid
    re-implementing the whole dataclass tree – and to remain forward compatible
    with upstream updates – we keep the config as a thin dictionary-backed object
    that supports attribute access.
    """

    def __init__(self, **kwargs: Any):
        for key, value in kwargs.items():
            setattr(self, key, value)

    @classmethod
    def from_dict(cls, params: Dict[str, Any]) -> "FlexibleConfig":
        return cls(**params)

    def to_dict(self) -> Dict[str, Any]:
        return self.__dict__


AudioConfig = FlexibleConfig
TalkerConfig = FlexibleConfig
Code2WavConfig = FlexibleConfig


@dataclass
class ModelConfig(BaseModelConfig):
    """
    Top-level configuration for Qwen3-Omni-MoE.

    The thinker stack reuses the Qwen3-VL-MoE vision+language components,
    therefore we keep their configs unchanged.  Additional modules (audio
    encoder, talker, code2wav) are represented with the flexible config so that
    new parameters coming from HF configs are preserved automatically.
    """

    model_type: str = "qwen3_omni_moe"

    text_config: Dict[str, Any] = field(default_factory=dict)
    vision_config: Dict[str, Any] = field(default_factory=dict)
    audio_config: Dict[str, Any] = field(default_factory=dict)

    talker_config: Optional[TalkerConfig] = None
    code2wav_config: Optional[Code2WavConfig] = None

    # Multimodal token ids – defaults mirror the values used in the HF configs.
    image_token_id: int = 151655
    video_token_id: int = 151656
    audio_token_id: int = 151657
    vision_start_token_id: int = 151652
    vision_end_token_id: int = 151653
    vision_token_id: int = 151654
    audio_start_token_id: int = 151658
    audio_end_token_id: int = 151659
    position_id_per_seconds: int = 25

    # Misc
    ignore_index: int = -100
    enable_audio_output: bool = False
    pad_token_id: Optional[int] = None
    eos_token_id: Optional[int] = None

    def __post_init__(self) -> None:
        # Ensure nested configs are instances of the proper classes.  `update_module_configs`
        # in `mlx_vlm.utils` will overwrite these with the actual objects loaded
        # from `config.json`, so during the initial construction we just need to
        # make sure the attributes exist.
        if isinstance(self.text_config, dict):
            self.text_config = ThinkerTextConfig.from_dict(self.text_config)
        if isinstance(self.vision_config, dict):
            self.vision_config = ThinkerVisionConfig.from_dict(self.vision_config)
        if isinstance(self.audio_config, dict):
            self.audio_config = AudioConfig.from_dict(self.audio_config)
        if isinstance(self.talker_config, dict):
            self.talker_config = TalkerConfig.from_dict(self.talker_config)
        if isinstance(self.code2wav_config, dict):
            self.code2wav_config = Code2WavConfig.from_dict(self.code2wav_config)

    @classmethod
    def from_dict(cls, params: Dict[str, Any]) -> "ModelConfig":
        filtered = {k: v for k, v in params.items() if k in cls.__dataclass_fields__}
        return cls(**filtered)
