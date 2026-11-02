"""LiquidAI d1-omni: d1's System One API (see ../d1_lfm3_vl) on LFM3-Omni."""

from dataclasses import dataclass

from ..d1_lfm3_vl.config import D1Config
from ..d1_lfm3_vl.d1 import D1, REFUSAL
from ..lfm3_omni.config import ModelConfig as Lfm3OmniConfig
from ..lfm3_omni.config import TextConfig, VisionConfig
from ..lfm3_omni.lfm3_omni import Model as Lfm3OmniModel


@dataclass
class ModelConfig(D1Config, Lfm3OmniConfig):
    pass


class Model(D1, Lfm3OmniModel):
    def generate_audio(self, *args, **kwargs):
        raise ValueError(REFUSAL)

    stream_slots = generate_audio


__all__ = ["Model", "ModelConfig", "TextConfig", "VisionConfig"]
