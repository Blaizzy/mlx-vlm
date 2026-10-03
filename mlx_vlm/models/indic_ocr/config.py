"""IndicOCR (bodhan-ai/indic-ocr) configuration.

OCR-only configs reuse Qwen3.5. Combined checkpoints embed both stage configs
as ``layout_config`` and ``ocr_config`` and load as an IndicOCRParser.
The repository provides a standard shard index pointing to both stages.
"""

from dataclasses import dataclass
from typing import List, Optional, Union

from ..base import BaseModelConfig
from ..pp_doclayout_v3.config import ModelConfig as LayoutConfig
from ..qwen3_5.config import TextConfig, VisionConfig


@dataclass
class ModelConfig(BaseModelConfig):
    text_config: TextConfig
    vision_config: VisionConfig
    model_type: str = "indic_ocr"
    ignore_index: int = -100
    image_token_id: int = 262155
    video_token_id: int = 262156
    image_token_index: Optional[int] = None
    video_token_index: Optional[int] = None
    vision_start_token_id: int = 262153
    vision_end_token_id: int = 262154
    vocab_size: int = 262157
    eos_token_id: Optional[Union[int, List[int]]] = None
    quantization: Optional[dict] = None
    quantization_config: Optional[dict] = None

    def __post_init__(self):
        if self.image_token_index is None:
            self.image_token_index = self.image_token_id
        if self.video_token_index is None:
            self.video_token_index = self.video_token_id

    @classmethod
    def from_dict(cls, params):
        params = dict(params)
        if "layout_config" in params or "ocr_config" in params:
            return ParserConfig.from_dict(params)
        if not params.get("text_config") and "stages" in params:
            raise ValueError(
                "indic_ocr wrapper repo detected (stages.layout/stages.ocr). "
                "Use a repository revision with embedded layout_config and ocr_config "
                "and a standard safetensors index, or load the weights/ocr or "
                "weights/layout stage directly."
            )
        from ..qwen3_vl.config import _config_kwargs, _maybe_deserialize_config

        params["vision_config"] = _maybe_deserialize_config(
            VisionConfig, params.get("vision_config")
        )
        params["text_config"] = _maybe_deserialize_config(
            TextConfig, params.get("text_config"), require_all_fields=True
        )
        return cls(**_config_kwargs(cls, params))


@dataclass
class ParserConfig(BaseModelConfig):
    layout_config: LayoutConfig
    ocr_config: ModelConfig
    model_type: str = "indic_ocr"
    quantization: Optional[dict] = None
    quantization_config: Optional[dict] = None
    ocr_model_path: Optional[str] = None
    weight_mapping: Optional[dict] = None

    @classmethod
    def from_dict(cls, params):
        if not params.get("layout_config") or not params.get("ocr_config"):
            raise ValueError(
                "Combined IndicOCR configs require layout_config and ocr_config"
            )
        return cls(
            layout_config=LayoutConfig.from_dict(params["layout_config"]),
            ocr_config=ModelConfig.from_dict(params["ocr_config"]),
            quantization=params.get("quantization"),
            quantization_config=params.get("quantization_config"),
            ocr_model_path=params.get("ocr_model_path"),
            weight_mapping=params.get("weight_mapping"),
        )
