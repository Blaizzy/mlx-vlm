"""
Processor for DiffusionGemma4 checkpoints.

Reuses the MLX-native Gemma4 processor (image processor + tokenizer + image
token expansion + ``mm_token_type_ids``), so no torch/torchvision is pulled in.
Audio and video inputs are rejected: this model port supports text and images.
"""

from typing import List, Optional, Union

from transformers.feature_extraction_utils import BatchFeature
from transformers.tokenization_utils_base import PreTokenizedInput, TextInput

from ..gemma4.processing_gemma4 import Gemma4Processor


class DiffusionGemma4Processor(Gemma4Processor):
    model_type = "diffusion_gemma4"

    def __call__(
        self,
        images=None,
        text: Optional[
            Union[
                TextInput,
                PreTokenizedInput,
                List[TextInput],
                List[PreTokenizedInput],
            ]
        ] = None,
        audio=None,
        videos=None,
        **kwargs,
    ) -> BatchFeature:
        if audio is not None or videos is not None:
            raise ValueError(
                "DiffusionGemma4 audio/video inputs are not supported yet."
            )
        if text is None and images is None:
            raise ValueError(
                "Provide `text` and/or `images` for DiffusionGemma4Processor."
            )
        return super().__call__(images=images, text=text, **kwargs)


__all__ = ["DiffusionGemma4Processor"]

from ..base import install_auto_processor_patch

install_auto_processor_patch(
    ["diffusion_gemma4", "diffusion_gemma"],
    DiffusionGemma4Processor,
)
