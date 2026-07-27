"""Torch-free processor for Mage-VL.

The checkpoint's own remote-code processor (`processing_mage_vl.MageVLProcessor` on the hub) is
torch-native — its ``__call__`` builds torch tensors, so mlx-vlm's ``prepare_inputs`` dies with
``ones_like(): argument 'input' must be Tensor, not array``. This in-repo processor follows the
``qwen3_vl`` pattern: numpy/mlx end to end, installed over AutoProcessor for ``model_type
mage_vl`` so ``mlx_vlm.load`` returns it without ``trust_remote_code``.

Image path only. Upstream's video path is either codec-derived (the ``codec-video-prep`` native
extension — manylinux wheels only, cannot run on macOS) or a torch-based frame sampler; a
torch-free video path is a follow-up.

``patch_positions`` is deliberately NOT emitted: the model derives positions from
``image_grid_thw`` in the processor's 2x2 block order, and that derivation is pinned
row-identical to the reference processor's output by ``test_mage_vl_positions.py``.
"""

from typing import List, Optional, Union

import numpy as np
from transformers.feature_extraction_utils import BatchFeature
from transformers.processing_utils import ProcessorMixin
from transformers.tokenization_utils_base import PreTokenizedInput, TextInput

from ..base import install_auto_processor_patch, load_chat_template, to_mlx

VISION_START = "<|vision_start|>"
VISION_END = "<|vision_end|>"
IMAGE_PAD = "<|image_pad|>"


class MageVLProcessor(ProcessorMixin):
    attributes = ["image_processor", "tokenizer"]
    image_processor_class = "AutoImageProcessor"
    tokenizer_class = "AutoTokenizer"

    def __init__(
        self, image_processor=None, tokenizer=None, chat_template=None, **kwargs
    ):
        if chat_template is None and tokenizer is not None:
            chat_template = getattr(tokenizer, "chat_template", None)
        super().__init__(image_processor, tokenizer, chat_template=chat_template)
        self.spatial_merge_size = getattr(image_processor, "merge_size", 2)
        self.image_token = IMAGE_PAD

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, **kwargs):
        from transformers import AutoImageProcessor, AutoTokenizer

        kwargs.pop("use_fast", None)
        kwargs.pop("trust_remote_code", None)
        tokenizer = AutoTokenizer.from_pretrained(
            pretrained_model_name_or_path, **kwargs
        )
        load_chat_template(tokenizer, pretrained_model_name_or_path)
        # preprocessor_config.json declares Qwen2VLImageProcessor (patch 16, merge 2,
        # temporal_patch_size 1, CLIP statistics) — framework-agnostic, numpy-native.
        image_processor = AutoImageProcessor.from_pretrained(
            pretrained_model_name_or_path, use_fast=False
        )
        return cls(image_processor=image_processor, tokenizer=tokenizer)

    def __call__(
        self,
        text: Union[TextInput, PreTokenizedInput, List[TextInput]] = None,
        images=None,
        videos=None,
        audio=None,
        return_tensors: Optional[str] = None,
        **kwargs,
    ) -> BatchFeature:
        if videos is not None:
            raise NotImplementedError(
                "mage_vl video input needs the upstream codec pipeline (not available as a "
                "torch-free path yet); image input only for now"
            )
        if isinstance(text, str):
            text = [text]
        text = list(text or [])

        image_inputs = {}
        if images is not None:
            image_inputs = dict(
                self.image_processor(images=images, return_tensors="np")
            )
            grids = np.asarray(image_inputs["image_grid_thw"])

            # Expand each <|image_pad|> placeholder to its grid's merged-token count
            # (t*h*w / merge^2) — mirrors the reference processor's `_expand_image_pads`.
            merge = self.spatial_merge_size
            counts = (grids[:, 0] * grids[:, 1] * grids[:, 2]) // (merge * merge)
            img_idx = 0

            def expand(s: str) -> str:
                nonlocal img_idx
                while IMAGE_PAD in s and img_idx < len(counts):
                    s = s.replace(
                        IMAGE_PAD, "<|placeholder|>" * int(counts[img_idx]), 1
                    )
                    img_idx += 1
                return s.replace("<|placeholder|>", IMAGE_PAD)

            text = [expand(s) for s in text]

        # mlx-vlm's process_inputs passes padding in kwargs; hardcoding it too collides.
        padding = kwargs.pop("padding", True)
        text_inputs = self.tokenizer(
            text, return_tensors="np", padding=padding, **kwargs
        )
        return BatchFeature(data=to_mlx({**dict(text_inputs), **image_inputs}))

    def batch_decode(self, *args, **kwargs):
        return self.tokenizer.batch_decode(*args, **kwargs)

    def decode(self, *args, **kwargs):
        return self.tokenizer.decode(*args, **kwargs)

    @property
    def model_input_names(self):
        return ["input_ids", "attention_mask", "pixel_values", "image_grid_thw"]


__all__ = ["MageVLProcessor"]

install_auto_processor_patch("mage_vl", MageVLProcessor)
