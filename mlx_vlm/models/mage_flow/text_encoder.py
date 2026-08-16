from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import mlx.core as mx
import numpy as np
from PIL import Image
from transformers import AutoProcessor, AutoTokenizer

from mlx_vlm.models.qwen3_vl import processing_qwen3_vl  # noqa: F401
from mlx_vlm.models.qwen3_vl.qwen3_vl import Model as Qwen3VLModel

GENERATION_TEMPLATE = (
    "<|im_start|>system\n"
    "Describe the image by detailing the color, shape, size, texture, quantity, "
    "text, spatial relationships of the objects and background:"
    "<|im_end|>\n<|im_start|>user\n{}<|im_end|>\n"
    "<|im_start|>assistant\n"
)
EDIT_TEMPLATE = (
    "<|im_start|>system\n"
    "Describe the key features of the input image (color, shape, size, texture,"
    " objects, background), then explain how the user's text instruction should "
    "alter or modify the image. Generate a new image that meets the user's "
    "requirements while maintaining consistency with the original input where "
    "appropriate.<|im_end|>\n<|im_start|>user\n{}<|im_end|>\n"
    "<|im_start|>assistant\n"
)
IMAGE_PLACEHOLDER = "<|vision_start|><|image_pad|><|vision_end|>"


def _to_mx(value):
    if isinstance(value, mx.array):
        return value
    if isinstance(value, np.ndarray):
        return mx.array(value)
    return mx.array(np.asarray(value))


def resize_long_edge(image: Image.Image, maximum: int | None = 384) -> Image.Image:
    image = image.convert("RGB")
    if maximum is None or maximum <= 0 or max(image.size) <= maximum:
        return image
    scale = maximum / max(image.size)
    size = (
        max(1, int(round(image.width * scale))),
        max(1, int(round(image.height * scale))),
    )
    return image.resize(size, Image.Resampling.BICUBIC)


class MageFlowTextEncoder:
    def __init__(
        self,
        *,
        model: Qwen3VLModel,
        model_path: str | Path,
        max_length: int = 2048,
    ) -> None:
        self.model = model
        self.model_path = Path(model_path).expanduser()
        self.max_length = max_length
        self.tokenizer = AutoTokenizer.from_pretrained(
            str(self.model_path / "text_encoder"),
            local_files_only=True,
            use_fast=True,
        )
        self.processor = AutoProcessor.from_pretrained(
            str(self.model_path / "text_encoder"),
            local_files_only=True,
        )

    def count_tokens(self, prompt: str, *, edit: bool = False) -> int:
        formatted = (EDIT_TEMPLATE if edit else GENERATION_TEMPLATE).format(prompt)
        return len(
            self.tokenizer(
                formatted,
                truncation=False,
                add_special_tokens=True,
            )["input_ids"]
        )

    def _hidden_states(self, inputs: dict) -> mx.array:
        input_ids = _to_mx(inputs["input_ids"]).astype(mx.int32)
        pixel_values = inputs.get("pixel_values")
        image_grid_thw = inputs.get("image_grid_thw")
        if pixel_values is not None:
            pixel_values = _to_mx(pixel_values)
        if image_grid_thw is not None:
            image_grid_thw = _to_mx(image_grid_thw).astype(mx.int32)
        features = self.model.get_input_embeddings(
            input_ids,
            pixel_values,
            image_grid_thw=image_grid_thw,
        )
        return self.model.language_model.model(
            input_ids,
            inputs_embeds=features.inputs_embeds,
            position_ids=features.position_ids,
            visual_pos_masks=features.visual_pos_masks,
            deepstack_visual_embeds=features.deepstack_visual_embeds,
        )

    def encode(self, prompt: str) -> mx.array:
        formatted = GENERATION_TEMPLATE.format(prompt)
        tokens = self.tokenizer(
            formatted,
            max_length=self.max_length + 34,
            truncation=True,
            return_tensors="np",
        )
        hidden = self._hidden_states(tokens)
        if hidden.shape[1] <= 34:
            raise ValueError("Mage-Flow prompt was empty after template trimming")
        return hidden[:, 34:]

    def encode_edit(
        self,
        prompt: str,
        images: Sequence[Image.Image],
        *,
        vl_cond_long_edge: int | None = 384,
    ) -> mx.array:
        refs = [resize_long_edge(image, vl_cond_long_edge) for image in images]
        prefix = "".join(
            f"Image {index}: {IMAGE_PLACEHOLDER}" for index in range(1, len(refs) + 1)
        )
        formatted = EDIT_TEMPLATE.format(prefix + prompt)
        inputs = self.processor(
            text=[formatted],
            images=refs,
            padding=True,
            return_tensors="np",
        )
        hidden = self._hidden_states(dict(inputs))
        if hidden.shape[1] <= 64:
            raise ValueError("Mage-Flow edit prompt was empty after template trimming")
        return hidden[:, 64:]


__all__ = [
    "EDIT_TEMPLATE",
    "GENERATION_TEMPLATE",
    "MageFlowTextEncoder",
    "resize_long_edge",
]
