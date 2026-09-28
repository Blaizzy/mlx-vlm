from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path

import mlx.core as mx
import numpy as np
from PIL import Image
from transformers import AutoTokenizer

from mlx_vlm.models.qwen3_vl.processing_qwen3_vl import (
    Qwen3VLImageProcessor,
    Qwen3VLProcessor,
)
from mlx_vlm.models.qwen3_vl.qwen3_vl import Model as Qwen3VLModel

SYSTEM_PROMPT = "Comprehend and analyze the provided prompt."

# The checkpoint expects the raw template string fed straight to the processor,
# not `apply_chat_template`; the two tokenize differently. The image prefix only
# appears in the edit (image-conditioned) template.
PROMPT_TEMPLATE_T2I = (
    f"<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n"
    "<|im_start|>user\n{}<|im_end|>\n"
    "<|im_start|>assistant\n"
)
PROMPT_TEMPLATE_TI2I = (
    f"<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n"
    "<|im_start|>user\n{}{}<|im_end|>\n"
    "<|im_start|>assistant\n"
)
IMAGE_PLACEHOLDER = "<|vision_start|><|image_pad|><|vision_end|>"


def _to_mx(value) -> mx.array:
    if isinstance(value, mx.array):
        return value
    if isinstance(value, np.ndarray):
        return mx.array(value)
    return mx.array(np.asarray(value))


class QwenImageTextEncoder:
    """Qwen3-VL prompt conditioner for Qwen-Image-2.1.

    Wraps the shared ``qwen3_vl`` model and returns the last decoder layer's
    hidden states *before* the final RMSNorm (``apply_final_norm=False``), with
    the leading system-prompt tokens dropped, matching the reference pipeline.
    """

    def __init__(
        self,
        *,
        model: Qwen3VLModel,
        model_path: str | Path,
        max_length: int = 1024,
    ) -> None:
        self.model = model
        self.model_path = Path(model_path).expanduser()
        self.max_length = max_length
        # Reuse the NumPy image processor so editing does not require PyTorch.
        self.processor_dir = str(self.model_path / "processor")
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.processor_dir, local_files_only=True, use_fast=True
        )
        self._processor = None
        # Number of leading system-turn tokens the reference drops from the hidden
        # states, derived by tokenizing the template's system prefix directly.
        system_prefix = f"<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n"
        self.drop_idx = len(
            self.tokenizer(system_prefix, add_special_tokens=False)["input_ids"]
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
            input_ids, pixel_values, image_grid_thw=image_grid_thw
        )
        return self.model.language_model.model(
            input_ids,
            inputs_embeds=features.inputs_embeds,
            position_ids=features.position_ids,
            visual_pos_masks=features.visual_pos_masks,
            deepstack_visual_embeds=features.deepstack_visual_embeds,
            apply_final_norm=False,
        )

    def encode(self, prompt: str) -> mx.array:
        formatted = PROMPT_TEMPLATE_T2I.format(prompt)
        tokens = self.tokenizer(
            formatted,
            max_length=self.max_length + self.drop_idx,
            truncation=True,
            return_tensors="np",
        )
        hidden = self._hidden_states(dict(tokens))
        if hidden.shape[1] <= self.drop_idx:
            raise ValueError("Qwen-Image prompt was empty after template trimming")
        return hidden[:, self.drop_idx :]

    @property
    def processor(self):
        if self._processor is None:
            config = json.loads(
                (Path(self.processor_dir) / "preprocessor_config.json").read_text()
            )
            size = config.get("size", {})
            config["min_pixels"] = config.get("min_pixels") or size.get(
                "shortest_edge", 65536
            )
            config["max_pixels"] = config.get("max_pixels") or size.get(
                "longest_edge", 16777216
            )
            self._processor = Qwen3VLProcessor(
                image_processor=Qwen3VLImageProcessor(**config),
                tokenizer=self.tokenizer,
            )
        return self._processor

    def encode_edit(
        self,
        prompt: str,
        images: Sequence[Image.Image],
    ) -> tuple[mx.array, mx.array]:
        refs = []
        for image in images:
            rgba = image.convert("RGBA")
            rgb = Image.new("RGB", rgba.size, "white")
            rgb.paste(rgba, mask=rgba.getchannel("A"))
            refs.append(rgb)
        prefix = " ".join(
            f"<image{index}>{IMAGE_PLACEHOLDER}" for index in range(1, len(refs) + 1)
        )
        formatted = PROMPT_TEMPLATE_TI2I.format(prefix, prompt)
        inputs = self.processor(
            text=[formatted],
            images=refs,
            padding=True,
            return_tensors="np",
        )
        hidden = self._hidden_states(dict(inputs))
        if hidden.shape[1] <= self.drop_idx:
            raise ValueError("Qwen-Image edit prompt was empty after template trimming")
        image_token_id = self.tokenizer.convert_tokens_to_ids("<|image_pad|>")
        image_pad_mask = _to_mx(inputs["input_ids"]) == image_token_id
        return hidden[:, self.drop_idx :], image_pad_mask[:, self.drop_idx :]


__all__ = [
    "IMAGE_PLACEHOLDER",
    "PROMPT_TEMPLATE_T2I",
    "PROMPT_TEMPLATE_TI2I",
    "QwenImageTextEncoder",
    "SYSTEM_PROMPT",
]
