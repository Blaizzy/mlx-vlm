"""Processor for Jina VLM in MLX-VLM."""

from typing import Dict, List, Literal, Optional, Union

import mlx.core as mx
import numpy as np
from PIL import Image
from transformers.processing_utils import ProcessorMixin

from .image_processor import ImageProcessor


class JinaVLMProcessor(ProcessorMixin):
    """Processor for Jina VLM that combines tokenizer and image processor."""

    tokenizer_class = ("Qwen2Tokenizer", "Qwen2TokenizerFast")
    attributes = ["tokenizer"]

    def __init__(
        self,
        tokenizer,
        image_token: str = "<|image|>",
        chat_template: Optional[str] = None,
        **kwargs,
    ):
        self.tokenizer = tokenizer
        self.image_token = image_token
        self._image_proc = ImageProcessor()  # Internal, not exposed as image_processor

        # Get image token ID
        self.image_token_id = self.tokenizer.convert_tokens_to_ids(image_token)

        super().__init__(tokenizer, **kwargs)

        # Set chat template AFTER super().__init__ - always set the default if not already set
        default_chat_template = (
            "{% for message in messages %}"
            "{% if message['role'] == 'system' %}{{ '<|system|>\n' + message['content'] + '\n' }}"
            "{% elif message['role'] == 'user' %}{{ '<|user|>\n' + message['content'] + '\n' }}"
            "{% elif message['role'] == 'assistant' %}{{ '<|assistant|>\n' + message['content'] + '\n' }}"
            "{% endif %}{% endfor %}"
            "{% if add_generation_prompt %}{{ '<|assistant|>\n' }}{% endif %}"
        )
        if chat_template is not None:
            self.tokenizer.chat_template = chat_template
        elif not self.tokenizer.chat_template:
            self.tokenizer.chat_template = default_chat_template

    @property
    def chat_template(self):
        return self.tokenizer.chat_template

    @chat_template.setter
    def chat_template(self, value):
        self.tokenizer.chat_template = value

    @property
    def pad_token(self):
        return self.tokenizer.pad_token

    @property
    def pad_token_id(self):
        return self.tokenizer.pad_token_id

    @property
    def eos_token(self):
        return self.tokenizer.eos_token

    @property
    def eos_token_id(self):
        return self.tokenizer.eos_token_id

    @property
    def bos_token(self):
        return self.tokenizer.bos_token

    @property
    def bos_token_id(self):
        return self.tokenizer.bos_token_id

    def encode(self, text: str, add_special_tokens: bool = False) -> List[int]:
        return self.tokenizer.encode(text, add_special_tokens=add_special_tokens)

    def decode(self, token_ids: List[int], **kwargs) -> str:
        return self.tokenizer.decode(token_ids, **kwargs)

    def batch_decode(self, token_ids, **kwargs) -> List[str]:
        return self.tokenizer.batch_decode(token_ids, **kwargs)

    def process_one(
        self,
        prompt: str,
        images: Optional[List[Image.Image]] = None,
        inference_mode: bool = True,
    ) -> Dict:
        """Process a single prompt with images."""
        if images is None:
            images = []

        # Process images
        if images:
            image_outputs = self._image_proc.preprocess(images)
            pixel_values_list = image_outputs["pixel_values"]
            image_tokens = image_outputs["image_tokens"]
            image_input_idx_list = image_outputs["image_input_idx"]
            image_masks_list = image_outputs["image_masks"]
        else:
            pixel_values_list = None
            image_tokens = []
            image_input_idx_list = None
            image_masks_list = None

        # Split prompt by image token
        text_splits = prompt.split(self.image_token)

        # Build input_ids with image tokens interleaved
        input_ids = []
        current_image_idx = 0
        updated_image_input_idx = []

        for i, text_part in enumerate(text_splits):
            # Encode text part
            if text_part:
                text_tokens = self.encode(text_part, add_special_tokens=False)
                input_ids.extend(text_tokens)

            # Add image tokens if not the last split and we have images
            if i < len(text_splits) - 1 and current_image_idx < len(image_tokens):
                # Get image tokens for this image
                img_tokens = image_tokens[current_image_idx]
                # Offset image_input_idx by current position
                if image_input_idx_list is not None and current_image_idx < len(
                    image_input_idx_list
                ):
                    offset_idx = image_input_idx_list[current_image_idx] + len(
                        input_ids
                    )
                    updated_image_input_idx.append(offset_idx)
                input_ids.extend(img_tokens.tolist())
                current_image_idx += 1

        input_ids = mx.array(input_ids)

        result = {
            "input_ids": input_ids[None, :],  # Add batch dimension
            "attention_mask": mx.ones_like(input_ids)[None, :],
        }

        if pixel_values_list is not None and len(pixel_values_list) > 0:
            # Stack pixel values: (n_crops, n_patches, patch_dim)
            result["pixel_values"] = mx.array(np.stack(pixel_values_list))
            # Stack image_input_idx: (n_images, tokens_per_image)
            result["image_input_idx"] = mx.array(np.stack(updated_image_input_idx))
            # Stack image_masks: (n_crops, n_patches)
            result["image_masks"] = mx.array(np.stack(image_masks_list))

        return result

    def __call__(
        self,
        text: Optional[Union[str, List[str]]] = None,
        images: Optional[Union[Image.Image, List[Image.Image]]] = None,
        inference_mode: bool = True,
        return_tensors: Literal["np", "mx", "pt"] = "mx",
        **kwargs,
    ) -> Dict:
        """Process text and images for Jina VLM.

        When called with just text (like a tokenizer), returns tokenizer output.
        When called with text and images, returns full processed inputs.

        Args:
            text: Input text or list of texts
            images: Input image or list of images
            inference_mode: Whether in inference mode
            return_tensors: Type of tensors to return

        Returns:
            Dictionary containing processed inputs
        """
        # If called with just text (like a tokenizer), delegate to tokenizer
        if text is not None and images is None:
            return self.tokenizer(text, **kwargs)

        if text is None:
            raise ValueError("Text must be provided")

        # Normalize inputs
        if isinstance(text, str):
            texts = [text]
        else:
            texts = text

        if images is None:
            images_list = [None] * len(texts)
        elif isinstance(images, Image.Image):
            images_list = [[images]]
        elif isinstance(images, list) and len(images) > 0:
            if isinstance(images[0], Image.Image):
                # Single list of images for single prompt
                images_list = [images]
            else:
                images_list = images
        else:
            images_list = [None] * len(texts)

        # Process each text-image pair
        batch_results = []
        for prompt, imgs in zip(texts, images_list):
            result = self.process_one(prompt, imgs, inference_mode)
            batch_results.append(result)

        # Collate results
        if len(batch_results) == 1:
            return batch_results[0]
        else:
            return self._collate_batch(batch_results)

    def _collate_batch(self, batch_results: List[Dict]) -> Dict:
        """Collate multiple results into a batch."""
        # Get max sequence length
        max_len = max(r["input_ids"].shape[1] for r in batch_results)

        padded_input_ids = []
        padded_attention_mask = []

        for r in batch_results:
            seq_len = r["input_ids"].shape[1]
            pad_len = max_len - seq_len

            if pad_len > 0:
                input_ids = mx.concatenate(
                    [mx.full((1, pad_len), self.pad_token_id), r["input_ids"]], axis=1
                )
                attention_mask = mx.concatenate(
                    [mx.zeros((1, pad_len)), r["attention_mask"]], axis=1
                )
            else:
                input_ids = r["input_ids"]
                attention_mask = r["attention_mask"]

            padded_input_ids.append(input_ids)
            padded_attention_mask.append(attention_mask)

        result = {
            "input_ids": mx.concatenate(padded_input_ids, axis=0),
            "attention_mask": mx.concatenate(padded_attention_mask, axis=0),
        }

        # Combine pixel values if present
        all_pixel_values = []
        all_image_input_idx = []
        all_image_masks = []

        for r in batch_results:
            if "pixel_values" in r:
                all_pixel_values.append(r["pixel_values"])
                all_image_input_idx.append(r["image_input_idx"])
                all_image_masks.append(r["image_masks"])

        if all_pixel_values:
            result["pixel_values"] = mx.concatenate(all_pixel_values, axis=0)
            result["image_input_idx"] = mx.concatenate(all_image_input_idx, axis=0)
            result["image_masks"] = mx.concatenate(all_image_masks, axis=0)

        return result
