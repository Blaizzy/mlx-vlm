import json
import warnings

import mlx.core as mx

from ..models.base import to_mlx
from ..prompt_utils import MODEL_CONFIG, apply_chat_template

NATIVE_PREPROCESS_MODELS = set(MODEL_CONFIG.keys())


class VisionDataset:
    """Simplified dataset class for Vision LLMs"""

    def __init__(
        self,
        hf_dataset,
        config,
        processor,
        image_resize_shape=None,
    ):
        self.dataset = hf_dataset
        self.processor = processor
        self.config = config
        self.image_resize_shape = image_resize_shape

    def __len__(self):
        if hasattr(self.dataset, "__len__"):
            return len(self.dataset)
        raise TypeError("Streaming dataset has no length")

    def __getitem__(self, idx):
        return self.process(self.dataset[idx])

    def process(self, item):
        """Process a single item from the dataset"""
        from mlx_vlm.utils import prepare_inputs, process_inputs_with_fallback

        # Handle images
        images = item.get("images", item.get("image", []))
        if not isinstance(images, list):
            images = [images] if images else []

        # Handle audio
        audio = item.get("audio", item.get("audios", []))
        if not isinstance(audio, list):
            audio = [audio] if audio else []

        # Get conversations
        conversations = item.get("messages", item.get("conversations"))

        model_type = self.config.get("model_type")

        num_images = len(images)
        num_audios = len(audio)

        prompts = []
        # Format prompt using apply_chat_template for consistency with inference
        if isinstance(conversations, list) and isinstance(conversations[0], list):
            for conversation in conversations:
                if model_type == "pixtral":
                    conversation = [json.loads(i) for i in conversation]
                    if len(conversations) > 1:
                        warnings.warn(
                            "Pixtral batch processing is not supported yet. Set batch size to 1."
                        )

                prompt = apply_chat_template(
                    self.processor,
                    self.config,
                    conversation,
                    add_generation_prompt=False,
                    num_images=num_images,
                    num_audios=num_audios,
                )
                prompts.append(prompt)

        else:
            if model_type == "pixtral":
                conversations = [json.loads(i) for i in conversations]
            prompt = apply_chat_template(
                self.processor,
                self.config,
                conversations,
                add_generation_prompt=False,
                num_images=num_images,
                num_audios=num_audios,
            )
            prompts.append(prompt)

        # Prepare inputs - always pass images so the processor computes pixel values
        image_token_index = self.config.get("image_token_index") or self.config.get(
            "image_token_id"
        )
        if not image_token_index:
            raise ValueError(
                "Config must contain 'image_token_index' or 'image_token_id'"
            )

        # Prefer native multimodal processor for known model families to keep
        # image/audio placeholders and model-specific metadata aligned.
        if model_type in NATIVE_PREPROCESS_MODELS:
            try:
                inputs = process_inputs_with_fallback(
                    processor=self.processor,
                    prompts=prompts,
                    images=images if images else None,
                    audio=audio if audio else None,
                    add_special_tokens=False,
                )
                if "images" in inputs and "pixel_values" not in inputs:
                    inputs["pixel_values"] = inputs.pop("images")
            except Exception:
                # Fall back to legacy preparation path when processor-native
                # processing is unavailable for a specific model implementation.
                inputs = prepare_inputs(
                    processor=self.processor,
                    images=images if images else None,
                    audio=audio if audio else None,
                    prompts=prompts,
                    image_token_index=image_token_index,
                    resize_shape=self.image_resize_shape,
                )
        else:
            inputs = prepare_inputs(
                processor=self.processor,
                images=images if images else None,
                audio=audio if audio else None,
                prompts=prompts,
                image_token_index=image_token_index,
                resize_shape=self.image_resize_shape,
            )

        # Native preprocessing may return torch tensors when it falls back from
        # return_tensors="mlx"; normalize all tensor-like outputs.
        inputs = to_mlx(inputs)

        return {
            "pixel_values": inputs.get("pixel_values"),
            "input_ids": inputs["input_ids"],
            "attention_mask": inputs.get(
                "attention_mask", mx.ones_like(inputs["input_ids"])
            ),
            **{
                k: v
                for k, v in inputs.items()
                if k not in ["input_ids", "pixel_values", "attention_mask"]
            },
        }


class PreferenceVisionDataset:
    """Dataset for preference-based training (ORPO, DPO).

    Expected item keys: ``chosen``, ``rejected``, and optionally ``images`` / ``image``.
    Each of ``chosen`` / ``rejected`` can be:
    - a list of message dicts (processed via ``apply_chat_template``)
    - a plain string (encoded directly)
    """

    def __init__(self, hf_dataset, config, processor, image_resize_shape=None):
        self.dataset = hf_dataset
        self.processor = processor
        self.config = config
        self.image_resize_shape = image_resize_shape

    def __len__(self):
        if hasattr(self.dataset, "__len__"):
            return len(self.dataset)
        raise TypeError("Streaming dataset has no length")

    def __getitem__(self, idx):
        return self.process(self.dataset[idx])

    def process(self, item):
        from mlx_vlm.utils import prepare_inputs, process_inputs_with_fallback

        images = item.get("images", item.get("image", []))
        if not isinstance(images, list):
            images = [images] if images else []

        image_token_index = self.config.get("image_token_index") or self.config.get(
            "image_token_id"
        )
        if not image_token_index:
            raise ValueError(
                "Config must contain 'image_token_index' or 'image_token_id'"
            )

        num_images = len(images)

        model_type = self.config.get("model_type")

        result = {}
        for key in ("chosen", "rejected"):
            sequence = item[key]
            if isinstance(sequence, str):
                prompt = sequence
            else:
                prompt = apply_chat_template(
                    self.processor,
                    self.config,
                    sequence,
                    add_generation_prompt=False,
                    num_images=num_images,
                )
            if model_type in NATIVE_PREPROCESS_MODELS:
                try:
                    inputs = process_inputs_with_fallback(
                        processor=self.processor,
                        prompts=[prompt],
                        images=images if images else None,
                        audio=None,
                        add_special_tokens=False,
                    )
                    if "images" in inputs and "pixel_values" not in inputs:
                        inputs["pixel_values"] = inputs.pop("images")
                except Exception:
                    inputs = prepare_inputs(
                        processor=self.processor,
                        images=images if images else None,
                        audio=None,
                        prompts=[prompt],
                        image_token_index=image_token_index,
                        resize_shape=self.image_resize_shape,
                    )
            else:
                inputs = prepare_inputs(
                    processor=self.processor,
                    images=images if images else None,
                    audio=None,
                    prompts=[prompt],
                    image_token_index=image_token_index,
                    resize_shape=self.image_resize_shape,
                )

            inputs = to_mlx(inputs)

            result[f"{key}_input_ids"] = inputs["input_ids"]
            result[f"{key}_attention_mask"] = inputs.get(
                "attention_mask", mx.ones_like(inputs["input_ids"])
            )
            if inputs.get("pixel_values") is not None:
                result[f"{key}_pixel_values"] = inputs["pixel_values"]

        return result
