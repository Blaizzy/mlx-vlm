import json
import warnings

import mlx.core as mx
import numpy as np

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
        train_on_completions=False,
    ):
        self.dataset = hf_dataset
        self.processor = processor
        self.config = config
        self.image_resize_shape = image_resize_shape
        self.train_on_completions = train_on_completions

    def _token_length(self, prompt, images, audio, image_token_index):
        from mlx_vlm.utils import prepare_inputs, process_inputs_with_fallback

        model_type = self.config.get("model_type")
        inputs = None
        if model_type in NATIVE_PREPROCESS_MODELS:
            try:
                inputs = process_inputs_with_fallback(
                    processor=self.processor,
                    prompts=[prompt],
                    images=images if images else None,
                    audio=audio if audio else None,
                    add_special_tokens=False,
                )
                if "images" in inputs and "pixel_values" not in inputs:
                    inputs["pixel_values"] = inputs.pop("images")
            except Exception:
                pass

        if inputs is None:
            inputs = prepare_inputs(
                processor=self.processor,
                images=images if images else None,
                audio=audio if audio else None,
                prompts=[prompt],
                image_token_index=image_token_index,
                resize_shape=self.image_resize_shape,
            )

        return np.array(inputs["input_ids"]).reshape(-1).shape[0]

    def _completion_prefix(self, conversation, num_images, num_audios):
        if not isinstance(conversation, list) or not conversation:
            return None

        last = conversation[-1]
        if not isinstance(last, dict) or last.get("role") != "assistant":
            return None

        return apply_chat_template(
            self.processor,
            self.config,
            conversation[:-1],
            add_generation_prompt=True,
            num_images=num_images,
            num_audios=num_audios,
        )

    def __len__(self):
        if hasattr(self.dataset, "__len__"):
            return len(self.dataset)
        raise TypeError("Streaming dataset has no length")

    def __getitem__(self, idx):
        return self.process(self.dataset[idx])

    def process(self, item):
        """Process a single item from the dataset"""
        from mlx_vlm.utils import prepare_inputs, process_inputs_with_fallback

        images = item.get("images", item.get("image", []))
        if not isinstance(images, list):
            images = [images] if images else []

        audio = item.get("audio", item.get("audios", []))
        if not isinstance(audio, list):
            audio = [audio] if audio else []

        conversations = item.get("messages", item.get("conversations"))

        model_type = self.config.get("model_type")

        num_images = len(images)
        num_audios = len(audio)

        prompts = []
        completion_prefixes = [] if self.train_on_completions else None
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
                if self.train_on_completions:
                    completion_prefixes.append(
                        self._completion_prefix(conversation, num_images, num_audios)
                    )

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
            if self.train_on_completions:
                completion_prefixes.append(
                    self._completion_prefix(conversations, num_images, num_audios)
                )

        image_token_index = self.config.get("image_token_index") or self.config.get(
            "image_token_id"
        )
        if not image_token_index:
            raise ValueError(
                "Config must contain 'image_token_index' or 'image_token_id'"
            )

        inputs = None
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
                pass

        if inputs is None:
            inputs = prepare_inputs(
                processor=self.processor,
                images=images if images else None,
                audio=audio if audio else None,
                prompts=prompts,
                image_token_index=image_token_index,
                resize_shape=self.image_resize_shape,
            )

        inputs = to_mlx(inputs)
        completion_mask = None
        if completion_prefixes and any(
            prefix is not None for prefix in completion_prefixes
        ):
            rows = []
            for row_idx, prefix in enumerate(completion_prefixes):
                row = mx.zeros_like(inputs["input_ids"][row_idx])
                if prefix is not None:
                    prefix_len = self._token_length(
                        prefix, images, audio, image_token_index
                    )
                    row = mx.where(mx.arange(row.shape[0]) >= prefix_len, 1, row)
                rows.append(row)
            completion_mask = mx.stack(rows)

        return {
            "pixel_values": inputs.get("pixel_values"),
            "input_ids": inputs["input_ids"],
            "attention_mask": inputs.get(
                "attention_mask", mx.ones_like(inputs["input_ids"])
            ),
            **(
                {"completion_mask": completion_mask}
                if completion_mask is not None
                else {}
            ),
            **{
                k: v
                for k, v in inputs.items()
                if k
                not in [
                    "input_ids",
                    "pixel_values",
                    "attention_mask",
                    "completion_mask",
                ]
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
            inputs = None
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
                    pass

            if inputs is None:
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
