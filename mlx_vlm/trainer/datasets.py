import json
import warnings

import mlx.core as mx
import numpy as np


def get_prompt(model_type, processor, conversation):
    if model_type == "paligemma":
        return conversation

    tokenizer = getattr(processor, "tokenizer", processor)
    chat_template = getattr(tokenizer, "chat_template", None)

    apply_fn = (
        tokenizer.apply_chat_template
        if chat_template
        else getattr(processor, "apply_chat_template", None)
    )

    if apply_fn is None:
        raise ValueError("Processor/Tokenizer has no apply_chat_template method.")

    return apply_fn(
        conversation,
        tokenize=False,
        add_generation_prompt=False,
    )


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
        from mlx_vlm.utils import prepare_inputs

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

        prompts = []
        # Format prompt based on model type
        if isinstance(conversations, list) and isinstance(conversations[0], list):
            for conversation in conversations:
                if model_type == "pixtral":
                    conversation = [json.loads(i) for i in conversation]
                    if len(conversations) > 1:
                        warnings.warn(
                            "Pixtral batch processing is not supported yet. Set batch size to 1."
                        )

                prompt = get_prompt(model_type, self.processor, conversation)
                prompts.append(prompt)

        else:
            if model_type == "pixtral":
                conversations = [json.loads(i) for i in conversations]
            prompt = get_prompt(model_type, self.processor, conversations)
            prompts.append(prompt)

        # Prepare inputs
        image_token_index = self.config.get("image_token_index") or self.config.get(
            "image_token_id"
        )
        if not image_token_index:
            raise ValueError(
                "Config must contain 'image_token_index' or 'image_token_id'"
            )

        use_embedded_images = (
            model_type.startswith("gemma")
            or model_type.startswith("qwen")
            or model_type == "smolvlm"
        )

        if use_embedded_images and images:
            # For models that embed images inline (Qwen, Gemma, SmolVLM):
            # 1. Tokenize text (contains raw image placeholder tokens)
            # 2. Process images separately to get pixel_values
            # 3. Expand placeholder tokens to match vision encoder output
            from mlx_vlm.utils import process_image

            tokenizer = getattr(self.processor, "tokenizer", self.processor)
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token

            encoded = tokenizer(
                prompts,
                padding=True,
                return_tensors="np",
                add_special_tokens=False,
            )
            input_ids = mx.array(encoded["input_ids"])
            attention_mask = mx.array(encoded["attention_mask"])

            # Squeeze batch dimension (iterate_batches expects 1D)
            if input_ids.ndim == 2 and input_ids.shape[0] == 1:
                input_ids = input_ids.squeeze(0)
            if attention_mask.ndim == 2 and attention_mask.shape[0] == 1:
                attention_mask = attention_mask.squeeze(0)

            image_processor = getattr(self.processor, "image_processor", None)
            processed_images = [
                process_image(img, self.image_resize_shape, image_processor)
                for img in images
            ]

            img_inputs = image_processor(images=processed_images, return_tensors="np")

            result = {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
            }

            if "pixel_values" in img_inputs:
                result["pixel_values"] = mx.array(img_inputs["pixel_values"])

            if "image_grid_thw" in img_inputs:
                thw = np.array(img_inputs["image_grid_thw"])
                result["image_grid_thw"] = mx.array(thw)

                # Expand image placeholder tokens to match vision encoder output.
                # The vision encoder produces (prod(thw) // merge_size^2) features
                # per image, but the tokenized text only has a few placeholder tokens.
                vision_config = getattr(self.config, "vision_config", None)
                if vision_config is None and isinstance(self.config, dict):
                    vision_config = self.config.get("vision_config", {})
                merge_size = getattr(vision_config, "spatial_merge_size", None)
                if merge_size is None and isinstance(vision_config, dict):
                    merge_size = vision_config.get("spatial_merge_size", 2)
                if merge_size is None:
                    merge_size = 2

                n_features = 0
                for row in thw:
                    n_features += int(np.prod(row)) // (merge_size**2)

                ids_np = np.array(input_ids.tolist()).flatten()
                n_placeholder = int((ids_np == image_token_index).sum())

                if n_placeholder > 0 and n_placeholder != n_features:
                    new_ids = []
                    expanded = False
                    for tok in ids_np:
                        if tok == image_token_index:
                            if not expanded:
                                new_ids.extend([image_token_index] * n_features)
                                expanded = True
                        else:
                            new_ids.append(int(tok))
                    result["input_ids"] = mx.array(new_ids)
                    result["attention_mask"] = mx.ones(len(new_ids), dtype=mx.int32)

            return result
        else:
            inputs = prepare_inputs(
                processor=self.processor,
                images=None if use_embedded_images else (images if images else None),
                audio=audio if audio else None,
                prompts=prompts,
                image_token_index=image_token_index,
                resize_shape=self.image_resize_shape,
            )

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
        from mlx_vlm.utils import prepare_inputs

        images = item.get("images", item.get("image", []))
        if not isinstance(images, list):
            images = [images] if images else []

        model_type = self.config.get("model_type")

        image_token_index = self.config.get("image_token_index") or self.config.get(
            "image_token_id"
        )
        if not image_token_index:
            raise ValueError(
                "Config must contain 'image_token_index' or 'image_token_id'"
            )

        use_embedded_images = (
            model_type.startswith("gemma")
            or model_type.startswith("qwen")
            or model_type == "smolvlm"
        )
        # Pass images for all models — embedded-image models need pixel_values
        # for the vision encoder to produce actual features during training
        images_for_inputs = images if images else None

        result = {}
        for key in ("chosen", "rejected"):
            sequence = item[key]
            prompt = (
                sequence
                if isinstance(sequence, str)
                else get_prompt(model_type, self.processor, sequence)
            )
            inputs = prepare_inputs(
                processor=self.processor,
                images=images_for_inputs,
                audio=None,
                prompts=[prompt],
                image_token_index=image_token_index,
                resize_shape=self.image_resize_shape,
            )
            result[f"{key}_input_ids"] = inputs["input_ids"]
            result[f"{key}_attention_mask"] = inputs.get(
                "attention_mask", mx.ones_like(inputs["input_ids"])
            )
            if inputs.get("pixel_values") is not None:
                result[f"{key}_pixel_values"] = inputs["pixel_values"]

        return result
