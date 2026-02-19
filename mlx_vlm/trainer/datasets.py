import json
import warnings

import mlx.core as mx


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
        image_processor=None,
        image_resize_shape=None,
    ):
        self.dataset = hf_dataset
        self.processor = processor
        self.config = config
        self.image_processor = image_processor
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
