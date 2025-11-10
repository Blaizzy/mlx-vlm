import warnings
import json
import mlx.core as mx


def get_prompt(model_type, processor, conversation):
    if model_type == "paligemma":
        return conversation

    if "chat_template" in processor.__dict__.keys():
        prompt = processor.apply_chat_template(
            conversation,
            tokenize=False,
            add_generation_prompt=False,
        )
    elif "tokenizer" in processor.__dict__.keys():
        prompt = processor.tokenizer.apply_chat_template(
            conversation,
            tokenize=False,
            add_generation_prompt=False,
        )

    return prompt


class Dataset:
    def __init__(
        self,
        hf_dataset,
        config,
        processor,
        image_processor=None,
        take=None,
        split=None,
        image_resize_shape=None,
    ):
        if split is not None:
            self.dataset = hf_dataset[split]
        else:
            self.dataset = hf_dataset
        if take is not None:
            self.dataset = self.dataset.take(take)
        self.processor = processor
        self.config = config
        self.image_processor = image_processor
        self.image_resize_shape = image_resize_shape

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        from mlx_vlm.utils import prepare_inputs

        item = self.dataset[idx]

        images = item.get("images", item.get("image", None))
        conversations = item.get("messages", item.get("conversations"))
        if images in (None, "", []):
            images = []
        elif not isinstance(images, list):
            images = [images]
        prompts = []

        if isinstance(conversations, list) and isinstance(conversations[0], list):
            for conversation in conversations:
                if self.config["model_type"] == "pixtral":
                    conversation = [json.loads(i) for i in conversation]
                    if len(conversations) > 1:
                        warnings.warn(
                            "Pixtral batch processing is not supported yet. Set batch size to 1."
                        )

                prompt = get_prompt(
                    self.config["model_type"], self.processor, conversation
                )
                prompts.append(prompt)

        else:
            if self.config["model_type"] == "pixtral":
                conversations = [json.loads(i) for i in conversations]
            prompt = get_prompt(
                self.config["model_type"], self.processor, conversations
            )
            prompts.append(prompt)

        image_token_index = self.config.get("image_token_index") or self.config.get("image_token_id")
        if image_token_index is None:
            raise ValueError("Config must contain either 'image_token_index' or 'image_token_id'")
        
        # Separate image processing from text processing
        if self.image_processor is not None and images:
            pixel_values = self.image_processor(images=images, return_tensors="np")["pixel_values"]
            pixel_values = mx.array(pixel_values)
        else:
            pixel_values = None

        inputs = prepare_inputs(
            processor=self.processor,
            images=images,
            audio=None,
            prompts=prompts,
            image_token_index=image_token_index,
            resize_shape=self.image_resize_shape,
        )
        text_inputs = self.processor(text=prompts, return_tensors="np", padding=True)
        input_ids = mx.array(text_inputs["input_ids"])
        mask = mx.array(text_inputs.get("attention_mask", mx.ones_like(input_ids)))
        kwargs = {
            k: v
            for k, v in inputs.items()
            if k not in ["input_ids", "pixel_values", "attention_mask"]
        }

        if mask is None:
            mask = mx.ones_like(input_ids)

        return {
            "pixel_values": pixel_values,
            "input_ids": input_ids,
            "attention_mask": mask,
            **kwargs,
        }