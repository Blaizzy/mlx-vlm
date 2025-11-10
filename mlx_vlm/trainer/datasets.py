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
        
        # Use prepare_inputs to process everything together
        inputs = prepare_inputs(
            processor=self.processor,
            images=images,  # Pass images directly here
            audio=None,
            prompts=prompts,
            image_token_index=image_token_index,
            resize_shape=self.image_resize_shape,
        )
        
        # Extract what we need from inputs
        input_ids = inputs.get("input_ids")
        pixel_values = inputs.get("pixel_values")
        mask = inputs.get("attention_mask")
        
        # Get any additional kwargs
        kwargs = {
            k: v
            for k, v in inputs.items()
            if k not in ["input_ids", "pixel_values", "attention_mask"]
        }
        
        # Ensure mask exists
        if mask is None:
            mask = mx.ones_like(input_ids)
        
        return {
            "pixel_values": pixel_values,
            "input_ids": input_ids,
            "attention_mask": mask,
            **kwargs,
        }