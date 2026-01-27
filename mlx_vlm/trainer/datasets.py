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
        image_resize_shape=None,
        train_on_completions=False,
        tokenizer=None,
        prompt_format=None,
        assistant_id="assistant",
    ):
        self.dataset = hf_dataset
        self.processor = processor
        self.config = config
        self.image_processor = image_processor
        self.image_resize_shape = image_resize_shape
        self.train_on_completions = train_on_completions
        self.tokenizer = tokenizer or getattr(processor, "tokenizer", None)
        self.prompt_format = prompt_format
        self.assistant_id = assistant_id

    def process(self, *args, **kwargs):
        """
        Backwards-compatible wrapper so callers can call dataset.process(...).
        Delegates to the underlying processor or tokenizer if available.
        """
        proc = getattr(self, "processor", None)
        if proc is None:
            raise AttributeError("Dataset has no 'processor' to delegate 'process' to.")

        if hasattr(proc, "process"):
            return proc.process(*args, **kwargs)

        tokenizer = getattr(proc, "tokenizer", None)
        if tokenizer and hasattr(tokenizer, "process"):
            return tokenizer.process(*args, **kwargs)

        raise AttributeError(
            "Processor object does not implement a 'process' method to delegate to."
        )

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        from mlx_vlm.utils import prepare_inputs

        def build_prompt_from_format(prompt_format, item, images):
            from mlx_vlm.lora import apply_prompt_format
            question = item.get("question") or item.get("input") or item.get("prompt")
            answer = item.get("answer") or item.get("output") or item.get("response")
            image = images[0] if images else None
            audio = item.get("audio")
            return apply_prompt_format(prompt_format, image, question, answer, audio)

        def extract_images(item):
            images = item.get("images", item.get("image", None))
            if images in (None, "", []):
                return []
            if not isinstance(images, list):
                return [images]
            return images

        def extract_conversations(item):
            return item.get("messages", item.get("conversations"))

        def flatten_conversations(conversations):
            if len(conversations) > 0 and isinstance(conversations[0], list):
                return [msg for conv in conversations for msg in conv]
            return conversations

        def get_last_assistant_message(conversations, assistant_id):
            flat = flatten_conversations(conversations) if isinstance(conversations, list) else []
            for msg in reversed(flat):
                if isinstance(msg, dict) and msg.get("role") == assistant_id:
                    return msg.get("content")
            return None

        item = self.dataset[idx]
        images = extract_images(item)
        conversations = extract_conversations(item)
        prompts = []

        if self.prompt_format is not None:
            prompts.append(build_prompt_from_format(self.prompt_format, item, images))
        elif isinstance(conversations, list) and isinstance(conversations[0], list):
            for conversation in conversations:
                if self.config["model_type"] == "pixtral":
                    conversation = [json.loads(i) for i in conversation]
                    if len(conversations) > 1:
                        warnings.warn("Pixtral batch processing is not supported yet. Set batch size to 1.")
                prompts.append(get_prompt(self.config["model_type"], self.processor, conversation))
        else:
            if self.config["model_type"] == "pixtral":
                conversations = [json.loads(i) for i in conversations]
            prompts.append(get_prompt(self.config["model_type"], self.processor, conversations))

        image_token_index = self.config.get("image_token_index") or self.config.get("image_token_id")
        if image_token_index is None:
            raise ValueError("Config must contain either 'image_token_index' or 'image_token_id'")

        inputs = prepare_inputs(
            processor=self.processor,
            images=images if images else None,
            audio=None,
            prompts=prompts,
            image_token_index=image_token_index,
            resize_shape=self.image_resize_shape,
        )
        input_ids = inputs.get("input_ids")
        pixel_values = inputs.get("pixel_values")
        mask = inputs.get("attention_mask") or mx.ones_like(input_ids)
        kwargs = {k: v for k, v in inputs.items() if k not in ["input_ids", "pixel_values", "attention_mask"]}

        if self.train_on_completions and self.tokenizer:
            if getattr(self, "prompt_key", None) is not None:
                prompt_text = item[self.prompt_key]
            else:
                completion = get_last_assistant_message(conversations, self.assistant_id)
                if completion is None:
                    raise ValueError(f"Could not find assistant message with role '{self.assistant_id}' in conversations/messages.")
                prompt_text = completion

            if self.prompt_format is not None:
                prompt_for_offset = build_prompt_from_format(self.prompt_format, item, images)
                if "chat_template" in self.processor.__dict__:
                    prompt_for_offset_str = self.processor.apply_chat_template(
                        prompt_for_offset, tokenize=False, add_generation_prompt=False)
                elif "tokenizer" in self.processor.__dict__:
                    prompt_for_offset_str = self.processor.tokenizer.apply_chat_template(
                        prompt_for_offset, tokenize=False, add_generation_prompt=False)
                else:
                    prompt_for_offset_str = str(prompt_for_offset)
                offset = len(prompt_for_offset_str)
            else:
                if self.config["model_type"] == "paligemma":
                    prompt_for_offset = prompt_text
                elif "chat_template" in self.processor.__dict__:
                    prompt_for_offset = self.processor.apply_chat_template(
                        [{"role": "user", "content": prompt_text}], tokenize=False, add_generation_prompt=False)
                elif "tokenizer" in self.processor.__dict__:
                    prompt_for_offset = self.processor.tokenizer.apply_chat_template(
                        [{"role": "user", "content": prompt_text}], tokenize=False, add_generation_prompt=False)
                else:
                    prompt_for_offset = prompt_text
                offset = len(prompt_for_offset)
            return (input_ids, offset)

        return {"pixel_values": pixel_values, "input_ids": input_ids, "attention_mask": mask, **kwargs}