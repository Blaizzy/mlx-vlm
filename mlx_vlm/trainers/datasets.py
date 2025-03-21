from typing import Optional, Any
import warnings
import logging
import json

import mlx.core as mx

from datasets import load_dataset
from ..prompt_utils import apply_chat_template

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


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


class SFTDataset:
    def __init__(
        self,
        hf_dataset,
        config,
        processor,
        image_processor=None,
        take=None,
        split=None,
        image_resize_shape=None,
        images_key: str = "image",
        messages_key: str = "messages"
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
        self.images_key = images_key
        self.messages_key = messages_key

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        from mlx_vlm.utils import prepare_inputs

        item = self.dataset[idx]

        images = item[self.images_key]
        if isinstance(images, list) and len(images) > 0 and isinstance(images[0], list):
            images = [img for sublist in images for img in sublist]
        conversations = item[self.messages_key]
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

        image_token_index = self.config["image_token_index"]

        inputs = prepare_inputs(
            self.processor,
            images,
            prompts,
            image_token_index,
            self.image_resize_shape,
        )
        input_ids = inputs["input_ids"]
        pixel_values = inputs["pixel_values"]
        mask = inputs["attention_mask"]
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
    

def prepare_dataset(
    dataset,
    config,
    processor,
    args,
    promtp_field: str = "question",
    completion_field: str = "output",
    new_image_field: str = "image",
    messages_field: str = "messages"
):
    needs_message_transform = False
    if messages_field in dataset.column_names:
        messages_key = messages_field
    elif "conversations" in dataset.column_names:
        messages_key = "conversations"
    else:
        needs_message_transform = True

    needs_image_fiel_renaming = new_image_field in dataset.column_names and "images" not in dataset.column_names
    
    if needs_message_transform:
        def transform_to_messages(example):
            messages = [
                {"role": "user", "content": example[promtp_field]},
                {"role": "assistant", "content": example[completion_field]}
            ]
            example["messages"] = messages
            return example
        messages_key = messages_field
        dataset = dataset.map(transform_to_messages)

    if needs_image_fiel_renaming:
        image_field = new_image_field

    if "images" not in dataset.column_names and new_image_field not in dataset.column_names:
        raise ValueError("Dataset must have an 'images' column")
    
    first_example = dataset[0]
    first_message = first_example[messages_key][0] if len(first_example[messages_key]) > 0 else None
    needs_format_conversion = False
    if first_message and "from" in first_message and "value" in first_message and "role" not in first_message:
        needs_format_conversion = True
    if needs_format_conversion:
        logger.info(f"\033[32mConverting message format from 'from/value' to 'role/content'\033[0m")
        def transform_conversation_format(example):
            transformed_messages = []
            for msg in example[messages_key]:
                if "from" in msg and "value" in msg:
                    if msg["from"] == "human":
                        role = "user"
                    elif msg["from"] in ["assistant", "gpt"]:
                        role = "assistant"
                    elif msg["from"] == "system":
                        role = "system"
                    else:
                        role = msg["from"]
                    transformed_messages.append({"role": role, "content": msg["value"]})
                else:
                    transformed_messages.append(msg)
            
            example[messages_key] = transformed_messages
            return example
        dataset = dataset.map(transform_conversation_format)

    if args.apply_chat_template:
        logger.info(f"\033[32mApplying chat template to the dataset\033[0m")
        def process_data(examples):
            if config["model_type"] == "pixtral":
                conversations = apply_chat_template(
                    config=config,
                    processor=processor,
                    prompt=examples[messages_key],
                    return_messages=True,
                )
                examples[messages_key] = [
                    json.dumps(item, ensure_ascii=False) for item in conversations
                ]
            else:
                examples[messages_key] = apply_chat_template(
                    config=config,
                    processor=processor,
                    prompt=examples[messages_key],
                    return_messages=True,
                )
            return examples

        dataset = dataset.map(process_data)

    return dataset, messages_key, image_field


def load_and_prepare_dataset(
    config,
    args,
    processor,
    image_processor,
    image_resize_shape: Optional[Any] = None,
    path: Optional[str] = None,
    split: Optional[str] = None,
    type: str = "sft"
):
    logger.info(f"\033[32mLoading dataset from {args.dataset}\033[0m")
    loaded_dataset = load_dataset(args.dataset, name=args.dataset_config, split=args.split)

    logger.info(f"\033[32mPreparing and maping dataset\033[0m")
    prepared_dataset, messages_key, image_field = prepare_dataset(
        dataset=loaded_dataset,
        config=config,
        processor=processor,
        args=args
    )

    if type == "sft": # will be args.train_type later for dpo and so on
        return SFTDataset(
            prepared_dataset,
            config,
            processor,
            image_processor=image_processor,
            image_resize_shape=args.image_resize_shape,
            messages_key=messages_key,
            images_key=image_field
        )
    else:
        raise ValueError("training type musst be 'sft',")