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

        images = item["images"]
        conversations = item["messages"]
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
    image_field: str = "image"
):
    needs_message_transform = "messages" not in dataset.column_names
    needs_image_transform = "images" not in dataset.column_names and image_field in dataset.column_names
    
    if needs_message_transform:
        def transform_to_messages(example):
            messages = [
                {"role": "user", "content": example[promtp_field]},
                {"role": "assistant", "content": example[completion_field]}
            ]
            example["messages"] = messages
            return example
        dataset = dataset.map(transform_to_messages)

    if needs_image_transform:
        def rename_image_column(example):
            example["images"] = example[image_field]
            return example
        dataset = dataset.map(rename_image_column)

    if "messages" not in dataset.column_names:
        raise ValueError("Dataset must have a 'messages' column")
    if "images" not in dataset.column_names:
        raise ValueError("Dataset must have an 'images' column")

    if args.apply_chat_template:
        logger.info(f"\033[32mApplying chat template to the dataset\033[0m")
        def process_data(examples):
            if config["model_type"] == "pixtral":
                conversations = apply_chat_template(
                    config=config,
                    processor=processor,
                    prompt=examples["messages"],
                    return_messages=True,
                )
                examples["messages"] = [
                    json.dumps(item, ensure_ascii=False) for item in conversations
                ]
            else:
                examples["messages"] = apply_chat_template(
                    config=config,
                    processor=processor,
                    prompt=examples["messages"],
                    return_messages=True,
                )
            return examples

        dataset = dataset.map(process_data)
    
    return dataset


def load_and_prepare_dataset(
    config,
    args,
    processor,
    image_processor,
    path: str,
    split: str,
    image_resize_shape,
    type: str = "sft"
):
    logger.info(f"\033[32mLoading dataset from {args.dataset}\033[0m")
    loaded_dataset = load_dataset(args.dataset, split=args.split)

    logger.info(f"\033[32mPreparing and maping dataset\033[0m")
    prepared_dataset = prepare_dataset(
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
            image_resize_shape=args.image_resize_shape
        )
    else:
        raise ValueError("training type musst be 'sft',")