from typing import Optional, Any
import warnings
import logging
import json

from datasets import load_dataset
from ..prompt_utils import apply_chat_template

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def get_prompt(model_type, processor, conversation):
    if model_type == "paligemma":
        return conversation

    if hasattr(processor, "apply_chat_template"):
        return processor.apply_chat_template(
            conversation,
            tokenize=False,
            add_generation_prompt=False,
        )
    elif hasattr(processor, "tokenizer"):
        return processor.tokenizer.apply_chat_template(
            conversation,
            tokenize=False,
            add_generation_prompt=False,
        )
    else:
        raise ValueError("Processor has no chat template method available.")


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
        self.dataset = hf_dataset[split] if split is not None else hf_dataset
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
        item = self.dataset[idx]

        # Flatten image structure if needed
        images = item[self.images_key]
        if isinstance(images, list) and images and isinstance(images[0], list):
            images = [img for sublist in images for img in sublist]

        # Handle conversation format
        conversations = item[self.messages_key]
        prompts = []

        # For Pixtral, we decode JSON strings
        if self.config["model_type"] == "pixtral":
            if isinstance(conversations[0], str):
                conversations = [json.loads(c) for c in conversations]
            else:
                conversations = [conversations]

        # Single conversation (ideal)
        if isinstance(conversations, list) and isinstance(conversations[0], dict):
            prompt = get_prompt(self.config["model_type"], self.processor, conversations)
            prompts.append(prompt)

        # Multiple conversations per sample (not typical, but handled)
        elif isinstance(conversations, list) and isinstance(conversations[0], list):
            warnings.warn(
                "Multiple conversations detected in one sample — assuming batch size = 1"
            )
            for convo in conversations:
                prompt = get_prompt(self.config["model_type"], self.processor, convo)
                prompts.append(prompt)

        else:
            raise ValueError(f"Unexpected format in 'messages': {type(conversations)}")

        # Prepare model inputs
        try:
            from ..utils import prepare_inputs
            inputs = prepare_inputs(
                self.processor,
                images,
                prompts,
                self.config["image_token_index"],
                self.image_resize_shape,
            )

            input_ids = inputs["input_ids"]
            pixel_values = inputs["pixel_values"]
            attention_mask = inputs["attention_mask"]

            other_inputs = {
                k: v for k, v in inputs.items()
                if k not in ["input_ids", "pixel_values", "attention_mask"]
            }

            return {
                "pixel_values": pixel_values,
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                **other_inputs,
            }

        except Exception as e:
            logger.warning(f"Skipping sample at index {idx} due to error: {e}")
            return self.__getitem__((idx + 1) % len(self))


class GRPODataset:
    """
    Dataset wrapper for GRPO training data with VLM support.
    Each example must have 'prompt', 'answer', and 'image' fields.
    Returns model-ready inputs, including pixel values and tokenized sequences.
    """
    def __init__(
        self,
        hf_dataset,
        config,
        processor,
        image_processor=None,
        take=None,
        split=None,
        image_resize_shape=None,
        images_key: str = "images",
        prompt_key: str = "problem",
        answer_key: str = "answer",
        system_key: str = "system",
        system_prompt: str = None,
        type_key: str = "type",
        use_chat_template: bool = False,
        use_prompt: bool = False
    ):
        self.dataset = hf_dataset[split] if split is not None else hf_dataset
        if take is not None:
            self.dataset = self.dataset.take(take)

        self.processor = processor
        self.config = config
        self.image_processor = image_processor
        self.image_resize_shape = image_resize_shape
        self.images_key = images_key
        self.prompt_key = prompt_key
        self.answer_key = answer_key
        self.system_key = system_key
        self.type_key = type_key
        self.use_chat_template = use_chat_template
        self.use_prompt = use_prompt
        self.system_prompt = system_prompt

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]

        image = item[self.images_key]
        prompt_str = str(item[self.prompt_key])
        answer_str = str(item[self.answer_key])
        type_info = item.get(self.type_key, None)

        if self.use_chat_template:
            default_system_str = (
                "A conversation between User and Assistant. The user asks a question, and the Assistant solves it. "
                "The assistant first thinks about the reasoning process in the mind and then provides the user with the answer. "
                "The reasoning process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, "
                "i.e., <think> reasoning process here </think><answer> answer here </answer>."
            )
            system_str = self.system_prompt or item.get(self.system_key, default_system_str)
            full_prompt = [
                {'role': 'system', 'content': system_str},
                {'role': 'user', 'content': prompt_str}
            ]
            prompt_tokens = self.processor.apply_chat_template(
                full_prompt,
                add_generation_prompt=True
            )
            answer_tokens = self.processor.tokenizer.encode(answer_str)
        else:
            if self.use_prompt:
                full_prompt_str = (
                    "A conversation between User and Assistant. The user asks a question, and the Assistant solves it. "
                    "The assistant first thinks about the reasoning process in the mind and then provides the user with the answer. "
                    "The reasoning process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, "
                    f"i.e., <think> reasoning process here </think><answer> answer here </answer>. User: {prompt_str} Assistant: "
                )
                prompt_tokens = self.processor.tokenizer.encode(full_prompt_str)
            else:
                prompt_tokens = self.processor.tokenizer.encode(prompt_str)
            answer_tokens = self.processor.tokenizer.encode(answer_str)

        from ..utils import prepare_inputs
        prompt_text = self.processor.tokenizer.decode(prompt_tokens)
        answer_text = self.processor.tokenizer.decode(answer_tokens)
        inputs = prepare_inputs(
            self.processor,
            image,
            [prompt_text + answer_text],
            self.config["image_token_index"],
            self.image_resize_shape,
        )

        input_ids = inputs["input_ids"]
        pixel_values = inputs["pixel_values"]
        attention_mask = inputs["attention_mask"]
        other_inputs = {k: v for k, v in inputs.items() if k not in ["input_ids", "pixel_values", "attention_mask"]}

        return {
            "input_ids": input_ids,
            "answer_ids": answer_tokens,
            "pixel_values": pixel_values,
            "attention_mask": attention_mask,
            "prompt_str": prompt_str,
            "answer_str": answer_str,
            "type": type_info,
            **other_inputs
        }


def prepare_dataset(
    dataset,
    config,
    processor,
    args,
    prompt_field: str = "question",
    completion_field: str = "output",
    new_image_field: str = "images",
    messages_field: str = "messages"
):
    needs_message_transform = False

    if messages_field in dataset.column_names:
        messages_key = messages_field
    elif "conversations" in dataset.column_names:
        messages_key = "conversations"
    else:
        needs_message_transform = True

    if needs_message_transform:
        def transform_to_messages(example):
            messages = [
                {"role": "user", "content": example[prompt_field]},
                {"role": "assistant", "content": example[completion_field]}
            ]
            example["messages"] = messages
            return example

        messages_key = messages_field
        dataset = dataset.map(transform_to_messages)

    if new_image_field not in dataset.column_names:
        raise ValueError(f"Dataset must have a '{new_image_field}' column")

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

    return dataset, messages_key, new_image_field


def load_and_prepare_dataset(
    config,
    args,
    processor,
    image_processor,
    image_resize_shape: Optional[Any] = None,
    path: Optional[str] = None,
    split: Optional[str] = None,
):
    logger.info(f"\033[32mLoading dataset from {args.dataset}\033[0m")
    loaded_dataset = load_dataset(args.dataset, name=args.dataset_config, split=args.split)

    if args.train_mode == "sft":
        logger.info(f"\033[32mPreparing and maping dataset\033[0m")
        prepared_dataset, messages_key, image_field = prepare_dataset(
            prompt_field = "question",
            completion_field = "output",
            new_image_field = "image",
            messages_field = "messages",
            dataset=loaded_dataset,
            config=config,
            processor=processor,
            args=args
        )
        return SFTDataset(
            prepared_dataset,
            config,
            processor,
            image_processor=image_processor,
            image_resize_shape=args.image_resize_shape,
            messages_key=messages_key,
            images_key=image_field
        )
    elif args.train_mode == "grpo":
        logger.info(f"\033[32mPreparing and maping dataset\033[0m")
        prepared_dataset, messages_key, image_field = prepare_dataset(
            prompt_field = "problem",
            completion_field = "answer",
            new_image_field = "images",
            messages_field = "messages",
            dataset=loaded_dataset,
            config=config,
            processor=processor,
            args=args
        )
        return GRPODataset(
            prepared_dataset,
            config,
            processor,
            image_processor=image_processor,
            image_resize_shape=args.image_resize_shape,
            # messages_key=messages_key,
            images_key=image_field
        )
    else:
        raise ValueError("training type musst be 'sft',")