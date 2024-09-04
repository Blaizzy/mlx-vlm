import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from pprint import pprint
from typing import Union

import mlx.core as mx
import mlx.nn as nn
import numpy as np
from mlx.utils import tree_flatten
from PIL import Image

from mlx_vlm.prompt_utils import get_message_json
from mlx_vlm.utils import prepare_inputs


class Dataset:
    def __init__(
        self,
        hf_dataset,
        config,
        processor,
        image_processor=None,
        take=None,
        split="train",
    ):
        self.dataset = hf_dataset[split]
        if take is not None:
            self.dataset = self.dataset.take(take)
        self.processor = processor
        self.config = config
        self.image_processor = image_processor

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]

        # Process image data
        image = item["image"]

        conversations = item["conversations"]
        # check if conversation is a list of list
        if isinstance(conversations, list) and isinstance(conversations[0], list):
            prompts = []
            for conversation in conversations:
                if "chat_template" in self.processor.__dict__.keys():
                    prompts.append(
                        self.processor.apply_chat_template(conversation, tokenize=False)
                    )

                elif "tokenizer" in self.processor.__dict__.keys():
                    if self.config["model_type"] != "paligemma":
                        prompts.append(
                            self.processor.tokenizer.apply_chat_template(
                                conversation, tokenize=False
                            )
                        )
                else:
                    raise ValueError(
                        "Processor does not have 'chat_template' or 'tokenizer' attribute."
                    )

        else:
            if "chat_template" in self.processor.__dict__.keys():
                prompts = self.processor.apply_chat_template(
                    conversations, tokenize=False
                )

            elif "tokenizer" in self.processor.__dict__.keys():
                if self.config["model_type"] != "paligemma":
                    prompts = self.processor.tokenizer.apply_chat_template(
                        conversations, tokenize=False
                    )
            else:
                raise ValueError(
                    "Processor does not have 'chat_template' or 'tokenizer' attribute."
                )

        image_token_index = self.config["image_token_index"]
        input_ids, pixel_values, mask = prepare_inputs(
            self.image_processor, self.processor, image, prompts, image_token_index
        )

        if mask is None:
            mask = mx.ones_like(input_ids)

        return {
            "pixel_values": pixel_values,
            "input_ids": input_ids,
            "attention_mask": mask,
        }


def grad_checkpoint(layer):
    """
    Update all instances of type(layer) to use gradient checkpointing.
    """
    fn = type(layer).__call__

    def checkpointed_fn(model, *args, **kwargs):
        def inner_fn(params, *args, **kwargs):
            model.update(params)
            return fn(model, *args, **kwargs)

        return mx.checkpoint(inner_fn)(model.trainable_parameters(), *args, **kwargs)

    type(layer).__call__ = checkpointed_fn


@dataclass
class TrainingArgs:
    batch_size: int = field(default=4, metadata={"help": "Minibatch size."})
    iters: int = field(default=100, metadata={"help": "Iterations to train for."})
    val_batches: int = field(
        default=25,
        metadata={
            "help": "Number of validation batches, -1 uses the entire validation set."
        },
    )
    steps_per_report: int = field(
        default=10,
        metadata={"help": "Number of training steps between loss reporting."},
    )
    steps_per_eval: int = field(
        default=200, metadata={"help": "Number of training steps between validations."}
    )
    steps_per_save: int = field(
        default=100, metadata={"help": "Save the model every number steps"}
    )
    max_seq_length: int = field(
        default=2048, metadata={"help": "Maximum sequence length."}
    )
    adapter_file: str = field(
        default="adapters.safetensors",
        metadata={"help": "Save/load path for the trained adapter weights."},
    )
    grad_checkpoint: bool = field(
        default=False,
        metadata={"help": "Use gradient checkpointing to reduce memory use."},
    )


def default_loss(model, inputs, targets, lengths):
    logits = model(inputs)
    logits = logits.astype(mx.float32)

    length_mask = mx.arange(inputs.shape[1])[None, :] < lengths[:, None]

    ce = nn.losses.cross_entropy(logits, targets) * length_mask
    ntoks = length_mask.sum()
    ce = ce.sum() / ntoks

    return ce, ntoks


class Trainer:
    def __init__(self, model, optimizer):
        self.model = model
        self.optimizer = optimizer

    def loss_fn(self, model, batch):
        pixel_values = batch["pixel_values"]
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        labels = mx.where(
            attention_mask == 1, input_ids, -100
        )  # Only compute loss on non-padded tokens

        logits = model(input_ids, pixel_values, attention_mask)

        # Ensure logits and labels have the same sequence length
        min_length = min(logits.shape[1], labels.shape[1])
        logits = logits[:, :min_length, :]
        labels = labels[:, :min_length]
        attention_mask = attention_mask[:, :min_length]

        # Shift logits and labels for next-token prediction
        shift_logits = logits[:, :-1, :]
        shift_labels = labels[:, 1:]
        shift_attention_mask = attention_mask[:, 1:]

        # Flatten the tensors
        flat_logits = shift_logits.reshape(-1, shift_logits.shape[-1])
        flat_labels = shift_labels.reshape(-1)
        flat_attention_mask = shift_attention_mask.reshape(-1)

        # Compute loss only on non-padded tokens
        ce = nn.losses.cross_entropy(flat_logits, flat_labels, reduction="none")
        ce = (ce * flat_attention_mask).sum() / flat_attention_mask.sum()

        return ce

    def train_step(self, batch):
        loss_and_grad_fn = nn.value_and_grad(self.model, self.loss_fn)
        loss, grads = loss_and_grad_fn(self.model, batch)
        self.optimizer.update(self.model, grads)
        return loss

    @mx.compile
    def train_epoch(self, dataloader):
        total_loss = 0
        for batch in dataloader:
            loss = self.train_step(batch)
            total_loss += loss
        return total_loss / len(dataloader)


def save_adapter(
    model: nn.Module,
    adapter_file: Union[str, Path],
):
    flattened_tree = tree_flatten(model.trainable_parameters())
    mx.save_safetensors(str(adapter_file), dict(flattened_tree))
