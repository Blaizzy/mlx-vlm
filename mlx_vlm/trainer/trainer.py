import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Union

import mlx.core as mx
import mlx.nn as nn
import numpy as np
from mlx.utils import tree_flatten
from PIL import Image


class ImageTextDataset:
    def __init__(self, image_dir, caption_file, img_size=(224, 224)):
        self.image_dir = image_dir
        self.img_size = img_size
        self.image_captions = []
        self.unique_captions = set()

        with open(caption_file, "r") as f:
            for line in f:
                image_name, caption = line.strip().split(",")
                self.image_captions.append((image_name, caption))
                self.unique_captions.add(caption)

        self.caption_to_id = {
            caption: i for i, caption in enumerate(self.unique_captions)
        }

    def __len__(self):
        return len(self.image_captions)

    def __getitem__(self, idx):
        image_name, caption = self.image_captions[idx]
        image_path = os.path.join(self.image_dir, image_name)

        image = Image.open(image_path).convert("RGB")
        image = image.resize(self.img_size)
        image_array = np.array(image).astype(np.float32) / 255.0

        caption_id = self.caption_to_id[caption]

        return mx.array(image_array), mx.array(caption_id, dtype=mx.int32)


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
    def __init__(self, model, optimizer, loss_fn):
        self.model = model
        self.optimizer = optimizer
        self.loss_fn = loss_fn

    def train_step(self, batch):
        images, labels = batch

        def loss_fn(model):
            logits = model(images)
            return self.loss_fn(logits, labels)

        loss, grads = mx.value_and_grad(loss_fn)(self.model)
        self.optimizer.update(self.model, grads)
        return loss

    @mx.compile
    def train_epoch(self, dataloader):
        total_loss = 0
        for batch in dataloader:
            loss = self.train_step(batch)
            total_loss += loss
        return total_loss / len(dataloader)

    def evaluate(self, dataloader):
        correct = total = 0
        for images, labels in dataloader:
            logits = self.model(images)
            predictions = mx.argmax(logits, axis=1)
            correct += mx.sum(predictions == labels)
            total += labels.size
        return correct / total


def save_adapter(
    model: nn.Module,
    adapter_file: Union[str, Path],
):
    flattened_tree = tree_flatten(model.trainable_parameters())
    mx.save_safetensors(str(adapter_file), dict(flattened_tree))
