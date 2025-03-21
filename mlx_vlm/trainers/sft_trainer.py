import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Union

import mlx.core as mx
import mlx.nn as nn
import numpy as np
from mlx.utils import tree_flatten, tree_map


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
    def __init__(
        self,
        model,
        optimizer,
        train_on_completions=False,
        assistant_id=77091,
        clip_gradients=None,
    ):
        self.model = model
        self.optimizer = optimizer
        self.train_on_completions = train_on_completions
        self.assistant_id = assistant_id
        self.clip_gradients = clip_gradients

    def loss_fn(self, model, batch):
        pixel_values = batch["pixel_values"]
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        lengths = mx.sum(attention_mask, axis=1)
        labels = input_ids[:, 1:]

        batch_size, seq_length = input_ids.shape

        if self.train_on_completions:
            weight_mask = mx.ones_like(attention_mask)

            assistant_response_index = np.where(input_ids == self.assistant_id)[1]
            range_matrix = mx.repeat(
                mx.expand_dims(mx.arange(seq_length), 0), batch_size, axis=0
            )
            assistant_mask = range_matrix <= mx.array(assistant_response_index).reshape(
                -1, 1
            )
            # Apply the mask to weight_mask
            weight_mask = mx.where(
                assistant_mask, mx.zeros_like(weight_mask), weight_mask
            )[:, 1:]
        else:
            weight_mask = None

        input_ids = input_ids[:, :-1]

        kwargs = {
            k: v
            for k, v in batch.items()
            if k not in ["input_ids", "pixel_values", "attention_mask"]
        }

        # Forward pass
        outputs = model(input_ids, pixel_values, attention_mask, **kwargs)

        # Cast to float32
        logits = outputs.logits.astype(mx.float32)

        # Ensure logits and labels have the same sequence length
        def align_logits_with_labels(logits, labels):
            if logits.shape[1] < labels.shape[1]:
                pad_length = labels.shape[1] - logits.shape[1]
                pad_width = ((0, 0), (0, pad_length), (0, 0))
                return mx.pad(logits, pad_width, mode="constant", constant_values=-100)
            elif logits.shape[1] > labels.shape[1]:
                return logits[:, -labels.shape[1] :, :]
            return logits

        logits = align_logits_with_labels(logits, labels)

        length_mask = mx.arange(input_ids.shape[1])[None, :] < lengths[:, None]

        # Compute loss only on non-padded tokens
        ce = (
            nn.losses.cross_entropy(
                logits,
                labels,
                weights=weight_mask,
            )
            * length_mask
        )
        ntoks = length_mask.sum()
        ce = ce.sum() / ntoks

        return ce

    def train_step(self, batch):
        loss_and_grad_fn = nn.value_and_grad(self.model, self.loss_fn)
        loss, grads = loss_and_grad_fn(self.model, batch)

        # Add gradient clipping
        if self.clip_gradients is not None:
            grads = tree_map(
                lambda g: mx.clip(g, -self.clip_gradients, self.clip_gradients), grads
            )

        # Update parameters based on whether we're using LoRA or full training
        if hasattr(self.model, 'trainable_parameters'):
            # LoRA mode - update only trainable parameters
            trainable_params = self.model.trainable_parameters(self.model)
            self.optimizer.update(trainable_params, grads)
        else:
            # Full weight training - update all parameters
            self.optimizer.update(self.model.parameters(), grads)

        return loss

    @mx.compile
    def train_epoch(self, dataloader):
        total_loss = 0
        for batch in dataloader:
            loss = self.train_step(batch)
            mx.eval(self.model, self.optimizer.state)
            total_loss += loss
        return total_loss / len(dataloader)
    
    @mx.compile
    def evaluate(self, dataloader, num_batches=-1):
        total_loss = 0
        batch_count = 0
        for i, batch in enumerate(dataloader):
            if num_batches > 0 and i >= num_batches:
                break
            loss = self.loss_fn(self.model, batch)
            total_loss += loss
            batch_count += 1
        
        return total_loss / batch_count if batch_count > 0 else 0


def save_adapter(
    model: nn.Module,
    adapter_file: Union[str, Path],
):
    path = Path(adapter_file)
    if hasattr(model.config, "lora"):
        with open(path.parent / "adapter_config.json", "w") as f:
            json.dump(model.config.lora, f)
    flattened_tree = tree_flatten(model.trainable_parameters())
    mx.save_safetensors(str(adapter_file), dict(flattened_tree))


def save_full_model(model: nn.Module, save_path: Union[str, Path]):
    path = Path(save_path)
    path.mkdir(exist_ok=True, parents=True)
    with open(path / "config.json", "w") as f:
        json.dump(model.config.to_dict(), f)
    mx.save_safetensors(str(path / "model.safetensors"), dict(tree_flatten(model.parameters())))