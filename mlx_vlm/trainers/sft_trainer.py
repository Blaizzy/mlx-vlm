import time
from dataclasses import dataclass, field

import mlx.core as mx
import mlx.nn as nn
import numpy as np

from mlx.nn.utils import average_gradients
from mlx.utils import tree_map

from .utils import grad_checkpoint
from .callback import TrainingCallback

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


def default_loss(model, inputs, targets, lengths, pixel_values=None, mask=None):
    # Call the model and get the output object
    outputs = model(inputs, pixel_values=pixel_values, mask=mask)
    
    # Extract logits from the output object and remove the last token
    logits = outputs.logits[:, :-1].astype(mx.float32)

    # Adjust length mask to match target length
    length_mask = mx.arange(targets.shape[1])[None, :] < (lengths - 1)[:, None]

    ce = nn.losses.cross_entropy(logits, targets) * length_mask
    ntoks = length_mask.sum()
    ce = ce.sum() / ntoks

    return ce, ntoks


def loss_with_completions(
    model, inputs, targets, lengths, pixel_values=None, mask=None, 
    assistant_id=77091, train_on_completions=False
):
    batch_size, seq_length = inputs.shape
    
    # Adjust for the shifted target length
    target_seq_length = seq_length - 1
    
    if train_on_completions:
        weight_mask = mx.ones_like(mx.arange(target_seq_length)[None, :] < (lengths - 1)[:, None])
        
        assistant_response_index = np.where(inputs == assistant_id)[1]
        range_matrix = mx.repeat(
            mx.expand_dims(mx.arange(target_seq_length), 0), batch_size, axis=0
        )
        assistant_mask = range_matrix <= mx.array(assistant_response_index).reshape(
            -1, 1
        )
        # Apply the mask to weight_mask
        weight_mask = mx.where(
            assistant_mask, mx.zeros_like(weight_mask), weight_mask
        )
    else:
        weight_mask = None

    # Call the model and get the output object
    outputs = model(inputs, pixel_values=pixel_values, mask=mask)
    
    # Extract logits from the output object and remove the last token
    logits = outputs.logits[:, :-1].astype(mx.float32)

    # Adjust length mask to match target length
    length_mask = mx.arange(targets.shape[1])[None, :] < (lengths - 1)[:, None]

    ce = nn.losses.cross_entropy(logits, targets, weights=weight_mask) * length_mask
    ntoks = length_mask.sum()
    ce = ce.sum() / ntoks

    return ce, ntoks


def iterate_batches(
    dataset,
    processor,
    batch_size,
    max_seq_length,
    train=False,
):
    step = mx.distributed.init().size()
    if batch_size % step != 0:
        raise ValueError("The batch size must be divisible by the number of workers")
    
    local_batch_size = batch_size // step
    dataset_size = len(dataset)
    indices = list(range(dataset_size))

    while True:
        if train:
            np.random.shuffle(indices)
        
        for start_idx in range(0, dataset_size - local_batch_size + 1, local_batch_size):
            batch_indices = indices[start_idx:start_idx + local_batch_size]
            batch_samples = [dataset[idx] for idx in batch_indices]
            
            # Debug information to understand data shapes
            sample_shapes = []
            for sample in batch_samples:
                if isinstance(sample["input_ids"], mx.array):
                    input_shape = sample["input_ids"].shape
                else:
                    input_shape = np.asarray(sample["input_ids"]).shape
                sample_shapes.append(input_shape)
            
            # Convert all inputs to numpy arrays for consistent handling
            processed_samples = []
            for sample in batch_samples:
                if isinstance(sample["input_ids"], mx.array):
                    input_ids = sample["input_ids"].tolist()
                else:
                    input_ids = sample["input_ids"]
                
                # Handle case where input_ids might already be a batch
                if isinstance(input_ids, list) and len(input_ids) > 0:
                    if isinstance(input_ids[0], list):
                        # It's already a batch, flatten it
                        input_ids = input_ids[0]
                
                processed_sample = {
                    "input_ids": np.array(input_ids, dtype=np.int32),
                    "pixel_values": sample.get("pixel_values")
                }
                processed_samples.append(processed_sample)
            
            # Get max length for this batch
            max_length = min(
                max(len(sample["input_ids"]) for sample in processed_samples),
                max_seq_length
            )
            
            # Pad to multiple of 8
            pad_to = 8
            max_length_padded = pad_to * ((max_length + pad_to - 1) // pad_to)
            
            # Create properly sized batch arrays
            batch_input_ids = np.zeros((local_batch_size, max_length_padded), dtype=np.int32)
            batch_mask = np.zeros((local_batch_size, max_length_padded), dtype=np.int32)
            
            # Fill the batch arrays
            for i, sample in enumerate(processed_samples):
                seq_length = min(len(sample["input_ids"]), max_seq_length)
                batch_input_ids[i, :seq_length] = sample["input_ids"][:seq_length]
                batch_mask[i, :seq_length] = 1
            
            # Build the final batch dictionary
            batch_dict = {
                "input_ids": mx.array(batch_input_ids),
                "mask": mx.array(batch_mask),
                "lengths": mx.array([min(len(sample["input_ids"]), max_length_padded) for sample in processed_samples])
            }
            
            # Handle pixel values
            pixel_values = [sample.get("pixel_values") for sample in processed_samples]
            if all(p is not None for p in pixel_values):
                # Make sure pixel values are properly stacked
                if isinstance(pixel_values[0], mx.array):
                    batch_dict["pixel_values"] = mx.stack(pixel_values)
                else:
                    batch_dict["pixel_values"] = mx.array(np.stack(pixel_values))
            elif all(p is None for p in pixel_values):
                batch_dict["pixel_values"] = None
            else:
                raise ValueError("Mixed presence of pixel_values across samples; ensure consistency.")

            yield batch_dict
        
        if not train:
            break


def train_sft(
    model,
    processor,
    optimizer,
    dataset,
    args: TrainingArgs = TrainingArgs(),
    loss: callable = default_loss,
    training_callback: TrainingCallback = None,
    clip_gradients=None,
    train_on_completions=False,
    assistant_id=77091
):
    dataset_iterator = iterate_batches(
        dataset,
        processor,
        args.batch_size,
        args.max_seq_length,
        train=True
    )

    world = mx.distributed.init()
    world_size = world.size()
    rank = world.rank()
    if world_size > 1:
        print(f"Node {rank} of {world_size}")

    if args.grad_checkpoint:
        grad_checkpoint(model.layers[0])

    state = [model.state, optimizer.state]

    def step(batch):
        # Derive lengths from attention mask
        lengths = batch["mask"].sum(axis=1)
        
        # Get targets (shifted input_ids)
        targets = batch["input_ids"][:, 1:]
        
        if train_on_completions:
            (lvalue, toks), grad = loss_value_and_grad(
                model,
                batch["input_ids"],
                targets,
                lengths,
                assistant_id=assistant_id, 
                train_on_completions=True
            )
        else:
            (lvalue, toks), grad = loss_value_and_grad(
                model,
                batch["input_ids"],
                targets,
                lengths
            )

        if clip_gradients is not None:
            grad = tree_map(
                lambda g: mx.clip(g, -clip_gradients, clip_gradients), grad
            )

        grad = average_gradients(grad)
        optimizer.update(model, grad)

        return lvalue, toks

    if train_on_completions:
        loss_value_and_grad = nn.value_and_grad(model, loss_with_completions)
    else:
        loss_value_and_grad = nn.value_and_grad(model, loss)

    losses = 0
    n_tokens = 0
    steps = 0
    trained_tokens = 0
    train_time = 0
    
    # Keep track of batches for training
    batch_count = 0
    
    # Main training loop
    for it in range(1, args.iters + 1):
        tic = time.perf_counter()
        
        # Get next batch - handle dataset exhaustion by wrapping back to start
        try:
            batch = next(dataset_iterator)
        except StopIteration:
            # Reset the iterator
            dataset_iterator = iter(dataset_iterator)
            batch = next(dataset_iterator)
        
        # Training step
        lvalue, toks = step(batch)
        losses += lvalue
        n_tokens += toks
        steps += 1
        mx.eval(state, losses, n_tokens)
        train_time += time.perf_counter() - tic
        batch_count += 1

        # Rest of the reporting and saving logic remains the same
        if it % args.steps_per_report == 0 or it == args.iters:
            train_loss = mx.distributed.all_sum(losses, stream=mx.cpu).item()
            train_loss /= steps * mx.distributed.init().size()
            n_tokens = mx.distributed.all_sum(n_tokens, stream=mx.cpu).item()
            learning_rate = optimizer.learning_rate.item()
            it_sec = args.steps_per_report / train_time
            tokens_sec = float(n_tokens) / train_time
            trained_tokens += n_tokens
            peak_mem = mx.get_peak_memory() / 1e9
            if training_callback is not None:
                train_info = {
                    "iteration": it,
                    "train_loss": train_loss,
                    "learning_rate": learning_rate,
                    "iterations_per_second": it_sec,
                    "tokens_per_second": tokens_sec,
                    "trained_tokens": trained_tokens,
                    "peak_memory": peak_mem,
                }
                training_callback.on_train_loss_report(train_info)

            losses = 0
            n_tokens = 0
            steps = 0
            train_time = 0