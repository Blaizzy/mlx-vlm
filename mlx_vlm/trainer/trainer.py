# Copyright © 2024 MLX-VLM

import json
import time
from dataclasses import dataclass, field
from functools import partial
from pathlib import Path
from typing import Union

import mlx.core as mx
import mlx.nn as nn
import numpy as np
from mlx.nn.utils import average_gradients
from mlx.utils import tree_flatten, tree_map
from tqdm import tqdm

from .utils import grad_checkpoint, Colors, get_learning_rate

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
    learning_rate: float = field(
        default=1e-5,
        metadata={"help": "Learning rate."},
    )
    grad_clip: float = field(
        default=1.0,
        metadata={"help": "Gradient clipping value."},
    )
    warmup_steps: int = field(
        default=100,
        metadata={"help": "Number of warmup steps for learning rate."},
    )
    min_learning_rate: float = field(
        default=1e-6,
        metadata={"help": "Minimum learning rate after decay."},
    )


def default_loss(model, inputs, targets, lengths, train_on_completions=False, assistant_id=77091):
    outputs = model(inputs)
    logits = outputs.logits.astype(mx.float32)
    _, seq_len = targets.shape
    steps = mx.arange(seq_len)[None, :]
    base_mask = steps < lengths[:, None]
    if train_on_completions:
        eq = (inputs == assistant_id)
        idxs = mx.arange(seq_len)[None, :]
        last_ass_idx = mx.where(eq, idxs, mx.full(idxs.shape, -1)).max(axis=1)
        comp_mask = steps > last_ass_idx[:, None]
        mask = base_mask & comp_mask
    else:
        mask = base_mask
    
    weight_mask = mask.astype(mx.float32)
    
    ce = nn.losses.cross_entropy(logits, targets) * weight_mask
    ntoks = mx.maximum(weight_mask.sum(), 1)
    ce = ce.sum() / ntoks
    return ce, ntoks

def iterate_batches(dataset, batch_size, max_seq_length, train=False):
    # Simple indices without sorting
    indices = list(range(len(dataset)))
    
    if len(dataset) < batch_size:
        raise ValueError(
            f"Dataset must have at least batch_size={batch_size} "
            f"examples but only has {len(dataset)}."
        )
    
    # Distributed training support
    offset = mx.distributed.init().rank()
    step = mx.distributed.init().size()
    if batch_size % step != 0:
        raise ValueError("The batch size must be divisible by the number of workers")
    
    # Create batch indices
    batch_indices = [
        indices[i + offset : i + offset + batch_size : step]
        for i in range(0, len(indices) - batch_size + 1, batch_size)
    ]
    
    while True:
        # Shuffle batch order if training
        if train:
            batch_order = np.random.permutation(len(batch_indices))
        else:
            batch_order = range(len(batch_indices))
        
        for batch_idx in batch_order:
            # Load batch data
            batch_data = []
            batch_lengths = []
            
            for idx in batch_indices[batch_idx]:
                try:
                    item = dataset[idx]
                    if "input_ids" in item:
                        input_ids = item["input_ids"]
                        
                        # Convert MLX array to numpy and flatten if needed
                        if hasattr(input_ids, 'shape'):
                            if isinstance(input_ids, mx.array):
                                input_ids_np = np.array(input_ids)
                            else:
                                input_ids_np = input_ids
                            
                            # Flatten if multidimensional
                            if len(input_ids_np.shape) > 1:
                                input_ids_np = input_ids_np.flatten()
                        else:
                            input_ids_np = np.array(input_ids)
                        
                        seq_len = len(input_ids_np)
                        
                        # Truncate if needed
                        if seq_len > max_seq_length:
                            input_ids_np = input_ids_np[:max_seq_length]
                            seq_len = max_seq_length
                        
                        # Handle attention mask
                        attention_mask_np = None
                        if "attention_mask" in item:
                            attention_mask = item["attention_mask"]
                            if hasattr(attention_mask, 'shape'):
                                if isinstance(attention_mask, mx.array):
                                    attention_mask_np = np.array(attention_mask)
                                else:
                                    attention_mask_np = attention_mask
                                
                                if len(attention_mask_np.shape) > 1:
                                    attention_mask_np = attention_mask_np.flatten()
                            else:
                                attention_mask_np = np.array(attention_mask)
                            
                            attention_mask_np = attention_mask_np[:seq_len]
                        
                        processed_item = {
                            "input_ids": input_ids_np,
                            "attention_mask": attention_mask_np
                        }
                        
                        batch_data.append(processed_item)
                        batch_lengths.append(seq_len)
                        
                except Exception as e:
                    print(f"Warning: Error loading item {idx}: {e}")
                    continue
            
            if not batch_data:
                continue
        
            # MLX-LM style padding with pad_to=32
            pad_to = 32
            max_length_in_batch = max(batch_lengths)
            
            # Pad to nearest multiple of pad_to
            padded_length = 1 + pad_to * ((max_length_in_batch + pad_to - 1) // pad_to)
            padded_length = min(padded_length, max_seq_length)
            
            # Create padded batch arrays
            actual_batch_size = len(batch_data)
            input_ids_batch = np.zeros((actual_batch_size, padded_length), dtype=np.int32)
            attention_mask_batch = np.zeros((actual_batch_size, padded_length), dtype=np.int32)
            
            # Fill batch arrays
            for i, (item, length) in enumerate(zip(batch_data, batch_lengths)):
                truncated_length = min(length, padded_length)
                
                input_ids_batch[i, :truncated_length] = item["input_ids"][:truncated_length]
                
                if item["attention_mask"] is not None:
                    attention_mask_batch[i, :truncated_length] = item["attention_mask"][:truncated_length]
                else:
                    attention_mask_batch[i, :truncated_length] = 1
            
            # Convert to MLX arrays
            batch = {
                "input_ids": mx.array(input_ids_batch),
                "attention_mask": mx.array(attention_mask_batch)
            }
            
            yield batch
        
        if not train:
            break
        
        mx.clear_cache()


def evaluate(
    model,
    dataset,
    batch_size,
    num_batches,
    max_seq_length=2048,
    loss_fn=default_loss,
    train_on_completions=False,
    assistant_id=77091,
):
    """
    Evaluate the model on validation dataset.
    """
    model.eval()
    all_losses = mx.array(0.0)
    ntokens = mx.array(0)
    
    loss_fn_partial = partial(
        loss_fn,
        train_on_completions=train_on_completions,
        assistant_id=assistant_id
    )
    
    index_iterator = iter(range(num_batches)) if num_batches != -1 else iter(int, 1)
    
    for _, batch in tqdm(
        zip(
            index_iterator,
            iterate_batches(
                dataset=dataset,
                batch_size=batch_size,
                max_seq_length=max_seq_length,
            ),
        ),
        desc="Calculating loss...",
        total=min(len(dataset) // batch_size, num_batches) if num_batches != -1 else len(dataset) // batch_size,
    ):
        inputs = batch["input_ids"][:, :-1]
        targets = batch["input_ids"][:, 1:]
        if "attention_mask" in batch:
            lengths = batch["attention_mask"].sum(axis=1)
        else:
            lengths = mx.full((inputs.shape[0],), inputs.shape[1])
        losses, toks = loss_fn_partial(model, inputs, targets, lengths)
        all_losses += losses * toks
        ntokens += toks
        mx.eval(all_losses, ntokens)
    
    all_losses = mx.distributed.all_sum(all_losses, stream=mx.cpu)
    ntokens = mx.distributed.all_sum(ntokens, stream=mx.cpu)

    mx.clear_cache()
    
    return (all_losses / mx.maximum(ntokens, 1)).item()


def train(
    model,
    optimizer,
    train_dataset,
    val_dataset = None,
    args: TrainingArgs = TrainingArgs(),
    loss_fn=default_loss,
    train_on_completions=False,
    assistant_id=77091,
):
    """
    Main training function for vision-language models.
    """
    # Set memory limit if using Metal
    if mx.metal.is_available():
        mx.set_wired_limit(mx.metal.device_info()["max_recommended_working_set_size"])
    
    print(f"{Colors.HEADER}Starting training..., iterations: {args.iters}{Colors.ENDC}")

    # Initialize distributed training
    world = mx.distributed.init()
    world_size = world.size()
    rank = world.rank()
    if world_size > 1:
        print(f"Node {rank} of {world_size}")

    if val_dataset is None and rank == 0:
        print(f"{Colors.OKBLUE}No validation dataset provided — training will run without validation.{Colors.ENDC}")
    
    # Enable gradient checkpointing if requested
    if args.grad_checkpoint:
        if hasattr(model, 'layers'):
            grad_checkpoint(model.layers[0])
    
    # Create loss function with partial application
    loss_fn_partial = partial(
        loss_fn,
        train_on_completions=train_on_completions,
        assistant_id=assistant_id
    )
    
    # Compile the training step (like MLX-LM)
    state = [model.state, optimizer.state, mx.random.state]
    
    @partial(mx.compile, inputs=state, outputs=state)
    def step(batch):
        inputs = batch["input_ids"][:, :-1]
        targets = batch["input_ids"][:, 1:]
        lengths = batch["attention_mask"].sum(axis=1)

        (lvalue, toks), grad = loss_value_and_grad(model, inputs, targets, lengths)

        # Gradient clipping
        if args.grad_clip is not None:
            grad = tree_map(
                lambda g: mx.clip(g, -args.grad_clip, args.grad_clip),
                grad
            )

        # Average gradients for distributed training
        grad = average_gradients(grad)

        # Sanitize gradients before optimizer update
        grad = clean_gradients(grad)
        grad = prune_to_structure(grad, model.trainable_parameters())

        # Update model
        optimizer.update(model, grad)

        return lvalue, toks
    
    loss_value_and_grad = nn.value_and_grad(model, loss_fn_partial)
    
    def clean_gradients(grads):
        """Remove non-trainable / transient gradients like rope_deltas, cache etc."""
        if isinstance(grads, dict):
            cleaned = {}
            for key, value in grads.items():
                if key in ['rope_deltas', 'position_ids', 'cache', 'attention_mask']:
                    continue
                if isinstance(value, dict):
                    nested = clean_gradients(value)
                    if nested:
                        cleaned[key] = nested
                else:
                    cleaned[key] = value
            return cleaned
        return grads

    def prune_to_structure(grads, ref):
        """Prune gradient tree to match the structure of the reference (trainable parameters).
        This avoids KeyError in optimizers when grads contain extra keys.
        """
        if isinstance(grads, dict) and isinstance(ref, dict):
            pruned = {}
            for k, v in grads.items():
                if k in ref:
                    pruned_child = prune_to_structure(v, ref[k])
                    if pruned_child is not None:
                        pruned[k] = pruned_child
            return pruned
        # For non-dict leaves, just return as-is
        return grads
    
    # Training metrics
    model.train()
    losses = 0
    n_tokens = 0
    steps = 0
    trained_tokens = 0
    train_time = 0
    
    # Main training loop
    for it, batch in zip(
        range(1, args.iters + 1),
        iterate_batches(
            dataset=train_dataset,
            batch_size=args.batch_size,
            max_seq_length=args.max_seq_length,
            train=True,
        ),
    ):
        tic = time.perf_counter()

        # Validation (only if a validation dataset is provided)
        if val_dataset is not None and (it == 1 or it % args.steps_per_eval == 0 or it == args.iters):
            tic_val = time.perf_counter()
            val_loss = evaluate(
                model=model,
                dataset=val_dataset,
                batch_size=args.batch_size,
                num_batches=args.val_batches,
                max_seq_length=args.max_seq_length,
                loss_fn=loss_fn,
                train_on_completions=train_on_completions,
                assistant_id=assistant_id,
            )
            model.train()
            val_time = time.perf_counter() - tic_val

            if rank == 0:
                print(
                    f"{Colors.OKCYAN}Iter {it}: "
                    f"Val loss {val_loss:.3f}, "
                    f"Val took {val_time:.3f}s{Colors.ENDC}",
                    flush=True,
                )

            tic = time.perf_counter()

        # Training step
        lvalue, toks = step(batch)
        mx.clear_cache()
        losses += lvalue
        n_tokens += toks
        steps += 1
        mx.eval(state, losses, n_tokens)
        train_time += time.perf_counter() - tic

        # Report training metrics
        if it % args.steps_per_report == 0 or it == args.iters:
            train_loss = mx.distributed.all_sum(losses, stream=mx.cpu).item()
            train_loss /= steps * world_size
            n_tokens_total = mx.distributed.all_sum(n_tokens, stream=mx.cpu).item()
            learning_rate = optimizer.learning_rate.item() if hasattr(optimizer.learning_rate, 'item') else args.learning_rate
            it_sec = args.steps_per_report / train_time
            tokens_sec = float(n_tokens_total) / train_time
            trained_tokens += n_tokens_total
            peak_mem = mx.get_peak_memory() / 1e9

            if rank == 0:
                print(
                    f"Iter {it}: Train loss {Colors.OKGREEN}{train_loss:.3f}{Colors.ENDC}, "
                    f"Learning Rate {learning_rate:.3e}, "
                    f"It/sec {it_sec:.3f}, "
                    f"Tokens/sec {tokens_sec:.3f}, "
                    f"Trained Tokens {trained_tokens}, "
                    f"Peak mem {peak_mem:.3f} GB",
                    flush=True,
                )

            # Reset metrics
            losses = 0
            n_tokens = 0
            steps = 0
            train_time = 0

        # Save checkpoint
        if it % args.steps_per_save == 0 and rank == 0:
            save_adapter(model, args.adapter_file)
            checkpoint = (
                Path(args.adapter_file).parent / f"{it:07d}_adapters.safetensors"
            )
            save_adapter(model, checkpoint)
            print(
                f"{Colors.OKBLUE}Iter {it}: Saved adapter weights to "
                f"{args.adapter_file} and {checkpoint}.{Colors.ENDC}",
                flush=True,
            )
    
    # Save final weights
    if rank == 0:
        save_adapter(model, args.adapter_file)
        print(f"{Colors.OKGREEN}Saved final adapter weights to {args.adapter_file}.{Colors.ENDC}")


def save_adapter(model: nn.Module, adapter_file: Union[str, Path]):
    """Save adapter weights and config."""
    path = Path(adapter_file)
    path.parent.mkdir(parents=True, exist_ok=True)
    
    # Save adapter config if available
    if hasattr(model, 'config') and hasattr(model.config, "lora"):
        with open(path.parent / "adapter_config.json", "w") as f:
            json.dump(model.config.lora, f, indent=2)
    
    # Save weights
    flattened_tree = tree_flatten(model.trainable_parameters())
    mx.save_safetensors(str(adapter_file), dict(flattened_tree))