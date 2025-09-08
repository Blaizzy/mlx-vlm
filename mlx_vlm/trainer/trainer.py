# Copyright © 2024 MLX-VLM

import json
import time
import warnings
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


class Colors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


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
        
        image_token_index = getattr(self.config, "image_token_index", "image_token_id")
        
        inputs = prepare_inputs(
            processor=self.processor,
            images=images,
            audio=None,
            prompts=prompts,
            image_token_index=image_token_index,
            resize_shape=self.image_resize_shape
        )
        
        return inputs


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
    learning_rate: float = field(
        default=1e-5,
        metadata={"help": "Learning rate."},
    )
    grad_clip: float = field(
        default=None,
        metadata={"help": "Gradient clipping value."},
    )


def default_loss(model, inputs, targets, lengths, train_on_completions=False, assistant_id=77091):
    outputs = model(inputs)
    logits = outputs.logits.astype(mx.float32)

    length_mask = mx.arange(inputs.shape[1])[None, :] < lengths[:, None]

    # Build weight_mask according to train_on_completions and assistant_id
    if train_on_completions:
        weight_mask = mx.array(length_mask)
        for i in range(inputs.shape[0]):
            mask = (inputs[i] == assistant_id)
            has_any = mx.sum(mask) > 0
            if bool(has_any.item()):
                first_idx = int(mx.argmax(mask).item())
                weight_mask[i, :first_idx] = 0
            else:
                weight_mask[i, :] = 0
    else:
        weight_mask = length_mask

    ce = nn.losses.cross_entropy(logits, targets) * weight_mask
    ntoks = weight_mask.sum()
    ce = ce.sum() / ntoks
    mx.clear_cache()
    return ce, ntoks


def iterate_batches(dataset, batch_size, max_seq_length, train=False):
    indices = list(range(len(dataset)))
    
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
        if train:
            batch_order = np.random.permutation(len(batch_indices))
        else:
            batch_order = range(len(batch_indices))
        
        for batch_idx in batch_order:
            batch_data = []
            for idx in batch_indices[batch_idx]:
                try:
                    item = dataset[idx]
                    
                    if "input_ids" in item:
                        if len(item["input_ids"]) > max_seq_length:
                            item["input_ids"] = item["input_ids"][:max_seq_length]
                            if "attention_mask" in item:
                                item["attention_mask"] = item["attention_mask"][:max_seq_length]
                    
                    batch_data.append(item)
                except Exception as e:
                    print(f"Warning: Error loading item {idx}: {e}")
                    continue
            
            if not batch_data:
                continue
            
            if batch_size == 1 and len(batch_data) == 1:
                yield batch_data[0]
            else:
                batch = {}
                for key in batch_data[0].keys():
                    try:
                        values = [item[key] for item in batch_data]
                        if all(isinstance(v, mx.array) for v in values):
                            batch[key] = mx.stack(values)
                        else:
                            batch[key] = values
                    except:
                        batch[key] = values[0] if len(values) == 1 else values
                
                yield batch
                del batch_data
                mx.clear_cache()
        
        if not train:
            break

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
    val_dataset,
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
    
    # Compile training step
    state = [model.state, optimizer.state, mx.random.state]
    
    def step(batch):
        # Forward and backward pass
        inputs = batch["input_ids"][:, :-1]
        targets = batch["input_ids"][:, 1:]
        if "attention_mask" in batch:
            lengths = batch["attention_mask"].sum(axis=1)
        else:
            lengths = mx.full((inputs.shape[0],), inputs.shape[1])
        (lvalue, toks), grad = loss_value_and_grad(model, inputs, targets, lengths)

        grad = clean_gradients(grad)
        
        # Gradient clipping
        if args.grad_clip is not None:
            grad = tree_map(
                lambda g: mx.clip(g, -args.grad_clip, args.grad_clip),
                grad
            )
        
        # All reduce the gradients if running in distributed mode
        grad = average_gradients(grad)
        
        # Model update
        optimizer.update(model, grad)

        return lvalue, toks
    loss_value_and_grad = nn.value_and_grad(model, loss_fn_partial)
    
    def clean_gradients(grads):
        """Remove non-trainable gradients like rope_deltas."""
        if isinstance(grads, dict):
            cleaned = {}
            for key, value in grads.items():
                if key in ['rope_deltas', 'position_ids', 'cache', 'attention_mask']:
                    continue
                elif isinstance(value, dict):
                    cleaned_value = clean_gradients(value)
                    if cleaned_value:
                        cleaned[key] = cleaned_value
                else:
                    cleaned[key] = value
            return cleaned
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
        
        # Validation
        if it == 1 or it % args.steps_per_eval == 0 or it == args.iters:
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