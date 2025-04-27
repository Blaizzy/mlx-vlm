import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Generator, List, Optional, Tuple, Union
import logging

import mlx.core as mx
import mlx.nn as nn
import numpy as np
from mlx.utils import tree_flatten, tree_map


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


from .grpo_reward_functions import (
    RewardFunctions,
    r1_accuracy_reward_func,
    r1_count_xml,
    r1_extract_xml_answer,
    r1_int_reward_func,
    r1_soft_format_reward_func,
    r1_strict_format_reward_func,
)

from .sft_trainer import TrainingArgs, average_gradients, grad_checkpoint
from .callback import TrainingCallback


@dataclass
class GRPOTrainingArgs(TrainingArgs):
    group_size: int = field(
        default=4,
        metadata={"help": "Number of responses per prompt."},
    )
    beta: float = field(default=0.1, metadata={"help": "KL penalty coefficient."})
    epsilon: float = field(
        default=1e-4, metadata={"help": "The Epsilon for numerical stability."}
    )
    max_completion_length: int = field(
        default=512, metadata={"help": "Number of Generations."}
    )
    reference_model_path: str = field(
        default=None,
        metadata={
            "help": "Path to reference model weights. If None, uses the same model."
        },
    )
    temperature: float = field(
        default=0.8,
        metadata={
            "help": "Temperature for sampling. The higher the temperature, the more random the completions."
        },
    )
    reward_weights: Optional[List[float]] = field(
        default=None,
        metadata={
            "help": "Weights for each reward function. Must match the number of reward functions. If `None`, all rewards are weighted equally with weight `1.0`."
        },
    )


def get_per_token_logps(model: nn.Module, inputs, lengths, **kwargs):
    output = model(inputs, mask=None, pixel_values=None, **kwargs)
    logits = output.logits
    logits = logits[:, :-1, :]
    targets = inputs[:, 1:]
    per_token_logps = []
    for i in range(logits.shape[0]):
        seq_len = int(lengths[i]) - 1
        seq_logits = logits[i, :seq_len]
        seq_targets = targets[i, :seq_len]
        log_probs = nn.log_softmax(seq_logits, axis=-1)
        token_log_probs = mx.take_along_axis(
            log_probs, seq_targets.reshape(seq_len, 1), axis=-1
        ).squeeze(-1)
        per_token_logps.append(token_log_probs)
    mx.eval(logits)
    return per_token_logps

def generate_grpo(
    model: nn.Module,
    processor,
    prompt_tokens=None,
    prompt_masks: Optional[List[List[int]]] = None,
    images: Union[str, List[str]] = None,
    max_tokens: int = 512,
    group_size: int = 2,
    temperature: float = 0.8,
    batch_size: int = 1,
    end_token: str = "</answer>",
    other_inputs: Optional[List[Dict]] = None
):
    from ..utils import StoppingCriteria
    if prompt_masks is not None and len(prompt_masks) != len(prompt_tokens):
        raise ValueError("prompt_masks length must match prompt_tokens length")

    end_sequence = mx.array(processor.tokenizer.encode(end_token))
    stopping_criteria = StoppingCriteria(end_sequence.tolist(), tokenizer=processor.tokenizer)
    total_samples = len(prompt_tokens)
    all_completions = []
    all_completion_texts = []
    batch_indices = []

    try:
        for prompt_idx in range(total_samples):
            prompt_array = mx.array(prompt_tokens[prompt_idx])
            prompt_tensor = prompt_array.reshape((1, -1))

            if prompt_masks is not None:
                m = prompt_masks[prompt_idx]
                mask_tensor = mx.array(m).reshape((1, -1))
                mask_tensor = mx.repeat(mask_tensor, group_size, axis=0)
            else:
                mask_tensor = None

            if images:
                pixel = images[prompt_idx]
                if (
                    pixel is None
                    or not isinstance(pixel, mx.array)
                    or pixel.size == 0
                    or (hasattr(pixel, 'shape') and 0 in pixel.shape)
                ):
                    pixel_values = None
                else:
                    if pixel.ndim == 3:
                        pixel = pixel[None, ...]
                    pixel_values = pixel
            else:
                pixel_values = None

            if other_inputs is not None:
                kwargs = other_inputs[prompt_idx]
            else:
                kwargs = {}

            batch_results = []
            for g_idx in range(group_size):
                from ..utils import generate_step

                current_tokens = []
                for result in generate_step(
                    input_ids=prompt_tensor,
                    model=model,
                    pixel_values=pixel_values,
                    mask=mask_tensor,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    **kwargs
                ):
                    token_id = result[0]
                    if token_id is None:
                        break
                    current_tokens.append(token_id)
                    if stopping_criteria(token_id):
                        break

                if current_tokens:
                    batch_results.append(mx.array(current_tokens, dtype=mx.int32))

            for completion_ids in batch_results:
                batch_indices.append(prompt_idx)
                completion_text = processor.tokenizer.decode(completion_ids.tolist())
                all_completions.append(mx.stop_gradient(completion_ids))
                all_completion_texts.append(completion_text)

    finally:
        mx.clear_cache()

    return all_completions, all_completion_texts, batch_indices


def grpo_loss(
    model,
    ref_model,
    processor,
    batch,
    all_completions=None,
    all_completion_texts=None,
    batch_indices=None,
    reward_funcs: Optional[List[RewardFunctions]] = None,
    beta: float = 0.1,
    epsilon: float = 1e-4,
    reward_weights: Optional[List[float]] = None,
):
    prompt_tokens, _, prompt_text, answer_text, type_info, images, other_inputs = batch

    if not all_completions:
        raise ValueError(
            "No completions were generated. Please check your model and inputs."
        )

    expanded_answers = []
    expanded_prompts = []
    expanded_types = []
    unique_prompt_indices = sorted(set(batch_indices))
    grouped_completions = {idx: [] for idx in unique_prompt_indices}

    for i, completion_idx in enumerate(batch_indices):
        grouped_completions[completion_idx].append(i)

    ordered_completions = []
    ordered_completion_texts = []
    ordered_batch_indices = []

    for prompt_idx in unique_prompt_indices:
        completion_indices = grouped_completions[prompt_idx]
        for idx in completion_indices:
            ordered_completions.append(all_completions[idx])
            ordered_completion_texts.append(all_completion_texts[idx])
            ordered_batch_indices.append(prompt_idx)
            expanded_answers.append(answer_text[prompt_idx])
            expanded_prompts.append(prompt_text[prompt_idx])
            expanded_types.append(type_info[prompt_idx] if type_info is not None else None)

    all_completions = ordered_completions
    all_completion_texts = ordered_completion_texts
    batch_indices = ordered_batch_indices
    max_length = max(ids.shape[0] for ids in all_completions)
    padded_completions = []
    attention_masks = []

    for completion_ids in all_completions:
        completion_tensor = mx.array(completion_ids.tolist())
        padding_length = max_length - completion_tensor.shape[0]
        if padding_length > 0:
            padding = mx.zeros((padding_length,), dtype=completion_tensor.dtype)
            padded_ids = mx.concatenate([completion_tensor, padding])
            mask = mx.concatenate(
                [mx.ones_like(completion_tensor), mx.zeros_like(padding)]
            )
        else:
            padded_ids = completion_tensor
            mask = mx.ones_like(completion_tensor)
        padded_completions.append(padded_ids)
        attention_masks.append(mask)

    inputs = mx.stack(padded_completions)
    attention_mask = mx.stack(attention_masks)
    lengths = attention_mask.sum(axis=1)
        
    if other_inputs is not None:
        kwargs = other_inputs[0]
    else:
        kwargs = {}
    token_log_probs = get_per_token_logps(model, inputs, lengths, **kwargs)
    mx.eval(token_log_probs)

    if ref_model is None:
        ref_token_log_probs = token_log_probs
    else:
        ref_token_log_probs = get_per_token_logps(ref_model, inputs, lengths, **kwargs)
        mx.eval(ref_token_log_probs)

    max_len = max(x.shape[0] for x in token_log_probs)
    padded_log_probs = []
    padded_ref_log_probs = []

    for i in range(len(token_log_probs)):
        seq_len = token_log_probs[i].shape[0]
        padding = mx.zeros((max_len - seq_len,))

        padded_log_probs.append(mx.concatenate([token_log_probs[i], padding]))
        padded_ref_log_probs.append(mx.concatenate([ref_token_log_probs[i], padding]))

    token_log_probs = mx.stack(padded_log_probs)
    ref_token_log_probs = mx.stack(padded_ref_log_probs)

    all_func_rewards = []
    for reward_func in reward_funcs:
        raw_rewards = reward_func(
            prompts=expanded_prompts,
            completions=all_completion_texts,
            answer=expanded_answers,
            types=expanded_types
        )
        if raw_rewards is None:
            processed_rewards = [float('nan')] * len(all_completion_texts)
        else:
            processed_rewards = [float(r) if r is not None else float('nan') for r in raw_rewards]
        func_rewards = mx.array(processed_rewards)
        all_func_rewards.append(func_rewards)

    rewards = mx.stack(all_func_rewards, axis=1)

    all_nan_rows = mx.all(mx.isnan(rewards), axis=1)
    if mx.any(all_nan_rows):
        nan_row_idx = mx.argmax(all_nan_rows).item()
        warning_msg = (
            f"All reward functions returned None for prompt: {expanded_prompts[nan_row_idx]}, "
            f"completion: {all_completion_texts[nan_row_idx]}, "
            f"answer: {expanded_answers[nan_row_idx]}. "
            "Please ensure that at least one reward function returns a valid reward."
        )
        raise RuntimeError(warning_msg)

    if reward_weights is not None:
        if len(reward_weights) != len(reward_funcs):
            raise ValueError(
                f"Number of reward weights ({len(reward_weights)}) must match number of reward "
                f"functions ({len(reward_funcs)})"
            )
        reward_weights = mx.array(reward_weights, dtype=mx.float32)
    else:
        reward_weights = mx.ones(len(reward_funcs), dtype=mx.float32)

    valid_reward_mask = ~mx.isnan(rewards)
    rewards_no_nan = mx.where(valid_reward_mask, rewards, mx.zeros_like(rewards))
    rewards = (rewards_no_nan * mx.expand_dims(reward_weights, 0)).sum(axis=1)

    num_unique_prompts = len(unique_prompt_indices)

    rewards_by_prompt = [[] for _ in range(num_unique_prompts)]
    for i, prompt_idx in enumerate(batch_indices):
        prompt_position = unique_prompt_indices.index(prompt_idx)
        rewards_by_prompt[prompt_position].append(rewards[i])

    advantages = mx.zeros_like(rewards)
    for i, prompt_rewards in enumerate(rewards_by_prompt):
        if len(prompt_rewards) > 1:
            prompt_rewards = mx.array(prompt_rewards)
            mean_reward = mx.mean(prompt_rewards)
            std_reward = mx.std(prompt_rewards)
            indices = [
                j
                for j, idx in enumerate(batch_indices)
                if idx == unique_prompt_indices[i]
            ]
            for j, idx in enumerate(indices):
                advantages[idx] = (prompt_rewards[j] - mean_reward) / (
                    std_reward + epsilon
                )
        else:
            idx = batch_indices.index(unique_prompt_indices[i])
            advantages[idx] = 0.0

    # Compute KL divergence using Schulman's approximator
    kl_div = (
        mx.exp(ref_token_log_probs - token_log_probs)
        - (ref_token_log_probs - token_log_probs)
        - 1
    )

    # Create mask for valid tokens
    length_mask = mx.arange(inputs.shape[1] - 1)[None, :] < (lengths[:, None] - 1)

    # Compute policy ratio
    policy_ratio = mx.exp(
        mx.array(token_log_probs - mx.stop_gradient(ref_token_log_probs))
    )

    # Apply PPO like clipping
    policy_ratio_cliped = mx.clip(policy_ratio, 1 - epsilon, 1 + epsilon)

    # Calculate both unclipped and clipped objectives
    unclipped_obj = policy_ratio * advantages.reshape(-1, 1)
    clipped_obj = policy_ratio_cliped * advantages.reshape(-1, 1)

    # Take the minimum (pessimistic bound)
    per_token_loss = -mx.minimum(unclipped_obj, clipped_obj)

    # Add KL penalty if beta is non-zero
    if beta != 0.0:
        per_token_loss = per_token_loss + beta * kl_div

    # Average over tokens
    loss = (per_token_loss).sum() / length_mask.sum()

    # Calculate mean KL divergence for metrics
    mean_kl = ((kl_div * length_mask).sum(axis=1) / length_mask.sum(axis=1)).mean()

    reward_metrics = {}
    for i, reward_func in enumerate(reward_funcs):
        func_name = reward_func.__name__
        raw_rewards = reward_func(
            prompts=expanded_prompts,
            completions=all_completion_texts,
            answer=expanded_answers,
        )
        valid_mask = ~mx.isnan(mx.array([reward if reward is not None else float('nan') for reward in raw_rewards]))
        valid_rewards = mx.array([reward for reward in raw_rewards if reward is not None and not mx.isnan(reward)])
        if len(valid_rewards) > 0:
            reward_metrics[f"{func_name}_mean"] = mx.mean(valid_rewards)
            reward_metrics[f"{func_name}_std"] = mx.std(valid_rewards) if len(valid_rewards) > 1 else mx.zeros(1)
            reward_metrics[f"{func_name}_coverage"] = valid_mask.sum() / len(raw_rewards)
        else:
            reward_metrics[f"{func_name}_mean"] = float('nan')
            reward_metrics[f"{func_name}_std"] = float('nan')
            reward_metrics[f"{func_name}_coverage"] = 0.0

    grouped_rewards_mean = mx.array(
        [mx.mean(mx.array(rewards)) for rewards in rewards_by_prompt]
    )
    grouped_rewards_std = mx.array(
        [
            mx.std(mx.array(rewards)) if len(rewards) > 1 else mx.zeros(1)
            for rewards in rewards_by_prompt
        ]
    )

    metrics = {
        "total_rewards_mean": mx.mean(rewards),
        "total_rewards_std": mx.std(rewards),
        "grouped_rewards_mean": mx.mean(grouped_rewards_mean),
        "grouped_rewards_std": mx.mean(grouped_rewards_std),
        "kl": mean_kl,
        "average_generated_tokens": len(all_completion_texts[-1]) // len(batch_indices),
        **reward_metrics,
    }

    mx.clear_cache()

    return loss, length_mask.sum(axis=1).sum(), metrics


def iterate_grpo_batches(dataset, batch_size, max_seq_length, train=False):
    if not dataset or not isinstance(dataset[0], dict):
        raise ValueError("Dataset must be a list of dictionaries")

    def length_key(i):
        return len(dataset[i]["input_ids"])

    idx = sorted(range(len(dataset)), key=length_key)

    if len(dataset) < batch_size:
        raise ValueError(
            f"Dataset must have at least batch_size={batch_size} "
            f"examples but only has {len(dataset)}."
        )

    step = mx.distributed.init().size()
    if batch_size % step != 0:
        raise ValueError("The batch size must be divisible by the number of workers")

    def batch_index_generator():
        for i in range(0, len(idx) - batch_size + 1, batch_size):
            yield idx[i : i + batch_size : step]

    while True:
        indices = (
            np.random.permutation(list(batch_index_generator()))
            if train
            else batch_index_generator()
        )

        for batch_idx in indices:
            current_batch = [dataset[int(j)] for j in batch_idx]

            prompts_tokens = [item["input_ids"] for item in current_batch]
            answers_tokens = [item["answer_ids"] for item in current_batch]
            prompts_text = [item["prompt_str"] for item in current_batch]
            answers_text = [item["answer_str"] for item in current_batch]
            types = [item.get("type", None) for item in current_batch]
            images = [item.get("pixel_values", None) for item in current_batch]
            prompt_masks = [item.get("attention_mask") for item in current_batch]

            other_inputs = []
            for item in current_batch:
                oi = {}
                if "image_grid_thw" in item:
                    oi["image_grid_thw"] = item["image_grid_thw"]
                if "video_grid_thw" in item:
                    oi["video_grid_thw"] = item["video_grid_thw"]
                # Add more keys here if needed later
                other_inputs.append(oi)

            if any(len(p) > max_seq_length for p in prompts_tokens):
                print(
                    f"[WARNING] Some prompts are longer than {max_seq_length} tokens. "
                    "Long prompts will be truncated."
                )

            yield prompts_tokens, answers_tokens, prompts_text, answers_text, types, images, prompt_masks, other_inputs

        if not train:
            break


def train_grpo(
    model: nn.Module,
    ref_model: Optional[nn.Module],
    processor,
    optimizer,
    dataset,
    reward_funcs: Optional[List[RewardFunctions]] = [
        r1_accuracy_reward_func,
        r1_int_reward_func,
        r1_strict_format_reward_func,
        r1_soft_format_reward_func,
        r1_count_xml,
    ],
    args: GRPOTrainingArgs = GRPOTrainingArgs(),
    loss_fn: callable = grpo_loss,
    training_callback: TrainingCallback = None,
    clip_gradients=None
):
    world = mx.distributed.init()
    world_size = world.size()
    rank = world.rank()
    if world_size > 1:
        print(f"Node {rank} of {world_size}")

    if args.grad_checkpoint:
        grad_checkpoint(model.layers[0])

    state = [model.state, optimizer.state]

    def step(batch):
        prompt_tokens, targets, prompt_lens, target_lens, type_info, images, prompt_masks, other_inputs = batch

        all_completions, all_completion_texts, batch_indices = generate_grpo(
            model=model,
            processor=processor,
            prompt_tokens=prompt_tokens,
            prompt_masks=prompt_masks,
            images=images,
            other_inputs=other_inputs,
            max_tokens=args.max_completion_length,
            group_size=args.group_size,
            temperature=args.temperature,
            batch_size=args.batch_size
        )
        if not all_completions:
            print("[WARNING] Retrying generation due to empty completions")
            all_completions, all_completion_texts, batch_indices = generate_grpo(
                model=model,
                processor=processor,
                prompt_tokens=prompt_tokens,
                prompt_masks=prompt_masks,
                images=images,
                max_tokens=args.max_completion_length,
                group_size=args.group_size,
                temperature=args.temperature,
                batch_size=args.batch_size
            )
            if not all_completions:
                raise ValueError("Still no completions after retry. Check model and inputs.")

        (loss, toks, metrics), grad = loss_value_and_grad(
            model,
            processor=processor,
            batch=(prompt_tokens, targets, prompt_lens, target_lens, type_info, images, other_inputs),
            all_completions=all_completions,
            all_completion_texts=all_completion_texts,
            batch_indices=batch_indices,
            reward_funcs=reward_funcs,
            beta=args.beta,
            epsilon=args.epsilon,
            ref_model=ref_model
        )

        if clip_gradients is not None:
            grad = tree_map(
                lambda g: mx.clip(g, -clip_gradients, clip_gradients), grad
            )

        grad = average_gradients(grad)
        optimizer.update(model, grad)

        return loss, toks, metrics

    loss_value_and_grad = nn.value_and_grad(model, loss_fn)

    losses = 0
    n_tokens = 0
    steps = 0
    trained_tokens = 0
    accumulated_metrics = {
        "total_rewards_mean": 0,
        "total_rewards_std": 0,
        "grouped_rewards_mean": 0,
        "grouped_rewards_std": 0,
        "kl": 0,
        'average_generated_tokens': 0
    }
    for reward_func in reward_funcs:
        func_name = reward_func.__name__
        accumulated_metrics[f"{func_name}_mean"] = 0
        accumulated_metrics[f"{func_name}_std"] = 0
        accumulated_metrics[f"{func_name}_coverage"] = 0

    start = time.perf_counter()
    for it, batch in zip(
        range(1, args.iters + 1),
        iterate_grpo_batches(
            dataset=dataset,
            batch_size=args.batch_size,
            max_seq_length=args.max_seq_length,
            train=True,
        ),
    ):
        loss, toks, metrics = step(batch)
        losses += loss
        n_tokens += toks
        steps += 1

        mx.clear_cache()

        for k, v in metrics.items():
            accumulated_metrics[k] += v

        mx.eval(state, losses, n_tokens)

        if it % args.steps_per_report == 0 or it == args.iters:
            stop = time.perf_counter()

            train_loss = mx.distributed.all_sum(losses, stream=mx.cpu).item()
            train_loss /= steps * mx.distributed.init().size()
            avg_metrics = {
                k: v / (steps * world_size) for k, v in accumulated_metrics.items()
            }
            n_tokens = mx.distributed.all_sum(n_tokens, stream=mx.cpu).item()
            learning_rate = optimizer.learning_rate.item()
            it_sec = args.steps_per_report / (stop - start)
            tokens_sec = float(n_tokens) / (stop - start)
            trained_tokens += n_tokens
            peak_mem = mx.get_peak_memory() / 1e9

            if rank == 0:
                train_metrics_str = (
                    f"Train loss {float(train_loss):.3f}, "
                    f"Total rewards mean {float(avg_metrics['total_rewards_mean']):.3f}, "
                    f"Total rewards std {float(avg_metrics['total_rewards_std']):.3f}, "
                    f"Grouped rewards mean {float(avg_metrics['grouped_rewards_mean']):.3f}, "
                    f"Grouped rewards std {float(avg_metrics['grouped_rewards_std']):.3f}, "
                    f"Average Generated Tokens {float(avg_metrics['average_generated_tokens'])}, "
                    f"KL {float(avg_metrics['kl']):.3f}"
                )

                for i, reward_func in enumerate(reward_funcs):
                    func_name = reward_func.__name__
                    train_metrics_str += (
                        f", {func_name} mean {float(avg_metrics[f'{func_name}_mean']):.3f}, "
                        f"{func_name} std {float(avg_metrics[f'{func_name}_std']):.3f}"
                    )

                print(
                    f"Iter {it}: {train_metrics_str}, "
                    f"Learning Rate {learning_rate:.3e}, "
                    f"It/sec {it_sec:.3f}, "
                    f"Tokens/sec {tokens_sec:.3f}, "
                    f"Peak mem {peak_mem:.3f} GB",
                    flush=True,
                )

            if training_callback is not None:
                training_callback.on_train_loss_report(
                    {
                        "iteration": it,
                        "train_loss": train_loss,
                        **{f"train_{k}": v for k, v in avg_metrics.items()},
                        "learning_rate": learning_rate,
                        "iterations_per_second": it_sec,
                        "tokens_per_second": tokens_sec,
                        "trained_tokens": trained_tokens,
                        "peak_memory": peak_mem,
                    }
                )

            losses = 0
            n_tokens = 0
            steps = 0
            accumulated_metrics = {k: 0 for k in accumulated_metrics}
            start = time.perf_counter()

        if it % args.steps_per_save == 0:
            adapter_weights = dict(tree_flatten(model.trainable_parameters()))
            mx.save_safetensors(str(args.adapter_file), adapter_weights)
            checkpoint = (
                Path(args.adapter_file).parent / f"{it:07d}_adapters.safetensors"
            )
            mx.save_safetensors(str(checkpoint), adapter_weights)
            print(
                f"Iter {it}: Saved adapter weights to "
                f"{args.adapter_file} and {checkpoint}."
            )

    adapter_weights = dict(tree_flatten(model.trainable_parameters()))
    mx.save_safetensors(str(args.adapter_file), adapter_weights)
    print(f"Saved final weights to {args.adapter_file}.")
