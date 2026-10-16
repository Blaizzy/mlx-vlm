"""Online text/vision GRPO using the shared training lifecycle.

Adapted from Goekdeniz-Guelmez/mlx-lm-lora's GRPO optimizations: exact rollout
IDs, CPU rewards, completion-only scoring, and synchronized gradient chunks.
"""

from functools import partial

import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten, tree_map

from mlx_vlm.trainer.common.task import TrainingTask
from mlx_vlm.trainer.common.utils import (
    enable_model_gradient_checkpointing,
    iterate_batch_indices,
)
from mlx_vlm.trainer.datasets.loading import CacheDataset
from mlx_vlm.trainer.vlm.grpo.dataset import GRPOVisionDataset
from mlx_vlm.trainer.vlm.grpo.loss import completion_logps, grpo_objective
from mlx_vlm.trainer.vlm.grpo.rewards import (
    accuracy_reward,
    calculate_rewards_and_advantages,
)
from mlx_vlm.trainer.vlm.sft.dataset import _config_mapping, _get_option
from mlx_vlm.trainer.vlm.sft.runtime import _model_type, vlm_batch_metrics


def _cache_frozen_vision(model, rows):
    """Reuse Qwen image features across a group, without caching trainable vision."""
    if _model_type(model) != "qwen3_5" or tree_flatten(
        model.vision_tower.trainable_parameters()
    ):
        return rows
    result = []
    for row in rows:
        inputs = dict(row["inputs"])
        if (
            inputs.get("pixel_values") is not None
            and "cached_image_features" not in inputs
        ):
            pixels = inputs["pixel_values"].astype(
                model.vision_tower.patch_embed.proj.weight.dtype
            )
            features, _ = model.vision_tower(pixels, inputs.get("image_grid_thw"))
            features = mx.stop_gradient(features)
            mx.eval(features)
            inputs["cached_image_features"] = features
        result.append(dict(row, inputs=inputs))
    return result


def generate_grpo(model, tokenizer, rows, args):
    """Bound KV concurrency to one completion, retaining EOS in training IDs."""
    from mlx_vlm.generate import generate_step

    # These tokens represent processor-expanded input features, never generated
    # media. Sampling them would create image slots without matching features.
    config = _config_mapping(model.config)
    vocab_size = _get_option(
        config.get("text_config"), "vocab_size", config.get("vocab_size")
    )
    media_token_ids = {
        config.get(name)
        for name in (
            "image_token_index",
            "image_token_id",
            "video_token_index",
            "video_token_id",
        )
        if isinstance(config.get(name), int)
        and config[name] >= 0
        and (vocab_size is None or config[name] < vocab_size)
    }
    logit_bias = {token: -float("inf") for token in media_token_ids}

    eos = getattr(tokenizer, "eos_token_ids", None)
    if eos is None:
        eos = getattr(tokenizer, "eos_token_id", None)
    eos = set(
        eos if isinstance(eos, (list, tuple, set)) else ([] if eos is None else [eos])
    )
    end_ids = (
        tokenizer.encode(args.end_answer_token, add_special_tokens=False)
        if args.end_answer_token
        else []
    )
    completions, texts, indices = [], [], []
    training = model.training
    try:
        model.eval()
        rows = _cache_frozen_vision(model, rows)
        for index, row in enumerate(rows):
            inputs = row["inputs"]
            ids = inputs["input_ids"]
            if ids.ndim != 1 or ids.size < 1:
                raise ValueError(
                    "GRPO prepared prompts require non-empty 1D input_ids."
                )
            if ids.size + args.max_completion_length > args.max_seq_length:
                raise ValueError(
                    "GRPO prompt plus max_completion_length exceeds max_seq_length; resize images or increase the limit. Image prompts are never truncated."
                )
            media = {
                k: v
                for k, v in inputs.items()
                if k not in {"input_ids", "attention_mask"}
            }
            for _ in range(args.group_size):
                tokens = []
                options = dict(media)
                pixels = options.pop("pixel_values", None)
                stream = generate_step(
                    ids[None, :],
                    model,
                    pixels,
                    inputs["attention_mask"][None, :],
                    max_tokens=args.max_completion_length,
                    temperature=args.temperature,
                    top_p=args.top_p,
                    top_k=args.top_k,
                    min_p=args.min_p,
                    logit_bias=logit_bias or None,
                    **options,
                )
                try:
                    for token, _ in stream:
                        token = int(token)
                        tokens.append(token)
                        if token in eos or (
                            end_ids and tokens[-len(end_ids) :] == end_ids
                        ):
                            break
                finally:
                    stream.close()
                if not tokens:
                    raise ValueError("GRPO generation produced an empty completion.")
                completions.append(mx.array(tokens, dtype=mx.int32))
                texts.append(
                    tokenizer.decode(tokens[:-1] if tokens[-1] in eos else tokens)
                )
                indices.append(index)
        return completions, texts, indices
    finally:
        model.train(training)


def prepare_rollout(model, tokenizer, rows, args, reward_funcs, reward_weights):
    rows = _cache_frozen_vision(model, rows)
    completions, texts, indices = generate_grpo(model, tokenizer, rows, args)
    advantages, metrics = calculate_rewards_and_advantages(
        reward_funcs,
        [rows[i]["prompt"] for i in indices],
        texts,
        [rows[i]["answer"] for i in indices],
        indices,
        metadata=[rows[i].get("metadata", {}) for i in indices],
        reward_weights=reward_weights,
        epsilon=args.advantage_epsilon,
    )
    samples = []
    for position, (completion, index) in enumerate(zip(completions, indices)):
        inputs = rows[index]["inputs"]
        prompt = inputs["input_ids"]
        ids = mx.concatenate([prompt, completion])[None, :]
        batch = dict(inputs, input_ids=ids, attention_mask=mx.ones_like(ids))
        samples.append(
            {
                "batch": batch,
                "start": prompt.size - 1,
                "advantage": advantages[position : position + 1],
            }
        )
    return {"samples": samples, "reward_metrics": metrics}


def make_task(
    model,
    args,
    processing_class=None,
    dataset_config=None,
    *,
    reference_model=None,
    reward_funcs=None,
    reward_weights=None,
):
    if processing_class is None:
        raise ValueError("GRPO requires processing_class for decoding rollouts.")
    tokenizer = getattr(processing_class, "tokenizer", processing_class)
    if args.beta and reference_model is None:
        raise ValueError(
            "GRPO beta > 0 requires a separate frozen reference_model; use beta=0 to skip KL."
        )
    if reference_model is model:
        raise ValueError("GRPO requires a separate reference_model.")
    if reference_model is not None:
        from mlx_vlm.trainer.vlm.dpo.trainer import _validate_reference_model

        _validate_reference_model(model, reference_model)
        reference_model.freeze()
        reference_model.eval()
    for _, module in model.named_modules():
        if isinstance(module, nn.Dropout) and module._p_1 < 1:
            raise ValueError(
                "On-policy GRPO requires dropout=0 for consistent rollout/scoring probabilities."
            )
    reward_funcs = [accuracy_reward] if reward_funcs is None else list(reward_funcs)
    if not reward_funcs or any(not callable(fn) for fn in reward_funcs):
        raise ValueError("reward_funcs must contain callable reward functions.")
    if reward_weights is not None and len(reward_weights) != len(reward_funcs):
        raise ValueError("reward_weights must match reward_funcs.")

    def sample_loss(policy, sample):
        batch, start = sample["batch"], sample["start"]
        logps = completion_logps(policy, batch, start, args.logit_chunk_size)
        reference_batch = {
            key: value for key, value in batch.items() if key != "cached_image_features"
        }
        ref = (
            mx.stop_gradient(
                completion_logps(
                    reference_model, reference_batch, start, args.logit_chunk_size
                )
            )
            if args.beta
            else mx.stop_gradient(logps)
        )
        loss, metrics = grpo_objective(
            logps,
            ref,
            mx.ones_like(logps, dtype=mx.bool_),
            sample["advantage"],
            beta=args.beta,
            epsilon=args.epsilon,
            epsilon_high=args.epsilon_high,
            max_tokens=args.max_completion_length,
            importance_sampling_level=args.importance_sampling_level,
            grpo_loss_type=args.grpo_loss_type,
        )
        metrics.update(vlm_batch_metrics(batch, policy))
        metrics["num_tokens"] = mx.array(logps.size)
        metrics["num_reference_tokens"] = mx.array(
            batch["input_ids"].size - 1 if args.beta else 0
        )
        return loss, metrics

    def compute_rollout(policy, rollout, compute, with_grad=False):
        # Group rewards are fixed before chunking. Match the full objective's
        # denominator exactly for unequal completion lengths (especially BNPO).
        samples = rollout["samples"]
        total_tokens = sum(
            sample["batch"]["input_ids"].shape[1] - sample["start"] - 1
            for sample in samples
        )
        loss, metrics, gradients = mx.array(0.0), {}, None
        training = policy.training
        try:
            # Qwen's training path provides differentiable recurrence ops.
            policy.train()
            for sample in samples:
                result = compute(policy, sample)
                if with_grad:
                    (value, info), grad = result
                else:
                    value, info = result
                token_count = (
                    sample["batch"]["input_ids"].shape[1] - sample["start"] - 1
                )
                weight = (
                    token_count / total_tokens
                    if args.grpo_loss_type == "bnpo"
                    else 1 / len(samples)
                )
                loss = loss + value * weight
                for key, val in info.items():
                    factor = (
                        1
                        if key.startswith("num_")
                        else (
                            token_count / total_tokens
                            if key in {"clip_ratio", "kl_clip_ratio"}
                            else 1 / len(samples)
                        )
                    )
                    metrics[key] = metrics.get(key, 0) + val * factor
                if with_grad:
                    scaled = tree_map(lambda g: g * weight, grad)
                    gradients = (
                        scaled
                        if gradients is None
                        else tree_map(lambda a, b: a + b, gradients, scaled)
                    )
                    del grad, scaled
                del result, value, info
                # Release this completion's graph before building the next one.
                mx.eval(loss, metrics, gradients)
            metrics.update(rollout["reward_metrics"])
            metrics["weight"] = mx.array(
                total_tokens if args.grpo_loss_type == "bnpo" else len(samples)
            )
            metrics["average_generated_tokens"] = mx.array(total_tokens / len(samples))
            metrics["hit_max_tokens_ratio"] = mx.array(
                sum(
                    s["batch"]["input_ids"].shape[1] - s["start"] - 1
                    == args.max_completion_length
                    for s in samples
                )
                / len(samples)
            )
            return ((loss, metrics), gradients) if with_grad else (loss, metrics)
        finally:
            policy.train(training)

    def value_and_grad_factory(policy, compiled):
        value_grad = nn.value_and_grad(policy, sample_loss)
        # Capture the Module; mx.compile flattens a Module argument to a dict.
        compute_sample = lambda sample: value_grad(policy, sample)
        if compiled:
            compute_sample = mx.compile(
                compute_sample,
                inputs=[policy.state]
                + ([reference_model.state] if reference_model is not None else []),
                outputs=[policy.state],
            )
        compute = lambda policy, sample: compute_sample(sample)
        return partial(compute_rollout, compute=compute, with_grad=True)

    def batches(dataset, train=False, seed=None):
        world = mx.distributed.init()
        for indices in iterate_batch_indices(
            len(dataset),
            args.batch_size,
            length_key=lambda i: (
                dataset.itemlen(i)
                if hasattr(dataset, "itemlen")
                else dataset[i]["inputs"]["input_ids"].size
            ),
            train=train,
            seed=seed,
            rank=world.rank(),
            world_size=world.size(),
        ):
            yield prepare_rollout(
                model,
                tokenizer,
                [dataset[i] for i in indices],
                args,
                reward_funcs,
                reward_weights,
            )

    return TrainingTask(
        loss=partial(compute_rollout, compute=sample_loss),
        batches=batches,
        prepare_dataset=lambda rows: CacheDataset(
            GRPOVisionDataset(
                rows, model.config, processing_class, config=dataset_config or args
            ),
            max_size=args.cache_size,
            process=False,
        ),
        checkpoint=enable_model_gradient_checkpointing,
        unit="completions",
        state=[reference_model.state] if reference_model is not None else [],
        value_and_grad_factory=value_and_grad_factory,
    )
