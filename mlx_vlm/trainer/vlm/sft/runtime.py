"""Supervised fine-tuning for vision-language models."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping, Sequence
from functools import partial
from typing import Any

import mlx.core as mx
import mlx.nn as nn
import numpy as np

from mlx_vlm.trainer.common.utils import (
    enable_model_gradient_checkpointing,
    evaluate_weighted_batches,
    iterate_batch_indices,
    save_trainable_weights,
)
from mlx_vlm.trainer.losses.cross_entropy import cross_entropy
from mlx_vlm.trainer.vlm.config import VLMTrainingArgs

_TEXT_FIELDS = frozenset({"input_ids", "attention_mask", "completion_mask", "labels"})
_IMAGE_FIELDS = ("pixel_values", "images", "image_embeds")
_AUDIO_VIDEO_FIELDS = (
    "audio_features",
    "input_features",
    "input_values",
    "audio",
    "audios",
    "video_values",
    "videos",
)
_SIGNATURE_FIELDS = ("pixel_values", "image_embeds", *_AUDIO_VIDEO_FIELDS)


def _model_type(model: nn.Module) -> str | None:
    model_type = getattr(model, "model_type", None)
    if model_type is not None:
        return model_type
    config = getattr(model, "config", None)
    if isinstance(config, Mapping):
        return config.get("model_type")
    return getattr(config, "model_type", None)


def _logits_from_output(output: Any) -> mx.array:
    if isinstance(output, Mapping):
        logits = output.get("logits")
    else:
        logits = getattr(output, "logits", output)
    if not isinstance(logits, mx.array):
        raise TypeError(
            "A VLM must return an mx.array or an output containing mx.array "
            f"logits; received {type(output)!r}."
        )
    return logits


def forward_vlm_logits(
    model: nn.Module,
    input_ids: mx.array,
    attention_mask: mx.array,
    batch: Mapping[str, Any],
) -> mx.array:
    """Run a VLM with its processor-produced multimodal fields and return logits."""
    model_inputs = {
        key: value for key, value in batch.items() if key not in _TEXT_FIELDS
    }
    pixel_values = model_inputs.pop("pixel_values", None)
    if _model_type(model) == "gemma4_unified":
        attention_mask = None
    return _logits_from_output(
        model(input_ids, pixel_values, attention_mask, **model_inputs)
    ).astype(mx.float32)


def _completion_mask(batch: Mapping[str, Any], assistant_id: int | None) -> mx.array:
    """Get completion positions, falling back to an explicit assistant token."""
    if "completion_mask" in batch:
        return batch["completion_mask"]
    if assistant_id is None:
        raise ValueError(
            "Completion-only loss requires `completion_mask` in the batch or an "
            "assistant_id for fallback mask construction."
        )

    input_ids = batch["input_ids"]
    matches = input_ids == assistant_id
    positions = mx.arange(input_ids.shape[1])[None, :]
    first_assistant = mx.argmax(matches.astype(mx.int32), axis=1)[:, None]
    has_assistant = mx.any(matches, axis=1)[:, None]
    return mx.logical_and(positions > first_assistant, has_assistant).astype(mx.int32)


def vlm_sft_loss(
    model: nn.Module,
    batch: Mapping[str, Any],
    *,
    train_on_completions: bool = False,
    assistant_id: int | None = 77091,
) -> tuple[mx.array, dict[str, mx.array]]:
    """Compute masked causal cross-entropy for a VLM batch.

    The batch stores unshifted ``input_ids``. The model receives all but the
    last token and predicts the remaining targets. Padding and (optionally)
    prompt tokens are excluded from the shared cross-entropy objective.
    """
    required = {"input_ids", "attention_mask"}
    missing = required.difference(batch)
    if missing:
        raise KeyError("Missing VLM batch fields: " + ", ".join(sorted(missing)))

    full_input_ids = batch["input_ids"]
    attention_mask = batch["attention_mask"]
    if full_input_ids.ndim != 2 or full_input_ids.shape[1] < 2:
        raise ValueError(
            "input_ids must have shape [batch, sequence] with length >= 2."
        )
    if attention_mask.shape != full_input_ids.shape:
        raise ValueError("attention_mask must have the same shape as input_ids.")

    input_ids = full_input_ids[:, :-1]
    targets = full_input_ids[:, 1:]
    model_attention_mask = attention_mask[:, :-1]
    loss_mask = attention_mask[:, 1:].astype(mx.bool_)
    if train_on_completions:
        completion_mask = _completion_mask(batch, assistant_id)
        if completion_mask.shape != full_input_ids.shape:
            raise ValueError("completion_mask must have the same shape as input_ids.")
        loss_mask = mx.logical_and(loss_mask, completion_mask[:, 1:].astype(mx.bool_))

    logits = forward_vlm_logits(
        model,
        input_ids,
        model_attention_mask,
        batch,
    )
    if logits.ndim != 3 or logits.shape[:-1] != targets.shape:
        raise ValueError(
            "VLM logits must align position-for-position with shifted targets; "
            f"got logits {logits.shape} and targets {targets.shape}."
        )

    loss, metrics = cross_entropy(
        logits, targets, loss_mask, vision_mask=vlm_vision_mask(model, batch)
    )
    return loss, {**metrics, **vlm_batch_metrics(batch)}


def _squeeze_batch(value: Any) -> Any:
    if (
        isinstance(value, (mx.array, np.ndarray))
        and value.ndim > 0
        and value.shape[0] == 1
    ):
        return value[0]
    return value


def _as_array(value: Any) -> mx.array:
    return value if isinstance(value, mx.array) else mx.array(value)


def _collate_feature(values: Sequence[Any], name: str) -> Any:
    values = [_squeeze_batch(value) for value in values]
    if all(isinstance(value, mx.array) for value in values):
        arrays = values
        if all(array.shape == arrays[0].shape for array in arrays):
            return mx.stack(arrays)
        if name in _IMAGE_FIELDS and all(
            array.ndim == arrays[0].ndim
            and array.ndim > 0
            and array.shape[1:] == arrays[0].shape[1:]
            for array in arrays
        ):
            return mx.concatenate(arrays, axis=0)
        if all(
            array.ndim == arrays[0].ndim
            and array.ndim > 0
            and array.shape[:-1] == arrays[0].shape[:-1]
            for array in arrays
        ):
            axis = arrays[0].ndim - 1
        elif all(
            array.ndim == arrays[0].ndim
            and array.ndim > 0
            and array.shape[1:] == arrays[0].shape[1:]
            for array in arrays
        ):
            axis = 0
        else:
            raise ValueError(
                f"Cannot collate VLM field {name!r} with shapes "
                f"{[array.shape for array in arrays]}."
            )
        target_length = max(array.shape[axis] for array in arrays)
        padded = []
        for array in arrays:
            padding = [(0, 0)] * array.ndim
            padding[axis] = (0, target_length - array.shape[axis])
            padded.append(mx.pad(array, padding))
        return mx.stack(padded)

    if all(isinstance(value, np.ndarray) for value in values):
        return _collate_feature([mx.array(value) for value in values], name)
    if all(np.isscalar(value) for value in values):
        try:
            return mx.array(values)
        except (TypeError, ValueError):
            pass
    if all(value == values[0] for value in values):
        return values[0]
    return values


def collate_vlm_batch(
    samples: Sequence[Mapping[str, Any]],
    max_seq_length: int,
    *,
    pad_to_multiple: int = 32,
    pad_token_id: int = 0,
    image_token_id: int | None = None,
) -> dict[str, Any]:
    """Pad text fields and collate processor outputs without breaking media alignment."""
    if not samples:
        raise ValueError("Cannot collate an empty VLM batch.")
    if max_seq_length < 2 or pad_to_multiple < 1:
        raise ValueError("max_seq_length >= 2 and pad_to_multiple >= 1 are required.")

    ids = []
    masks = []
    completions = []
    has_completion = any("completion_mask" in sample for sample in samples)
    for sample_index, sample in enumerate(samples):
        token_ids = np.asarray(_squeeze_batch(sample["input_ids"])).reshape(-1)
        if len(token_ids) < 2:
            raise ValueError(f"VLM row {sample_index} contains fewer than two tokens.")
        has_audio_or_video = any(
            sample.get(key) is not None for key in _AUDIO_VIDEO_FIELDS
        )
        has_media = has_audio_or_video or any(
            sample.get(key) is not None for key in _IMAGE_FIELDS
        )
        if len(token_ids) > max_seq_length and has_audio_or_video:
            raise ValueError(
                f"VLM row {sample_index} contains audio or video media and exceeds "
                f"max_seq_length={max_seq_length}. Media placeholder alignment is "
                "not preserved by truncation; increase max_seq_length or shorten "
                "the example during preprocessing."
            )
        if len(token_ids) > max_seq_length and has_media and image_token_id is None:
            raise ValueError(
                f"VLM row {sample_index} contains media but its model config has no "
                "image token ID for a safe truncation check. Increase "
                "max_seq_length so the row is not truncated."
            )
        if image_token_id is not None and len(token_ids) > max_seq_length:
            total = int(np.count_nonzero(token_ids == image_token_id))
            kept = int(np.count_nonzero(token_ids[:max_seq_length] == image_token_id))
            if kept != total:
                raise ValueError(
                    f"Truncating VLM row {sample_index} at max_seq_length="
                    f"{max_seq_length} would remove image placeholders ({kept} of "
                    f"{total} retained) while its image features remain. Increase "
                    "max_seq_length or reduce the image resolution."
                )
        ids.append(token_ids[:max_seq_length].astype(np.int32, copy=False))

        mask = sample.get("attention_mask")
        if mask is None:
            mask_array = np.ones(len(token_ids), dtype=np.int32)
        else:
            mask_array = np.asarray(_squeeze_batch(mask)).reshape(-1)
            if mask_array.shape != token_ids.shape:
                raise ValueError("Each attention_mask must align with input_ids.")
        masks.append(mask_array[:max_seq_length].astype(np.int32, copy=False))

        completion = sample.get("completion_mask")
        if completion is not None:
            completion_array = np.asarray(_squeeze_batch(completion)).reshape(-1)
            if completion_array.shape != token_ids.shape:
                raise ValueError("Each completion_mask must align with input_ids.")
            completion_array = completion_array[:max_seq_length]
            if not np.any(completion_array):
                raise ValueError(
                    f"VLM row {sample_index} has no completion tokens within "
                    f"max_seq_length={max_seq_length}."
                )
            if not np.any(completion_array[1:]):
                raise ValueError(
                    f"VLM row {sample_index} has no supervised next-token targets "
                    f"within max_seq_length={max_seq_length}. Increase the limit "
                    "or shorten the prompt/completion."
                )
            completions.append(completion_array.astype(np.int32, copy=False))
        elif has_completion:
            completions.append(
                np.zeros(min(len(token_ids), max_seq_length), dtype=np.int32)
            )

    longest = max(map(len, ids))
    padded_length = max(
        2,
        ((longest + pad_to_multiple - 1) // pad_to_multiple) * pad_to_multiple,
    )
    padded_length = min(padded_length, max_seq_length)
    input_batch = np.full((len(samples), padded_length), pad_token_id, dtype=np.int32)
    attention_batch = np.zeros((len(samples), padded_length), dtype=np.int32)
    completion_batch = np.zeros((len(samples), padded_length), dtype=np.int32)
    for row, (token_ids, mask) in enumerate(zip(ids, masks)):
        input_batch[row, : len(token_ids)] = token_ids
        attention_batch[row, : len(mask)] = mask
        if has_completion:
            completion_batch[row, : len(completions[row])] = completions[row]

    batch: dict[str, Any] = {
        "input_ids": mx.array(input_batch),
        "attention_mask": mx.array(attention_batch),
    }
    if has_completion:
        batch["completion_mask"] = mx.array(completion_batch)

    extra_keys = (
        set().union(*(sample.keys() for sample in samples)).difference(_TEXT_FIELDS)
    )
    for key in sorted(extra_keys):
        values = [sample.get(key) for sample in samples]
        if all(value is None for value in values):
            batch[key] = None
            continue
        if any(value is None for value in values):
            raise ValueError(
                f"VLM field {key!r} is present for only some examples in a batch. "
                "Use homogeneous media rows or batch_size=1."
            )
        if key in {"image_grid_thw", "video_grid_thw"}:
            grids = []
            for value in values:
                grid = _as_array(_squeeze_batch(value))
                if grid.ndim == 1:
                    grid = grid[None, :]
                if grid.ndim != 2 or grid.shape[1] != 3:
                    raise ValueError(
                        f"Expected {key} with shape [num_media, 3], got {grid.shape}."
                    )
                grids.append(grid)
            batch[key] = mx.concatenate(grids, axis=0)
            continue
        batch[key] = _collate_feature(values, key)
    return batch


def _image_token_id(dataset: Any) -> int | None:
    config = getattr(dataset, "config", None)
    if isinstance(config, Mapping):
        value = config.get("image_token_index")
        if value is None:
            value = config.get("image_token_id")
    else:
        value = getattr(config, "image_token_index", None)
        if value is None:
            value = getattr(config, "image_token_id", None)
    return None if value is None else int(value)


def _sample_signature(sample: Mapping[str, Any]) -> tuple[str, ...]:
    return tuple(key for key in _SIGNATURE_FIELDS if sample.get(key) is not None)


def iterate_vlm_batches(
    dataset: Any,
    batch_size: int,
    max_seq_length: int,
    *,
    train: bool = False,
    pad_to_multiple: int = 32,
    pad_token_id: int = 0,
    rank: int | None = None,
    world_size: int | None = None,
    seed: int | None = None,
) -> Iterable[dict[str, Any]]:
    """Length-grouped, modality-homogeneous batches, sharded across ranks."""
    if rank is None or world_size is None:
        world = mx.distributed.init()
        rank, world_size = world.rank(), world.size()
    signature = getattr(dataset, "media_signature", None)
    if signature is None:

        def signature(index):
            return _sample_signature(dataset[index])

    length = getattr(dataset, "itemlen", None)
    if length is None:

        def length(index):
            return dataset[index]["input_ids"].shape[-1]

    batches = iterate_batch_indices(
        len(dataset),
        batch_size,
        length_key=length,
        group_key=signature,
        train=train,
        rank=rank,
        world_size=world_size,
        seed=seed,
    )
    for indices in batches:
        yield collate_vlm_batch(
            [dataset[index] for index in indices],
            max_seq_length,
            pad_to_multiple=pad_to_multiple,
            pad_token_id=pad_token_id,
            image_token_id=_image_token_id(dataset),
        )


def evaluate_vlm(
    model: nn.Module,
    dataset: Any,
    batch_size: int,
    num_batches: int,
    max_seq_length: int = 2048,
    *,
    loss_fn: Callable = vlm_sft_loss,
    train_on_completions: bool = False,
    assistant_id: int | None = 77091,
    iterate_batches: Callable = iterate_vlm_batches,
    pad_to_multiple: int = 32,
    pad_token_id: int = 0,
) -> float:
    """Evaluate token-weighted VLM cross-entropy on a finite split."""
    if len(dataset) == 0:
        raise ValueError("Cannot evaluate an empty VLM validation dataset.")
    batches = iterate_batches(
        dataset,
        batch_size,
        max_seq_length,
        train=False,
        pad_to_multiple=pad_to_multiple,
        pad_token_id=pad_token_id,
    )

    return evaluate_weighted_batches(
        model,
        batches,
        partial(
            loss_fn,
            train_on_completions=train_on_completions,
            assistant_id=assistant_id,
        ),
        num_batches,
        empty_error="VLM evaluation produced zero supervised tokens.",
    )


_enable_vlm_gradient_checkpointing = enable_model_gradient_checkpointing
_save_vlm_weights = save_trainable_weights


def vlm_vision_mask(model: nn.Module, batch: Mapping[str, Any]) -> mx.array:
    """Identify valid image/video slots in the actual causal forward inputs.

    Counts processor-expanded image/video token positions, rather than image
    boundary markers or the vision encoder's pre-merge patch sequence.
    """
    config = getattr(model, "config", None)
    input_ids = batch["input_ids"][:, :-1]
    mask = mx.zeros(input_ids.shape, dtype=mx.bool_)
    token_ids = set()
    for name in (
        "image_token_index",
        "image_token_id",
        "video_token_index",
        "video_token_id",
    ):
        token_id = (
            config.get(name)
            if isinstance(config, Mapping)
            else getattr(config, name, None)
        )
        if token_id is not None:
            token_ids.add(int(token_id))
    for token_id in sorted(token_ids):
        mask = mx.logical_or(mask, input_ids == token_id)
    return mx.logical_and(mask, batch["attention_mask"][:, :-1].astype(mx.bool_))


def vlm_batch_metrics(
    batch: Mapping[str, Any], model: nn.Module | None = None
) -> dict[str, mx.array]:
    """Count non-padding causal forward inputs, including prompt/media tokens.

    Preference batches include both policy sequences. DPO reference forwards
    and checkpoint recomputation are excluded from these data-volume counts.
    """
    sequences = (batch["chosen"], batch["rejected"]) if "chosen" in batch else (batch,)
    tokens = mx.array(0, dtype=mx.int32)
    vision_tokens = mx.array(0, dtype=mx.int32)
    slots, count = 0, 0
    for sequence in sequences:
        attention = sequence["attention_mask"][:, :-1]
        tokens += attention.astype(mx.bool_).sum()
        slots += attention.size
        count += attention.shape[0]
        if model is not None:
            vision_tokens += vlm_vision_mask(model, sequence).sum()
    metrics = {
        "num_processed_tokens": tokens,
        "num_padded_tokens": mx.array(slots, dtype=mx.int32) - tokens,
        "num_sequences": mx.array(count, dtype=mx.int32),
    }
    if model is not None:
        metrics["num_vision_tokens"] = vision_tokens
    return metrics


def _make_vlm_task(args, dataset_factory, *, loss_fn, iterate_batches, unit="tokens"):
    """Keep media caching, padding, and lifecycle hooks identical across VLM tasks."""
    from mlx_vlm.trainer.common.task import TrainingTask
    from mlx_vlm.trainer.datasets.loading import CacheDataset

    return TrainingTask(
        loss=loss_fn,
        batches=partial(
            iterate_batches,
            batch_size=args.batch_size,
            max_seq_length=args.max_seq_length,
            pad_to_multiple=args.pad_to_multiple,
            pad_token_id=args.pad_token_id,
        ),
        prepare_dataset=lambda rows: CacheDataset(
            dataset_factory(rows),
            max_size=args.cache_size,
            process=False,
        ),
        checkpoint=_enable_vlm_gradient_checkpointing,
        save_weights=_save_vlm_weights,
        unit=unit,
    )


def make_task(
    model,
    args,
    processing_class=None,
    dataset_config=None,
    *,
    loss_fn=vlm_sft_loss,
    iterate_batches=iterate_vlm_batches,
):
    """Bind VLM preprocessing and supervised cross-entropy."""
    from mlx_vlm.trainer.vlm.sft.dataset import VisionDataset

    return _make_vlm_task(
        args,
        lambda rows: VisionDataset(
            rows, model.config, processing_class, config=dataset_config or args
        ),
        loss_fn=partial(
            loss_fn,
            train_on_completions=args.train_on_completions,
            assistant_id=args.assistant_id,
        ),
        iterate_batches=iterate_batches,
    )


def train_vlm(
    model,
    optimizer,
    train_dataset,
    val_dataset=None,
    args=None,
    *,
    loss_fn=vlm_sft_loss,
    iterate_batches=iterate_vlm_batches,
    training_callback=None,
    seed=None,
):
    """Low-level MLX-LM-style API for prepared VLM examples."""
    from mlx_vlm.trainer.common.engine import train

    args = args or VLMTrainingArgs()
    args.validate()
    task = make_task(model, args, loss_fn=loss_fn, iterate_batches=iterate_batches)
    return train(
        model,
        optimizer,
        train_dataset,
        val_dataset,
        args,
        task,
        training_callback,
        seed,
    )
