# Copyright © 2026 MLX-VLM

from dataclasses import dataclass, field
from typing import Any

import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_map

from ..sft.runtime import _completion_mask, vlm_batch_metrics
from ..sft.trainer import TrainingArgs


@dataclass
class ORPOTrainingArgs(TrainingArgs):
    beta: float = field(
        default=0.1,
        metadata={"help": "Weight of the ORPO preference objective"},
    )
    eps: float = field(
        default=1e-8,
        metadata={"help": "Small constant for numerical stability in log calculations"},
    )


def get_logps(model, batch, train_on_completions=False, assistant_id=77091):
    """Keep the historical scoring signature over shared response scoring."""
    from ..preference.loss import sequence_logps

    completion = (
        _completion_mask(batch, assistant_id)
        if train_on_completions
        else batch["attention_mask"]
    )
    scores, counts, logits = sequence_logps(
        model, {**batch, "completion_mask": completion}, length_normalize=True
    )
    return scores, logits.sum() / mx.maximum(counts.sum(), 1)


def orpo_loss(
    chosen_logps,
    chosen_logits_mean,
    rejected_logps,
    rejected_logits_mean,
    chosen_masks,
    rejected_masks,
    beta=0.1,
    eps=1e-8,
):
    """Adapt the historical score-based API to the shared ORPO objective."""
    from ...losses.preference import _log1mexp, orpo_objective

    losses, metrics = orpo_objective(chosen_logps, rejected_logps, beta=beta, eps=eps)
    metrics.update(
        accuracies=metrics["preference_accuracy"],
        margins=beta * metrics["odds_margin"],
        sft_term=metrics["chosen_nll"],
        orpo_pref_term=metrics["preference_loss"],
        chosen_reward=mx.mean(beta * (chosen_logps - _log1mexp(chosen_logps, eps))),
        rejected_reward=mx.mean(
            beta * (rejected_logps - _log1mexp(rejected_logps, eps))
        ),
        policy_chosen_logps=metrics["chosen_logps"],
        policy_rejected_logps=metrics["rejected_logps"],
        chosen_logits_mean=chosen_logits_mean,
        rejected_logits_mean=rejected_logits_mean,
        num_tokens=chosen_masks.sum() + rejected_masks.sum(),
    )
    return mx.mean(losses), metrics


def iterate_batches(dataset, batch_size, max_seq_length, train=False):
    """Adapt historical prefixed rows to the shared paired batch iterator."""
    from ..preference.dataset import iterate_vlm_preference_batches

    class PairedRows:
        config = getattr(dataset, "config", None)

        def __len__(self):
            return len(dataset)

        def __getitem__(self, index):
            row = dataset[index]
            if "chosen" in row and "rejected" in row:
                return row
            return {
                candidate: {
                    key.removeprefix(candidate + "_"): value
                    for key, value in row.items()
                    if key.startswith(candidate + "_")
                }
                for candidate in ("chosen", "rejected")
            }

    return iterate_vlm_preference_batches(
        PairedRows(), batch_size, max_seq_length, train=train
    )


def _legacy_task(args, loss_fn, train_on_completions, assistant_id):
    """Adapt historical batches and score-based losses to the common engine."""
    from ...common.task import TrainingTask
    from ..sft.mtp import add_mtp_loss
    from ..sft.runtime import _enable_vlm_gradient_checkpointing, _save_vlm_weights

    def objective(model, batch):
        chosen, rejected = batch["chosen"], batch["rejected"]
        chosen_logps, chosen_logits = get_logps(
            model, chosen, train_on_completions, assistant_id
        )
        rejected_logps, rejected_logits = get_logps(
            model, rejected, train_on_completions, assistant_id
        )
        loss, metrics = loss_fn(
            chosen_logps,
            chosen_logits,
            rejected_logps,
            rejected_logits,
            chosen["attention_mask"],
            rejected["attention_mask"],
            beta=args.beta,
            **({"eps": args.eps} if loss_fn is orpo_loss else {}),
        )
        metrics.update(vlm_batch_metrics(batch, model))
        mask = chosen["attention_mask"][:, 1:]
        if train_on_completions:
            mask = mx.logical_and(mask, _completion_mask(chosen, assistant_id)[:, 1:])
        return add_mtp_loss(
            model, chosen, loss, metrics, args.mtp_loss_weight, loss_mask=mask
        )

    return TrainingTask(
        loss=objective,
        batches=lambda rows, train=False, seed=None: iterate_batches(
            rows, args.batch_size, args.max_seq_length, train=train
        ),
        checkpoint=_enable_vlm_gradient_checkpointing,
        save_weights=_save_vlm_weights,
        unit="preference_pairs",
    )


def _shared_args(args):
    from dataclasses import fields

    from .config import ORPOTrainingArgs as SharedArgs

    return SharedArgs(
        **{
            field.name: getattr(args, field.name)
            for field in fields(SharedArgs)
            if hasattr(args, field.name)
        }
    )


def evaluate_orpo(
    model,
    dataset,
    batch_size,
    num_batches,
    max_seq_length=2048,
    loss_fn=orpo_loss,
    train_on_completions=False,
    assistant_id=77091,
    return_metrics=False,
    mtp_loss_weight=0.0,
):
    """Evaluate historical preference batches with the common engine."""
    from ...common.engine import evaluate

    args = _shared_args(
        ORPOTrainingArgs(
            batch_size=batch_size,
            val_batches=num_batches,
            max_seq_length=max_seq_length,
            mtp_loss_weight=mtp_loss_weight,
        )
    )
    task = _legacy_task(args, loss_fn, train_on_completions, assistant_id)
    return evaluate(model, dataset, task, args, return_metrics=return_metrics)


def train_orpo(
    model,
    optimizer,
    train_dataset,
    val_dataset=None,
    args=None,
    loss_fn=orpo_loss,
    train_on_completions=False,
    assistant_id=77091,
    *,
    training_callback=None,
    seed=None,
):
    """Train through the common engine, adapting historical calls when needed.

    Modern ORPOTrainingArgs use paired batches from vlm.preference.dataset.
    The historical args class retains support for prefixed dataset fields and
    custom score-based losses.
    """
    from ...common.engine import train
    from .config import ORPOTrainingArgs as SharedArgs
    from .runtime import train_orpo as train_shared

    if args is None or isinstance(args, SharedArgs):
        if loss_fn is not orpo_loss or train_on_completions or assistant_id != 77091:
            raise ValueError(
                "Legacy loss/masking options require legacy ORPOTrainingArgs."
            )
        return train_shared(
            model,
            optimizer,
            train_dataset,
            val_dataset,
            args,
            training_callback=training_callback,
            seed=seed,
        )
    shared_args = _shared_args(args)
    task = _legacy_task(shared_args, loss_fn, train_on_completions, assistant_id)
    if args.grad_clip is not None:

        def gradients(policy, compile):
            compute = nn.value_and_grad(policy, task.loss)

            def clipped(*values):
                result, grad = compute(*values)
                return result, tree_map(
                    lambda g: mx.clip(g, -args.grad_clip, args.grad_clip), grad
                )

            return clipped

        task.value_and_grad_factory = gradients
    return train(
        model,
        optimizer,
        train_dataset,
        val_dataset,
        shared_args,
        task,
        training_callback,
        seed,
    )


@dataclass
class ORPOTrainer:
    """Configure and run the functional preference-training recipe."""

    model: nn.Module
    optimizer: Any
    train_dataset: Any
    val_dataset: Any = None
    args: ORPOTrainingArgs = field(default_factory=ORPOTrainingArgs)
    train_on_completions: bool = False
    assistant_id: int = 77091

    def fit(self):
        """Run training with the configured model, data, and options."""
        return train_orpo(
            model=self.model,
            optimizer=self.optimizer,
            train_dataset=self.train_dataset,
            val_dataset=self.val_dataset,
            args=self.args,
            train_on_completions=self.train_on_completions,
            assistant_id=self.assistant_id,
        )
