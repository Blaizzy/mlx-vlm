"""The small contract between a modality/objective and the training loop."""

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any


@dataclass
class TrainingTask:
    """Bind model-specific work without inheriting or copying a trainer.

    ``loss(model, batch, *loss_args)`` returns (mean loss, metrics).
    ``batches(dataset, train=False, seed=None)`` yields finite evaluation
    batches or repeating training batches. Weights are tokens, pairs, or
    examples, stored as ``metrics['weight']`` (default 1), and determine the
    reported/evaluated mean, not gradient scaling. Scalar ``num_*`` metrics
    are summed; other scalar metrics use weighted means. Metric dictionaries
    must have the same keys across distributed ranks for each batch.
    Auxiliary models belong in ``state`` when compilation is enabled.
    """

    loss: Callable
    batches: Callable
    prepare_dataset: Callable | None = None
    checkpoint: Callable | None = None
    save_weights: Callable | None = None
    loss_args: Callable | None = None
    unit: str = "examples"
    state: list[Any] = field(default_factory=list)

    def objective(self, model, batch):
        extra = self.loss_args(batch) if self.loss_args is not None else ()
        return self.loss(model, batch, *extra)
