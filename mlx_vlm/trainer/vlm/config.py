"""Sequence and media settings shared by VLM algorithms."""

import math
from dataclasses import dataclass, field

from mlx_vlm.trainer.common.config import CoreTrainingArgs


@dataclass
class VLMTrainingArgs(CoreTrainingArgs):
    """Common training options and VLM sequence/batching settings."""

    batch_size: int = field(default=1, metadata={"help": "Global minibatch size."})
    max_seq_length: int = 2048
    pad_to_multiple: int = 32
    pad_token_id: int = 0
    train_on_completions: bool = False
    assistant_id: int | None = 77091
    mtp_loss_weight: float = field(
        default=0.0,
        kw_only=True,
        metadata={"help": "Auxiliary next-next-token loss weight (0 disables MTP)."},
    )

    def validate(self) -> None:
        super().validate()
        if not math.isfinite(self.mtp_loss_weight) or self.mtp_loss_weight < 0:
            raise ValueError("mtp_loss_weight must be finite and non-negative.")
        if self.max_seq_length < 2:
            raise ValueError("max_seq_length must be at least 2 for next-token loss.")
        if self.pad_to_multiple < 1:
            raise ValueError("pad_to_multiple must be at least 1.")
