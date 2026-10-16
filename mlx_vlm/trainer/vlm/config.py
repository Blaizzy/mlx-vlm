"""Sequence and media settings shared by VLM algorithms."""

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

    def validate(self) -> None:
        super().validate()
        if self.max_seq_length < 2:
            raise ValueError("max_seq_length must be at least 2 for next-token loss.")
        if self.pad_to_multiple < 1:
            raise ValueError("pad_to_multiple must be at least 1.")
