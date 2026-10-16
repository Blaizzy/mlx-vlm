"""Odds ratio preference optimization settings."""

from dataclasses import dataclass

from mlx_vlm.trainer.vlm.config import VLMTrainingArgs


@dataclass
class ORPOTrainingArgs(VLMTrainingArgs):
    """VLM training settings for reference-free ORPO."""

    beta: float = 0.1
    eps: float = 1e-06

    def validate(self) -> None:
        super().validate()
        if self.beta <= 0:
            raise ValueError("ORPO beta must be greater than zero.")
        if not 0 < self.eps < 1:
            raise ValueError("ORPO eps must be in the interval (0, 1).")
