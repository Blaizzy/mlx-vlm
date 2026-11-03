"""Direct preference optimization settings."""

from dataclasses import dataclass

from mlx_vlm.trainer.vlm.config import VLMTrainingArgs


@dataclass
class DPOTrainingArgs(VLMTrainingArgs):
    """VLM training settings for reference-based DPO."""

    beta: float = 0.1
    loss_type: str = "sigmoid"
    delta: float = 50.0

    def validate(self) -> None:
        super().validate()
        if self.beta <= 0:
            raise ValueError("DPO beta must be greater than zero.")
        if self.loss_type not in {"sigmoid", "hinge", "ipo", "dpop"}:
            raise ValueError("DPO loss_type must be sigmoid, hinge, ipo, or dpop.")
        if self.delta < 0:
            raise ValueError("DPO delta must be nonnegative.")
