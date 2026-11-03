"""Online group-relative optimization settings."""

import math
from dataclasses import dataclass

from mlx_vlm.trainer.vlm.config import VLMTrainingArgs


@dataclass
class GRPOTrainingArgs(VLMTrainingArgs):
    group_size: int = 4
    beta: float = 0.0
    epsilon: float = 1e-4
    epsilon_high: float | None = None
    advantage_epsilon: float = 1e-4
    max_completion_length: int = 512
    temperature: float = 0.8
    top_p: float = 0.95
    top_k: int = 20
    min_p: float = 0.0
    grpo_loss_type: str = "grpo"
    importance_sampling_level: str = "token"
    logit_chunk_size: int = 128
    end_answer_token: str = ""

    def validate(self):
        super().validate()
        if self.group_size < 2:
            raise ValueError("group_size must be at least 2.")
        for name in ("max_completion_length", "logit_chunk_size"):
            if getattr(self, name) < 1:
                raise ValueError(f"{name} must be positive.")
        if self.max_completion_length >= self.max_seq_length:
            raise ValueError(
                "max_seq_length must leave room for prompt and completion."
            )
        for name in ("beta", "epsilon", "temperature", "min_p"):
            value = getattr(self, name)
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and nonnegative.")
        if self.epsilon_high is not None and (
            not math.isfinite(self.epsilon_high) or self.epsilon_high < 0
        ):
            raise ValueError("epsilon_high must be finite and nonnegative or None.")
        if not math.isfinite(self.advantage_epsilon) or self.advantage_epsilon <= 0:
            raise ValueError("advantage_epsilon must be finite and positive.")
        if not 0 < self.top_p <= 1 or not 0 <= self.min_p <= 1 or self.top_k < 0:
            raise ValueError("Invalid top_p, min_p, or top_k sampling settings.")
        if self.grpo_loss_type not in {"grpo", "bnpo", "dr_grpo"}:
            raise ValueError("grpo_loss_type must be grpo, bnpo, or dr_grpo.")
        if self.importance_sampling_level not in {"token", "sequence"}:
            raise ValueError("importance_sampling_level must be token or sequence.")
