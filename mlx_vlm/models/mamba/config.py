import math
from dataclasses import dataclass

from ..base import BaseModelConfig


@dataclass
class ModelConfig(BaseModelConfig):
    model_type: str
    vocab_size: int
    hidden_size: int
    intermediate_size: int
    state_size: int
    num_hidden_layers: int
    conv_kernel: int
    use_bias: bool
    use_conv_bias: bool
    time_step_rank: int
    tie_word_embeddings: bool = True
    use_bcdt_rms: bool = False
    mixer_rms_eps: float = 1e-6

    def __post_init__(self):
        if self.time_step_rank == "auto":
            self.time_step_rank = math.ceil(self.hidden_size / 16)
        if self.model_type == "falcon_mamba":
            self.use_bcdt_rms = True
