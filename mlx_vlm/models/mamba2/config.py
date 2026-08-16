import math
from dataclasses import dataclass
from typing import Optional, Tuple, Union

from ..base import BaseModelConfig


@dataclass
class ModelConfig(BaseModelConfig):
    model_type: str
    num_heads: int
    head_dim: int
    vocab_size: int
    hidden_size: int
    state_size: int
    num_hidden_layers: int
    layer_norm_epsilon: float
    conv_kernel: int
    n_groups: int
    use_bias: bool
    use_conv_bias: bool
    tie_word_embeddings: bool
    time_step_limit: Tuple[float, float]
    time_step_rank: Union[int, str]
    intermediate_size: Optional[int] = None
    ssm_state_size: Optional[int] = None
    max_position_embeddings: int = 2056

    def __post_init__(self):
        if self.time_step_rank == "auto":
            self.time_step_rank = math.ceil(self.hidden_size / 16)
        if self.ssm_state_size is None:
            self.ssm_state_size = self.state_size
