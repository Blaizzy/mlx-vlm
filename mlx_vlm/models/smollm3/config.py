from dataclasses import dataclass
from typing import List, Optional

from ..llama.config import ModelConfig as LlamaModelConfig


@dataclass
class ModelConfig(LlamaModelConfig):
    no_rope_layer_interval: int = 4
    no_rope_layers: Optional[List[int]] = None

    def __post_init__(self):
        super().__post_init__()
        if self.no_rope_layers is None:
            self.no_rope_layers = [
                int((i + 1) % self.no_rope_layer_interval != 0)
                for i in range(self.num_hidden_layers)
            ]
