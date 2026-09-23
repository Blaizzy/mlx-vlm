from dataclasses import dataclass

from ..qwen3_5_moe.config import TextConfig


@dataclass
class ModelConfig(TextConfig):
    def __post_init__(self):
        if self.rope_parameters and "mrope_section" not in self.rope_parameters:
            head_dim = self.head_dim or self.hidden_size // self.num_attention_heads
            rotary_factor = self.rope_parameters.get("partial_rotary_factor", 1.0)
            self.rope_parameters = {
                **self.rope_parameters,
                "mrope_section": [int(head_dim * rotary_factor) // 2, 0, 0],
            }
        super().__post_init__()
