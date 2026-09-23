from dataclasses import dataclass
from typing import Dict, Optional

from ..qwen3_5.config import sanitize_quantization_config
from ..qwen3_5_moe.config import TextConfig


@dataclass
class ModelConfig(TextConfig):
    quantization: Optional[Dict] = None
    quantization_config: Optional[Dict] = None
    output_gate_type: Optional[str] = None

    def __post_init__(self):
        # The shared gated delta-net norm applies silu to the output gate.
        if self.output_gate_type not in (None, "swish", "silu"):
            raise ValueError(
                f"output_gate_type {self.output_gate_type!r} is not supported; "
                "the linear-attention output gate is silu"
            )
        if self.rope_parameters and "mrope_section" not in self.rope_parameters:
            # Many text-only checkpoints omit mrope_section. Without vision
            # grids every position axis carries the same index, so any split
            # is plain RoPE; give every rotary pair to the first axis.
            head_dim = self.head_dim or self.hidden_size // self.num_attention_heads
            rotary_factor = self.rope_parameters.get("partial_rotary_factor", 1.0)
            self.rope_parameters = {
                **self.rope_parameters,
                "mrope_section": [int(head_dim * rotary_factor) // 2, 0, 0],
            }
        super().__post_init__()
        quantization = self.quantization
        self.quantization = sanitize_quantization_config(quantization)
        if self.quantization_config == quantization:
            self.quantization_config = self.quantization
        else:
            self.quantization_config = sanitize_quantization_config(
                self.quantization_config
            )
