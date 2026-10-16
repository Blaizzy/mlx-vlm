import mlx.core as mx
import mlx.nn as nn

from ..qwen3_5_text import Model as Qwen3_5TextModel
from ..switch_layers import (
    EXPERT_WEIGHT_SUFFIXES,
    move_expert_projection,
    split_expert_projection,
)
from .config import ModelConfig
from .language import LanguageModel


class Model(Qwen3_5TextModel):
    def __init__(self, config: ModelConfig):
        nn.Module.__init__(self)
        self.config = config
        self.model_type = config.model_type
        self.language_model = LanguageModel(config, config)

    def sanitize(self, weights):
        prefix = "model.language_model."
        weights = {
            ("model." + key[len(prefix) :] if key.startswith(prefix) else key): value
            for key, value in weights.items()
            if not key.endswith(".input_global_scale")
        }
        return self._stack_experts(super().sanitize(weights))

    def _stack_experts(self, weights):
        """Split fused gate/up experts, or stack per-expert tensors, into
        the SwitchGLU layout."""
        for layer_idx in range(self.config.num_hidden_layers):
            prefix = f"language_model.model.layers.{layer_idx}.mlp"
            gate_up_key = f"{prefix}.experts.gate_up_proj"
            if gate_up_key in weights or f"{gate_up_key}.weight" in weights:
                for name in ("gate_up_proj", "down_proj"):
                    key = f"{prefix}.experts.{name}"
                    if f"{key}_scales" in weights:
                        weights[f"{key}.scales"] = weights.pop(f"{key}_scales")
                split_expert_projection(
                    weights,
                    gate_up_key,
                    [f"{prefix}.switch_mlp.gate_proj", f"{prefix}.switch_mlp.up_proj"],
                )
                move_expert_projection(
                    weights,
                    f"{prefix}.experts.down_proj",
                    f"{prefix}.switch_mlp.down_proj",
                )
            elif f"{prefix}.experts.0.up_proj.weight" in weights:
                for name in ("up_proj", "down_proj", "gate_proj"):
                    for suffix in EXPERT_WEIGHT_SUFFIXES:
                        if f"{prefix}.experts.0.{name}.{suffix}" not in weights:
                            continue
                        weights[f"{prefix}.switch_mlp.{name}.{suffix}"] = mx.stack(
                            [
                                weights.pop(f"{prefix}.experts.{e}.{name}.{suffix}")
                                for e in range(self.config.num_experts)
                            ]
                        )
        return weights
