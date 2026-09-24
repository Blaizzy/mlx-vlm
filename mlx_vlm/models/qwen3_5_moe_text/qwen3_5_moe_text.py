import mlx.core as mx
import mlx.nn as nn

from ..qwen3_5_text import Model as Qwen3_5TextModel
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
            if gate_up_key in weights:
                gate_up = weights.pop(gate_up_key)
                if gate_up.ndim != 3:
                    raise ValueError(
                        f"{gate_up_key} has shape {gate_up.shape}; expected "
                        "[num_experts, 2 * moe_intermediate_size, hidden_size]"
                    )
                mid = gate_up.shape[-2] // 2
                weights[f"{prefix}.switch_mlp.gate_proj.weight"] = gate_up[..., :mid, :]
                weights[f"{prefix}.switch_mlp.up_proj.weight"] = gate_up[..., mid:, :]
                weights[f"{prefix}.switch_mlp.down_proj.weight"] = weights.pop(
                    f"{prefix}.experts.down_proj"
                )
            elif f"{prefix}.experts.0.up_proj.weight" in weights:
                for name in ("up_proj", "down_proj", "gate_proj"):
                    for suffix in ("weight", "scales"):
                        if f"{prefix}.experts.0.{name}.{suffix}" not in weights:
                            continue
                        weights[f"{prefix}.switch_mlp.{name}.{suffix}"] = mx.stack(
                            [
                                weights.pop(f"{prefix}.experts.{e}.{name}.{suffix}")
                                for e in range(self.config.num_experts)
                            ]
                        )
        return weights
