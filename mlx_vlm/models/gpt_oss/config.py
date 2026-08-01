from dataclasses import dataclass
from typing import Any, Dict, Optional

from ..base import BaseModelConfig


def sanitize_quantization_config(quantization):
    """Remap per-layer quantization keys onto the wrapped module paths.

    gpt-oss checkpoints are a mixed per-layer quant: the MoE experts are mxfp4
    (scales only) while the non-expert layers are 8-bit ``affine`` (scales +
    ``biases``). ``config.json`` keys those per-layer overrides by the unwrapped
    module path (``model.layers.N...``, ``model.embed_tokens``, ``lm_head``),
    but at load time ``gpt_oss.Model.sanitize`` moves every tensor under
    ``language_model.`` -- so ``nn.quantize`` sees ``language_model.model...``
    paths. Without remapping the config keys, the per-layer entries never match,
    the non-expert layers inherit the top-level mxfp4 mode (no ``biases`` slot),
    and their affine ``biases`` have nowhere to land -> ``load_weights(strict)``
    aborts with "parameters not in model".
    """
    if not isinstance(quantization, dict):
        return quantization

    def remap(key):
        if key.startswith("language_model."):
            return key
        if key.startswith("model.") or key == "lm_head" or key.startswith("lm_head."):
            return f"language_model.{key}"
        return key

    return {remap(k): v for k, v in quantization.items()}


@dataclass
class ModelConfig(BaseModelConfig):
    model_type: str = "gpt_oss"
    num_hidden_layers: int = 36
    num_local_experts: int = 128
    num_experts_per_tok: int = 4
    vocab_size: int = 201088
    rms_norm_eps: float = 1e-05
    hidden_size: int = 2880
    intermediate_size: int = 2880
    head_dim: int = 64
    num_attention_heads: int = 64
    num_key_value_heads: int = 8
    sliding_window: int = 128
    rope_theta: int = 150000
    rope_scaling: Any = None
    layer_types: list = None
    quantization: Optional[Dict] = None
    quantization_config: Optional[Dict] = None

    def __post_init__(self):
        quantization = self.quantization
        self.quantization = sanitize_quantization_config(quantization)
        if self.quantization_config == quantization:
            self.quantization_config = self.quantization
        else:
            self.quantization_config = sanitize_quantization_config(
                self.quantization_config
            )
