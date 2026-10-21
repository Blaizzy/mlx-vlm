import mlx.nn as nn

from ..step3p7.step3p7 import Model as StepModel
from ..step3p7.vision import VisionModel
from .language import LanguageModel


class Model(StepModel):
    def __init__(self, config):
        nn.Module.__init__(self)
        self.config = config
        self.vision_model = VisionModel(config.vision_config)
        self.language_model = LanguageModel(config.text_config, config)
        self.vit_large_projector = nn.Linear(
            config.vision_config.width * 4,
            config.text_config.hidden_size,
            bias=config.projector_bias,
        )

    def sanitize(self, weights):
        # Raw StepFun RMSNorm weights are zero-centered. LayerNorm weights
        # (including the sparse indexer's K norm) are ordinary multiplicative scales.
        norms = (
            "input_layernorm.weight",
            "post_attention_layernorm.weight",
            "q_norm.weight",
            "k_norm.weight",
            "sparse_indexer_q_norm.weight",
        )
        out = {}
        for key, value in weights.items():
            raw = key.startswith("model.") and not key.startswith("model.vision_model.")
            if raw and (
                key in ("model.norm.weight", "model.language_model.norm.weight")
                or key.endswith(norms)
                and not key.endswith("sparse_indexer_k_norm.weight")
            ):
                value = value + 1
            out[key] = value
        return super().sanitize(out)
