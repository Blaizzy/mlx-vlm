import mlx.core as mx
import numpy as np

from ..pixtral import LanguageModel
from ..pixtral import Model as PixtralModel
from ..pixtral import ModelConfig, TextConfig, VisionConfig, VisionModel


class Model(PixtralModel):
    def __init__(self, config: ModelConfig):
        super().__init__(config)

    def sanitize(self, weights):
        sanitized_weights = {}
        for k, v in weights.items():
            # Rename attention weight keys from wq/wk/wv/wo to q_proj/k_proj/v_proj/o_proj
            if k.endswith(".attention.wq.weight"):
                new_key = k.replace(".attention.wq.weight", ".attention.q_proj.weight")
                sanitized_weights[new_key] = v
            elif k.endswith(".attention.wk.weight"):
                new_key = k.replace(".attention.wk.weight", ".attention.k_proj.weight")
                sanitized_weights[new_key] = v
            elif k.endswith(".attention.wv.weight"):
                new_key = k.replace(".attention.wv.weight", ".attention.v_proj.weight")
                sanitized_weights[new_key] = v
            elif k.endswith(".attention.wo.weight"):
                new_key = k.replace(".attention.wo.weight", ".attention.o_proj.weight")
                sanitized_weights[new_key] = v

            # Rename feed_forward weight keys from w1/w2/w3 to gate_proj/down_proj/up_proj
            elif k.endswith(".feed_forward.w1.weight"):
                new_key = k.replace(
                    ".feed_forward.w1.weight", ".feed_forward.gate_proj.weight"
                )
                sanitized_weights[new_key] = v
            elif k.endswith(".feed_forward.w2.weight"):
                new_key = k.replace(
                    ".feed_forward.w2.weight", ".feed_forward.down_proj.weight"
                )
                sanitized_weights[new_key] = v
            elif k.endswith(".feed_forward.w3.weight"):
                new_key = k.replace(
                    ".feed_forward.w3.weight", ".feed_forward.up_proj.weight"
                )
                sanitized_weights[new_key] = v
            # Rename vision_encoder to vision_tower.vision_model
            elif k.startswith("vision_encoder."):
                new_key = k.replace("vision_encoder.", "vision_tower.vision_model.")
                sanitized_weights[new_key] = v

            elif k.startswith("layers."):
                new_key = k.replace("layers.", "language_model.model.layers.")
                sanitized_weights[new_key] = v
            else:
                sanitized_weights[k] = v
