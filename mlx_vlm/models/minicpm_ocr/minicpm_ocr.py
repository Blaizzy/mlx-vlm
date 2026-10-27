import mlx.core as mx
import mlx.nn as nn

from ..minicpmv4_6.minicpmv4_6 import Model as MiniCPMVModel
from ..qwen3_5.qwen3_5 import NORM_WEIGHT_SUFFIXES
from .precision import preserve_norm_precision


class Model(MiniCPMVModel):
    def __init__(self, config):
        super().__init__(config)
        preserve_norm_precision(self.language_model)
        # The OCR reference uses tanh GELU in the intermediate window merger.
        self.vit_merger.act = nn.GELU(approx="tanh")

    def sanitize(self, weights):
        # HF evaluates (1 + weight) in float32 even for BF16 checkpoints.
        weights = {
            key: (
                value.astype(mx.float32)
                if key.endswith(NORM_WEIGHT_SUFFIXES)
                else value
            )
            for key, value in weights.items()
        }
        return super().sanitize(weights)

    @property
    def cast_predicate(self):
        predicate = self.language_model.cast_predicate
        return lambda path: predicate(path) and not path.endswith(NORM_WEIGHT_SUFFIXES)

    def get_input_embeddings(self, input_ids=None, pixel_values=None, **kwargs):
        mode = kwargs.get("downsample_mode", self.config.downsample_mode)
        if mode not in ("4x", "16x"):
            raise ValueError("downsample_mode must be '4x' or '16x'.")
        result = super().get_input_embeddings(input_ids, pixel_values, **kwargs)
        mask = kwargs.get("attention_mask", kwargs.get("mask"))
        # Carry explicit positions through chunked prefill, including padding.
        positions, deltas = self.language_model.get_rope_index(
            input_ids, attention_mask=mask
        )
        self.language_model._position_ids = positions
        self.language_model._rope_deltas = deltas
        result.position_ids = positions
        result.rope_deltas = deltas
        return result
