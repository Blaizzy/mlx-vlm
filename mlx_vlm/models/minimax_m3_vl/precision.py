"""Inference precision helpers for MiniMax M3.1 Flash."""

import mlx.core as mx
import mlx.nn as nn


class CachedFP32Linear(nn.Linear):
    """Reuse the FP32 inference cast without changing checkpoint parameters."""

    def __call__(self, x):
        if (
            self.training
            or x.dtype != mx.float32
            or self.weight.dtype not in (mx.bfloat16, mx.float16)
        ):
            self._fp32_weight_cache = None
            return super().__call__(x)

        # Private state is excluded from Module.parameters()/checkpoint saves.
        # Identity invalidation also covers load_weights, set_dtype and updates.
        cached = getattr(self, "_fp32_weight_cache", None)
        if cached is None or cached[0] is not self.weight:
            cached = (self.weight, self.weight.astype(mx.float32))
            self._fp32_weight_cache = cached
        output = x @ cached[1].T
        if "bias" in self:
            output = output + self.bias
        return output
