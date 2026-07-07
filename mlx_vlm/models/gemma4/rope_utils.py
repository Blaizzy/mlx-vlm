# Re-export rope utilities from mlx-lm with ProportionalRoPE support.

from typing import Optional

import mlx.core as mx
import mlx.nn as nn


class ProportionalRoPE(nn.Module):
    """Proportional RoPE for Gemma 4 full-attention layers.

    Frequencies are computed relative to the full head dimension. HF pads the
    non-rotary inverse-frequency slots with zeros; for mx.fast.rope, the
    equivalent denominator-frequency slots are padded with infinity.
    """

    def __init__(
        self,
        dims: int,
        traditional: bool = False,
        base: float = 10000.0,
        scaling_config: Optional[dict] = None,
    ):
        super().__init__()
        self.dims = dims
        self.traditional = traditional

        scaling_config = scaling_config or {}
        factor = scaling_config.get("factor", 1.0)
        partial_rotary_factor = scaling_config.get("partial_rotary_factor", 1.0)

        self.rope_angles = int(partial_rotary_factor * dims // 2)
        nope_angles = dims // 2 - self.rope_angles

        if self.rope_angles > 0:
            exponents = mx.arange(0, 2 * self.rope_angles, 2, dtype=mx.float32) / dims
            freqs = factor * mx.power(base, exponents)
            if nope_angles > 0:
                freqs = mx.concatenate(
                    [freqs, mx.full((nope_angles,), mx.inf, dtype=mx.float32)]
                )
            self._freqs = freqs
        else:
            self._freqs = mx.full((dims // 2,), mx.inf, dtype=mx.float32)
        self.eval_cached_arrays()

    @property
    def freqs(self):
        return self._freqs

    def eager_eval_arrays(self):
        return [self._freqs]

    def eval_cached_arrays(self):
        arrays = self.eager_eval_arrays()
        if arrays:
            mx.eval(*arrays)

    def __call__(self, x, offset=0):
        if self.rope_angles <= 0:
            return x

        return mx.fast.rope(
            x,
            self.dims,
            traditional=self.traditional,
            base=None,
            scale=1.0,
            offset=offset,
            freqs=self._freqs,
        )


def initialize_rope(
    dims: int,
    base: float,
    traditional: bool,
    scaling_config: Optional[dict] = None,
    max_position_embeddings: Optional[int] = None,
):
    """Initialize the appropriate RoPE variant based on scaling_config."""
    if scaling_config is not None:
        rope_type = scaling_config.get("type") or scaling_config.get(
            "rope_type", "default"
        )
    else:
        rope_type = "default"

    if rope_type == "proportional":
        return ProportionalRoPE(
            dims=dims,
            traditional=traditional,
            base=base,
            scaling_config=scaling_config,
        )

    # Default: standard RoPE
    return nn.RoPE(dims, traditional=traditional, base=base)
