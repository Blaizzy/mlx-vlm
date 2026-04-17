# Re-export rope utilities from mlx-lm with ProportionalRoPE support.

from typing import Optional

import mlx.core as mx
import mlx.nn as nn


class ProportionalRoPE(nn.Module):
    """Proportional RoPE for Gemma 4 full-attention layers.

    Frequencies are computed relative to the full head dimension (not just the
    rotated portion), and rotation is applied to the first rotated_dims//2
    elements of each half of the head — matching HF's rotate_half convention.
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

        rope_angles = int(partial_rotary_factor * dims // 2)
        self.rotated_dims = 2 * rope_angles

        if self.rotated_dims > 0:
            exponents = mx.arange(0, self.rotated_dims, 2, dtype=mx.float32) / dims
            self._freqs = factor * (base**exponents)
        else:
            self._freqs = None

    def __call__(self, x, offset=0):
        if self.rotated_dims <= 0:
            return x

        head = x[..., : self.dims]
        tail = x[..., self.dims :]
        half = self.dims // 2

        left = head[..., :half]
        right = head[..., half:]
        rotated = mx.concatenate(
            [left[..., : self.rotated_dims // 2], right[..., : self.rotated_dims // 2]],
            axis=-1,
        )
        rotated = mx.fast.rope(
            rotated,
            self.rotated_dims,
            traditional=self.traditional,
            base=None,
            scale=1.0,
            offset=offset,
            freqs=self._freqs,
        )

        left = mx.concatenate(
            [
                rotated[..., : self.rotated_dims // 2],
                left[..., self.rotated_dims // 2 :],
            ],
            axis=-1,
        )
        right = mx.concatenate(
            [
                rotated[..., self.rotated_dims // 2 :],
                right[..., self.rotated_dims // 2 :],
            ],
            axis=-1,
        )
        head = mx.concatenate([left, right], axis=-1)

        if tail.shape[-1] == 0:
            return head
        return mx.concatenate([head, tail], axis=-1)


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
