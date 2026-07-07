# Re-export rope utilities from mlx-lm with ProportionalRoPE support.

from typing import Optional

import mlx.core as mx
import mlx.nn as nn


class ProportionalRoPE(nn.Module):
    """Proportional RoPE for Gemma 4 full-attention layers.

    Frequencies are computed relative to the full head dimension. HF then pads
    the non-rotary frequency slots with zeros and applies rotate_half over the
    whole head, so only the matching prefix in each half rotates.
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
            inv_freq = 1.0 / mx.power(base, exponents)
            if nope_angles > 0:
                inv_freq = mx.concatenate(
                    [inv_freq, mx.zeros((nope_angles,), dtype=mx.float32)]
                )
            self._inv_freq = inv_freq / factor
        else:
            self._inv_freq = mx.zeros((dims // 2,), dtype=mx.float32)
        self.eval_cached_arrays()

    @property
    def freqs(self):
        return self._inv_freq

    def eager_eval_arrays(self):
        return [self._inv_freq]

    def eval_cached_arrays(self):
        arrays = self.eager_eval_arrays()
        if arrays:
            mx.eval(*arrays)

    def __call__(self, x, offset=0):
        if self.rope_angles <= 0:
            return x

        head = x[..., : self.dims]
        tail = x[..., self.dims :]
        half = self.dims // 2

        positions = mx.arange(head.shape[-2], dtype=mx.float32) + offset
        freqs = positions[..., None] * self._inv_freq
        emb = mx.concatenate([freqs, freqs], axis=-1)
        cos = mx.cos(emb).astype(head.dtype)
        sin = mx.sin(emb).astype(head.dtype)
        cos = cos.reshape((1,) * (head.ndim - 2) + cos.shape)
        sin = sin.reshape((1,) * (head.ndim - 2) + sin.shape)

        rotated = mx.concatenate([-head[..., half:], head[..., :half]], axis=-1)
        head = (head * cos) + (rotated * sin)

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
