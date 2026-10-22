"""Bounded compilation of stateless Flint inference stages."""

from functools import lru_cache

import mlx.core as mx
import mlx.nn as nn

from ..switch_layers import QuantizedSwitchLinear, SwitchLinear


class _StreamKey:
    def __init__(self):
        self.stream = mx.default_stream(mx.default_device())

    def __hash__(self):
        # MLX can return equal Stream wrappers with different object hashes.
        return 0

    def __eq__(self, other):
        return isinstance(other, _StreamKey) and self.stream == other.stream


def linear_signature(module):
    """Describe static projection options; custom modules execute normally."""
    if type(module) not in (
        nn.Linear,
        nn.QuantizedLinear,
        SwitchLinear,
        QuantizedSwitchLinear,
    ):
        return None
    options = vars(module)
    return (
        type(module),
        options.get("bits"),
        options.get("group_size"),
        options.get("mode"),
    )


class CompiledStage:
    def __init__(self, function, module):
        # Custom Metal operations need concrete output shapes. Bound the number
        # of graphs, including variable prefill lengths in a persistent server.
        # Parameters remain implicit inputs, never captured weight snapshots.
        @lru_cache(maxsize=8)
        def compile_stage(signature, shapes, stream):
            return mx.compile(lambda *arrays: function(*arrays), inputs=module)

        self._cache = compile_stage

    def __call__(self, *arrays, signature):
        shapes = tuple((a.shape, a.dtype) for a in arrays)
        return self._cache(signature, shapes, _StreamKey())(*arrays)
