"""DeepSeek verification policies with the serving model's normal traversal."""

from copy import copy

import mlx.core as mx
import mlx.nn as nn

from ...models.base import create_attention_mask
from ...models.cache import CacheList
from ...models.deepseek_v4.language import (
    CompressedAttention,
    LocalAttention,
    SparseCompressedAttention,
)
from ...models.quantized_verifier import (
    exact_quantized_selected_linear,
    exact_quantized_switch_linear,
    singleton_quantized_linear,
)
from ..cache_state import start_speculative_cache
from ..ops.hyper_connection import SpeculativeHyperConnection
from ..ops.linear import native_batch_linear


class _Timewise:
    """Retain the native Bx1 reduction order for small dense operations."""

    def __init__(self, module, axis=1):
        self.module = module
        self.axis = axis

    def __call__(self, x, *args):
        def at(value, index):
            slices = [slice(None)] * value.ndim
            slices[self.axis] = slice(index, index + 1)
            return mx.contiguous(value[tuple(slices)])

        outputs = [
            self.module(
                at(x, index),
                *(at(arg, index) for arg in args),
            )
            for index in range(x.shape[self.axis])
        ]
        if isinstance(outputs[0], (tuple, list)):
            return tuple(
                mx.concatenate(parts, axis=self.axis) for parts in zip(*outputs)
            )
        return mx.concatenate(outputs, axis=self.axis)


class _Projection:
    def __init__(self, module):
        self.module = module

    def __call__(self, x):
        if x.shape[0] == 1:
            output = singleton_quantized_linear(self.module, x)
            if output is not None:
                return output
        return native_batch_linear(self.module, x)


class _Attention:
    def _attend(self, x, q, kv, q_residual, mask, cache, offset):
        if cache is None:
            return super()._attend(x, q, kv, q_residual, mask, cache, offset)
        local_cache = cache[0] if isinstance(cache, CacheList) else cache
        outputs = []
        for index in range(x.shape[1]):
            part = mx.contiguous(x[:, index : index + 1])
            # Both the physical window order and the attention reduction must
            # match decode. Recompute the model's mask after each cache update.
            part_mask = create_attention_mask(
                part,
                local_cache,
                window_size=self.config.sliding_window,
                return_array=True,
            )
            output = super()._attend(
                part,
                mx.contiguous(q[:, :, index : index + 1]),
                mx.contiguous(kv[:, :, index : index + 1]),
                mx.contiguous(q_residual[:, index : index + 1]),
                part_mask,
                cache,
                offset + index,
            )
            mx.async_eval(output)
            outputs.append(output)
        return mx.concatenate(outputs, axis=2)


class _LocalAttention(_Attention, LocalAttention):
    pass


class _CompressedAttention(_Attention, CompressedAttention):
    pass


class _SparseCompressedAttention(_Attention, SparseCompressedAttention):
    pass


_ATTENTION_VIEWS = {
    LocalAttention: _LocalAttention,
    CompressedAttention: _CompressedAttention,
    SparseCompressedAttention: _SparseCompressedAttention,
}


def _view(module, cls=None):
    view = copy(module)
    if cls is not None:
        view.__class__ = cls
    for name, value in module.items():
        if isinstance(value, (nn.Linear, nn.QuantizedLinear)):
            view[name] = _Projection(value)
    return view


class _Switch:
    """Share expert reads across time without changing decode's route sorting."""

    def __init__(self, module):
        self.module = module

    def __call__(self, x, indices):
        if x.shape[0] * indices.shape[-1] < 64:
            up = exact_quantized_switch_linear(self.module.up_proj, x, indices)
            gate = exact_quantized_switch_linear(self.module.gate_proj, x, indices)
            if up is not None and gate is not None:
                activated = self.module.activation(up, gate)
                down = exact_quantized_selected_linear(
                    self.module.down_proj, activated, indices
                )
                if down is not None:
                    return down
        return _Timewise(self.module)(x, indices)


class DeepseekV4SpeculativeTarget:
    def __init__(self, language_model):
        self.language_model = language_model
        self.verifier = copy(language_model)
        self.verifier.model = copy(language_model.model)
        layers = []
        for layer in language_model.model.layers:
            view = copy(layer)
            view.attn_hc = _view(layer.attn_hc, SpeculativeHyperConnection)
            view.ffn_hc = _view(layer.ffn_hc, SpeculativeHyperConnection)
            view.attn = _view(layer.attn, _ATTENTION_VIEWS[type(layer.attn)])
            view.attn.wo_a = _Timewise(layer.attn.wo_a, axis=2)
            view.ffn = copy(layer.ffn)
            view.ffn.gate = _Timewise(layer.ffn.gate)
            view.ffn.shared_experts = _view(layer.ffn.shared_experts)
            view.ffn.switch_mlp = _Switch(layer.ffn.switch_mlp)
            layers.append(view)
        self.verifier.model.layers = layers
        self.verifier.model.hc_head = _Timewise(language_model.model.hc_head)
        self.verifier.lm_head = _Timewise(language_model.lm_head)

    def __getattr__(self, name):
        return getattr(self.language_model, name)

    def train(self, mode=True):
        self.language_model.train(mode)
        self.verifier.train(mode)
        return self

    def eval(self):
        return self.train(False)

    def __call__(self, inputs, cache=None, **kwargs):
        if not kwargs.pop("speculative_verify", False):
            return self.language_model(inputs, cache=cache, **kwargs)

        transaction = start_speculative_cache(cache or [], inputs.shape[1])
        try:
            output = self.verifier(inputs, cache=cache, **kwargs)
            output.gdn_states = transaction
            return output
        except BaseException:
            transaction.abort()
            raise
