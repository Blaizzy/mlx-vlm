from dataclasses import replace
from math import ceil
from typing import Any, Optional

import mlx.core as mx
import numpy as np

from ..base import LanguageModelOutput
from ..cache import KVCache
from ..deepseekocr.language import DeepseekV2DecoderLayer, DeepseekV2Model
from ..deepseekocr.language import LanguageModel as DeepseekLanguageModel
from .config import TextConfig
from .dash import (
    DashAttention,
    DashParams,
    DashRope,
    build_context,
    is_dash_config,
    make_dash_cache,
)


class RingSlidingKVCache(KVCache):
    """Unlimited-OCR R-SWA cache.

    Upstream keeps the full prompt/prefill cache, then appends generated tokens
    until a small decode ring is full. Afterwards, new decode keys/values
    overwrite that ring while absolute positions keep increasing.

    The language model sets ``prefill_length`` before processing any chunks.
    """

    def memory_profile(self, token_count):
        profile = super().memory_profile(token_count)
        return replace(
            profile,
            fixed_bytes=ceil((self.step - 1) * profile.bytes_per_token),
            step=1,
        )

    def __init__(self, window_size: int):
        super().__init__()
        self.window_size = int(window_size)
        self.prefill_length: Optional[int] = None
        self._ring_pos = 0

    def update_and_fetch(self, keys, values):
        seq_len = int(keys.shape[2])

        if self.prefill_length is None:
            if seq_len > 1:
                return super().update_and_fetch(keys, values)
            self.prefill_length = self.offset

        if self.keys is None or self.offset < self.prefill_length + self.window_size:
            keys, values = super().update_and_fetch(keys, values)
            if self.offset >= self.prefill_length + self.window_size:
                self._ring_pos = 0
            return keys, values

        for idx in range(seq_len):
            slot = self.prefill_length + self._ring_pos
            self.keys[..., slot : slot + 1, :] = keys[..., idx : idx + 1, :]
            self.values[..., slot : slot + 1, :] = values[..., idx : idx + 1, :]
            self._ring_pos = (self._ring_pos + 1) % self.window_size

        self.offset += seq_len
        return (
            self.keys[..., : self.prefill_length + self.window_size, :],
            self.values[..., : self.prefill_length + self.window_size, :],
        )

    def make_mask(self, N: int, return_array: bool = False, **kwargs):
        kwargs.pop("window_size", None)
        # Prefill/warmup is standard causal attention. Steady-state decode uses
        # q_len=1 and attends over the retained prompt plus ring slots, matching
        # upstream SlidingWindowLlamaAttention.
        if self.prefill_length is not None and self.offset >= (
            self.prefill_length + self.window_size
        ):
            if N == 1 and not return_array:
                return None
        return super().make_mask(
            N, return_array=return_array, window_size=None, **kwargs
        )

    @property
    def state(self):
        if self.keys is None:
            return None, None
        end = (
            self.offset
            if self.prefill_length is None
            else min(self.offset, self.prefill_length + self.window_size)
        )
        return self.keys[..., :end, :], self.values[..., :end, :]

    @state.setter
    def state(self, v):
        self.keys, self.values = v
        self.offset = 0 if self.keys is None else self.keys.shape[2]
        self.prefill_length = None
        self._ring_pos = 0

    @property
    def meta_state(self):
        return tuple(
            map(
                str,
                (
                    self.window_size,
                    -1 if self.prefill_length is None else self.prefill_length,
                    self.offset,
                    self._ring_pos,
                ),
            )
        )

    @meta_state.setter
    def meta_state(self, v):
        window_size, prefill_length, offset, ring_pos = map(int, v)
        self.window_size = window_size
        self.prefill_length = None if prefill_length < 0 else prefill_length
        self.offset = offset
        self._ring_pos = ring_pos


def _match_cache_dtype(h, cache):
    """Keep decode activations at the precision established during prefill."""
    if cache:
        keys = getattr(cache[0], "keys", None)
        # Quantized caches retain the input floating dtype in their scales.
        if isinstance(keys, (tuple, list)):
            keys = keys[1]
        if isinstance(keys, mx.array) and mx.issubdtype(keys.dtype, mx.floating):
            h = h.astype(mx.result_type(h.dtype, keys.dtype))
    return h


class DeepseekV2RingSlidingModel(DeepseekV2Model):
    def __call__(
        self,
        x: mx.array,
        mask: Optional[mx.array] = None,
        inputs_embeds: Optional[mx.array] = None,
        cache: Optional[Any] = None,
    ) -> mx.array:
        if inputs_embeds is None:
            h = self.embed_tokens(x)
        else:
            h = inputs_embeds

        if cache is None:
            cache = [None] * len(self.layers)

        h = _match_cache_dtype(h, cache)

        if mask is None:
            mask = self._make_attention_mask(h, cache[0])

        for layer, c in zip(self.layers, cache):
            h = layer(h, mask, c)

        return self.norm(h)

    def _make_attention_mask(self, h: mx.array, cache: Optional[Any]):
        if cache is not None and hasattr(cache, "make_mask"):
            return cache.make_mask(h.shape[1], return_array=False, window_size=None)
        if h.shape[1] == 1:
            return None
        return "causal"


class DashDecoderLayer(DeepseekV2DecoderLayer):
    def __init__(self, config: TextConfig, layer_idx: int, params: DashParams):
        super().__init__(config, layer_idx)
        self.self_attn = DashAttention(config, layer_idx, params)

    def __call__(self, x, ctx, cache):
        h = x + self.self_attn(self.input_layernorm(x), ctx, cache)
        return h + self.mlp(self.post_attention_layernorm(h))


class DeepseekV2DashModel(DeepseekV2Model):
    """Unlimited-OCR-2 decoder: page2d RoPE + DASH sparse attention."""

    def __init__(self, config: TextConfig, params: DashParams):
        super().__init__(config)
        self.layers = [
            DashDecoderLayer(config, idx, params)
            for idx in range(config.num_hidden_layers)
        ]
        head_dim = config.hidden_size // config.num_attention_heads
        self._rope = DashRope(params, head_dim, config)

    def __call__(self, x, inputs_embeds=None, cache=None, prompt_ids=None):
        h = self.embed_tokens(x) if inputs_embeds is None else inputs_embeds
        h = _match_cache_dtype(h, cache)
        state = cache[0].dash_state
        offset = cache[0].offset
        if offset == 0:
            state.start(_prompt_ids(prompt_ids, x, h.shape[1]))
        ctx = build_context(state, self._rope, x, offset, h.shape[1], h.dtype)
        for layer, c in zip(self.layers, cache):
            h = layer(h, ctx, c)
        return self.norm(h)


def _prompt_ids(stashed, inputs, length):
    """Full prompt ids: the processor's (chunked prefill sees slices) or ``inputs``."""
    current = None if inputs is None else np.array(inputs).reshape(-1)
    if stashed is not None:
        stashed = np.asarray(stashed).reshape(-1)
        if current is None or (
            stashed.shape[0] >= length
            and np.array_equal(stashed[: current.shape[0]], current)
        ):
            return stashed
    if current is None:
        raise ValueError("Unlimited-OCR-2 needs the prompt token ids.")
    return current


class LanguageModel(DeepseekLanguageModel):
    def __init__(self, config: TextConfig):
        super().__init__(config)
        self.dash_params = None
        if is_dash_config(config.sparse_attention_config):
            self.dash_params = DashParams.from_config(
                config.sparse_attention_config,
                config.num_hidden_layers,
                config.hidden_size // config.num_attention_heads,
            )
            self.model = DeepseekV2DashModel(config, self.dash_params)
        else:
            self.model = DeepseekV2RingSlidingModel(config)
        self._prompt_ids = None

    def __call__(
        self,
        inputs: mx.array,
        inputs_embeds: Optional[mx.array] = None,
        mask: Optional[mx.array] = None,
        cache: Optional[Any] = None,
        **kwargs,
    ):
        if self.dash_params is None:
            if (
                cache
                and isinstance(cache[0], RingSlidingKVCache)
                and (cache[0].offset == 0 or self._prompt_ids is not None)
            ):
                length = (inputs if inputs_embeds is None else inputs_embeds).shape[1]
                if self._prompt_ids is not None:
                    ids = None if inputs is None else inputs[0]
                    length = len(_prompt_ids(self._prompt_ids, ids, length))
                # Like DASH, fix the boundary before the first chunk. A final
                # single-token chunk still belongs to this persistent prefix.
                for entry in cache:
                    entry.prefill_length = entry.offset + length
            # The boundary now belongs to the cache, including restored caches.
            self._prompt_ids = None
            return super().__call__(
                inputs, inputs_embeds=inputs_embeds, mask=mask, cache=cache, **kwargs
            )
        if cache is None:
            cache = self.make_cache()
        h = self.model(
            inputs,
            inputs_embeds=inputs_embeds,
            cache=cache,
            prompt_ids=self._prompt_ids,
        )
        keep = kwargs.get("logits_to_keep")
        if keep:
            h = h[:, -int(keep) :]
        return LanguageModelOutput(logits=self.lm_head(h))

    def sanitize(self, weights):
        if self.dash_params is not None:
            # The warmup-stage lightning indexer is unused at DASH inference.
            weights = {
                k: v for k, v in weights.items() if ".self_attn.indexer." not in k
            }
        return super().sanitize(weights)

    def make_cache(self):
        if self.dash_params is not None:
            return make_dash_cache(self.dash_params, len(self.layers))
        window_size = self.config.sliding_window_size or self.config.sliding_window
        if window_size is None:
            return [KVCache() for _ in self.layers]
        return [RingSlidingKVCache(window_size) for _ in self.layers]


__all__ = ["LanguageModel", "RingSlidingKVCache"]
