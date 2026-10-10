"""Chunked, cancellable Clef prefill with exact-state prefix reuse.

The engine is owned by one inference thread. Requests at the same position are
prefilled together without padding, preserving Qwen's recurrent state semantics.
The joint head still runs separately for each request's schema.
"""

import hashlib
from collections import OrderedDict
from dataclasses import dataclass
from threading import Event

import mlx.core as mx
import numpy as np

from ...decision import DecisionCancelled, check_cancelled, context_limit
from ..cache import ArraysCache, KVCache
from .clef import format_result
from .processing_clef import encode_record


def _clone_cache(caches):
    result = []
    for source in caches:
        target = (
            ArraysCache(len(source.state))
            if isinstance(source, ArraysCache)
            else KVCache()
        )
        target.state = [mx.array(value) for value in source.state]
        result.append(target)
    mx.eval([c.state for c in result])
    return result


class PrefixCache:
    """Byte-bounded LRU containing both recurrent/KV state and final hidden states."""

    def __init__(self, max_bytes=256 * 1024**2):
        if max_bytes < 0:
            raise ValueError("Prefix cache budget cannot be negative")
        self.max_bytes = max_bytes
        self.entries = OrderedDict()
        self.nbytes = self.hits = self.misses = 0

    def get(self, key):
        entry = self.entries.get(key)
        if entry is None:
            self.misses += 1
            return None
        self.hits += 1
        self.entries.move_to_end(key)
        caches, hidden, _ = entry
        return _clone_cache(caches), hidden

    def put(self, key, caches, hidden):
        size = hidden.nbytes + sum(v.nbytes for c in caches for v in c.state)
        if size > self.max_bytes or not self.max_bytes:
            return
        if key in self.entries:
            self.nbytes -= self.entries.pop(key)[2]
        while self.entries and self.nbytes + size > self.max_bytes:
            self.nbytes -= self.entries.popitem(last=False)[1][2]
        # Copy out compact arrays: neither future cache writes nor a view into a
        # larger request/batch may mutate or pin memory in the retained prefix.
        saved = _clone_cache(caches)
        hidden = mx.array(hidden)
        mx.eval(hidden)
        self.entries[key] = saved, hidden, size
        self.nbytes += size

    def clear(self):
        self.entries.clear()
        self.nbytes = 0


def _prefix_key(record, namespace):
    digest = hashlib.sha256()
    digest.update(
        np.asarray(record.input_ids[: record.prefix_length], dtype=np.int32).tobytes()
    )
    for name, value in sorted((record.media or {}).items()):
        array = np.asarray(value)
        digest.update(name.encode())
        digest.update(str((array.dtype, array.shape)).encode())
        digest.update(array.tobytes())
    return namespace, digest.digest()


@dataclass
class DecisionState:
    record: object
    request: dict
    ids: mx.array
    embeds: mx.array
    positions: mx.array
    cache: list
    hidden: list
    offset: int
    embed_offset: int
    key: object
    cancelled: Event
    cached_tokens: int = 0

    @property
    def done(self):
        return self.offset == len(self.record.input_ids)


class DecisionEngine:
    def __init__(
        self,
        model,
        processor,
        *,
        max_length=None,
        prefill_step_size=512,
        cache_bytes=256 * 1024**2,
    ):
        if prefill_step_size < 1:
            raise ValueError("Prefill step size must be positive")
        self.model, self.processor = model, processor
        self.max_length = context_limit(model.config, max_length)
        self.prefill_step_size = prefill_step_size
        self.prefix_cache = PrefixCache(cache_bytes)
        self.batch_steps = self.prefill_tokens = 0

    def prepare(self, request, *, namespace=None, cancelled=None):
        cancelled = cancelled if cancelled is not None else Event()
        check_cancelled(cancelled)
        record = encode_record(
            getattr(self.processor, "tokenizer", self.processor),
            request,
            max_length=self.max_length,
            processor=self.processor,
            checkpoint=lambda: check_cancelled(cancelled),
        )
        ids = mx.array(record.input_ids)[None]
        key = (
            _prefix_key(record, namespace)
            if record.prefix_length and self.prefix_cache.max_bytes
            else None
        )
        saved = (
            self.prefix_cache.get(key)
            if record.prefix_length and self.prefix_cache.max_bytes
            else None
        )
        media = {
            k: mx.array(v)
            for k, v in (record.media or {}).items()
            if k
            in (
                "pixel_values",
                "pixel_values_videos",
                "image_grid_thw",
                "video_grid_thw",
            )
        }
        if saved is None:
            features = self.model.get_input_embeddings(ids, **media)
            embeds, positions = features.inputs_embeds, features.position_ids
            caches, hidden, offset = self.model.language_model.make_cache(), [], 0
        else:
            caches, prefix_hidden = saved
            offset = record.prefix_length
            hidden = [prefix_hidden]
            embeds = self.model.language_model.model.embed_tokens(ids[:, offset:])
            positions, _ = self.model.language_model.get_rope_index(
                ids, media.get("image_grid_thw"), media.get("video_grid_thw")
            )
        if positions.ndim == 2:
            positions = mx.broadcast_to(positions[None], (3,) + positions.shape)
        mx.eval(embeds, positions)
        check_cancelled(cancelled)
        return DecisionState(
            record,
            request,
            ids,
            embeds,
            positions,
            caches,
            hidden,
            offset,
            offset,
            key,
            cancelled,
            offset,
        )

    def step(self, states):
        """Advance equal-offset requests by one bounded, unpadded GPU batch."""
        if not states:
            return
        if len({s.offset for s in states}) != 1:
            raise ValueError("A prefill batch must start at the same offset")
        offset = states[0].offset
        # Splitting at the state/schema boundary is useful only when saving a
        # reusable prefix. With caching disabled it repeats the backbone pass
        # for short requests without retaining anything for a future request.
        lengths = [
            min(
                self.prefill_step_size,
                (
                    s.record.prefix_length
                    if self.prefix_cache.max_bytes and offset < s.record.prefix_length
                    else len(s.record.input_ids)
                )
                - offset,
            )
            for s in states
        ]
        length = min(lengths)
        if length <= 0:
            raise ValueError("Cannot prefill a completed request")

        def checkpoint(value):
            if all(s.cancelled.is_set() for s in states):
                raise DecisionCancelled("Decision batch cancelled")
            mx.eval(value)
            if all(s.cancelled.is_set() for s in states):
                raise DecisionCancelled("Decision batch cancelled")

        checkpoint([])
        caches = (
            states[0].cache
            if len(states) == 1
            else [
                type(group[0]).merge(group) for group in zip(*(s.cache for s in states))
            ]
        )
        ids = mx.concatenate([s.ids[:, offset : offset + length] for s in states])
        embeds = mx.concatenate(
            [
                s.embeds[:, offset - s.embed_offset : offset - s.embed_offset + length]
                for s in states
            ]
        )
        positions = mx.concatenate(
            [s.positions[:, :, offset : offset + length] for s in states], axis=1
        )
        hidden = self.model.language_model.model(
            ids,
            inputs_embeds=embeds,
            position_ids=positions,
            cache=caches,
            checkpoint=checkpoint,
        )
        mx.eval(hidden, [c.state for c in caches])
        self.batch_steps += 1
        self.prefill_tokens += length * len(states)
        for index, state in enumerate(states):
            state.cache = (
                caches if len(states) == 1 else [c.extract(index) for c in caches]
            )
            state.hidden.append(hidden[index : index + 1])
            state.offset += length
            if (
                state.offset == state.record.prefix_length
                and not state.cancelled.is_set()
                and self.prefix_cache.max_bytes
            ):
                self.prefix_cache.put(
                    state.key, state.cache, mx.concatenate(state.hidden, axis=1)
                )

    def finish(self, state):
        check_cancelled(state.cancelled)
        if not state.done:
            raise ValueError("Cannot score an incomplete decision")

        def checkpoint(value):
            check_cancelled(state.cancelled)
            mx.eval(value)
            check_cancelled(state.cancelled)

        logits = self.model.score_record(
            mx.concatenate(state.hidden, axis=1),
            state.ids,
            state.record,
            checkpoint=checkpoint,
        )
        checkpoint(logits)
        output = format_result(state.record, state.request["questions"], logits)
        output["cached_tokens"] = state.cached_tokens
        return output

    def decide(self, record, **kwargs):
        state = self.prepare(record, **kwargs)
        while not state.done:
            self.step([state])
        return self.finish(state)
