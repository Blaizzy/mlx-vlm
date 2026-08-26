import math
from functools import lru_cache, partial
from typing import Any, List, Optional, Tuple

import mlx.core as mx
import mlx.nn as nn
from mlx.nn.layers.distributed import shard_inplace

from ..cache import ArraysCache, CacheList, KVCache, _BaseCache
from ..qwen3_5.language import LanguageModel as Qwen3_5LanguageModel
from ..qwen3_5.language import (
    Qwen3_5Attention,
    Qwen3_5GatedDeltaNet,
    Qwen3_5RMSNormGated,
    Qwen3_5RotaryEmbedding,
    _create_qwen3_5_attention_mask,
    _create_qwen3_5_ssm_mask,
    _precise_swiglu,
)
from ..qwen3_5_moe.language import Qwen3_5MoeSparseMoeBlock
from ..rope_utils import rotate_half
from .config import LINEAR_ATTENTION, SPARSE_ATTENTION, ModelConfig, TextConfig


@partial(mx.compile, shapeless=True)
def _precise_sigmoid_gate(h, gate, x):
    gate = mx.sigmoid(gate.astype(mx.float32))
    return (gate * x.astype(mx.float32)).astype(h.dtype)


class Qwen4ExpRMSNormGated(Qwen3_5RMSNormGated):
    """Gated RMSNorm whose output gate activation is configurable."""

    def __init__(self, hidden_size: int, eps: float = 1e-6, activation: str = "silu"):
        super().__init__(hidden_size, eps)
        self.activation = activation

    def __call__(
        self, hidden_states: mx.array, gate: Optional[mx.array] = None
    ) -> mx.array:
        x = mx.fast.rms_norm(hidden_states, self.weight, self.eps)
        if gate is None:
            return x.astype(hidden_states.dtype)
        if self.activation == "sigmoid":
            return _precise_sigmoid_gate(hidden_states, gate, x)
        return _precise_swiglu(hidden_states, gate, x)


class Qwen4ExpGroupRMSNorm(nn.Module):
    """RMSNorm normalising each ``group_size``-wide slice of the last axis.

    Used by the hyper-connections and PLE, where the last axis holds
    ``hc_count`` concatenated residual streams that must be normalised apart.
    """

    def __init__(self, dims: int, group_size: int, eps: float = 1e-6):
        super().__init__()
        if dims % group_size != 0:
            raise ValueError(
                f"dims ({dims}) must be divisible by group_size ({group_size})"
            )
        self.weight = mx.ones((dims,))
        self.eps = eps
        self.group_size = group_size

    def __call__(self, x: mx.array) -> mx.array:
        shape = x.shape
        grouped = x.reshape(*shape[:-1], shape[-1] // self.group_size, self.group_size)
        normed = mx.fast.rms_norm(grouped, None, self.eps)
        return normed.reshape(shape) * self.weight


def _apply_partial_rope(x: mx.array, cos: mx.array, sin: mx.array) -> mx.array:
    """Rotate the leading ``cos.shape[-1]`` features of ``x`` and pass the rest through."""
    rotary_dim = cos.shape[-1]
    rot, passthrough = x[..., :rotary_dim], x[..., rotary_dim:]
    rot = (rot * cos) + (rotate_half(rot) * sin)
    if passthrough.shape[-1] == 0:
        return rot.astype(x.dtype)
    return mx.concatenate([rot.astype(x.dtype), passthrough], axis=-1)


class Qwen4ExpGatedResidual(nn.Module):
    """Hyper-connection block.

    Collapses the ``hc_count`` residual streams into the single stream a
    sub-layer consumes, and (when ``use_combine``) also returns the untouched
    streams plus the per-stream weights used to inject the sub-layer output back.
    """

    def __init__(self, args: TextConfig, use_combine: bool = True):
        super().__init__()
        self.hc_count = args.hc_count
        self.hidden_size = args.hidden_size
        hc_hidden_size = self.hc_count * self.hidden_size
        self.use_combine = use_combine

        self.hc_norm = Qwen4ExpGroupRMSNorm(
            hc_hidden_size, self.hidden_size, eps=args.rms_norm_eps
        )
        self.input_mix_weight_down = nn.Linear(
            hc_hidden_size, args.hc_lowrank, bias=False
        )
        self.input_mix_weight_up = nn.Linear(
            args.hc_lowrank, hc_hidden_size, bias=False
        )
        if use_combine:
            self.block_inject_weight = nn.Linear(
                hc_hidden_size, self.hc_count, bias=False
            )

    def __call__(self, hyper_input: mx.array):
        normed = self.hc_norm(hyper_input)
        mix = nn.silu(self.input_mix_weight_down(normed) / self.hc_count)
        mix = mx.sigmoid(self.input_mix_weight_up(mix))

        streams = normed.reshape(*normed.shape[:-1], self.hc_count, self.hidden_size)
        mixed = (mix.reshape(streams.shape) * streams).mean(axis=-2)
        if not self.use_combine:
            return mixed

        inject = 2 * mx.sigmoid(self.block_inject_weight(normed) / self.hc_count)
        return mixed, hyper_input, inject


def _hc_inject(x: mx.array, hyper_input: mx.array, inject: mx.array) -> mx.array:
    """Broadcast a sub-layer output back onto every residual stream."""
    injection = x[..., None, :] * inject[..., None]
    return hyper_input + injection.reshape(*injection.shape[:-2], -1)


# --- Per-Layer Embedding (PLE) ------------------------------------------------

_MASK64 = (1 << 64) - 1
_SPLITMIX_GAMMA = 0x9E3779B97F4A7C15
_SPLITMIX_M1 = 0xBF58476D1CE4E5B9
_SPLITMIX_M2 = 0x94D049BB133111EB
_PRIME_1 = 10007


def _splitmix64(value: int) -> int:
    value = (value + _SPLITMIX_GAMMA) & _MASK64
    value = ((value ^ (value >> 30)) * _SPLITMIX_M1) & _MASK64
    value = ((value ^ (value >> 27)) * _SPLITMIX_M2) & _MASK64
    return (value ^ (value >> 31)) & _MASK64


@lru_cache(maxsize=None)
def build_layer_multipliers(
    unigram_vocab_size: int, ngram_size: int, ple_layer_index: int, seed: int
) -> Tuple[int, ...]:
    """Per-position hash multipliers, derived deterministically from ``seed``."""
    multiplier_max = ((1 << 63) - 1) // max(unigram_vocab_size, 1)
    half_bound = max(1, multiplier_max // 2)
    base_seed = seed + _PRIME_1 * ple_layer_index
    return tuple(
        2
        * (_splitmix64((base_seed + _SPLITMIX_GAMMA * (i + 1)) & _MASK64) % half_bound)
        + 1
        for i in range(ngram_size)
    )


def _is_prime(value: int) -> bool:
    if value < 2:
        return False
    if value % 2 == 0:
        return value == 2
    for divisor in range(3, math.isqrt(value) + 1, 2):
        if value % divisor == 0:
            return False
    return True


def _find_nth_prime_after(start: int, count: int) -> int:
    prime = start
    for _ in range(count):
        prime += 1
        while not _is_prime(prime):
            prime += 1
    return prime


@lru_cache(maxsize=None)
def build_ngram_head_tables(
    ngram_heads: int, ple_layer_index: int, vocab_size_base: int
) -> Tuple[Tuple[int, ...], Tuple[int, ...]]:
    """A distinct prime vocabulary size, and its offset, for every hashed head."""
    sizes, offsets, total = [], [], 0
    for head_idx in range(ngram_heads):
        global_head_idx = ple_layer_index * ngram_heads + head_idx
        size = _find_nth_prime_after(vocab_size_base - 1, global_head_idx + 1)
        sizes.append(size)
        offsets.append(total)
        total += size
    return tuple(sizes), tuple(offsets)


class Qwen4ExpNGramEmbedding(nn.Module):
    """Hashed n-gram embedding.

    Every token is hashed together with its predecessors (within the same
    eos-delimited segment) into ``heads_per_ngram`` independent buckets per
    n-gram order, and the looked-up vectors are concatenated.
    """

    def __init__(self, args: TextConfig, embedding_dim: int, ple_layer_index: int):
        super().__init__()
        self.ngram_size = args.ngram_size
        self.context_len = self.ngram_size - 1
        self.heads_per_ngram = args.heads_per_ngram
        self.ngram_heads = self.context_len * self.heads_per_ngram
        self.eos_token_id = args.ple_eos_token_id

        self._multipliers = build_layer_multipliers(
            args.vocab_size, self.ngram_size, ple_layer_index, args.seed
        )
        sizes, offsets = build_ngram_head_tables(
            self.ngram_heads, ple_layer_index, args.ngram_vocab_size_base
        )
        self._head_vocab_sizes = mx.array(sizes, dtype=mx.int64)
        self._head_offsets = mx.array(offsets, dtype=mx.int64)

        divisor = args.make_ngram_vocab_size_divisible_by
        total = sizes[-1] + offsets[-1]
        padded_vocab_size = math.ceil(total / divisor) * divisor
        self.ngram_embedding = nn.Embedding(
            padded_vocab_size, embedding_dim // self.ngram_heads
        )

    def _shift_right_ignore_eos(self, token_ids: mx.array, shift: int) -> mx.array:
        """``token_ids`` shifted right by ``shift``, not crossing eos boundaries."""
        if shift == 0:
            return token_ids
        B, S = token_ids.shape
        positions = mx.arange(S)
        eos_positions = mx.where(token_ids == self.eos_token_id, positions, -1)
        previous_eos = mx.concatenate(
            [
                mx.full((B, 1), -1, dtype=eos_positions.dtype),
                mx.cummax(eos_positions, axis=1)[:, :-1],
            ],
            axis=1,
        )
        position_in_segment = positions - (previous_eos + 1)
        source_positions = positions - shift
        shifted = mx.take_along_axis(
            token_ids,
            mx.broadcast_to(mx.maximum(source_positions, 0)[None], (B, S)),
            axis=1,
        )
        valid = (position_in_segment >= shift) & (source_positions >= 0)[None]
        return mx.where(valid, shifted, mx.array(self.eos_token_id, token_ids.dtype))

    def __call__(
        self,
        input_ids: mx.array,
        cache: Optional[Any] = None,
        target_verify: bool = False,
    ) -> mx.array:
        token_ids = input_ids.astype(mx.int64)
        B, L = token_ids.shape

        # The previous `context_len` tokens are kept in the cache exactly the way
        # a short-conv state is: they are the left context of the n-gram hashes.
        previous = cache[3] if cache is not None else None
        if previous is None or previous.shape[0] != B:
            previous = mx.full((B, self.context_len), self.eos_token_id, dtype=mx.int64)
        elif previous.shape[1] != self.context_len:
            # A verify pass left its whole window behind; the live history is the
            # tail of it. Reading the tail also means a missed rollback degrades
            # to "everything accepted" rather than a shape error.
            previous = previous[:, -self.context_len :]
        history = mx.concatenate([previous, token_ids], axis=1)
        if cache is not None:
            # Keep the whole window while verifying so a rejected block can be
            # rolled back by slicing -- see `LanguageModel.rollback_speculative_cache`.
            cache[3] = history if target_verify else history[:, -self.context_len :]

        shifted = [
            self._shift_right_ignore_eos(history, shift)
            for shift in range(self.ngram_size)
        ]

        blocks = []
        for ngram in range(2, self.ngram_size + 1):
            start = (ngram - 2) * self.heads_per_ngram
            end = start + self.heads_per_ngram
            mixed = shifted[0] * self._multipliers[0]
            for position in range(1, ngram):
                mixed = mx.bitwise_xor(
                    mixed, shifted[position] * self._multipliers[position]
                )
            sizes = self._head_vocab_sizes[start:end].reshape(1, 1, -1)
            offsets = self._head_offsets[start:end].reshape(1, 1, -1)
            blocks.append(mx.remainder(mixed[..., None], sizes) + offsets)

        ngram_ids = mx.concatenate(blocks, axis=-1)[:, -L:]
        return self.ngram_embedding(ngram_ids.astype(mx.int32)).reshape(B, L, -1)


class Qwen4ExpPLELayer(nn.Module):
    """Injects hashed n-gram features into every hyper-connection stream."""

    def __init__(self, args: TextConfig, ple_layer_index: int):
        super().__init__()
        self.hidden_size = args.hidden_size
        self.hc_count = args.hc_count
        hc_hidden_size = self.hidden_size * self.hc_count

        self.ple_embedding = Qwen4ExpNGramEmbedding(
            args, args.ple_embed_dim, ple_layer_index
        )
        self.short_conv_state_len = (args.ple_conv_kernel_size - 1) * args.ngram_size
        self.key_proj = nn.Linear(args.ple_embed_dim, hc_hidden_size, bias=False)
        self.value_proj = nn.Linear(args.ple_embed_dim, self.hidden_size, bias=False)
        self.norm_key = Qwen4ExpGroupRMSNorm(
            hc_hidden_size, self.hidden_size, eps=args.rms_norm_eps
        )
        self.norm_query = Qwen4ExpGroupRMSNorm(
            hc_hidden_size, self.hidden_size, eps=args.rms_norm_eps
        )
        self.norm_conv = Qwen4ExpGroupRMSNorm(
            hc_hidden_size, self.hidden_size, eps=args.rms_norm_eps
        )
        self.conv1d = nn.Conv1d(
            hc_hidden_size,
            hc_hidden_size,
            kernel_size=args.ple_conv_kernel_size,
            groups=hc_hidden_size,
            dilation=args.ngram_size,
            bias=False,
        )

    def _short_conv(
        self, x: mx.array, cache: Optional[Any], target_verify: bool = False
    ) -> mx.array:
        B, _, C = x.shape
        n_keep = self.short_conv_state_len
        state = cache[2] if cache is not None else None
        if state is None or state.shape[0] != B:
            state = mx.zeros((B, n_keep, C), dtype=x.dtype)
        elif state.shape[1] != n_keep:
            # A verify pass left its whole window behind; the live taps are its
            # tail. Slicing on read also keeps a missed rollback from turning into
            # a convolution shape error.
            state = state[:, -n_keep:]
        conv_input = mx.concatenate([state, x], axis=1)
        if cache is not None:
            # Keep the whole window while verifying so a rejected block can be
            # rolled back by slicing -- see `LanguageModel.rollback_speculative_cache`.
            cache[2] = conv_input if target_verify else conv_input[:, -n_keep:]
        return nn.silu(self.conv1d(conv_input))

    def __call__(
        self,
        hidden_states: mx.array,
        input_ids: mx.array,
        cache: Optional[Any] = None,
        mask: Optional[mx.array] = None,
        target_verify: bool = False,
    ) -> mx.array:
        embeddings = self.ple_embedding(input_ids, cache, target_verify)
        B, L, _ = hidden_states.shape
        stream_shape = (B, L, self.hc_count, self.hidden_size)

        keys = self.norm_key(self.key_proj(embeddings)).reshape(stream_shape)
        queries = self.norm_query(hidden_states).reshape(stream_shape)
        value = self.value_proj(embeddings)

        gate = (keys * queries).sum(axis=-1, keepdims=True) / math.sqrt(
            self.hidden_size
        )
        gate = mx.sign(gate) * mx.sqrt(mx.maximum(mx.abs(gate), 1e-6))
        gated = (mx.sigmoid(gate) * value[..., None, :]).reshape(B, L, -1)
        gated_normed = self.norm_conv(gated)

        conv_mask = None
        if mask is not None:
            conv_mask = mask[:, -L:][..., None].astype(gated.dtype)
            gated = gated * conv_mask
            gated_normed = gated_normed * conv_mask

        out = gated + self._short_conv(gated_normed, cache, target_verify)
        return out if conv_mask is None else out * conv_mask


# --- Sparse (QSA) attention ---------------------------------------------------

# Cap on the intermediate (B, L, indexer_heads, n_blocks) float32 score tensor,
# i.e. 16 MiB -- the same bound the reference implementation uses.
_MAX_SCORE_ELEMENTS = 1 << 22


class Qwen4ExpBlockCache(_BaseCache):
    """The QSA indexer's keys: one row per token, plus the pooled block keys.

    A block key is a pure function of ``ratio`` consecutive token keys and the
    absolute position of the first of them, so blocks never move -- block ``b``
    always covers positions ``[b * ratio, (b + 1) * ratio)``. That makes `trim`
    exact and O(1): drop the tail tokens and keep the first ``offset // ratio``
    blocks.

    Which is why the per-token keys are kept rather than dropped once pooled, the
    way :class:`PoolingCache` drops them. Without them a trim could not rebuild
    the partial trailing block, and every trim -- speculative rollback, prefix
    reuse -- would have to give the indexer up for the rest of the sequence. They
    are cheap next to what they sit beside: one head, keys only, and no rotation
    per token -- a block is rotated by the position of its first token, so only
    those rows are worth keeping.
    """

    step = 256

    def __init__(self, ratio: int):
        self.ratio = max(int(ratio), 1)
        self.index_keys = None
        self.block_rotation = None
        self.offset = 0
        self.pooled = None
        self.n_pooled = 0

    @property
    def n_leading(self) -> int:
        """How many positions so far start a block, filled or not."""
        return (self.offset + self.ratio - 1) // self.ratio

    @property
    def blocks(self):
        """The pooled block keys: a view of the step-allocated buffer."""
        return None if self.pooled is None else self.pooled[:, : self.n_pooled]

    @property
    def leading_rotation(self):
        return (
            None
            if self.block_rotation is None
            else self.block_rotation[:, : self.n_leading]
        )

    def _room_for(self, buf, need, like, width):
        """A buffer with room for `need` rows, and whether it is a fresh one.

        Every distinct array shape is a distinct CUDA graph, so these grow in
        `step` chunks and are written in place. Growing by concatenation instead
        would mint a new shape on every step that completes a block and thrash the
        graph cache -- at a 12k-token prompt that is thousands of shapes.
        """
        if buf is not None and need <= buf.shape[1]:
            return buf, False
        capacity = ((need + self.step - 1) // self.step) * self.step
        return mx.zeros((like.shape[0], capacity, width), like.dtype), True

    def append(self, keys: mx.array, rotation: mx.array) -> None:
        """Store this step's raw keys, and the rope factors of block-leading ones."""
        _, L, D = keys.shape
        prev = self.offset
        end = prev + L

        buf, fresh = self._room_for(self.index_keys, end, keys, D)
        if fresh:
            if self.index_keys is not None:
                buf[:, :prev] = self.index_keys[:, :prev]
            self.index_keys = buf
        self.index_keys[:, prev:end] = keys

        # Block `b` always starts at absolute position `b * ratio`, so which rows
        # lead a block never changes -- they can be picked out on the way in and
        # survive any later trim.
        leading = rotation[:, (-prev) % self.ratio :: self.ratio]
        if leading.shape[1]:
            prior = (prev + self.ratio - 1) // self.ratio
            need = prior + leading.shape[1]
            buf, fresh = self._room_for(
                self.block_rotation, need, rotation, rotation.shape[-1]
            )
            if fresh:
                if self.block_rotation is not None:
                    buf[:, :prior] = self.block_rotation[:, :prior]
                self.block_rotation = buf
            self.block_rotation[:, prior:need] = leading
        self.offset = end

    def pending_windows(self):
        """``(keys, block_rotation)`` for blocks that are complete but not pooled."""
        complete = self.offset // self.ratio
        if complete <= self.n_pooled:
            return None
        lo, hi = self.n_pooled * self.ratio, complete * self.ratio
        return (
            self.index_keys[:, lo:hi],
            self.block_rotation[:, self.n_pooled : complete],
        )

    def commit(self, block_keys: mx.array) -> mx.array:
        prior = self.n_pooled
        need = prior + block_keys.shape[1]
        buf, fresh = self._room_for(self.pooled, need, block_keys, block_keys.shape[-1])
        if fresh:
            if self.pooled is not None:
                buf[:, :prior] = self.pooled[:, :prior]
            self.pooled = buf
        self.pooled[:, prior:need] = block_keys
        self.n_pooled = need
        return self.blocks

    def is_trimmable(self):
        return True

    def trim(self, n):
        # Only the logical lengths move: the buffers keep their capacity so the
        # next append reuses the shapes already in the graph cache.
        n = min(int(n), self.offset)
        self.offset -= n
        self.n_pooled = min(self.n_pooled, self.offset // self.ratio)
        return n

    @property
    def state(self):
        if self.index_keys is None:
            return (None, self.leading_rotation, self.blocks)
        return (self.index_keys[:, : self.offset], self.leading_rotation, self.blocks)

    @state.setter
    def state(self, v):
        self.index_keys, self.block_rotation, self.pooled = v
        self.offset = 0 if self.index_keys is None else self.index_keys.shape[1]
        self.n_pooled = 0 if self.pooled is None else self.pooled.shape[1]

    @property
    def meta_state(self):
        return str(self.ratio)

    @meta_state.setter
    def meta_state(self, v):
        self.ratio = int(v)

    def size(self):
        return self.offset

    def empty(self):
        return self.offset == 0

    @property
    def nbytes(self):
        total = 0 if self.pooled is None else self.pooled.nbytes
        if self.index_keys is not None:
            total += self.index_keys.nbytes
        if self.block_rotation is not None:
            total += self.block_rotation.nbytes
        return total

    def extract(self, idx):
        out = type(self)(self.ratio)
        out.state = tuple(
            None if a is None else mx.contiguous(a[idx : idx + 1]) for a in self.state
        )
        return out


class Qwen4ExpAttentionCache(CacheList):
    """The attention KV cache plus the QSA indexer's keys.

    Both halves trim, so the speculative rollback and prefix reuse need nothing
    from this class beyond what :class:`CacheList` already does -- see
    :class:`Qwen4ExpBlockCache` for why that is possible at all.
    """

    # Batched left-padded prompts put block boundaries at content-relative offsets,
    # which the position-relative pooling in the indexer cannot express. Rather than
    # poison the block cache the indexer switches off for the rest of that sequence
    # and attention stays dense -- a superset of the sparse selection, so still
    # correct, only slower. A class attribute because `from_state` builds instances
    # through `__new__`, skipping `__init__`; deliberately not part of `meta_state`,
    # since a restored cache re-latches the moment it sees such a batch.
    indexer_disabled = False

    def __init__(self, *caches, compress_ratio: int = 1):
        super().__init__(
            *(caches or (KVCache(), Qwen4ExpBlockCache(max(compress_ratio, 1))))
        )

    # The attention half is what the inherited `Qwen3_5Attention` talks to, so it
    # is proxied straight through; `caches[1]` is only ever touched by the indexer.
    @property
    def offset(self):
        return self.caches[0].offset

    @offset.setter
    def offset(self, value):
        self.caches[0].offset = value

    @property
    def keys(self):
        return self.caches[0].keys

    @keys.setter
    def keys(self, value):
        self.caches[0].keys = value

    @property
    def values(self):
        return self.caches[0].values

    @property
    def compress_ratio(self):
        return self.caches[1].ratio

    def update_and_fetch(self, keys, values):
        return self.caches[0].update_and_fetch(keys, values)

    def make_mask(self, *args, **kwargs):
        return self.caches[0].make_mask(*args, **kwargs)

    # `CacheList` forwards these to every child, but the block cache has no notion
    # of per-row padding and a plain `KVCache` has no `prepare` at all -- so the
    # inherited version raises the moment the batched speculative rollback probes
    # for it. Answer for the attention half only, which leaves a plain `KVCache`
    # looking exactly like the bare one qwen3_5 uses, and still forwards properly
    # if this cache is ever built over a batch-aware one.
    @property
    def prepare(self):
        return getattr(self.caches[0], "prepare", None)

    @property
    def finalize(self):
        return getattr(self.caches[0], "finalize", None)

    def extract(self, idx):
        return type(self)(*(c.extract(idx) for c in self.caches))

    @classmethod
    def from_state(cls, state, meta_state):
        # `CacheList.from_state` looks its children up in `cache.py`'s globals,
        # where `Qwen4ExpBlockCache` does not live.
        obj = cls.__new__(cls)
        _, metas = meta_state
        obj.caches = [
            child.from_state(s, m)
            for child, s, m in zip((KVCache, Qwen4ExpBlockCache), state, metas)
        ]
        return obj


class Qwen4ExpQSAIndexer(nn.Module):
    """Picks the key blocks each query is allowed to attend to.

    Keys are pooled over runs of ``compress_ratio`` consecutive tokens; the
    ``token_budget // compress_ratio`` highest scoring blocks are kept, plus the
    trailing partial block, which is always visible.
    """

    def __init__(self, args: TextConfig):
        super().__init__()
        self.n_heads = args.indexer_n_heads
        self.kv_heads = args.indexer_kv_heads
        self.head_dim = args.indexer_head_dim
        self.token_budget = args.indexer_budget
        self.compress_ratio = args.indexer_compress_ratio
        self.block_topk = self.token_budget // self.compress_ratio
        self.scale = self.head_dim**-0.5
        # Width of a single-query selection: every block of the budget plus the
        # longest possible trailing partial block. This is the buffer size the
        # reference implementation uses, and holding it constant keeps the shape of
        # the gathered KV identical from one decode step to the next.
        self.decode_width = (
            self.block_topk * self.compress_ratio + self.compress_ratio - 1
        )

        self.index_qk_proj = nn.Linear(
            args.hidden_size,
            (self.n_heads + self.kv_heads) * self.head_dim,
            bias=False,
        )
        self.q_layernorm = nn.RMSNorm(self.head_dim, eps=args.rms_norm_eps)
        self.k_layernorm = nn.RMSNorm(self.head_dim, eps=args.rms_norm_eps)

    def _pool_blocks(self, keys: mx.array, block_rotation: mx.array) -> mx.array:
        """Pool ``(B, n, compress_ratio, D)`` windows into one key per block.

        ``block_rotation`` is ``(B, n, ...)`` -- one row per block, already the
        rotation of that block's first token.
        """
        pooled = self.k_layernorm(
            keys.astype(mx.float32).mean(axis=2).astype(keys.dtype)
        )
        cos, sin = mx.split(block_rotation, 2, axis=-1)
        return _apply_partial_rope(pooled, cos, sin)

    def _ingest(
        self,
        keys: mx.array,
        cos: mx.array,
        sin: mx.array,
        cache: Optional[Any],
        offset: int,
    ) -> mx.array:
        """Fold this step's keys into the block cache and return every block key.

        The rotation is stored alongside the keys rather than re-derived from the
        block index: M-RoPE positions are not a plain arange once images are in the
        prompt, so a block has to remember the rotation of its own first token.
        """
        del offset  # block boundaries come from the cache's own offset
        ratio = self.compress_ratio
        rotation = mx.concatenate([cos, sin], axis=-1).astype(keys.dtype)
        blocks = cache[1] if cache is not None else None
        if blocks is None:
            usable = (keys.shape[1] // ratio) * ratio
            return self._pool_blocks_or_empty(
                keys[:, :usable], rotation[:, :usable:ratio], keys.dtype
            )

        blocks.append(keys, rotation)
        pending = blocks.pending_windows()
        if pending is None:
            view = blocks.blocks
            return view if view is not None else self._no_blocks(keys)
        return blocks.commit(self._pool_blocks_or_empty(*pending, keys.dtype))

    def _no_blocks(self, like: mx.array) -> mx.array:
        return mx.zeros((like.shape[0], 0, self.head_dim), like.dtype)

    def _pool_blocks_or_empty(
        self, keys: mx.array, block_rotation: mx.array, dtype
    ) -> mx.array:
        if keys.size == 0:
            return mx.zeros((keys.shape[0], 0, self.head_dim), dtype)
        return self._pool_blocks(
            mx.unflatten(keys, 1, (-1, self.compress_ratio)), block_rotation
        )

    def _score_blocks(
        self, queries: mx.array, block_keys: mx.array, complete_blocks: mx.array
    ) -> mx.array:
        """Top ``block_topk`` block indices per query, chunked over the query axis.

        The intermediate ``(B, L, indexer_heads, n_blocks)`` float32 score tensor is
        what bounds memory during a long prefill, so it is capped the same way the
        reference implementation caps it.
        """
        B, L = queries.shape[0], queries.shape[1]
        n_blocks = block_keys.shape[1]
        block_positions = mx.arange(n_blocks)
        keys_t = block_keys[:, None].swapaxes(-1, -2).astype(mx.float32)

        per_query = B * self.n_heads * max(n_blocks, 1)
        chunk = min(L, max(1, _MAX_SCORE_ELEMENTS // per_query))

        selected = []
        for begin in range(0, L, chunk):
            stop = min(begin + chunk, L)
            scores = mx.matmul(queries[:, begin:stop].astype(mx.float32), keys_t)
            scores = mx.maximum(scores, 0).sum(axis=2) * self.scale
            visible = block_positions < complete_blocks[begin:stop][:, None]
            scores = mx.where(visible, scores, -mx.inf)
            # Only the *set* of winning blocks matters downstream, so partitioning is
            # enough -- the same choice deepseek_v32, deepseek_v4, glm_moe_dsa and
            # minimax_m3_vl make. Blocks tied at the cut-off are picked arbitrarily:
            # the reference's topk resolves such ties towards the higher block index,
            # which no ordering here reproduces, so a full sort buys nothing.
            selected.append(
                mx.argpartition(-scores, kth=self.block_topk - 1, axis=-1)[
                    ..., : self.block_topk
                ]
            )
        return selected[0] if len(selected) == 1 else mx.concatenate(selected, axis=1)

    def select(
        self,
        x: mx.array,
        cos: mx.array,
        sin: mx.array,
        cache: Optional[Any],
        offset: int,
    ) -> Optional[Tuple[mx.array, mx.array, int]]:
        """Score the complete key blocks and keep the best ``block_topk`` per query.

        Returns ``None`` while every complete block still fits in the budget, in
        which case attention stays plain causal. Otherwise returns
        ``(block_indices, query_positions, kv_length)`` with ``block_indices`` of
        shape ``(B, L, block_topk)`` in arbitrary order.
        """
        B, L, _ = x.shape
        queries, keys = mx.split(
            self.index_qk_proj(x), [self.n_heads * self.head_dim], axis=-1
        )

        # `kv_heads` is 1 (the config rejects anything else), so the projection's
        # trailing axis is exactly one key per token.
        block_keys = self._ingest(
            keys.reshape(B, L, self.head_dim), cos, sin, cache, offset
        )

        if block_keys.shape[1] <= self.block_topk:
            # Every complete block fits in the budget: plain causal attention. The
            # keys still had to reach the cache for later steps, but the query half
            # of the indexer is dead work here, so it is only built below.
            return None

        queries = self.q_layernorm(queries.reshape(B, L, self.n_heads, self.head_dim))
        queries = _apply_partial_rope(queries, cos[:, :, None, :], sin[:, :, None, :])

        query_positions = offset + mx.arange(L)
        complete_blocks = (query_positions + 1) // self.compress_ratio
        block_indices = self._score_blocks(queries, block_keys, complete_blocks)
        return block_indices, query_positions, offset + L

    def block_mask(self, selection: Tuple[mx.array, mx.array, int]) -> mx.array:
        """A dense ``(B, 1, L, kv_length)`` mask for the selection.

        Used whenever more than one query is in flight: each of them picks a
        different set of blocks, so there is no rectangular gather to do and the
        mask is the cheaper encoding.
        """
        block_indices, query_positions, kv_length = selection
        ratio = self.compress_ratio

        selected = mx.put_along_axis(
            mx.zeros((*block_indices.shape[:-1], kv_length // ratio), dtype=mx.bool_),
            block_indices,
            mx.array(True),
            axis=-1,
        )
        selected = mx.repeat(selected, ratio, axis=-1)
        if selected.shape[-1] < kv_length:
            selected = mx.concatenate(
                [
                    selected,
                    mx.zeros(
                        (*selected.shape[:-1], kv_length - selected.shape[-1]),
                        dtype=mx.bool_,
                    ),
                ],
                axis=-1,
            )

        key_positions = mx.arange(kv_length)
        causal = key_positions <= query_positions[:, None]
        complete_blocks = (query_positions + 1) // ratio
        tail = causal & (key_positions >= (complete_blocks * ratio)[:, None])
        return ((selected & causal) | tail)[:, None]

    def gather_indices(
        self, selection: Tuple[mx.array, mx.array, int]
    ) -> Tuple[mx.array, mx.array]:
        """Key positions a single query attends to, as ``(indices, valid)``.

        Both are ``(B, 1, decode_width)``. The lone query of a decode step sees
        every complete block, so all ``block_topk`` winners are causal and in
        range; only the trailing partial block can come up short, and its unused
        slots are pointed at position 0 and masked out through ``valid``.
        """
        block_indices, _, kv_length = selection
        ratio = self.compress_ratio
        batch_shape = block_indices.shape[:-1]

        tokens = (block_indices[..., None] * ratio + mx.arange(ratio)).reshape(
            *batch_shape, -1
        )
        tail_offsets = mx.arange(ratio - 1)
        tail = mx.broadcast_to(
            kv_length - kv_length % ratio + tail_offsets, (*batch_shape, ratio - 1)
        )
        valid = mx.concatenate(
            [
                mx.ones(tokens.shape, dtype=mx.bool_),
                mx.broadcast_to(
                    tail_offsets < kv_length % ratio, (*batch_shape, ratio - 1)
                ),
            ],
            axis=-1,
        )
        indices = mx.concatenate([tokens, tail], axis=-1)
        return mx.where(valid, indices, 0), valid

    def __call__(
        self,
        x: mx.array,
        cos: mx.array,
        sin: mx.array,
        cache: Optional[Any],
        offset: int,
    ) -> Optional[mx.array]:
        """The selection as a single dense mask, for callers that want just that."""
        selection = self.select(x, cos, sin, cache, offset)
        return None if selection is None else self.block_mask(selection)


def _is_gatherable(cache) -> bool:
    """True when the KV behind ``cache`` is plain and unpacked, so it can be indexed.

    Quantized and rotating caches hold their keys in a packed or rotated layout that
    indexing the time axis would silently corrupt, so those keep the mask.
    """
    return isinstance(cache, Qwen4ExpAttentionCache) and type(cache[0]) is KVCache


def _gather_along_time(x: mx.array, indices: mx.array) -> mx.array:
    """Select ``(B, 1, N)`` key positions out of a ``(B, heads, T, D)`` cache."""
    return mx.take_along_axis(
        x,
        mx.broadcast_to(
            indices[:, :, :, None],
            (x.shape[0], x.shape[1], indices.shape[-1], x.shape[3]),
        ),
        axis=2,
    )


class Qwen4ExpAttention(Qwen3_5Attention):
    def __init__(self, args: TextConfig):
        super().__init__(args)
        self.indexer = Qwen4ExpQSAIndexer(args) if args.uses_indexer else None

    def post_cache(self, keys, values, mask, cache):
        # Handed over on the cache rather than on `self`: `nn.Module.__setattr__`
        # routes tuples into the module dict, which would put a live cache into
        # `model.state`. The qwen3_5 mask helpers stash on the cache for the same
        # reason.
        gather = getattr(cache, "_qsa_pending", None)
        if cache is not None:
            cache._qsa_pending = None
            pooled = cache[1].blocks if self.indexer is not None else None
            if pooled is not None and cache.keys is not None:
                # On steps below the budget nothing reads the block keys, so pin
                # their update to the attention keys to keep both in one evaluation
                # instead of letting the graph pile up (as deepseek_v32 and
                # glm_moe_dsa do).
                cache.keys = mx.depends(cache.keys, (pooled,))

        if gather is None:
            return keys, values, mask

        indices, valid = gather
        keys = _gather_along_time(keys, indices)
        values = _gather_along_time(values, indices)
        # `mask` is None on this path -- the gather only runs for a single query,
        # whose causal mask is implicit, and `valid` covers the padded tail slots.
        return keys, values, valid[:, None]

    def __call__(
        self,
        x: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional[Any] = None,
        position_ids: Optional[mx.array] = None,
        position_embeddings: Optional[tuple[mx.array, mx.array]] = None,
        target_verify: bool = False,
    ) -> mx.array:
        gather = None

        if self.indexer is not None and position_embeddings is not None:
            offset = cache.offset if cache is not None else 0
            if isinstance(offset, mx.array):
                offset = None if offset.ndim > 0 else int(offset)
            if offset is None and cache is not None:
                # Batched left-padded prompts put block boundaries at
                # content-relative offsets, which the position-relative pooling in
                # the indexer cannot express. Rather than poison the block cache,
                # give up on the indexer for the rest of this sequence: dense
                # attention is a superset of the sparse selection, so it stays
                # correct, only slower.
                cache.indexer_disabled = True
            if offset is not None and not (
                cache is not None and cache.indexer_disabled
            ):
                selection = self.indexer.select(x, *position_embeddings, cache, offset)
                if selection is not None:
                    if (
                        x.shape[1] == 1
                        and mask is None
                        and not target_verify
                        and _is_gatherable(cache)
                    ):
                        # Decode: shrink the cache to the selected keys instead of
                        # masking all of it. `post_cache` does the gather, because
                        # the keys only exist after the parent's cache update.
                        gather = self.indexer.gather_indices(selection)
                    else:
                        sparse_mask = self.indexer.block_mask(selection)
                        mask = (
                            sparse_mask
                            if mask is None or isinstance(mask, str)
                            else mask & sparse_mask
                        )

        if cache is not None:
            cache._qsa_pending = gather
        return super().__call__(
            x,
            mask=mask,
            cache=cache,
            position_ids=position_ids,
            position_embeddings=position_embeddings,
            target_verify=target_verify,
        )


class Qwen4ExpGatedDeltaNet(Qwen3_5GatedDeltaNet):
    def __init__(self, args: TextConfig):
        super().__init__(args)
        self.norm = Qwen4ExpRMSNormGated(
            self.head_v_dim,
            eps=self.layer_norm_epsilon,
            activation=args.output_gate_type or args.hidden_act,
        )


class Qwen4ExpDecoderLayer(nn.Module):
    def __init__(self, args: TextConfig, layer_idx: int):
        super().__init__()
        self.is_linear = args.layer_types[layer_idx] == LINEAR_ATTENTION
        if self.is_linear:
            self.linear_attn = Qwen4ExpGatedDeltaNet(args)
        else:
            self.self_attn = Qwen4ExpAttention(args)
        self.mlp = Qwen3_5MoeSparseMoeBlock(args)

        # ple_layer_ids are one-indexed.
        self.ple = (
            Qwen4ExpPLELayer(args, args.ple_layer_ids.index(layer_idx + 1))
            if (layer_idx + 1) in args.ple_layer_ids
            else None
        )
        self.attn_hyper_connection = Qwen4ExpGatedResidual(args)
        self.mlp_hyper_connection = Qwen4ExpGatedResidual(args)

    def __call__(
        self,
        x: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional[Any] = None,
        position_ids: Optional[mx.array] = None,
        position_embeddings: Optional[tuple[mx.array, mx.array]] = None,
        ple_input_ids: Optional[mx.array] = None,
        gdn_sink: Optional[list] = None,
        target_verify: bool = False,
    ) -> mx.array:
        if self.ple is not None:
            x = x + self.ple(x, ple_input_ids, cache, mask, target_verify)

        h, hyper_input, inject = self.attn_hyper_connection(x)
        if self.is_linear:
            h = self.linear_attn(
                h, mask, cache, gdn_sink=gdn_sink, target_verify=target_verify
            )
        else:
            h = self.self_attn(
                h,
                mask=mask,
                cache=cache,
                position_ids=position_ids,
                position_embeddings=position_embeddings,
                target_verify=target_verify,
            )
        x = _hc_inject(h, hyper_input, inject)

        h, hyper_input, inject = self.mlp_hyper_connection(x)
        return _hc_inject(self.mlp(h, target_verify), hyper_input, inject)


class _StreamHolder:
    """Carries the trunk's pre-mixer streams out of a forward pass.

    Deliberately not a list/tuple/dict: `nn.Module.__setattr__` routes those into
    the module dict, which would put a live activation into `model.state`. An
    opaque object is invisible to `tree_flatten`.
    """

    __slots__ = ("value",)

    def __init__(self):
        self.value = None

    def take(self):
        value, self.value = self.value, None
        return value


class Qwen4ExpModel(nn.Module):
    def __init__(self, args: TextConfig):
        super().__init__()
        self.args = args
        self.embed_tokens = nn.Embedding(args.vocab_size, args.hidden_size)
        self.layers = [
            Qwen4ExpDecoderLayer(args=args, layer_idx=i)
            for i in range(args.num_hidden_layers)
        ]
        self.hyper_connection_mixer = Qwen4ExpGatedResidual(args, use_combine=False)
        self.rotary_emb = Qwen3_5RotaryEmbedding(
            int(args.head_dim * args.rope_parameters["partial_rotary_factor"]),
            max_position_embeddings=args.max_position_embeddings,
            base=args.rope_parameters["rope_theta"],
            mrope_section=args.rope_parameters["mrope_section"],
        )
        layer_types = args.layer_types[: args.num_hidden_layers]
        self.streams = _StreamHolder()
        self.ssm_idx = layer_types.index(LINEAR_ATTENTION)
        self.fa_idx = layer_types.index(SPARSE_ATTENTION)

    def __call__(
        self,
        inputs: mx.array,
        inputs_embeds: Optional[mx.array] = None,
        mask: Optional[mx.array] = None,
        cache=None,
        position_ids: Optional[mx.array] = None,
        capture_layer_ids: Optional[List[int]] = None,
        hidden_sink: Optional[list] = None,
        gdn_sink: Optional[list] = None,
    ):
        if self.args.ple_layer_ids and inputs is None:
            raise ValueError(
                "qwen4_exp PLE layers hash the raw token ids, so input_ids must be "
                "passed even when inputs_embeds is given"
            )

        h = self.embed_tokens(inputs) if inputs_embeds is None else inputs_embeds
        if cache is None:
            cache = [None] * len(self.layers)

        fa_mask = _create_qwen3_5_attention_mask(h, cache[self.fa_idx])
        ssm_mask = _create_qwen3_5_ssm_mask(h, cache[self.ssm_idx])

        if position_ids is None:
            offset = getattr(cache[self.fa_idx], "offset", 0)
            if isinstance(offset, mx.array):
                offset = int(offset.max().item())
            position_ids = mx.arange(offset, offset + h.shape[1])[None, None]
        if position_ids.ndim == 2:
            position_ids = position_ids[None]
        if position_ids.shape[0] == 1:
            position_ids = mx.broadcast_to(position_ids, (3, *position_ids.shape[1:]))
        position_embeddings = self.rotary_emb(h, position_ids)

        ple_input_ids = inputs
        if (
            self.args.ple_layer_ids
            and ssm_mask is not None
            and ple_input_ids is not None
        ):
            # Padding positions must not leak into the n-gram context.
            ple_input_ids = mx.where(
                ssm_mask, ple_input_ids, self.args.ple_eos_token_id
            )

        # One residual stream per hyper-connection.
        h = mx.tile(h, (1, 1, self.args.hc_count))

        capture_set = set(capture_layer_ids) if capture_layer_ids else set()
        for i, (layer, c) in enumerate(zip(self.layers, cache)):
            h = layer(
                h,
                mask=ssm_mask if layer.is_linear else fa_mask,
                cache=c,
                position_ids=position_ids,
                position_embeddings=position_embeddings,
                ple_input_ids=ple_input_ids,
                gdn_sink=gdn_sink,
                target_verify=gdn_sink is not None,
            )
            if hidden_sink is not None and i in capture_set:
                hidden_sink.append(h)

        # The mixer is lossy, so anything downstream that needs the streams has to
        # be handed them from here.
        self.streams.value = h
        return self.hyper_connection_mixer(h)


class LanguageModel(Qwen3_5LanguageModel):
    def __init__(self, args: TextConfig, config: ModelConfig = None):
        nn.Module.__init__(self)
        self.args = args
        self.config = config
        self.model_type = args.model_type
        self.model = Qwen4ExpModel(args)
        self._rope_deltas = None
        self._position_ids = None

        if not args.tie_word_embeddings:
            self.lm_head = nn.Linear(args.hidden_size, args.vocab_size, bias=False)

    def make_cache(self):
        caches = []
        for i, layer_type in enumerate(self.args.layer_types[: len(self.layers)]):
            if layer_type == LINEAR_ATTENTION:
                # 0/1: gated delta net conv + recurrent state.
                # 2/3: PLE short-conv state + n-gram token history.
                caches.append(
                    ArraysCache(size=4 if (i + 1) in self.args.ple_layer_ids else 2)
                )
            else:
                caches.append(
                    Qwen4ExpAttentionCache(
                        compress_ratio=self.args.indexer_compress_ratio or 1
                    )
                )
        return caches

    # --- speculative decoding -------------------------------------------------
    #
    # The MTP head is fed the hyper-connection streams the trunk hands over
    # *before* `hyper_connection_mixer` collapses them -- the mixer is lossy, so
    # the post-mixer hidden cannot stand in. So for qwen4_exp "the speculative
    # hidden state" means those streams, and every consumer that wants logits
    # from them applies the mixer first.

    @property
    def _stream_width(self) -> int:
        return self.args.hc_count * self.args.hidden_size

    def _streams_to_logits_hidden(self, hidden: mx.array) -> mx.array:
        if hidden.shape[-1] == self._stream_width:
            return self.model.hyper_connection_mixer(hidden)
        return hidden

    def __call__(self, inputs, inputs_embeds=None, mask=None, cache=None, **kwargs):
        capture = kwargs.get("capture_layer_ids")
        # Only *widen* a capture the caller already asked for. `capture_layer_ids`
        # is also what switches the shared trunk onto its target-verify path, so
        # introducing one where none was requested would drag an ordinary prefill
        # onto the per-token verify attention.
        wants_streams = bool(kwargs.get("return_hidden")) and capture is not None
        if wants_streams:
            kwargs["capture_layer_ids"] = list(
                dict.fromkeys(list(capture) + [self.args.num_hidden_layers - 1])
            )
        out = super().__call__(inputs, inputs_embeds, mask, cache, **kwargs)
        streams = self.model.streams.take()
        if wants_streams and out.hidden_states and len(out.hidden_states) >= 2:
            # Captures are appended in layer order and the trunk's own output is
            # appended after them, so the last layer's pre-mixer streams sit at
            # [-2]. Move them to [-1], where the runtime looks.
            out.hidden_states[-1] = out.hidden_states[-2]
        elif (
            kwargs.get("return_hidden")
            and capture is None
            and streams is not None
            and out.hidden_states
        ):
            # The MTP prefill asks for hidden with no capture at all
            # (`speculative_prefill_kwargs`), and the head needs the streams the
            # mixer destroys. Take them from the trunk rather than introducing a
            # capture: a capture also builds `gdn_sink`, which switches the whole
            # trunk onto its per-token target-verify path -- wrong for a prefill.
            out.hidden_states[-1] = streams
        return out

    def speculative_logits_from_hidden(self, hidden: mx.array) -> mx.array:
        return super().speculative_logits_from_hidden(
            self._streams_to_logits_hidden(hidden)
        )

    def speculative_argmax_from_hidden(self, hidden: mx.array):
        return super().speculative_argmax_from_hidden(
            self._streams_to_logits_hidden(hidden)
        )

    def rollback_speculative_cache(self, caches, gdn_states, accepted, block_size):
        """Undo a rejected draft, including the PLE states the trunk cannot.

        The shared implementation rebuilds the gated-delta-net conv and recurrent
        states (cache slots 0 and 1) from the captured intermediates. A PLE layer
        keeps two more states in the same cache -- its dilated short-conv taps and
        the n-gram token history -- which that code knows nothing about, so a
        rejected block would leave both carrying the discarded tokens.

        Both are plain sliding windows over their input, so during a verify pass
        the PLE keeps the *whole* window instead of just the live tail and the
        rollback is a slice: with ``kept`` tokens accepted out of the block, the
        state is the window shifted by ``kept``.

        The QSA block cache needs nothing here as long as every row kept the same
        number of tokens: both halves of :class:`Qwen4ExpAttentionCache` trim, so
        the base implementation has already rewound it.
        """
        accepted_max = super().rollback_speculative_cache(
            caches, gdn_states, accepted, block_size
        )
        kept = int(accepted_max) + 1

        if isinstance(accepted, int):
            accepted_list = [int(accepted)]
        elif isinstance(accepted, mx.array):
            accepted_list = [int(x) for x in accepted.reshape(-1).tolist()]
        else:
            accepted_list = [int(x) for x in accepted]
        if len(set(accepted_list)) > 1:
            # Rows kept different numbers of tokens. The trim is uniform, so the
            # shorter rows keep block keys built from tokens they rejected -- and
            # unlike the attention half, whose per-row tail the base implementation
            # zeroes, a block key cannot be fixed after the fact without re-pooling
            # that row alone. Give the indexer up for this sequence instead and let
            # attention run dense, which is a superset of the sparse selection.
            for cache in caches:
                if isinstance(cache, Qwen4ExpAttentionCache):
                    cache.indexer_disabled = True

        for layer_idx in self.args.ple_layer_ids:
            index = layer_idx - 1
            if index >= len(caches) or caches[index] is None:
                continue
            cache = caches[index]
            ple = self.layers[index].ple
            if ple is None:
                continue
            for slot, width in (
                (2, ple.short_conv_state_len),
                (3, ple.ple_embedding.context_len),
            ):
                state = cache[slot]
                if state is None or state.shape[1] <= width:
                    continue
                cache[slot] = state[:, kept : kept + width]
        return accepted_max

    def shard(self, group: Optional[mx.distributed.Group] = None) -> None:
        """Split the experts across `group`; everything else stays replicated.

        Only the MoE is worth splitting -- it is about two thirds of the released
        checkpoint, while attention plus the indexer together are under half a
        percent. And attention *cannot* be split here even if it were worth it:
        there are two key/value heads and the QSA indexer has exactly one, so
        there is nothing to divide, and every rank would have to agree on the
        selected blocks anyway or they would attend to different keys.

        The PLE n-gram table is the other large share. It is a hashed lookup
        rather than a matmul, so splitting it needs a different scheme (by row
        range, with a gather) and it stays replicated for now.

        What gets split is the expert *intermediate* dimension, matching
        `minimax_m3_vl` and `deepseek_v4`: every rank keeps all experts and routes
        identically, but computes only its slice of each one.
        """
        group = group or mx.distributed.init()
        n = group.size()
        if n == 1:
            return

        for name in ("moe_intermediate_size", "shared_expert_intermediate_size"):
            size = getattr(self.args, name)
            if size % n:
                raise ValueError(
                    "qwen4_exp tensor parallelism splits the expert intermediate "
                    f"dimension, so {name}={size} must be divisible by the group "
                    f"size {n}."
                )

        for layer in self.layers:
            moe = layer.mlp
            for experts in (moe.switch_mlp, moe.shared_expert):
                shard_inplace(experts.gate_proj, "all-to-sharded", group=group)
                shard_inplace(experts.up_proj, "all-to-sharded", group=group)
                shard_inplace(experts.down_proj, "sharded-to-all", group=group)
            # `down_proj` now yields a partial sum; the block reduces it.
            moe.sharding_group = group

    @property
    def quant_predicate(self):
        base = super().quant_predicate

        def predicate(path, module):
            # By far the largest tensor in the model: rows are
            # `ple_embed_dim // ngram_heads` wide, which is 160 for the released
            # config. The default group size of 64 does not divide that, so
            # without a narrower one this table silently stays in bf16 and then
            # dominates the converted model -- ~95 GiB of a ~170 GiB result.
            if path.endswith("ple_embedding.ngram_embedding"):
                return {"group_size": 32}
            # Four output features that weight every sublayer's contribution to
            # every residual stream. Tiny, and far too sensitive for 4 bits --
            # the same reason the router and the shared-expert gate are carved out.
            if path.endswith("block_inject_weight"):
                return {"bits": 8}
            return base(path, module) if base is not None else True

        return predicate

    @property
    def head_dim(self):
        return self.args.head_dim
