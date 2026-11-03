"""Mistral Large 4: latent attention, DSA, LeRoPE and expert parallelism.

Architecture follows Hugging Face's Apache-2.0 Mistral Large 4 reference.
Packed NVFP4 experts retain their original per-block and global scales.
"""

import math
from functools import partial

import mlx.core as mx
import mlx.nn as nn

from ..base import LanguageModelOutput
from ..cache import _BaseCache, create_causal_mask, dynamic_roll
from ..switch_layers import (
    QuantizedSwitchLinear,
    SwitchLinear,
    _gather_sort,
    _scatter_unsort,
)
from .config import TextConfig
from .mla_kernel import sparse_mla
from .moe_kernel import masked_qmv, supports_masked_qmv
from .verify_kernel import native_wide_linear


def decode_linear(module, x, group=None):
    """Use stable decode reductions, sharing complete output rows across ranks."""
    if (
        not module.training
        and isinstance(module, nn.Linear)
        and "bias" not in module
        and x.ndim == 3
        and x.shape[1] == 1
    ):
        output = native_wide_linear(module.weight, x, group)
        if output is not None:
            return output
    return module(x)


def router_logits(x, weight):
    x, weight = x.astype(mx.float32), weight.astype(mx.float32)
    if x.ndim == 3 and x.shape[1] == 1:
        output = native_wide_linear(weight, x)
        if output is not None:
            return output
    return x @ weight.T


class LatentCache(_BaseCache):
    step = 256

    def __init__(self):
        self.keys = self.values = self.index_keys = None
        self.offset = 0

    def _append(self, cache, values):
        start = self.size()
        end = start + values.shape[-2]
        if cache is None or end > cache.shape[-2]:
            capacity = ((values.shape[-2] + self.step - 1) // self.step) * self.step
            padding = mx.zeros(
                (*values.shape[:-2], capacity, values.shape[-1]), dtype=values.dtype
            )
            cache = (
                padding
                if cache is None
                else mx.concatenate([cache[..., :start, :], padding], axis=-2)
            )
        cache[..., start:end, :] = values
        return cache

    def update_latent(self, keys, values):
        end = self.size() + keys.shape[2]
        self.keys = self._append(self.keys, keys)
        self.values = self._append(self.values, values)
        return self.keys[..., :end, :], self.values[..., :end, :]

    def update_index(self, keys):
        self.index_keys = self._append(self.index_keys, keys)
        return self.index_keys[:, : self.size() + keys.shape[1]]

    def advance(self, n):
        self.offset += n

    def make_mask(self, n):
        return create_causal_mask(n, offset=self.size())[None, None]

    def to_batch(self, left_padding):
        if self.empty():
            return BatchLatentCache(left_padding)
        if len(left_padding) != 1:
            raise ValueError(
                "A populated LatentCache must contain one request to batch"
            )
        cache = BatchLatentCache.merge([self])
        padding = left_padding[0]
        cache._map(
            lambda x: mx.pad(x, [(0, 0)] * (x.ndim - 2) + [(padding, 0), (0, 0)])
        )
        cache.left_padding += padding
        cache._idx += padding
        return cache

    @classmethod
    def merge(cls, caches, prefix_lens=None):
        return BatchLatentCache.merge(caches, prefix_lens)

    @property
    def state(self):
        return [
            x[..., : self.size(), :]
            for x in (self.keys, self.values, self.index_keys)
            if x is not None
        ]

    @state.setter
    def state(self, value):
        self._restored_state = value

    @property
    def meta_state(self):
        return (
            str(self.offset),
            str(int(self.keys is not None)),
            str(int(self.index_keys is not None)),
        )

    @meta_state.setter
    def meta_state(self, value):
        offset, latent, index = map(int, value)
        state = iter(self._restored_state)
        self.keys = next(state) if latent else None
        self.values = next(state) if latent else None
        self.index_keys = next(state) if index else None
        self.offset = offset
        del self._restored_state

    def size(self):
        return self.offset

    def empty(self):
        return self.size() == 0

    @property
    def nbytes(self):
        return sum(
            x.nbytes for x in (self.keys, self.values, self.index_keys) if x is not None
        )

    def is_trimmable(self):
        return True

    def trim(self, n):
        n = min(n, self.size())
        self.advance(-n)
        return n


class BatchLatentCache(LatentCache):
    """Keep latent, rotary and indexer keys aligned across padded requests.

    ``_idx`` bounds the physical cache extent; ``offset`` holds each request's
    logical position. After ragged rollback, rows append at offset + padding
    without moving earlier keys across attention reduction boundaries. Shared
    KV/indexer layers still advance positions even when they own no tensors.
    """

    def __init__(self, left_padding):
        super().__init__()
        self.left_padding = mx.array(left_padding, dtype=mx.int32)
        self.offset = -self.left_padding
        self._idx = 0
        self._right_padding = None
        self._ragged = False

    def _append(self, cache, values):
        if not self._ragged:
            return super()._append(cache, values)
        end = self._idx + values.shape[-2]
        if cache is None or end > cache.shape[-2]:
            capacity = ((end + self.step - 1) // self.step) * self.step
            if cache is None:
                cache = mx.zeros(
                    (*values.shape[:-2], capacity, values.shape[-1]), values.dtype
                )
            else:
                cache = mx.pad(
                    cache,
                    [(0, 0)] * (cache.ndim - 2)
                    + [(0, capacity - cache.shape[-2]), (0, 0)],
                )
        positions = (self.offset + self.left_padding)[:, None] + mx.arange(
            values.shape[-2]
        )
        positions = positions.reshape(
            (values.shape[0],) + (1,) * (values.ndim - 3) + (values.shape[-2], 1)
        )
        return mx.put_along_axis(cache, positions, values, axis=values.ndim - 2)

    def _map(self, fn):
        for name in ("keys", "values", "index_keys"):
            value = getattr(self, name)
            if value is not None:
                setattr(self, name, fn(value))

    def size(self):
        return self._idx

    @property
    def batch_size(self):
        return self.offset.shape[0]

    def is_single_row(self):
        return self.batch_size == 1

    def advance(self, n):
        super().advance(n)
        self._idx += n

    def make_mask(self, n):
        if self._ragged:
            positions = (self.offset + self.left_padding)[:, None] + mx.arange(n)
            columns = mx.arange(self._idx + n)[None, None]
            return (
                (columns <= positions[..., None])
                & (columns >= self.left_padding[:, None, None])
            )[:, None]
        return create_causal_mask(n, offset=self._idx, left_padding=self.left_padding)

    def prepare(self, *, left_padding=None, lengths=None, right_padding=None):
        if left_padding is not None:
            if not self.empty():
                raise ValueError(
                    "Left padding can only be added to an empty BatchLatentCache"
                )
            padding = mx.array(left_padding, dtype=mx.int32)
            self.left_padding += padding
            self.offset -= padding
        if right_padding is not None and max(right_padding) > 0:
            self._right_padding = mx.array(right_padding, dtype=mx.int32)

    def finalize(self):
        if self._right_padding is not None:
            padding = self._right_padding
            self._map(
                lambda x: dynamic_roll(
                    x, padding.reshape((-1,) + (1,) * (x.ndim - 3)), axis=x.ndim - 2
                )
            )
            self.offset -= padding
            self.left_padding += padding
            self._right_padding = None

    def trim_right(self, padding):
        # Shared KV layers and index-only layers must advance the same logical
        # positions as layers owning latent keys, even when keys is None.
        self.offset -= mx.array(padding, mx.int32)
        self._ragged = True

    def filter(self, batch_indices):
        self.offset = self.offset[batch_indices]
        self.left_padding = self.left_padding[batch_indices]
        if self._right_padding is not None:
            self._right_padding = self._right_padding[batch_indices]
        # A chunk can consist entirely of padding for the remaining rows.
        removed = min(self.left_padding.min().item(), self._idx)
        self._map(lambda x: x[batch_indices][..., removed:, :])
        self.left_padding -= removed
        self._idx -= removed

    def extend(self, other):
        length = max(self._idx, other._idx)
        for name in ("keys", "values", "index_keys"):
            arrays = [getattr(c, name) for c in (self, other)]
            sample = next((x for x in arrays if x is not None), None)
            if sample is None:
                continue
            capacity = max(length, *(x.shape[-2] for x in arrays if x is not None))
            rows = []
            for c, x in zip((self, other), arrays):
                if x is None:
                    x = mx.zeros(
                        (c.batch_size, *sample.shape[1:-2], 0, sample.shape[-1]),
                        dtype=sample.dtype,
                    )
                x = x[..., : c._idx, :]
                padding = [(0, 0)] * (x.ndim - 2) + [
                    (length - x.shape[-2], capacity - length),
                    (0, 0),
                ]
                rows.append(mx.pad(x, padding))
            setattr(self, name, mx.concatenate(rows))
        self.left_padding = mx.concatenate(
            [c.left_padding + length - c._idx for c in (self, other)]
        )
        self.offset = mx.concatenate([self.offset, other.offset])
        self._idx = length
        self._ragged = self._ragged or other._ragged

    def extract(self, idx):
        cache = LatentCache()
        padding = self.left_padding[idx].item()
        length = max(0, self.offset[idx].item())
        for name in ("keys", "values", "index_keys"):
            value = getattr(self, name)
            if value is not None:
                setattr(
                    cache,
                    name,
                    mx.contiguous(
                        value[idx : idx + 1, ..., padding : padding + length, :]
                    ),
                )
        cache.offset = length
        return cache

    @classmethod
    def merge(cls, caches, prefix_lens=None):
        lengths = [c.size() for c in caches] if prefix_lens is None else prefix_lens
        length = max(lengths, default=0)
        padding = [length - n for n in lengths]
        cache = cls(padding)
        for name in ("keys", "values", "index_keys"):
            arrays = [getattr(c, name) for c in caches]
            sample = next((x for x in arrays if x is not None), None)
            if sample is None:
                continue
            rows = []
            for x, n, pad in zip(arrays, lengths, padding):
                if x is None:
                    x = mx.zeros(
                        (1, *sample.shape[1:-2], n, sample.shape[-1]),
                        dtype=sample.dtype,
                    )
                elif x.shape[0] != 1:
                    raise ValueError("LatentCache.merge expects single-request caches")
                rows.append(
                    mx.pad(x[..., :n, :], [(0, 0)] * (x.ndim - 2) + [(pad, 0), (0, 0)])
                )
            setattr(cache, name, mx.concatenate(rows))
        cache.advance(length)
        return cache

    @property
    def state(self):
        state = super().state + [self.offset, self.left_padding]
        if self._right_padding is not None:
            state.append(self._right_padding)
        return state

    @state.setter
    def state(self, value):
        self._restored_state = value

    @property
    def meta_state(self):
        return (
            str(self._idx),
            str(int(self.keys is not None)),
            str(int(self.index_keys is not None)),
            str(int(self._right_padding is not None)),
            str(int(self._ragged)),
        )

    @meta_state.setter
    def meta_state(self, value):
        self._idx, latent, index, right = map(int, value[:4])
        self._ragged = bool(int(value[4])) if len(value) > 4 else False
        state = iter(self._restored_state)
        self.keys = next(state) if latent else None
        self.values = next(state) if latent else None
        self.index_keys = next(state) if index else None
        self.offset, self.left_padding = next(state), next(state)
        self._right_padding = next(state) if right else None
        del self._restored_state


class LeRoPE(nn.Module):
    def __init__(self, args):
        super().__init__()
        dim = args.qk_rope_head_dim
        p = args.rope_parameters
        base = p.get("rope_theta", 10000.0)
        factor = p.get("factor", 1.0)
        original = p.get("original_max_position_embeddings", 512)

        def correction(rotations):
            return (
                dim
                * math.log(original / (rotations * 2 * math.pi))
                / (2 * math.log(base))
            )

        low = max(math.floor(correction(p.get("beta_fast", 32.0))), 0)
        high = min(math.ceil(correction(p.get("beta_slow", 1.0))), dim - 1)
        ramp = mx.clip(
            (mx.arange(dim // 2, dtype=mx.float32) - low) / max(high - low, 0.001), 0, 1
        )
        inv = base ** (-mx.arange(0, dim, 2, dtype=mx.float32) / dim)
        self._inv_freq = inv * ((1 - ramp) + ramp / factor)
        self.alpha = mx.zeros((dim // 2,))

    def __call__(self, positions, dtype):
        angles = (
            positions[..., None].astype(mx.float32)
            * self._inv_freq
            * mx.exp(self.alpha.astype(mx.float32))
        )
        return mx.cos(angles).astype(dtype), mx.sin(angles).astype(dtype)


def rotate(x, cos, sin):
    """Interleaved input pairs, half-split output (the Mistral convention)."""
    a, b = x[..., ::2], x[..., 1::2]
    return mx.concatenate([a * cos - b * sin, b * cos + a * sin], axis=-1)


class SoftmaxScaling(nn.Module):
    def __init__(self, heads):
        super().__init__()
        self.s = mx.zeros((heads,))
        self.b = mx.ones((heads,))

    def __call__(self, q, positions):
        log = mx.log(positions.astype(mx.float32) + 1).astype(q.dtype)
        scale = (
            self.s.astype(q.dtype)[None, :, None] * log[:, None, :]
            + self.b.astype(q.dtype)[None, :, None]
        )
        return q * scale[..., None]


def _indexer_scores(q, k, weights, mask, scale):
    scores = mx.maximum(
        q.astype(mx.float32) @ k[:, None].swapaxes(-1, -2).astype(mx.float32) * scale,
        0,
    )
    scores = (scores * weights[..., None]).sum(axis=1)
    return mx.where(mask[:, 0], scores, -float("inf"))


# Cache length changes during decode. Keep the head-dependent scale explicit
# so a shapeless trace can also be reused safely across model configurations.
_compiled_indexer_scores = mx.compile(_indexer_scores, shapeless=True)


class Indexer(nn.Module):
    def __init__(self, args, idx):
        super().__init__()
        self.heads = args.index_n_heads
        self.dim = args.index_head_dim
        self.rope_dim = args.qk_rope_head_dim
        self.use_rope = args.no_rope_layers[idx]
        self.topk = args.index_topk
        self.wq_b = nn.Linear(args.hidden_size, self.heads * self.dim, bias=False)
        self.q_input_norm = nn.LayerNorm(args.hidden_size, eps=1e-6)
        self.k_norm = nn.LayerNorm(self.dim, eps=1e-6)
        self.ssmax_scaling = SoftmaxScaling(self.heads)
        self._init_projections(args.hidden_size)

    def _init_projections(self, hidden_size):
        self.wk = nn.Linear(hidden_size, self.dim, bias=False)
        self.weights_proj = nn.Linear(hidden_size, self.heads, bias=False)

    def _project(self, x, need_weights):
        return decode_linear(self.wk, x), (
            decode_linear(self.weights_proj, x) if need_weights else None
        )

    def __call__(self, x, positions, cos, sin, cache, causal):
        b, t, _ = x.shape
        length = (cache.size() if cache is not None else 0) + t
        k, weights = self._project(x, length > self.topk)
        k = self.k_norm(k)
        if self.use_rope:
            k = mx.concatenate(
                [rotate(k[..., : self.rope_dim], cos, sin), k[..., self.rope_dim :]],
                axis=-1,
            )
        if cache is not None:
            k = cache.update_index(k)
        if k.shape[1] <= self.topk:
            return None  # All causal tokens are selected; retain keys for later chunks.
        q = (
            decode_linear(self.wq_b, self.q_input_norm(x))
            .reshape(b, t, self.heads, self.dim)
            .transpose(0, 2, 1, 3)
        )
        q = self.ssmax_scaling(q, positions)
        if self.use_rope:
            q = mx.concatenate(
                [
                    rotate(q[..., : self.rope_dim], cos[:, None], sin[:, None]),
                    q[..., self.rope_dim :],
                ],
                axis=-1,
            )
        weights = weights.transpose(0, 2, 1).astype(mx.float32) * self.heads**-0.5
        score_fn = _compiled_indexer_scores if t == 1 else _indexer_scores
        scores = score_fn(q, k, weights, causal, self.dim**-0.5)
        return mx.argpartition(-scores, kth=self.topk - 1, axis=-1)[..., : self.topk]


class PackedIndexer(Indexer):
    def _init_projections(self, hidden_size):
        self.wk_weights_proj = nn.Linear(hidden_size, self.dim + self.heads, bias=False)

    def _project(self, x, need_weights):
        projection = self.wk_weights_proj
        if isinstance(projection, nn.Linear) and (x.shape[1] != 1 or not need_weights):
            # Preserve prefill GEMM shapes and avoid unused short-prefix work.
            if not self.training and x.shape[1] == 1:
                k = native_wide_linear(projection.weight[: self.dim], x)
                if k is not None:
                    return k, None
            k = x @ projection.weight[: self.dim].T
            weights = x @ projection.weight[self.dim :].T if need_weights else None
            return k, weights
        return mx.split(decode_linear(projection, x), [self.dim], axis=-1)


class Attention(nn.Module):
    def __init__(self, args, idx):
        super().__init__()
        self.group = None
        self.idx = idx
        self.source = args.kv_source_layers[idx]
        self.heads = args.num_attention_heads
        self.rope_dim = args.qk_rope_head_dim if args.no_rope_layers[idx] else 0
        self.nope_dim = args.qk_nope_head_dim + (
            0 if self.rope_dim else args.qk_rope_head_dim
        )
        self.head_dim = self.nope_dim + self.rope_dim
        self.latent = args.kv_lora_rank
        self.value_dim = args.v_head_dim
        self.wq = nn.Linear(args.hidden_size, self.heads * self.head_dim, bias=False)
        self.wo = nn.Linear(self.heads * self.value_dim, args.hidden_size, bias=False)
        if self.source == idx:
            self.wkv_a_with_mqa = nn.Linear(
                args.hidden_size, self.latent + self.rope_dim, bias=False
            )
            self.kv_a_norm = nn.RMSNorm(self.latent, eps=args.rms_norm_eps)
        self.wkv_b = nn.Linear(
            self.latent, self.heads * (self.nope_dim + self.value_dim), bias=False
        )
        self.ssmax_scaling = SoftmaxScaling(self.heads)
        self.indexer = Indexer(args, idx) if args.indexer_types[idx] == "full" else None

    def _decode_attention(self, q, latent, kr, mask, topk=None):
        # Absorb the K projection into Q and apply the V projection after
        # attention, avoiding expansion of every selected cached latent.
        weight = self.wkv_b.weight.reshape(
            self.heads, self.nope_dim + self.value_dim, self.latent
        )
        # Keep projection rounding in the model dtype, with an FP32 attention
        # core, matching the reference's query and latent-output boundaries.
        q_latent = q[..., : self.nope_dim] @ weight[:, : self.nope_dim]
        scale = self.head_dim**-0.5
        indices = mx.sort(topk, axis=-1)[:, 0] if topk is not None else None
        if (
            not self.training
            and mx.default_device() == mx.gpu
            and mx.metal.is_available()
            and mask.shape[1] == 1
        ):
            if indices is None:
                indices = mx.broadcast_to(
                    mx.arange(latent.shape[2])[None], (q.shape[0], latent.shape[2])
                )
            # Fuse cache reads, scores, online softmax and latent accumulation.
            out = sparse_mla(
                q_latent,
                q[..., self.nope_dim :],
                latent,
                kr,
                indices,
                mask,
                scale,
                chunk=128,
            ).astype(q.dtype)
        else:
            if indices is not None:
                latent = mx.take_along_axis(latent, indices[:, None, :, None], axis=2)
                kr = mx.take_along_axis(kr, indices[:, None, :, None], axis=2)
                mask = mx.take_along_axis(mask, indices[:, None, None, :], axis=-1)
            q_latent = q_latent.astype(mx.float32)
            latent = latent.astype(mx.float32)
            kr = kr.astype(mx.float32)
            if self.rope_dim:
                pe_scores = (
                    q[..., self.nope_dim :].astype(mx.float32) @ kr.swapaxes(-1, -2)
                ) * scale
                mask = mx.where(mask, pe_scores, -float("inf"))
            out = mx.fast.scaled_dot_product_attention(
                q_latent, latent, latent, scale=scale, mask=mask
            ).astype(q.dtype)
        return out @ weight[:, self.nope_dim :].swapaxes(-1, -2)

    def __call__(self, x, positions, cos, sin, cache, shared, previous, mask):
        b, t, _ = x.shape
        q = (
            decode_linear(self.wq, x, self.group)
            .reshape(b, t, self.heads, self.head_dim)
            .transpose(0, 2, 1, 3)
        )
        if self.rope_dim:
            q = mx.concatenate(
                [
                    q[..., : self.nope_dim],
                    rotate(q[..., self.nope_dim :], cos[:, None], sin[:, None]),
                ],
                axis=-1,
            )
        if self.source == self.idx:
            kv = decode_linear(self.wkv_a_with_mqa, x)
            latent = self.kv_a_norm(kv[..., : self.latent])[:, None]
            kr = kv[..., self.latent :][:, None]
            if self.rope_dim:
                kr = rotate(kr, cos[:, None], sin[:, None])
            if cache is not None:
                latent, kr = cache.update_latent(latent, kr)
            shared[self.idx] = (latent, kr)
        else:
            latent, kr = shared[self.source]
        q = self.ssmax_scaling(q, positions)
        length = latent.shape[2]
        causal = (
            cache.make_mask(t)
            if cache is not None
            else create_causal_mask(t)[None, None]
        )
        causal = mx.broadcast_to(causal, (b, 1, t, length))
        if mask is not None:
            if mask.ndim == 2:
                causal = causal & mask[:, None, None, :length].astype(mx.bool_)
            elif mask.dtype == mx.bool_:
                causal = causal & mask
            else:
                raise ValueError("Mistral Large 4 expects a boolean attention mask")
        topk = (
            self.indexer(x, positions, cos, sin, cache, causal)
            if self.indexer is not None
            else previous
        )
        decode = t == 1 and isinstance(self.wkv_b, nn.Linear)
        if topk is not None and not decode:
            if t == 1:
                # Expand only selected latents during decode, in sequence order.
                indices = mx.sort(topk, axis=-1)[:, None]
                latent = mx.take_along_axis(latent, indices[..., 0, :, None], axis=2)
                kr = mx.take_along_axis(kr, indices[..., 0, :, None], axis=2)
                causal = mx.take_along_axis(causal, indices, axis=-1)
                length = latent.shape[2]
            else:
                selected = mx.zeros((b, t, length), dtype=mx.bool_)
                selected = mx.put_along_axis(selected, topk, mx.array(True), axis=-1)
                causal = causal & selected[:, None]
        if decode:
            out = self._decode_attention(q, latent, kr, causal, topk)
        else:
            kv = (
                self.wkv_b(latent)
                .reshape(b, length, self.heads, self.nope_dim + self.value_dim)
                .transpose(0, 2, 1, 3)
            )
            kn, values = kv[..., : self.nope_dim], kv[..., self.nope_dim :]
            keys = mx.concatenate(
                [kn, mx.broadcast_to(kr, (b, self.heads, length, self.rope_dim))],
                axis=-1,
            )
            out = mx.fast.scaled_dot_product_attention(
                q, keys, values, scale=self.head_dim**-0.5, mask=causal
            )
        out = out.transpose(0, 2, 1, 3).reshape(b, t, -1)
        if cache is not None:
            cache.advance(t)
        return decode_linear(self.wo, out, self.group), topk


class Experts(nn.Module):
    def __init__(self, args):
        super().__init__()
        n = args.expert_end - args.expert_start
        d, h = args.hidden_size, args.moe_intermediate_size
        linear = (
            partial(
                QuantizedSwitchLinear, bias=False, group_size=16, bits=4, mode="nvfp4"
            )
            if args.nvfp4
            else partial(SwitchLinear, bias=False)
        )
        self.w1 = linear(d, h, n)
        self.w3 = linear(d, h, n)
        self.w2 = linear(h, d, n)

    def __call__(self, x, indices):
        x = mx.expand_dims(x, (-2, -3))
        do_sort = indices.size >= 32
        idx = indices
        inv_order = None
        if do_sort:
            x, idx, inv_order = _gather_sort(x, indices)
        if self.training:
            idx = mx.stop_gradient(idx)
        gate = self.w1(x, idx, sorted_indices=do_sort)
        up = self.w3(x, idx, sorted_indices=do_sort)
        out = self.w2(nn.silu(gate) * up, idx, sorted_indices=do_sort)
        if do_sort:
            # Restore route order before weighting and expert-parallel reduction.
            out = _scatter_unsort(out, inv_order, indices.shape)
        return out.squeeze(-2)

    def masked(self, x, indices):
        """Evaluate local routes; negative indices denote non-local experts."""
        if (
            not self.training
            and x.ndim == 3
            and x.shape[1] == 1
            and x.dtype in (mx.float32, mx.float16, mx.bfloat16)
            and mx.default_device() == mx.gpu
            and mx.metal.is_available()
            and all(supports_masked_qmv(w) for w in (self.w1, self.w3, self.w2))
        ):
            gate = masked_qmv(x, self.w1, indices)
            up = masked_qmv(x, self.w3, indices)
            return masked_qmv(nn.silu(gate) * up, self.w2, indices)
        out = self(x, mx.maximum(indices, 0))
        return mx.where(indices[..., None] >= 0, out, 0)


class MLP(nn.Module):
    def __init__(self, dim, hidden):
        super().__init__()
        self.group = None
        self.w1 = nn.Linear(dim, hidden, bias=False)
        self.w3 = nn.Linear(dim, hidden, bias=False)
        self.w2 = nn.Linear(hidden, dim, bias=False)

    def __call__(self, x):
        return decode_linear(
            self.w2,
            nn.silu(decode_linear(self.w1, x, self.group))
            * decode_linear(self.w3, x, self.group),
            self.group,
        )


class PackedMLP(nn.Module):
    def __init__(self, dim, hidden):
        super().__init__()
        self.group = None
        self.gate_up_proj = nn.Linear(dim, 2 * hidden, bias=False)
        self.w2 = nn.Linear(hidden, dim, bias=False)

    def __call__(self, x):
        projection = self.gate_up_proj
        if x.shape[-2] != 1 and isinstance(projection, nn.Linear):
            # Keep prefill accumulation unchanged without duplicate weights.
            w1, w3 = mx.split(projection.weight, 2, axis=0)
            gate, up = x @ w1.T, x @ w3.T
        else:
            gate, up = mx.split(decode_linear(projection, x, self.group), 2, axis=-1)
        return decode_linear(self.w2, nn.silu(gate) * up, self.group)


class MoE(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.gate = nn.Linear(args.hidden_size, args.n_routed_experts, bias=False)
        self.experts = Experts(args)
        self.shared_experts = (
            MLP(args.hidden_size, args.moe_intermediate_size * args.n_shared_experts)
            if args.n_shared_experts
            else None
        )
        self.group = None
        self.start, self.end = args.expert_start, args.expert_end

    def route(self, x):
        a = self.args
        scores = mx.softmax(
            (
                x.astype(mx.float32) @ self.gate.weight.T.astype(mx.float32)
                if self.training
                else router_logits(x, self.gate.weight)
            ),
            axis=-1,
            precise=True,
        )
        choice = scores
        if a.n_group > 1:
            grouped = scores.reshape(*scores.shape[:-1], a.n_group, -1)
            group_scores = mx.sort(grouped, axis=-1)[..., -2:].sum(axis=-1)
            groups = mx.argpartition(-group_scores, kth=a.topk_group - 1, axis=-1)[
                ..., : a.topk_group
            ]
            keep = mx.any(mx.arange(a.n_group) == groups[..., None], axis=-2)
            choice = mx.where(
                mx.repeat(keep, scores.shape[-1] // a.n_group, axis=-1),
                scores,
                -float("inf"),
            )
        inds = mx.argpartition(-choice, kth=a.num_experts_per_tok - 1, axis=-1)[
            ..., : a.num_experts_per_tok
        ]
        # Ascending expert order matches the reference scatter accumulation.
        inds = mx.stop_gradient(mx.sort(inds, axis=-1))
        weights = mx.take_along_axis(scores, inds, axis=-1)
        weights = (
            weights
            / (weights.sum(axis=-1, keepdims=True) + 1e-20)
            * a.routed_scaling_factor
        )
        return inds, weights

    def __call__(self, x):
        a = self.args
        inds, weights = self.route(x)
        if self.group is None:
            if self.start or self.end != a.n_routed_experts:
                raise ValueError("An expert partition requires a distributed group")
            outputs = self.experts(x, inds)
        else:
            owned = (inds >= self.start) & (inds < self.end)
            local = mx.where(owned, inds.astype(mx.int32) - self.start, -1)
            outputs = self.experts.masked(x, local)
            # Preserve each route before reduction, as in MiniMax's EP path.
            outputs = mx.distributed.all_sum(outputs, group=self.group)
        weighted = (outputs.astype(mx.float32) * weights[..., None]).astype(x.dtype)
        y = weighted[..., 0, :]
        for i in range(1, a.num_experts_per_tok):
            y = y + weighted[..., i, :]
        if self.shared_experts is not None:
            y = y + self.shared_experts(x)
        return y


class DecoderLayer(nn.Module):
    def __init__(self, args, idx):
        super().__init__()
        self.attention = Attention(args, idx)
        self.attention_norm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)
        self.ffn_norm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)
        self.feed_forward = (
            MLP(args.hidden_size, args.intermediate_size)
            if idx < args.first_k_dense_replace
            else MoE(args)
        )

    def __call__(self, x, positions, cos, sin, cache, shared, topk, mask):
        h, topk = self.attention(
            self.attention_norm(x), positions, cos, sin, cache, shared, topk, mask
        )
        x = x + h
        return x + self.feed_forward(self.ffn_norm(x)), topk


class TextModel(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.embed_tokens = nn.Embedding(args.vocab_size, args.hidden_size)
        self.layers = [DecoderLayer(args, i) for i in range(args.num_hidden_layers)]
        self.norm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)
        self.rotary_emb = LeRoPE(args)

    def __call__(
        self,
        input_ids,
        inputs_embeds=None,
        cache=None,
        mask=None,
        capture_layer_ids=None,
        capture_sink=None,
    ):
        x = self.embed_tokens(input_ids) if inputs_embeds is None else inputs_embeds
        offset = cache[0].offset if cache else 0
        positions = mx.broadcast_to(
            mx.maximum(mx.array(offset).reshape(-1, 1) + mx.arange(x.shape[1]), 0),
            x.shape[:2],
        )
        cos, sin = self.rotary_emb(positions, x.dtype)
        shared, topk = {}, None
        cache = cache or [None] * len(self.layers)
        capture = set(capture_layer_ids or [])
        if len(capture) != len(capture_layer_ids or []) or any(
            i < 0 or i >= len(self.layers) for i in capture
        ):
            raise ValueError(
                "capture_layer_ids must be unique indices inside the model"
            )
        captured = {}
        for i, (layer, c) in enumerate(zip(self.layers, cache)):
            x, topk = layer(x, positions, cos, sin, c, shared, topk, mask)
            if i in capture:
                captured[i] = x
        if capture_sink is not None:
            capture_sink.extend(captured[i] for i in capture_layer_ids or [])
        return self.norm(x)


class LanguageModel(nn.Module):
    def __init__(self, args: TextConfig):
        super().__init__()
        self.group = None
        self.args = args
        self.model_type = args.model_type
        self.model = TextModel(args)
        self.lm_head = nn.Linear(args.hidden_size, args.vocab_size, bias=False)
        # Derived scalar buffers only; excluded from serialized model parameters.
        self._verification_scales = {}

    def __call__(
        self,
        inputs=None,
        inputs_embeds=None,
        mask=None,
        cache=None,
        capture_layer_ids=None,
        **kwargs,
    ):
        if kwargs.get("speculative_verify", False):
            from .speculative_verifier import verify_logits

            captured, x, transaction = self.speculative_verify_dflash_hidden(
                inputs, cache, capture_layer_ids or []
            )
            try:
                logits = verify_logits(self, x)
                return LanguageModelOutput(
                    logits=logits, hidden_states=captured, gdn_states=transaction
                )
            except BaseException:
                transaction.abort()
                raise
        captured = [] if capture_layer_ids is not None else None
        x = self.model(inputs, inputs_embeds, cache, mask, capture_layer_ids, captured)
        logits = decode_linear(self.lm_head, x, self.group)
        cap = self.args.final_logit_softcapping
        if cap:
            logits = mx.tanh(logits / cap) * cap
        return LanguageModelOutput(logits=logits, hidden_states=captured)

    def speculative_verify_dflash_hidden(self, inputs, cache, capture_layer_ids):
        from .speculative_verifier import verify_hidden

        return verify_hidden(self, inputs, cache, capture_layer_ids)

    def speculative_argmax_from_hidden(self, hidden):
        from .speculative_verifier import verify_logits

        logits = verify_logits(self, hidden)
        # Match generate_step's BF16 log-probability rounding and tie-breaking.
        logprobs = logits - mx.logsumexp(logits, axis=-1, keepdims=True)
        return mx.argmax(logprobs, axis=-1)

    def rollback_speculative_cache(self, cache, transaction, accepted, block_size):
        from ...speculative.cache_state import rollback_speculative_cache

        return rollback_speculative_cache(cache, transaction, accepted, block_size)

    @property
    def layers(self):
        return self.model.layers

    def make_cache(self):
        return [LatentCache() for _ in self.layers]

    def chunked_prefill_policy(self, **kwargs):
        # Captured post-layer features can be concatenated across prompt chunks.
        # Keep the same prefill/absorbed-decode boundary with and without DSpark.
        return True

    def shard(self, group=None):
        group = group or mx.distributed.init()
        n = self.args.n_routed_experts
        start, end = (
            n * group.rank() // group.size(),
            n * (group.rank() + 1) // group.size(),
        )
        for layer in self.layers:
            moe = layer.feed_forward
            if not isinstance(moe, MoE):
                continue
            if (moe.start, moe.end) != (start, end):
                if (moe.start, moe.end) != (0, n):
                    raise ValueError("Checkpoint partition does not match rank")
                for linear in (moe.experts.w1, moe.experts.w2, moe.experts.w3):
                    for key, value in list(linear.items()):
                        if isinstance(value, mx.array) and value.shape[0] == n:
                            linear[key] = value[start:end]
            moe.start, moe.end, moe.group = start, end, group
        # Dense decode shares output rows; prefill retains native GEMMs.
        self.group = group
        for layer in self.layers:
            layer.attention.group = group
            mlp = layer.feed_forward
            if isinstance(mlp, MoE):
                mlp = mlp.shared_experts
            if mlp is not None:
                mlp.group = group
