from __future__ import annotations

import math
from functools import lru_cache
from typing import NamedTuple, Optional

import mlx.core as mx
import numpy as np
from mlx_lm.models.cache import _BaseCache, create_attention_mask

DEFAULT_TURBOQUANT_SEED = 0
_EPS = 1e-6


class TurboQuantMSEState(NamedTuple):
    norms: mx.array
    indices: mx.array


class TurboQuantProdState(NamedTuple):
    norms: mx.array
    mse_indices: mx.array
    residual_norms: mx.array
    qjl_signs: mx.array


class TurboQuantSplitState(NamedTuple):
    low: object
    high: object


def _validate_bits(bits: float) -> float:
    bits = float(bits)
    if bits < 1:
        raise ValueError("TurboQuant requires kv_bits >= 1.")
    rounded = round(bits * 2) / 2
    if not math.isclose(bits, rounded, abs_tol=1e-6):
        raise ValueError(
            f"TurboQuant currently supports integer and .5 bit-widths, got {bits}."
        )
    return rounded


def turboquant_enabled(bits: Optional[float], scheme: Optional[str] = None) -> bool:
    if bits is None:
        return False
    if scheme == "turboquant":
        return True
    bits = float(bits)
    return not math.isclose(bits, round(bits), abs_tol=1e-6)


@lru_cache(maxsize=None)
def _rotation_matrix(dim: int, seed: int) -> mx.array:
    if dim <= 0:
        return mx.zeros((0, 0), dtype=mx.float32)
    if dim == 1:
        return mx.ones((1, 1), dtype=mx.float32)

    rng = np.random.default_rng(seed + dim * 7919)
    matrix = rng.standard_normal((dim, dim), dtype=np.float32)
    q, r = np.linalg.qr(matrix)
    q *= np.sign(np.diag(r))
    return mx.array(q.astype(np.float32))


@lru_cache(maxsize=None)
def _projection_matrix(dim: int, seed: int) -> mx.array:
    if dim <= 0:
        return mx.zeros((0, 0), dtype=mx.float32)
    rng = np.random.default_rng(seed + dim * 2971 + 17)
    matrix = rng.standard_normal((dim, dim), dtype=np.float32)
    return mx.array(matrix.astype(np.float32))


def _beta_pdf(grid: np.ndarray, dim: int) -> np.ndarray:
    if dim <= 1:
        pdf = np.ones_like(grid)
    else:
        coeff = math.gamma(dim / 2) / (
            math.sqrt(math.pi) * math.gamma((dim - 1) / 2)
        )
        pdf = coeff * np.power(np.clip(1.0 - grid**2, 0.0, None), (dim - 3) / 2)
    pdf_sum = pdf.sum()
    if pdf_sum == 0:
        return np.full_like(grid, 1.0 / len(grid))
    return pdf / pdf_sum


@lru_cache(maxsize=None)
def _codebook(dim: int, bits: int) -> mx.array:
    if bits <= 0:
        return mx.zeros((0,), dtype=mx.float32)
    levels = 1 << bits
    if dim <= 1:
        centroids = np.linspace(-1.0, 1.0, levels, dtype=np.float32)
        return mx.array(centroids)

    grid = np.linspace(-1.0 + 1e-6, 1.0 - 1e-6, 32768, dtype=np.float32)
    weights = _beta_pdf(grid, dim)
    cdf = np.cumsum(weights)
    quantiles = (np.arange(levels, dtype=np.float32) + 0.5) / levels
    centroids = np.interp(quantiles, cdf, grid).astype(np.float32)

    for _ in range(100):
        boundaries = np.empty(levels + 1, dtype=np.float32)
        boundaries[0] = -1.0
        boundaries[-1] = 1.0
        boundaries[1:-1] = 0.5 * (centroids[:-1] + centroids[1:])
        new_centroids = centroids.copy()
        for i in range(levels):
            if i == levels - 1:
                mask = (grid >= boundaries[i]) & (grid <= boundaries[i + 1])
            else:
                mask = (grid >= boundaries[i]) & (grid < boundaries[i + 1])
            bucket_weights = weights[mask]
            if bucket_weights.size == 0:
                continue
            total_weight = bucket_weights.sum()
            if total_weight > 0:
                new_centroids[i] = np.sum(bucket_weights * grid[mask]) / total_weight
        if np.max(np.abs(new_centroids - centroids)) < 1e-6:
            centroids = new_centroids
            break
        centroids = new_centroids

    return mx.array(centroids.astype(np.float32))


def _packed_width(length: int, bits: int) -> int:
    if length == 0 or bits == 0:
        return 0
    return (length * bits + 31) // 32


def _pack_lowbit(values: mx.array, bits: int) -> mx.array:
    if bits == 0:
        return mx.zeros((*values.shape[:-1], 0), dtype=mx.uint32)

    values = values.astype(mx.uint32)
    length = values.shape[-1]
    packed_width = _packed_width(length, bits)
    flat = values.reshape((-1, length))
    packed = mx.zeros((flat.shape[0], packed_width), dtype=mx.uint32)

    for idx in range(length):
        bit_offset = idx * bits
        word_idx = bit_offset // 32
        offset = bit_offset % 32
        packed[:, word_idx] |= flat[:, idx] << offset
        spill = offset + bits - 32
        if spill > 0:
            packed[:, word_idx + 1] |= flat[:, idx] >> (bits - spill)

    return packed.reshape((*values.shape[:-1], packed_width))


def _unpack_lowbit(packed: mx.array, bits: int, length: int) -> mx.array:
    if bits == 0:
        return mx.zeros((*packed.shape[:-1], 0), dtype=mx.uint32)

    packed = packed.astype(mx.uint32)
    flat = packed.reshape((-1, packed.shape[-1]))
    unpacked = mx.zeros((flat.shape[0], length), dtype=mx.uint32)
    mask = (1 << bits) - 1

    for idx in range(length):
        bit_offset = idx * bits
        word_idx = bit_offset // 32
        offset = bit_offset % 32
        value = flat[:, word_idx] >> offset
        spill = offset + bits - 32
        if spill > 0:
            value |= flat[:, word_idx + 1] << (bits - spill)
        unpacked[:, idx] = value & mask

    return unpacked.reshape((*packed.shape[:-1], length))


def _concat_state(lhs, rhs):
    if lhs is None:
        return rhs
    if rhs is None:
        return lhs
    if isinstance(lhs, TurboQuantMSEState):
        return TurboQuantMSEState(
            mx.concatenate([lhs.norms, rhs.norms], axis=2),
            mx.concatenate([lhs.indices, rhs.indices], axis=2),
        )
    if isinstance(lhs, TurboQuantProdState):
        return TurboQuantProdState(
            mx.concatenate([lhs.norms, rhs.norms], axis=2),
            mx.concatenate([lhs.mse_indices, rhs.mse_indices], axis=2),
            mx.concatenate([lhs.residual_norms, rhs.residual_norms], axis=2),
            mx.concatenate([lhs.qjl_signs, rhs.qjl_signs], axis=2),
        )
    if isinstance(lhs, TurboQuantSplitState):
        return TurboQuantSplitState(
            _concat_state(lhs.low, rhs.low),
            _concat_state(lhs.high, rhs.high),
        )
    raise TypeError(f"Unsupported TurboQuant state type: {type(lhs)!r}")


def _slice_state(state, end: int):
    if state is None:
        return None
    if isinstance(state, TurboQuantMSEState):
        return TurboQuantMSEState(state.norms[..., :end], state.indices[..., :end, :])
    if isinstance(state, TurboQuantProdState):
        return TurboQuantProdState(
            state.norms[..., :end],
            state.mse_indices[..., :end, :],
            state.residual_norms[..., :end],
            state.qjl_signs[..., :end, :],
        )
    if isinstance(state, TurboQuantSplitState):
        return TurboQuantSplitState(
            _slice_state(state.low, end),
            _slice_state(state.high, end),
        )
    raise TypeError(f"Unsupported TurboQuant state type: {type(state)!r}")


def _state_nbytes(state) -> int:
    if state is None:
        return 0
    if isinstance(state, TurboQuantSplitState):
        return _state_nbytes(state.low) + _state_nbytes(state.high)
    if isinstance(state, tuple):
        return sum(_state_nbytes(part) for part in state)
    if isinstance(state, mx.array):
        return state.nbytes
    return 0


def _state_length(state) -> int:
    if state is None:
        return 0
    if isinstance(state, TurboQuantSplitState):
        return _state_length(state.low)
    if isinstance(state, TurboQuantMSEState):
        return state.norms.shape[2]
    if isinstance(state, TurboQuantProdState):
        return state.norms.shape[2]
    raise TypeError(f"Unsupported TurboQuant state type: {type(state)!r}")


class _TurboQuantMSECodec:
    def __init__(self, dim: int, bits: int, seed: int):
        self.dim = dim
        self.bits = bits
        self.rotation = _rotation_matrix(dim, seed)
        self.rotation_t = self.rotation.transpose() if dim > 0 else self.rotation
        self.codebook = _codebook(dim, bits)

    def _quantize_unit(self, unit_vectors: mx.array) -> mx.array:
        if self.bits == 0:
            return mx.zeros((*unit_vectors.shape[:-1], 0), dtype=mx.uint32)

        rotated = mx.matmul(unit_vectors.astype(mx.float32), self.rotation_t)
        distances = mx.abs(rotated[..., None] - self.codebook)
        indices = mx.argmin(distances, axis=-1).astype(mx.uint32)
        return _pack_lowbit(indices, self.bits)

    def _dequantize_unit(self, packed_indices: mx.array) -> mx.array:
        if self.bits == 0:
            return mx.zeros((*packed_indices.shape[:-1], self.dim), dtype=mx.float32)

        indices = _unpack_lowbit(packed_indices, self.bits, self.dim).astype(mx.int32)
        rotated = mx.take(self.codebook, indices, axis=0)
        return mx.matmul(rotated, self.rotation)

    def quantize(self, vectors: mx.array) -> TurboQuantMSEState:
        norms = mx.linalg.norm(vectors.astype(mx.float32), axis=-1)
        safe_norms = mx.maximum(norms[..., None], _EPS)
        unit_vectors = mx.where(
            norms[..., None] > 0,
            vectors.astype(mx.float32) / safe_norms,
            mx.zeros(vectors.shape, dtype=mx.float32),
        )
        return TurboQuantMSEState(
            norms.astype(vectors.dtype),
            self._quantize_unit(unit_vectors),
        )

    def dequantize(self, state: TurboQuantMSEState) -> mx.array:
        unit_vectors = self._dequantize_unit(state.indices)
        return state.norms[..., None].astype(unit_vectors.dtype) * unit_vectors


class _TurboQuantProdCodec:
    def __init__(self, dim: int, bits: int, seed: int):
        self.dim = dim
        self.bits = bits
        self.mse_codec = _TurboQuantMSECodec(dim, max(bits - 1, 0), seed)
        self.projection = _projection_matrix(dim, seed + 1)
        self.projection_t = (
            self.projection.transpose() if dim > 0 else self.projection
        )
        self.scale = math.sqrt(math.pi / 2) / dim if dim > 0 else 0.0

    def quantize(self, vectors: mx.array) -> TurboQuantProdState:
        norms = mx.linalg.norm(vectors.astype(mx.float32), axis=-1)
        safe_norms = mx.maximum(norms[..., None], _EPS)
        unit_vectors = mx.where(
            norms[..., None] > 0,
            vectors.astype(mx.float32) / safe_norms,
            mx.zeros(vectors.shape, dtype=mx.float32),
        )

        mse_indices = self.mse_codec._quantize_unit(unit_vectors)
        mse_unit = self.mse_codec._dequantize_unit(mse_indices)
        residual = unit_vectors - mse_unit
        residual_norms = mx.linalg.norm(residual, axis=-1)
        projected = mx.matmul(residual, self.projection_t)
        signs = mx.where(projected >= 0, 1, 0).astype(mx.uint32)

        return TurboQuantProdState(
            norms.astype(vectors.dtype),
            mse_indices,
            residual_norms.astype(vectors.dtype),
            _pack_lowbit(signs, 1),
        )

    def dequantize(self, state: TurboQuantProdState) -> mx.array:
        mse_unit = self.mse_codec._dequantize_unit(state.mse_indices)
        sign_bits = _unpack_lowbit(state.qjl_signs, 1, self.dim).astype(mx.float32)
        signs = sign_bits * 2.0 - 1.0
        qjl_unit = self.scale * state.residual_norms[..., None].astype(
            mx.float32
        ) * mx.matmul(signs, self.projection)
        return state.norms[..., None].astype(mx.float32) * (mse_unit + qjl_unit)


def _select_outlier_indices(tensor: mx.array, avg_bits: float) -> tuple[np.ndarray, np.ndarray]:
    lower_bits = math.floor(avg_bits)
    upper_bits = math.ceil(avg_bits)
    if lower_bits == upper_bits:
        raise ValueError("Mixed-precision selection requires a fractional bit-width.")

    dim = tensor.shape[-1]
    high_count = int(round((avg_bits - lower_bits) * dim / (upper_bits - lower_bits)))
    high_count = max(1, min(dim - 1, high_count))

    scores = mx.mean(mx.abs(tensor.astype(mx.float32)), axis=(0, 1, 2))
    order = np.argsort(np.asarray(scores))
    high_idx = np.sort(order[-high_count:].astype(np.int32))
    low_mask = np.ones(dim, dtype=bool)
    low_mask[high_idx] = False
    low_idx = np.nonzero(low_mask)[0].astype(np.int32)
    return low_idx, high_idx


class _SplitCodec:
    def __init__(self, tensor: mx.array, bits: float, mode: str, seed: int):
        self.bits = bits
        self.mode = mode
        self.dim = tensor.shape[-1]
        self.lower_bits = math.floor(bits)
        self.upper_bits = math.ceil(bits)
        low_idx, high_idx = _select_outlier_indices(tensor, bits)
        self.low_idx = mx.array(low_idx, dtype=mx.int32)
        self.high_idx = mx.array(high_idx, dtype=mx.int32)

        concat_order = np.concatenate([low_idx, high_idx])
        self.restore_order = mx.array(np.argsort(concat_order), dtype=mx.int32)

        codec_cls = _TurboQuantProdCodec if mode == "prod" else _TurboQuantMSECodec
        self.low_codec = codec_cls(len(low_idx), self.lower_bits, seed)
        self.high_codec = codec_cls(len(high_idx), self.upper_bits, seed + 97)

    def quantize(self, tensor: mx.array) -> TurboQuantSplitState:
        low_tensor = mx.take(tensor, self.low_idx, axis=-1)
        high_tensor = mx.take(tensor, self.high_idx, axis=-1)
        return TurboQuantSplitState(
            self.low_codec.quantize(low_tensor),
            self.high_codec.quantize(high_tensor),
        )

    def dequantize(self, state: TurboQuantSplitState) -> mx.array:
        low_tensor = self.low_codec.dequantize(state.low)
        high_tensor = self.high_codec.dequantize(state.high)
        merged = mx.concatenate([low_tensor, high_tensor], axis=-1)
        return mx.take(merged, self.restore_order, axis=-1)


def _build_codec(tensor: mx.array, bits: float, mode: str, seed: int):
    bits = _validate_bits(bits)
    if math.isclose(bits, round(bits), abs_tol=1e-6):
        codec_cls = _TurboQuantProdCodec if mode == "prod" else _TurboQuantMSECodec
        return codec_cls(tensor.shape[-1], int(round(bits)), seed)
    return _SplitCodec(tensor, bits, mode, seed)


class TurboQuantKVCache(_BaseCache):
    def __init__(self, bits: float, seed: int = DEFAULT_TURBOQUANT_SEED):
        self.bits = _validate_bits(bits)
        self.seed = seed
        self.offset = 0
        self.keys = None
        self.values = None
        self.key_codec = None
        self.value_codec = None

    @classmethod
    def from_cache(
        cls, cache, bits: float, seed: int = DEFAULT_TURBOQUANT_SEED
    ) -> "TurboQuantKVCache":
        turbo_cache = cls(bits=bits, seed=seed)
        keys, values = cache.state
        if keys is not None:
            turbo_cache.update_and_fetch(keys, values)
        return turbo_cache

    def _ensure_codecs(self, keys: mx.array, values: mx.array):
        if self.key_codec is None:
            self.key_codec = _build_codec(keys, self.bits, mode="prod", seed=self.seed)
        if self.value_codec is None:
            self.value_codec = _build_codec(
                values, self.bits, mode="mse", seed=self.seed + 1
            )

    def update_and_fetch(self, keys: mx.array, values: mx.array):
        self._ensure_codecs(keys, values)
        new_keys = self.key_codec.quantize(keys)
        new_values = self.value_codec.quantize(values)

        if self.keys is not None and self.offset < _state_length(self.keys):
            self.keys = _slice_state(self.keys, self.offset)
            self.values = _slice_state(self.values, self.offset)

        self.keys = _concat_state(self.keys, new_keys)
        self.values = _concat_state(self.values, new_values)
        self.offset += keys.shape[2]
        return self.state

    def dequantize(self, keys_state=None, values_state=None):
        keys_state = self.keys if keys_state is None else keys_state
        values_state = self.values if values_state is None else values_state
        keys = self.key_codec.dequantize(keys_state).astype(mx.float32)
        values = self.value_codec.dequantize(values_state).astype(mx.float32)
        return keys, values

    def size(self):
        return self.offset

    @property
    def state(self):
        if self.keys is None:
            return None, None
        return _slice_state(self.keys, self.offset), _slice_state(self.values, self.offset)

    @state.setter
    def state(self, value):
        if value is None:
            self.keys, self.values = None, None
            self.offset = 0
            return
        self.keys, self.values = value
        self.offset = _state_length(self.keys)

    @property
    def meta_state(self):
        return tuple(map(str, (self.offset, self.bits, self.seed)))

    @meta_state.setter
    def meta_state(self, value):
        self.offset = int(value[0])
        self.bits = float(value[1])
        self.seed = int(value[2])

    def is_trimmable(self):
        return True

    def trim(self, n):
        n = min(self.offset, n)
        self.offset -= n
        return n

    def make_mask(self, *args, **kwargs):
        return create_attention_mask(*args, offset=self.offset, **kwargs)

    def empty(self):
        return self.keys is None

    @property
    def nbytes(self):
        return _state_nbytes((self.keys, self.values))
