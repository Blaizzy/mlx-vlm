"""Automatic Prefix Caching (APC) for mlx-vlm.

Hash-based, block-level KV cache reuse across requests. The KV cache is split
into fixed-size blocks (default 16 tokens). Each fully-filled block is
identified by a chained hash::

    block_hash[i] = H(block_hash[i-1], tuple(tokens[i*bs:(i+1)*bs]), extra_hash[i])

``extra_hash[i]`` carries multimodal context (e.g. an image content hash) so
identical token IDs with different images don't collide. ``H`` defaults to
Python's built-in ``hash`` (fast, deterministic within a single process). Set
``APC_HASH=sha256`` to opt into a stable cryptographic hash (~100-200 ns/tok
overhead).

Eviction is LRU with reference counting: blocks are kept alive while
``ref_cnt > 0`` and the free queue is a doubly-linked list embedded in
``APCBlock`` for O(1) move-to-tail. All blocks are pre-allocated as a pool
to avoid Python object churn. When ``APC_DISK_PATH`` is configured, full
blocks are also written to a shard-based SSD tier and can be restored after
process restart through a direct-read prompt-cache path.

Numerical note: APC itself is *exact*. The K/V tensors stored in the block
pool are byte-identical to what a fresh prefill would produce — the cache
introduces no approximation, it just retains tensors. However, cold-vs-warm
runs of the same prompt can produce slightly different logits because of
**batch non-invariance** in the attention kernel: a long Q (cold prefill,
e.g. 60 tokens) and a short Q (warm-start suffix, e.g. 13 tokens against
47 cached tokens) trigger different tile shapes / reduction orders inside
flash-attention, and floating-point matmul is non-associative. The
Thinking Machines analysis (2025) and Microsoft Research's LLM-42 paper
give the formal treatment. The same drift happens without prefix caching
any time dynamic batching changes the batch composition between two
identical requests — APC just makes it visible by giving a clean
cold/warm contrast. Warm-to-warm runs *are* deterministic: identical
prompts repeated under APC always produce identical text. For
bit-equivalent cold==warm, you need batch-invariant RMSNorm / matmul /
attention kernels (vLLM's ``--enable-batch-invariance``, SGLang with
FlashInfer/FA3), not a different cache design.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import queue
import re
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, List, Optional, Sequence, Tuple

import mlx.core as mx
import numpy as np

logger = logging.getLogger("mlx_vlm.apc")

DEFAULT_BLOCK_SIZE = 16
DEFAULT_NUM_BLOCKS = 2048
SEED_PARENT_HASH = 0


def _hash_use_sha256() -> bool:
    return os.environ.get("APC_HASH", "fast").lower() == "sha256"


def _hash_tokens(parent: int, tokens: Tuple[int, ...], extra: int) -> int:
    """Chain hash for a single block."""
    if _hash_use_sha256():
        h = hashlib.sha256()
        h.update(int(parent & ((1 << 64) - 1)).to_bytes(8, "little"))
        h.update(np.asarray(tokens, dtype=np.int32).tobytes())
        h.update(int(extra & ((1 << 64) - 1)).to_bytes(8, "little"))
        return int.from_bytes(h.digest()[:8], "little", signed=True)
    return hash((parent, tokens, extra))


def _stable_int_hash(*values: int) -> int:
    h = hashlib.sha256()
    for value in values:
        h.update(int(value & ((1 << 64) - 1)).to_bytes(8, "little"))
    return int.from_bytes(h.digest()[:8], "little", signed=True)


def tenant_scoped_hash(tenant: Optional[str], payload_hash: int = 0) -> int:
    """Stable APC salt for tenant-scoped multimodal context."""
    if not tenant:
        return int(payload_hash)
    tenant_bytes = str(tenant).encode("utf-8")
    h = hashlib.sha256()
    h.update(len(tenant_bytes).to_bytes(4, "little"))
    h.update(tenant_bytes)
    h.update(int(payload_hash & ((1 << 64) - 1)).to_bytes(8, "little"))
    return int.from_bytes(h.digest()[:8], "little", signed=True)


def _copy_mlx_array(x: mx.array) -> mx.array:
    """Materialize ``x`` into a fresh MLX-owned contiguous buffer."""
    return mx.contiguous(mx.array(x, dtype=x.dtype))


def _pad_kv_for_capacity(
    keys: mx.array,
    values: mx.array,
    *,
    offset: int,
    min_capacity_tokens: Optional[int],
    step: int,
) -> Tuple[mx.array, mx.array]:
    if min_capacity_tokens is None or min_capacity_tokens <= offset:
        return keys, values
    capacity = int(min_capacity_tokens)
    if step > 0:
        capacity = ((capacity + step - 1) // step) * step
    if capacity <= keys.shape[2]:
        return keys, values
    pad_tokens = capacity - keys.shape[2]
    k_shape = (*keys.shape[:2], pad_tokens, keys.shape[3])
    v_shape = (*values.shape[:2], pad_tokens, values.shape[3])
    keys = mx.concatenate([keys, mx.zeros(k_shape, dtype=keys.dtype)], axis=2)
    values = mx.concatenate([values, mx.zeros(v_shape, dtype=values.dtype)], axis=2)
    return keys, values


def _clone_cache_entry_for_apc(
    c: Any,
    *,
    min_capacity_tokens: Optional[int],
    eval_targets: List[mx.array],
) -> Optional[Any]:
    """Deep-copy one prompt-cache entry, preserving its concrete cache kind."""
    from mlx_lm.models import cache as lm_cache

    if isinstance(c, lm_cache.KVCache):
        out = type(c)()
        off = int(getattr(c, "offset", 0) or 0)
        if c.keys is not None and c.values is not None and off > 0:
            keys = _copy_mlx_array(c.keys[..., :off, :])
            values = _copy_mlx_array(c.values[..., :off, :])
            step = int(getattr(c, "step", getattr(type(c), "step", 256)) or 0)
            keys, values = _pad_kv_for_capacity(
                keys,
                values,
                offset=off,
                min_capacity_tokens=min_capacity_tokens,
                step=step,
            )
            out.keys = keys
            out.values = values
            out.offset = off
            eval_targets.extend([keys, values])
        return out

    if isinstance(c, lm_cache.RotatingKVCache):
        out = type(c)(
            max_size=int(getattr(c, "max_size")),
            keep=int(getattr(c, "keep", 0)),
        )
        out.offset = int(getattr(c, "offset", 0) or 0)
        out._idx = int(getattr(c, "_idx", 0) or 0)
        if c.keys is not None and c.values is not None:
            out.keys = _copy_mlx_array(c.keys)
            out.values = _copy_mlx_array(c.values)
            eval_targets.extend([out.keys, out.values])
        return out

    if isinstance(c, lm_cache.ChunkedKVCache):
        out = type(c)(chunk_size=int(getattr(c, "chunk_size")))
        out.offset = int(getattr(c, "offset", 0) or 0)
        out.start_position = int(getattr(c, "start_position", 0) or 0)
        if c.keys is not None and c.values is not None:
            out.keys = _copy_mlx_array(c.keys)
            out.values = _copy_mlx_array(c.values)
            eval_targets.extend([out.keys, out.values])
        return out

    if isinstance(c, lm_cache.ArraysCache):
        out = lm_cache.ArraysCache(len(c.cache))
        out.cache = []
        for state in c.cache:
            if state is None:
                out.cache.append(None)
                continue
            copied = _copy_mlx_array(state)
            out.cache.append(copied)
            eval_targets.append(copied)
        if c.left_padding is not None:
            out.left_padding = _copy_mlx_array(c.left_padding)
            eval_targets.append(out.left_padding)
        if c.lengths is not None:
            out.lengths = _copy_mlx_array(c.lengths)
            eval_targets.append(out.lengths)
        return out

    if isinstance(c, lm_cache.CacheList):
        copied = [
            _clone_cache_entry_for_apc(
                sub_c,
                min_capacity_tokens=min_capacity_tokens,
                eval_targets=eval_targets,
            )
            for sub_c in c.caches
        ]
        if any(sub_c is None for sub_c in copied):
            return None
        return lm_cache.CacheList(*copied)

    if isinstance(c, tuple):
        copied = [
            _clone_cache_entry_for_apc(
                sub_c,
                min_capacity_tokens=min_capacity_tokens,
                eval_targets=eval_targets,
            )
            for sub_c in c
        ]
        if any(sub_c is None for sub_c in copied):
            return None
        return tuple(copied)

    return None


def _clone_prompt_cache_for_apc(
    prompt_cache: Sequence[Any],
    *,
    min_capacity_tokens: Optional[int] = None,
) -> Optional[List[Any]]:
    eval_targets: List[mx.array] = []
    out: List[Any] = []
    for c in prompt_cache:
        copied = _clone_cache_entry_for_apc(
            c,
            min_capacity_tokens=min_capacity_tokens,
            eval_targets=eval_targets,
        )
        if copied is None:
            return None
        out.append(copied)
    if eval_targets:
        mx.eval(eval_targets)
    return out


def _clone_layer_major_kv_cache_for_apc(
    layer_keys: Sequence[mx.array],
    layer_values: Sequence[mx.array],
    prefix_len: int,
) -> Optional[List[Any]]:
    """Deep-copy layer-major K/V tensors into compact ``KVCache`` entries."""
    from mlx_lm.models.cache import KVCache

    if prefix_len <= 0 or len(layer_keys) != len(layer_values):
        return None
    eval_targets: List[mx.array] = []
    out: List[Any] = []
    for k, v in zip(layer_keys, layer_values):
        c = KVCache()
        c.keys = _copy_mlx_array(k[..., :prefix_len, :])
        c.values = _copy_mlx_array(v[..., :prefix_len, :])
        c.offset = prefix_len
        eval_targets.extend([c.keys, c.values])
        out.append(c)
    if eval_targets:
        mx.eval(eval_targets)
    return out


def _cache_entry_supports_exact_apc(c: Any) -> bool:
    from mlx_lm.models import cache as lm_cache

    if isinstance(
        c,
        (
            lm_cache.KVCache,
            lm_cache.RotatingKVCache,
            lm_cache.ChunkedKVCache,
            lm_cache.ArraysCache,
        ),
    ):
        return True
    if isinstance(c, lm_cache.CacheList):
        return all(_cache_entry_supports_exact_apc(sub_c) for sub_c in c.caches)
    if isinstance(c, tuple):
        return all(_cache_entry_supports_exact_apc(sub_c) for sub_c in c)
    return False


def _cache_entry_supports_block_apc(c: Any) -> bool:
    from mlx_lm.models import cache as lm_cache

    return isinstance(c, lm_cache.KVCache)


def _sequence_hash(token_ids: Sequence[int], extra_hash: int, block_size: int) -> int:
    h = hashlib.sha256()
    h.update(int(extra_hash & ((1 << 64) - 1)).to_bytes(8, "little"))
    h.update(int(block_size).to_bytes(4, "little", signed=False))
    arr = np.asarray([int(t) for t in token_ids], dtype=np.int32)
    h.update(int(arr.size).to_bytes(8, "little", signed=False))
    h.update(arr.tobytes())
    return int.from_bytes(h.digest()[:8], "little", signed=True)


def hash_image_payload(
    pixel_values: Optional[mx.array] = None,
    image_ref: Any = None,
) -> int:
    """Stable content hash of an image payload.

    Prefers hashing the actual ``pixel_values`` tensor (so resize/transform
    differences invalidate the cache). Falls back to hashing the source
    identifier (path / URL / repr).
    """
    if pixel_values is not None:
        try:
            arr = np.asarray(pixel_values).astype(np.float16, copy=False)
            digest = hashlib.sha256(arr.tobytes()).digest()
            return int.from_bytes(digest[:8], "little", signed=True)
        except Exception:
            pass

    if image_ref is None:
        return 0
    if isinstance(image_ref, (list, tuple)):
        h = SEED_PARENT_HASH
        for it in image_ref:
            h = _stable_int_hash(h, hash_image_payload(image_ref=it))
        return h
    if isinstance(image_ref, str):
        digest = hashlib.sha256(image_ref.encode("utf-8")).digest()
        return int.from_bytes(digest[:8], "little", signed=True)
    if isinstance(image_ref, bytes):
        return int.from_bytes(
            hashlib.sha256(image_ref).digest()[:8], "little", signed=True
        )
    digest = hashlib.sha256(repr(image_ref).encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "little", signed=True)


@dataclass
class APCBlock:
    """One fixed-size KV block. Holds per-layer K/V slabs once committed."""

    block_id: int
    block_hash: Optional[int] = None
    parent_hash: int = SEED_PARENT_HASH
    token_ids: Tuple[int, ...] = ()
    extra_hash: int = 0
    ref_cnt: int = 0
    keys: Optional[List[mx.array]] = None
    values: Optional[List[mx.array]] = None
    last_used: float = 0.0
    prev: Optional["APCBlock"] = None
    next: Optional["APCBlock"] = None


@dataclass
class APCExactCacheEntry:
    """Exact-prefix prompt-cache snapshot for custom cache layouts."""

    token_ids: Tuple[int, ...]
    extra_hash: int
    prompt_cache: List[Any]
    last_used: float


@dataclass(frozen=True)
class _DiskBlockSnapshot:
    """Immutable view of an APC block for the asynchronous disk writer."""

    block_hash: int
    parent_hash: int
    extra_hash: int
    token_ids: Tuple[int, ...]
    keys: List[mx.array]
    values: List[mx.array]


@dataclass(frozen=True)
class _DiskLayerMajorBlock:
    """Per-block metadata for a direct layer-major disk write."""

    block_hash: int
    parent_hash: int
    extra_hash: int
    token_ids: Tuple[int, ...]
    source_block_idx: int


@dataclass(frozen=True)
class _DiskLayerMajorSnapshot:
    """Direct shard snapshot from the live per-layer KV cache."""

    blocks: List[_DiskLayerMajorBlock]
    layer_keys: List[mx.array]
    layer_values: List[mx.array]
    block_size: int
    store_id: str
    segment_index: int
    segment_count: int


@dataclass(frozen=True)
class _DiskExactCacheSnapshot:
    cache_hash: int
    token_ids: Tuple[int, ...]
    extra_hash: int
    prompt_cache: List[Any]


@dataclass
class APCStats:
    hits: int = 0
    misses: int = 0
    matched_tokens: int = 0
    served_tokens: int = 0
    evictions: int = 0
    stores: int = 0
    pool_used: int = 0
    disk_hits: int = 0
    disk_writes: int = 0
    exact_hits: int = 0
    exact_stores: int = 0

    def snapshot(self, num_blocks: int, block_size: int) -> dict:
        denom = self.matched_tokens + self.served_tokens
        hit_rate = self.matched_tokens / denom if denom > 0 else 0.0
        return {
            "block_size": block_size,
            "num_blocks": num_blocks,
            "pool_used": self.pool_used,
            "lookups_hit": self.hits,
            "lookups_miss": self.misses,
            "matched_tokens": self.matched_tokens,
            "served_tokens": self.served_tokens,
            "token_hit_rate": hit_rate,
            "evictions": self.evictions,
            "stores": self.stores,
            "disk_hits": self.disk_hits,
            "disk_writes": self.disk_writes,
            "exact_hits": self.exact_hits,
            "exact_stores": self.exact_stores,
        }


def _safe_namespace(name: str) -> str:
    """Sanitize a model identifier into a filesystem-friendly directory name."""
    safe = re.sub(r"[^A-Za-z0-9._-]+", "_", name).strip("_") or "default"
    return safe[:128]


def _read_safetensors_header(path: Path) -> Optional[Tuple[dict, dict, int]]:
    """Read a safetensors header without touching tensor payload bytes.

    Returns ``(tensor_entries, metadata, data_start)`` where ``data_start`` is
    the absolute file offset of the tensor data buffer.
    """
    try:
        with open(path, "rb") as f:
            head_bytes = f.read(8)
            if len(head_bytes) < 8:
                return None
            header_size = int.from_bytes(head_bytes, "little")
            # Sanity-bound the header so a corrupted file can't trigger a
            # huge allocation.
            if header_size <= 0 or header_size > 64 * 1024 * 1024:
                return None
            header_bytes = f.read(header_size)
            if len(header_bytes) < header_size:
                return None
        header = json.loads(header_bytes)
        metadata = dict(header.pop("__metadata__", {}) or {})
        return header, metadata, 8 + header_size
    except (OSError, ValueError):
        return None


def _read_safetensors_metadata(path: Path) -> Optional[dict]:
    """Read only the ``__metadata__`` dict from a safetensors file, without
    mmap'ing the tensor payload. Used to populate the disk index on init.

    Returns ``None`` on any read/parse error.
    """
    header = _read_safetensors_header(path)
    if header is None:
        return None
    return header[1]


def _numel(shape: Sequence[int]) -> int:
    out = 1
    for dim in shape:
        out *= int(dim)
    return out


def _safetensors_dtype_info(dtype: str):
    """Return ``(numpy_dtype, mlx_dtype, bitcast_to)`` for supported dtypes."""
    if dtype == "BF16":
        return np.dtype("<u2"), mx.uint16, mx.bfloat16
    mapping = {
        "F16": (np.dtype("<f2"), mx.float16, None),
        "F32": (np.dtype("<f4"), mx.float32, None),
    }
    return mapping.get(dtype)


def _safetensors_tensor_bounds(
    entry: dict,
) -> Optional[Tuple[int, int, Tuple[int, ...]]]:
    try:
        start, end = entry["data_offsets"]
        shape = tuple(int(x) for x in entry["shape"])
        dtype_info = _safetensors_dtype_info(str(entry["dtype"]))
        if dtype_info is None:
            return None
        np_dtype, _, _ = dtype_info
        if int(end) < int(start):
            return None
        if _numel(shape) * np_dtype.itemsize != int(end) - int(start):
            return None
        return int(start), int(end), shape
    except (KeyError, TypeError, ValueError):
        return None


def _mlx_array_from_safetensors_bytes(buf, entry: dict) -> Optional[mx.array]:
    bounds = _safetensors_tensor_bounds(entry)
    if bounds is None:
        return None
    _, _, shape = bounds
    dtype_info = _safetensors_dtype_info(str(entry["dtype"]))
    if dtype_info is None:
        return None
    np_dtype, mlx_dtype, bitcast_to = dtype_info
    arr = np.frombuffer(buf, dtype=np_dtype, count=_numel(shape)).reshape(shape)
    out = mx.array(arr, dtype=mlx_dtype)
    if bitcast_to is not None:
        out = out.view(bitcast_to)
    return out


def _read_safetensors_tensor(
    path: Path, data_start: int, entry: dict
) -> Optional[mx.array]:
    bounds = _safetensors_tensor_bounds(entry)
    if bounds is None:
        return None
    start, end, _ = bounds
    try:
        with open(path, "rb") as f:
            f.seek(data_start + start)
            raw = f.read(end - start)
            if len(raw) != end - start:
                return None
    except OSError:
        return None
    return _mlx_array_from_safetensors_bytes(memoryview(raw), entry)


def _read_safetensors_axis0_slice_bytes(
    path: Path,
    data_start: int,
    entry: dict,
    axis0_start: int,
    axis0_end: int,
) -> Optional[Tuple[bytes, dict]]:
    bounds = _safetensors_tensor_bounds(entry)
    if bounds is None:
        return None
    start, _, shape = bounds
    if not shape:
        return None
    axis0_start = int(axis0_start)
    axis0_end = int(axis0_end)
    if axis0_start < 0 or axis0_end < axis0_start or axis0_end > shape[0]:
        return None
    dtype_info = _safetensors_dtype_info(str(entry["dtype"]))
    if dtype_info is None:
        return None
    np_dtype, _, _ = dtype_info
    row_bytes = _numel(shape[1:]) * np_dtype.itemsize
    byte_start = start + axis0_start * row_bytes
    byte_end = start + axis0_end * row_bytes
    try:
        with open(path, "rb") as f:
            f.seek(data_start + byte_start)
            raw = f.read(byte_end - byte_start)
            if len(raw) != byte_end - byte_start:
                return None
    except OSError:
        return None

    sliced_entry = dict(entry)
    sliced_entry["shape"] = [axis0_end - axis0_start, *shape[1:]]
    sliced_entry["data_offsets"] = [0, byte_end - byte_start]
    return raw, sliced_entry


def _read_safetensors_axis0_slice(
    path: Path,
    data_start: int,
    entry: dict,
    axis0_start: int,
    axis0_end: int,
) -> Optional[mx.array]:
    sliced = _read_safetensors_axis0_slice_bytes(
        path, data_start, entry, axis0_start, axis0_end
    )
    if sliced is None:
        return None
    raw, sliced_entry = sliced
    return _mlx_array_from_safetensors_bytes(memoryview(raw), sliced_entry)


def _free_ram_bytes() -> Optional[int]:
    """Best-effort reading of currently-available system RAM. Returns
    ``None`` when we can't tell, in which case the caller should treat the
    answer as "don't know — proceed".

    Uses ``psutil`` when available; falls back to ``vm_stat`` on macOS.
    Never raises.
    """
    try:
        import psutil  # type: ignore

        return int(psutil.virtual_memory().available)
    except Exception:
        pass
    # macOS fallback: parse vm_stat. Cheap; runs in <2ms typically.
    try:
        import subprocess

        out = subprocess.check_output(["vm_stat"], timeout=1.0).decode("utf-8")
        page_size = 16384  # default on Apple Silicon; refined below
        free_pages = 0
        inactive_pages = 0
        for line in out.splitlines():
            if "page size of" in line:
                # "Mach Virtual Memory Statistics: (page size of 16384 bytes)"
                m = re.search(r"page size of (\d+) bytes", line)
                if m:
                    page_size = int(m.group(1))
            elif line.startswith("Pages free:"):
                free_pages = int(line.split(":")[1].strip().rstrip("."))
            elif line.startswith("Pages inactive:"):
                inactive_pages = int(line.split(":")[1].strip().rstrip("."))
        return (free_pages + inactive_pages) * page_size
    except Exception:
        return None


class DiskBlockStore:
    """Persistent SSD-backed cold tier for APC blocks.

    Layout: one or more segment shards per ``store_kv_blocks`` call. Each
    shard is a single ``.safetensors`` file under ``<root>/<namespace>/``
    containing a bounded run of that call's blocks. Co-locating
    chain-contiguous blocks in segment files
    makes restoration O(few sequential reads) rather than O(num_blocks
    random reads) — the difference is roughly two orders of magnitude on
    Apple NVMe for ~200-block prefixes.

    Shard naming: ``shard_<32-hex>.safetensors``. The 32-hex stem is the
    SHA-256 prefix of the contained block hashes; identical stores
    naturally dedup. Inside the file:

      * tensor ``k{layer}`` / ``v{layer}`` per layer in runtime KV layout
      * metadata: ``block_hashes`` (csv), ``num_layers``, ``block_size``,
        and ``b{idx}_meta`` (JSON) per block carrying parent_hash,
        extra_hash, and the token tuple

    The in-memory index ``hash → (shard_path, block_idx)`` is rebuilt on
    init by scanning shards (cheap, just reads safetensors headers).
    Shard mmap'd handles are kept in a small LRU cache so siblings within
    a single restore don't re-mmap the same file.

    Writes go through a single background worker so the prefill hot path
    isn't blocked. Eviction is at segment-shard granularity (drop one
    bounded file).
    """

    SUFFIX = ".safetensors"
    SHARD_PREFIX = "shard_"
    EXACT_PREFIX = "exact_"
    SHARD_STEM_LEN = len(SHARD_PREFIX) + 32  # "shard_" + 32 hex chars
    EXACT_STEM_LEN = len(EXACT_PREFIX) + 32  # "exact_" + 32 hex chars
    # Eviction targets this fraction of max_bytes after a single sweep so
    # we don't thrash on every write near the cap.
    _EVICT_LOW_WATERMARK = 0.9

    def __init__(
        self,
        root: Path,
        namespace: str = "default",
        num_workers: int = 1,
        max_bytes: Optional[int] = None,
    ):
        self.dir = Path(root) / _safe_namespace(namespace)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.max_bytes = max_bytes
        self.evictions = 0  # cumulative shard deletions by _maybe_evict
        self._q: queue.Queue = queue.Queue(maxsize=4096)
        self._stop = threading.Event()
        # Track in-flight hashes (across pending shard writes) so a lookup
        # racing a write can wait briefly for the bytes to land.
        self._in_flight: dict[int, threading.Event] = {}
        self._in_flight_lock = threading.Lock()
        # block_hash -> (shard_path, block_idx_in_shard)
        self._index: dict[int, Tuple[Path, int]] = {}
        # exact full-prefix hash -> snapshot path
        self._exact_index: dict[int, Path] = {}
        self._index_lock = threading.RLock()
        # Direct-read mode avoids mmap-backed MLX arrays entirely. It parses
        # safetensors headers, reads only the requested block's byte ranges
        # with normal file I/O, then constructs MLX-managed arrays from those
        # bytes. Keep the old mmap path available for comparison.
        self._read_mode = os.environ.get("APC_DISK_READ_MODE", "direct").lower()
        if self._read_mode not in ("direct", "mmap"):
            logger.warning(
                "APC disk: unknown APC_DISK_READ_MODE=%r; using direct",
                self._read_mode,
            )
            self._read_mode = "direct"
        # Bounded LRU of parsed safetensors headers:
        # shard_path -> (tensor_entries, file_metadata, data_start).
        self._header_cache: "OrderedDict[Path, Tuple[dict, dict, int]]" = OrderedDict()
        self._header_cache_lock = threading.Lock()
        self._header_cache_max = int(os.environ.get("APC_DISK_HEADER_CACHE", 4))
        self._direct_max_overread_bytes = int(
            float(os.environ.get("APC_DISK_DIRECT_MAX_OVERREAD_MB", "8")) * (1 << 20)
        )
        # Bound layer-major shard size so disk eviction is segment-granular
        # instead of one huge all-or-nothing prefix file. A Qwen3-VL-4B block
        # is ~2.25 MiB, so 256 blocks is roughly a 576 MiB shard before the
        # small KV step padding.
        self._shard_max_blocks = max(
            1, int(os.environ.get("APC_DISK_SHARD_MAX_BLOCKS", "256"))
        )
        # Layer-major warm-disk restore concatenates segment shards one layer
        # at a time. Clearing MLX's allocator cache after each layer keeps the
        # temporary segment tensors from coexisting with the fully-restored
        # prompt cache. Set to 0 to disable, or a larger value to trade peak
        # memory for slightly lower restore overhead.
        self._restore_clear_every = max(
            0, int(os.environ.get("APC_DISK_RESTORE_CLEAR_EVERY", "1"))
        )
        # Bounded LRU of mmap'd shards: shard_path -> (arrays_dict, file_metadata).
        # Default capped at 2 — the within-restore working set is typically
        # one shard, occasionally two (for a multi-shard restore). Larger
        # caps risk pinning lots of materialised K/V tensors in unified
        # memory after evicted blocks have already been used. Override with
        # APC_DISK_MMAP_CACHE if you know what you're doing.
        self._mmap_cache: "OrderedDict[Path, Tuple[dict, dict]]" = OrderedDict()
        self._mmap_cache_lock = threading.Lock()
        self._mmap_cache_max = int(os.environ.get("APC_DISK_MMAP_CACHE", 2))

        n_orphans = self._cleanup_partials()
        if n_orphans:
            logger.info(
                "APC disk: removed %d orphaned partial file(s) from %s",
                n_orphans,
                self.dir,
            )
        # Build index from existing shards and compute current byte usage.
        self._disk_bytes = self._rebuild_index()

        self._workers = [
            threading.Thread(
                target=self._writer_loop, daemon=True, name=f"apc-disk-{i}"
            )
            for i in range(max(1, num_workers))
        ]
        for t in self._workers:
            t.start()

    # ---------- Naming + housekeeping ----------
    @classmethod
    def _is_canonical_shard(cls, path: Path) -> bool:
        stem = path.stem
        if not stem.startswith(cls.SHARD_PREFIX):
            return False
        rest = stem[len(cls.SHARD_PREFIX) :]
        if len(rest) != 32:
            return False
        return all(c in "0123456789abcdef" for c in rest)

    @classmethod
    def _is_canonical_exact(cls, path: Path) -> bool:
        stem = path.stem
        if not stem.startswith(cls.EXACT_PREFIX):
            return False
        rest = stem[len(cls.EXACT_PREFIX) :]
        if len(rest) != 32:
            return False
        return all(c in "0123456789abcdef" for c in rest)

    @classmethod
    def _is_canonical_store_file(cls, path: Path) -> bool:
        return cls._is_canonical_shard(path) or cls._is_canonical_exact(path)

    def _shard_path(self, shard_id: str) -> Path:
        return self.dir / f"{shard_id}{self.SUFFIX}"

    @staticmethod
    def _shard_id_for(block_hashes: Sequence[int]) -> str:
        h = hashlib.sha256()
        for bh in block_hashes:
            h.update(int(bh & ((1 << 64) - 1)).to_bytes(8, "little"))
        return f"{DiskBlockStore.SHARD_PREFIX}{h.hexdigest()[:32]}"

    @staticmethod
    def _exact_id_for(cache_hash: int) -> str:
        h = hashlib.sha256()
        h.update(int(cache_hash & ((1 << 64) - 1)).to_bytes(8, "little"))
        return f"{DiskBlockStore.EXACT_PREFIX}{h.hexdigest()[:32]}"

    def _cleanup_partials(self) -> int:
        """Delete anything in the dir that isn't a canonical shard (left
        over from a crashed write, or files from an older block-per-file
        layout that this class no longer recognises)."""
        n = 0
        for p in self.dir.glob(f"*{self.SUFFIX}"):
            if not p.is_file() or self._is_canonical_store_file(p):
                continue
            try:
                p.unlink()
                n += 1
            except OSError as e:
                logger.warning("APC disk: failed to remove partial %s: %s", p, e)
        return n

    def _rebuild_index(self) -> int:
        """Scan shards, populate ``_index``, return total bytes on disk.

        Uses a header-only safetensors read so each shard scan touches only
        the file's leading few KB — no MLX array construction, no mmap of
        the tensor payload. On a disk with hundreds of cached shards this
        keeps server-startup overhead and Python heap growth minimal.
        """
        total = 0
        with self._index_lock:
            self._index.clear()
            self._exact_index.clear()
            for p in self.dir.glob(f"*{self.SUFFIX}"):
                if not self._is_canonical_store_file(p):
                    continue
                try:
                    total += p.stat().st_size
                except OSError:
                    continue
                metadata = _read_safetensors_metadata(p)
                if metadata is None:
                    logger.warning("APC disk: shard %s unreadable, dropping", p)
                    try:
                        p.unlink()
                    except OSError:
                        pass
                    continue
                if self._is_canonical_exact(p):
                    try:
                        cache_hash = int(metadata.get("cache_hash", ""))
                    except (TypeError, ValueError):
                        continue
                    self._exact_index[cache_hash] = p
                    continue
                hashes_csv = metadata.get("block_hashes", "")
                if not hashes_csv:
                    continue
                try:
                    block_hashes = [int(x) for x in hashes_csv.split(",") if x]
                except ValueError:
                    continue
                for idx, bh in enumerate(block_hashes):
                    self._index[bh] = (p, idx)
        return total

    @property
    def disk_bytes(self) -> int:
        return self._disk_bytes

    @property
    def num_blocks_indexed(self) -> int:
        with self._index_lock:
            return len(self._index)

    @property
    def num_exact_indexed(self) -> int:
        with self._index_lock:
            return len(self._exact_index)

    @property
    def load_returns_detached(self) -> bool:
        return self._read_mode == "direct"

    def _maybe_evict(self) -> int:
        """Evict segment shards until under the low watermark.

        Stores are ordered by last-used time; segments within the same store
        are evicted tail-first so a partially-retained store still has a
        useful prefix.
        """
        if self.max_bytes is None or self._disk_bytes <= self.max_bytes:
            return 0
        target = int(self.max_bytes * self._EVICT_LOW_WATERMARK)
        # Don't evict shards whose blocks are still in-flight to other
        # callers (would race a pending writer).
        with self._in_flight_lock:
            in_flight_hashes = set(self._in_flight.keys())
        with self._index_lock:
            in_flight_paths = {
                self._index[h][0] for h in in_flight_hashes if h in self._index
            }
            in_flight_paths.update(
                self._exact_index[h] for h in in_flight_hashes if h in self._exact_index
            )

        candidates: list[dict[str, Any]] = []
        for p in self.dir.glob(f"*{self.SUFFIX}"):
            if not self._is_canonical_store_file(p) or p in in_flight_paths:
                continue
            try:
                st = p.stat()
            except OSError:
                continue
            store_id = str(p)
            segment_index = 0
            metadata = _read_safetensors_metadata(p)
            if metadata is not None:
                store_id = metadata.get("store_id", store_id)
                try:
                    segment_index = int(metadata.get("segment_index", "0"))
                except (TypeError, ValueError):
                    segment_index = 0
            candidates.append(
                {
                    # Header reads during eviction can update atime on some
                    # filesystems. Use mtime as our explicit last-used clock;
                    # the load paths call os.utime(), which updates it.
                    "last_used": st.st_mtime,
                    "size": st.st_size,
                    "path": p,
                    "store_id": store_id,
                    "segment_index": segment_index,
                }
            )
        store_last_used: dict[str, float] = {}
        for candidate in candidates:
            sid = candidate["store_id"]
            store_last_used[sid] = min(
                candidate["last_used"],
                store_last_used.get(sid, candidate["last_used"]),
            )
        candidates.sort(
            key=lambda c: (
                store_last_used[c["store_id"]],
                -int(c["segment_index"]),
                c["last_used"],
            )
        )

        evicted = 0
        for candidate in candidates:
            if self._disk_bytes <= target:
                break
            size = int(candidate["size"])
            p = candidate["path"]
            try:
                p.unlink()
            except OSError as e:
                logger.warning("APC disk: failed to evict %s: %s", p, e)
                continue
            self._disk_bytes -= size
            evicted += 1
            # Drop index + mmap entries pointing at this shard.
            with self._index_lock:
                stale = [h for h, (sp, _) in self._index.items() if sp == p]
                for h in stale:
                    del self._index[h]
                stale_exact = [h for h, sp in self._exact_index.items() if sp == p]
                for h in stale_exact:
                    del self._exact_index[h]
            with self._mmap_cache_lock:
                self._mmap_cache.pop(p, None)
            with self._header_cache_lock:
                self._header_cache.pop(p, None)
        if evicted:
            self.evictions += evicted
            logger.info(
                "APC disk: evicted %d shard(s); now %.1f MB / %.1f MB cap",
                evicted,
                self._disk_bytes / 1e6,
                self.max_bytes / 1e6,
            )
        return evicted

    # ---------- Header cache ----------
    def _open_shard_header(self, shard_path: Path):
        """Return parsed safetensors header info for a shard, cached."""
        with self._header_cache_lock:
            cached = self._header_cache.get(shard_path)
            if cached is not None:
                self._header_cache.move_to_end(shard_path)
                return cached
        parsed = _read_safetensors_header(shard_path)
        if parsed is None:
            logger.warning("APC disk shard header read failed for %s", shard_path)
            return None
        with self._header_cache_lock:
            self._header_cache[shard_path] = parsed
            self._header_cache.move_to_end(shard_path)
            while len(self._header_cache) > self._header_cache_max:
                self._header_cache.popitem(last=False)
        return parsed

    # ---------- mmap cache ----------
    def _open_shard(self, shard_path: Path):
        """Return (arrays_dict, file_metadata) for a shard, mmap-cached."""
        with self._mmap_cache_lock:
            cached = self._mmap_cache.get(shard_path)
            if cached is not None:
                self._mmap_cache.move_to_end(shard_path)
                return cached
        try:
            arrays, metadata = mx.load(str(shard_path), return_metadata=True)
        except Exception as e:
            logger.warning("APC disk shard load failed for %s: %s", shard_path, e)
            return None
        # Touch recency timestamp so LRU eviction prefers truly-cold shards.
        try:
            os.utime(shard_path, None)
        except OSError:
            pass
        bundle = (dict(arrays), dict(metadata))
        with self._mmap_cache_lock:
            self._mmap_cache[shard_path] = bundle
            self._mmap_cache.move_to_end(shard_path)
            while len(self._mmap_cache) > self._mmap_cache_max:
                self._mmap_cache.popitem(last=False)
        return bundle

    # ---------- Public API ----------
    def has(self, block_hash: int) -> bool:
        with self._index_lock:
            return block_hash in self._index

    def has_exact(self, cache_hash: int) -> bool:
        with self._index_lock:
            return cache_hash in self._exact_index

    def find_exact_prefix(
        self,
        token_ids: Sequence[int],
        *,
        extra_hash: int = 0,
        max_prefix_tokens: Optional[int] = None,
        min_prefix_tokens: int = 0,
    ) -> Optional[Tuple[int, int]]:
        token_tuple = tuple(int(t) for t in token_ids)
        max_len = len(token_tuple) - 1
        if max_prefix_tokens is not None and max_prefix_tokens > 0:
            max_len = min(max_len, int(max_prefix_tokens))
        if max_len <= min_prefix_tokens:
            return None

        with self._index_lock:
            entries = list(self._exact_index.items())

        best: Optional[Tuple[int, int]] = None
        for cache_hash, path in entries:
            parsed = self._open_shard_header(path)
            if parsed is None:
                continue
            _tensor_entries, metadata, _data_start = parsed
            if metadata.get("layout") != "exact_cache_v1":
                continue
            try:
                stored_extra = int(metadata.get("extra_hash", "0"))
                stored_tokens = tuple(
                    int(x) for x in metadata.get("token_ids", "").split(",") if x
                )
            except (TypeError, ValueError):
                continue
            prefix_len = len(stored_tokens)
            if (
                stored_extra != extra_hash
                or prefix_len <= min_prefix_tokens
                or prefix_len > max_len
                or token_tuple[:prefix_len] != stored_tokens
            ):
                continue
            if best is None or prefix_len > best[1]:
                best = (int(cache_hash), prefix_len)
        return best

    def load_exact_cache(
        self,
        cache_hash: int,
        *,
        wait_in_flight_ms: float = 0.0,
        min_capacity_tokens: Optional[int] = None,
    ) -> Optional[Tuple[Tuple[int, ...], int, List[Any]]]:
        with self._index_lock:
            path = self._exact_index.get(cache_hash)
        if path is None:
            if wait_in_flight_ms > 0:
                with self._in_flight_lock:
                    ev = self._in_flight.get(cache_hash)
                if ev is not None and ev.wait(wait_in_flight_ms / 1000.0):
                    with self._index_lock:
                        path = self._exact_index.get(cache_hash)
            if path is None:
                return None
        return self._load_exact_cache_file(
            path, min_capacity_tokens=min_capacity_tokens
        )

    def _load_exact_cache_file(
        self,
        path: Path,
        *,
        min_capacity_tokens: Optional[int],
    ) -> Optional[Tuple[Tuple[int, ...], int, List[Any]]]:
        parsed = self._open_shard_header(path)
        if parsed is None:
            return None
        tensor_entries, metadata, data_start = parsed
        if metadata.get("layout") != "exact_cache_v1":
            return None
        try:
            token_ids = tuple(
                int(x) for x in metadata.get("token_ids", "").split(",") if x
            )
            extra_hash = int(metadata.get("extra_hash", "0"))
            n_entries = int(metadata.get("num_entries", "0"))
        except (TypeError, ValueError):
            return None
        if n_entries <= 0:
            return None

        prompt_cache: List[Any] = []
        eval_targets: List[mx.array] = []
        for i in range(n_entries):
            loaded = self._load_exact_cache_entry(
                path,
                tensor_entries,
                metadata,
                data_start,
                f"c{i}",
                min_capacity_tokens=min_capacity_tokens,
                eval_targets=eval_targets,
            )
            if loaded is None:
                return None
            prompt_cache.append(loaded)
        if eval_targets:
            mx.eval(eval_targets)
        try:
            os.utime(path, None)
        except OSError:
            pass
        return token_ids, extra_hash, prompt_cache

    def _load_exact_cache_entry(
        self,
        path: Path,
        tensor_entries: dict,
        metadata: dict,
        data_start: int,
        prefix: str,
        *,
        min_capacity_tokens: Optional[int],
        eval_targets: List[mx.array],
    ) -> Optional[Any]:
        from mlx_lm.models import cache as lm_cache

        kind = metadata.get(f"{prefix}_kind")
        if kind == "kv":
            if metadata.get(f"{prefix}_empty", "0") == "1":
                c = lm_cache.KVCache()
                try:
                    c.offset = int(metadata.get(f"{prefix}_offset", "0"))
                except (TypeError, ValueError):
                    c.offset = 0
                return c
            k_entry = tensor_entries.get(f"{prefix}_k")
            v_entry = tensor_entries.get(f"{prefix}_v")
            if k_entry is None or v_entry is None:
                return None
            k = _read_safetensors_tensor(path, data_start, k_entry)
            v = _read_safetensors_tensor(path, data_start, v_entry)
            if k is None or v is None:
                return None
            try:
                off = int(metadata.get(f"{prefix}_offset", str(k.shape[2])))
                step = int(metadata.get(f"{prefix}_step", "256"))
            except (TypeError, ValueError):
                return None
            k, v = _pad_kv_for_capacity(
                k,
                v,
                offset=off,
                min_capacity_tokens=min_capacity_tokens,
                step=step,
            )
            c = lm_cache.KVCache()
            c.keys = k
            c.values = v
            c.offset = off
            eval_targets.extend([k, v])
            return c

        if kind == "rotating_kv":
            try:
                keep = int(metadata.get(f"{prefix}_keep", "0"))
                max_size = int(metadata[f"{prefix}_max_size"])
                offset = int(metadata.get(f"{prefix}_offset", "0"))
                idx = int(metadata.get(f"{prefix}_idx", "0"))
            except (KeyError, TypeError, ValueError):
                return None
            c = lm_cache.RotatingKVCache(max_size=max_size, keep=keep)
            c.offset = offset
            c._idx = idx
            if metadata.get(f"{prefix}_empty", "0") == "1":
                return c
            k_entry = tensor_entries.get(f"{prefix}_k")
            v_entry = tensor_entries.get(f"{prefix}_v")
            if k_entry is None or v_entry is None:
                return None
            k = _read_safetensors_tensor(path, data_start, k_entry)
            v = _read_safetensors_tensor(path, data_start, v_entry)
            if k is None or v is None:
                return None
            c.keys = k
            c.values = v
            eval_targets.extend([k, v])
            return c

        if kind == "chunked_kv":
            try:
                chunk_size = int(metadata[f"{prefix}_chunk_size"])
                offset = int(metadata.get(f"{prefix}_offset", "0"))
                start_position = int(metadata.get(f"{prefix}_start_position", "0"))
            except (KeyError, TypeError, ValueError):
                return None
            c = lm_cache.ChunkedKVCache(chunk_size=chunk_size)
            c.offset = offset
            c.start_position = start_position
            if metadata.get(f"{prefix}_empty", "0") == "1":
                return c
            k_entry = tensor_entries.get(f"{prefix}_k")
            v_entry = tensor_entries.get(f"{prefix}_v")
            if k_entry is None or v_entry is None:
                return None
            k = _read_safetensors_tensor(path, data_start, k_entry)
            v = _read_safetensors_tensor(path, data_start, v_entry)
            if k is None or v is None:
                return None
            c.keys = k
            c.values = v
            eval_targets.extend([k, v])
            return c

        if kind == "arrays":
            try:
                size = int(metadata.get(f"{prefix}_size", "0"))
            except (TypeError, ValueError):
                return None
            c = lm_cache.ArraysCache(size=size)
            states: List[Optional[mx.array]] = []
            for j in range(size):
                if metadata.get(f"{prefix}_s{j}_none", "0") == "1":
                    states.append(None)
                    continue
                entry = tensor_entries.get(f"{prefix}_s{j}")
                if entry is None:
                    return None
                state = _read_safetensors_tensor(path, data_start, entry)
                if state is None:
                    return None
                states.append(state)
                eval_targets.append(state)
            c.cache = states
            lp_entry = tensor_entries.get(f"{prefix}_left_padding")
            if lp_entry is not None:
                c.left_padding = _read_safetensors_tensor(path, data_start, lp_entry)
                if c.left_padding is None:
                    return None
                eval_targets.append(c.left_padding)
            lengths_entry = tensor_entries.get(f"{prefix}_lengths")
            if lengths_entry is not None:
                c.lengths = _read_safetensors_tensor(path, data_start, lengths_entry)
                if c.lengths is None:
                    return None
                eval_targets.append(c.lengths)
            return c

        if kind in ("cache_list", "tuple"):
            try:
                size = int(metadata.get(f"{prefix}_size", "0"))
            except (TypeError, ValueError):
                return None
            loaded = []
            for j in range(size):
                sub_c = self._load_exact_cache_entry(
                    path,
                    tensor_entries,
                    metadata,
                    data_start,
                    f"{prefix}_e{j}",
                    min_capacity_tokens=min_capacity_tokens,
                    eval_targets=eval_targets,
                )
                if sub_c is None:
                    return None
                loaded.append(sub_c)
            if kind == "cache_list":
                return lm_cache.CacheList(*loaded)
            return tuple(loaded)

        return None

    def load(
        self, block_hash: int, *, wait_in_flight_ms: float = 0.0
    ) -> Optional[Tuple[List[mx.array], List[mx.array], dict]]:
        """Read one block. Returns (keys, values, per-block metadata) or None.

        Per-block metadata is decoded from the shard's ``b{idx}_meta`` JSON
        entry and includes ``token_ids``, ``parent_hash``, ``extra_hash``,
        ``block_hash``.
        """
        with self._index_lock:
            entry = self._index.get(block_hash)
        if entry is None:
            if wait_in_flight_ms > 0:
                with self._in_flight_lock:
                    ev = self._in_flight.get(block_hash)
                if ev is not None and ev.wait(wait_in_flight_ms / 1000.0):
                    with self._index_lock:
                        entry = self._index.get(block_hash)
            if entry is None:
                return None
        shard_path, block_idx = entry
        if self._read_mode == "mmap":
            return self._load_mmap(shard_path, block_idx)
        return self._load_direct(shard_path, block_idx)

    def load_many(
        self, block_hashes: Sequence[int], *, wait_in_flight_ms: float = 0.0
    ) -> List[Optional[Tuple[List[mx.array], List[mx.array], dict]]]:
        """Read multiple blocks, preserving order.

        In direct mode, consecutive requests from the same shard are coalesced
        into larger byte-range reads. In mmap mode, fall back to one-at-a-time
        loads so the old comparison path stays simple and unchanged.
        """
        if not block_hashes:
            return []
        if self._read_mode == "mmap":
            return [
                self.load(h, wait_in_flight_ms=wait_in_flight_ms) for h in block_hashes
            ]

        entries: List[Optional[Tuple[Path, int]]] = []
        for h in block_hashes:
            with self._index_lock:
                entry = self._index.get(h)
            if entry is None and wait_in_flight_ms > 0:
                with self._in_flight_lock:
                    ev = self._in_flight.get(h)
                if ev is not None and ev.wait(wait_in_flight_ms / 1000.0):
                    with self._index_lock:
                        entry = self._index.get(h)
            entries.append(entry)

        out: List[Optional[Tuple[List[mx.array], List[mx.array], dict]]] = [None] * len(
            block_hashes
        )
        i = 0
        while i < len(entries):
            entry = entries[i]
            if entry is None:
                i += 1
                continue
            shard_path = entry[0]
            j = i + 1
            while (
                j < len(entries)
                and entries[j] is not None
                and entries[j][0] == shard_path
            ):
                j += 1
            block_indices = [entries[k][1] for k in range(i, j)]
            loaded = self._load_direct_many(shard_path, block_indices)
            out[i:j] = loaded
            i = j
        return out

    def _decode_block_metadata(self, file_metadata: dict, block_idx: int) -> dict:
        block_meta_str = file_metadata.get(f"b{block_idx}_meta")
        if not block_meta_str:
            return {}
        try:
            block_meta = json.loads(block_meta_str)
            # Coerce token_ids (encoded as comma-separated ints) to a string
            # for compatibility with the existing verify path.
            if isinstance(block_meta.get("token_ids"), list):
                block_meta["token_ids"] = ",".join(
                    str(int(t)) for t in block_meta["token_ids"]
                )
            if "extra_hash" in block_meta:
                block_meta["extra_hash"] = str(block_meta["extra_hash"])
            return block_meta
        except Exception:
            return {}

    def _load_mmap(
        self, shard_path: Path, block_idx: int
    ) -> Optional[Tuple[List[mx.array], List[mx.array], dict]]:
        bundle = self._open_shard(shard_path)
        if bundle is None:
            return None
        arrays, file_metadata = bundle
        try:
            num_layers = int(file_metadata.get("num_layers", "0"))
        except (TypeError, ValueError):
            return None
        try:
            keys = [arrays[f"b{block_idx}_k{l}"] for l in range(num_layers)]
            values = [arrays[f"b{block_idx}_v{l}"] for l in range(num_layers)]
        except KeyError as e:
            logger.warning("APC disk shard %s missing tensor: %s", shard_path, e)
            return None
        return keys, values, self._decode_block_metadata(file_metadata, block_idx)

    def _load_direct(
        self, shard_path: Path, block_idx: int
    ) -> Optional[Tuple[List[mx.array], List[mx.array], dict]]:
        loaded = self._load_direct_many(shard_path, [block_idx])
        return loaded[0] if loaded else None

    def _load_layer_major_segment(
        self,
        shard_path: Path,
        block_indices: Sequence[int],
        *,
        preserve_capacity: bool = False,
    ) -> Optional[Tuple[List[mx.array], List[mx.array], List[dict]]]:
        if not block_indices:
            return None
        start_idx = block_indices[0]
        if list(block_indices) != list(
            range(start_idx, start_idx + len(block_indices))
        ):
            return None
        parsed = self._open_shard_header(shard_path)
        if parsed is None:
            return None
        tensor_entries, file_metadata, data_start = parsed
        layout = file_metadata.get("layout")
        if layout == "token_major_v2":
            return self._load_token_major_segment(
                shard_path, tensor_entries, file_metadata, data_start, block_indices
            )
        if layout not in ("layer_major_v1", "layer_major_v2"):
            return None
        try:
            num_layers = int(file_metadata.get("num_layers", "0"))
            block_size = int(file_metadata.get("block_size", "0"))
        except (TypeError, ValueError):
            return None
        if num_layers <= 0 or block_size <= 0:
            return None

        token_start = start_idx * block_size
        token_end = token_start + len(block_indices) * block_size
        shard_n_blocks = len(
            [x for x in file_metadata.get("block_hashes", "").split(",") if x]
        )
        requested_to_shard_end = (
            shard_n_blocks > 0
            and start_idx == 0
            and start_idx + len(block_indices) >= shard_n_blocks
        )
        slice_end = (
            None
            if (
                preserve_capacity
                and layout == "layer_major_v2"
                and requested_to_shard_end
            )
            else token_end
        )
        keys: List[mx.array] = []
        values: List[mx.array] = []
        for l in range(num_layers):
            k_entry = tensor_entries.get(f"k{l}")
            v_entry = tensor_entries.get(f"v{l}")
            if k_entry is None or v_entry is None:
                return None
            k = _read_safetensors_tensor(shard_path, data_start, k_entry)
            v = _read_safetensors_tensor(shard_path, data_start, v_entry)
            if k is None or v is None:
                return None
            keys.append(k[..., token_start:slice_end, :])
            values.append(v[..., token_start:slice_end, :])

        metadata = [
            self._decode_block_metadata(file_metadata, idx) for idx in block_indices
        ]
        try:
            os.utime(shard_path, None)
        except OSError:
            pass
        mx.eval(keys + values)
        return keys, values, metadata

    def _load_layer_major_prefix_segments_layerwise(
        self,
        segments: Sequence[Tuple[Path, List[int]]],
        *,
        preserve_capacity: bool,
    ) -> Optional[Tuple[List[mx.array], List[mx.array], List[dict]]]:
        """Load layer-major segments without holding all segment tensors.

        The older restore path first read every segment's K/V for every layer,
        then concatenated all layers at once. For long prefixes this doubled
        peak MLX memory: segment tensors plus final KVCache tensors. This
        routine reads all segments for one layer, emits that layer's final K/V,
        clears temporary allocator state, and moves to the next layer.
        """
        if not segments:
            return None

        segment_infos: List[
            Tuple[Path, dict, dict, int, int, Optional[int], List[int]]
        ] = []
        metadata: List[dict] = []
        num_layers: Optional[int] = None
        block_size_ref: Optional[int] = None
        last_segment_idx = len(segments) - 1

        for segment_idx, (shard_path, block_indices) in enumerate(segments):
            if not block_indices:
                return None
            start_idx = block_indices[0]
            if list(block_indices) != list(
                range(start_idx, start_idx + len(block_indices))
            ):
                return None
            parsed = self._open_shard_header(shard_path)
            if parsed is None:
                return None
            tensor_entries, file_metadata, data_start = parsed
            layout = file_metadata.get("layout")
            if layout not in ("layer_major_v1", "layer_major_v2"):
                return None
            try:
                shard_layers = int(file_metadata.get("num_layers", "0"))
                block_size = int(file_metadata.get("block_size", "0"))
            except (TypeError, ValueError):
                return None
            if shard_layers <= 0 or block_size <= 0:
                return None
            if num_layers is None:
                num_layers = shard_layers
                block_size_ref = block_size
            elif shard_layers != num_layers or block_size != block_size_ref:
                return None

            token_start = start_idx * block_size
            token_end = token_start + len(block_indices) * block_size
            shard_n_blocks = len(
                [x for x in file_metadata.get("block_hashes", "").split(",") if x]
            )
            requested_to_shard_end = (
                shard_n_blocks > 0
                and start_idx == 0
                and start_idx + len(block_indices) >= shard_n_blocks
            )
            slice_end = (
                None
                if (
                    preserve_capacity
                    and segment_idx == last_segment_idx
                    and layout == "layer_major_v2"
                    and requested_to_shard_end
                )
                else token_end
            )
            segment_infos.append(
                (
                    shard_path,
                    tensor_entries,
                    file_metadata,
                    data_start,
                    token_start,
                    slice_end,
                    list(block_indices),
                )
            )
            metadata.extend(
                self._decode_block_metadata(file_metadata, idx) for idx in block_indices
            )
            try:
                os.utime(shard_path, None)
            except OSError:
                pass

        if num_layers is None:
            return None

        keys: List[mx.array] = []
        values: List[mx.array] = []
        for layer_idx in range(num_layers):
            k_parts: List[mx.array] = []
            v_parts: List[mx.array] = []
            for (
                shard_path,
                tensor_entries,
                _file_metadata,
                data_start,
                token_start,
                slice_end,
                _block_indices,
            ) in segment_infos:
                k_entry = tensor_entries.get(f"k{layer_idx}")
                v_entry = tensor_entries.get(f"v{layer_idx}")
                if k_entry is None or v_entry is None:
                    return None
                k = _read_safetensors_tensor(shard_path, data_start, k_entry)
                v = _read_safetensors_tensor(shard_path, data_start, v_entry)
                if k is None or v is None:
                    return None
                k_parts.append(k[..., token_start:slice_end, :])
                v_parts.append(v[..., token_start:slice_end, :])

            k_out = k_parts[0] if len(k_parts) == 1 else mx.concatenate(k_parts, axis=2)
            v_out = v_parts[0] if len(v_parts) == 1 else mx.concatenate(v_parts, axis=2)
            mx.eval(k_out, v_out)
            keys.append(k_out)
            values.append(v_out)
            del k_parts, v_parts, k_out, v_out
            if (
                self._restore_clear_every > 0
                and (layer_idx + 1) % self._restore_clear_every == 0
            ):
                mx.clear_cache()

        return keys, values, metadata

    def _load_token_major_segment(
        self,
        shard_path: Path,
        tensor_entries: dict,
        file_metadata: dict,
        data_start: int,
        block_indices: Sequence[int],
    ) -> Optional[Tuple[List[mx.array], List[mx.array], List[dict]]]:
        try:
            num_layers = int(file_metadata.get("num_layers", "0"))
            block_size = int(file_metadata.get("block_size", "0"))
        except (TypeError, ValueError):
            return None
        if num_layers <= 0 or block_size <= 0:
            return None

        start_idx = block_indices[0]
        token_start = start_idx * block_size
        token_end = token_start + len(block_indices) * block_size
        k_entry = tensor_entries.get("k_all")
        v_entry = tensor_entries.get("v_all")
        if k_entry is None or v_entry is None:
            return None

        k_all = _read_safetensors_axis0_slice(
            shard_path, data_start, k_entry, token_start, token_end
        )
        v_all = _read_safetensors_axis0_slice(
            shard_path, data_start, v_entry, token_start, token_end
        )
        if k_all is None or v_all is None:
            return None
        if len(k_all.shape) != 5 or len(v_all.shape) != 5:
            return None
        if k_all.shape[1] != num_layers or v_all.shape[1] != num_layers:
            return None

        keys = [mx.transpose(k_all[:, l, ...], (1, 2, 0, 3)) for l in range(num_layers)]
        values = [
            mx.transpose(v_all[:, l, ...], (1, 2, 0, 3)) for l in range(num_layers)
        ]
        metadata = [
            self._decode_block_metadata(file_metadata, idx) for idx in block_indices
        ]
        try:
            os.utime(shard_path, None)
        except OSError:
            pass
        mx.eval([k_all, v_all])
        return keys, values, metadata

    def _load_token_major_prefix_segments(
        self, segments: Sequence[Tuple[Path, List[int]]]
    ) -> Optional[Tuple[List[mx.array], List[mx.array], List[dict]]]:
        """Fast path for token-major shards.

        Concatenate raw token-major byte ranges before constructing MLX arrays.
        This avoids a first-request MLX compile of 72 per-layer concatenations
        when a prefix spans a common-prefix shard plus a request-specific shard.
        """
        if not segments:
            return None

        num_layers: Optional[int] = None
        block_size_ref: Optional[int] = None
        k_tail_shape: Optional[Tuple[int, ...]] = None
        v_tail_shape: Optional[Tuple[int, ...]] = None
        k_dtype: Optional[str] = None
        v_dtype: Optional[str] = None
        total_tokens = 0
        k_buf = bytearray()
        v_buf = bytearray()
        metadata: List[dict] = []

        for shard_path, block_indices in segments:
            parsed = self._open_shard_header(shard_path)
            if parsed is None:
                return None
            tensor_entries, file_metadata, data_start = parsed
            if file_metadata.get("layout") != "token_major_v2":
                return None
            try:
                shard_layers = int(file_metadata.get("num_layers", "0"))
                block_size = int(file_metadata.get("block_size", "0"))
            except (TypeError, ValueError):
                return None
            if shard_layers <= 0 or block_size <= 0:
                return None
            if num_layers is None:
                num_layers = shard_layers
                block_size_ref = block_size
            elif shard_layers != num_layers or block_size != block_size_ref:
                return None

            k_entry = tensor_entries.get("k_all")
            v_entry = tensor_entries.get("v_all")
            if k_entry is None or v_entry is None:
                return None
            start_idx = block_indices[0]
            token_start = start_idx * block_size
            token_end = token_start + len(block_indices) * block_size
            k_sliced = _read_safetensors_axis0_slice_bytes(
                shard_path, data_start, k_entry, token_start, token_end
            )
            v_sliced = _read_safetensors_axis0_slice_bytes(
                shard_path, data_start, v_entry, token_start, token_end
            )
            if k_sliced is None or v_sliced is None:
                return None
            k_raw, k_sliced_entry = k_sliced
            v_raw, v_sliced_entry = v_sliced
            k_shape = tuple(int(x) for x in k_sliced_entry["shape"])
            v_shape = tuple(int(x) for x in v_sliced_entry["shape"])
            if len(k_shape) != 5 or len(v_shape) != 5:
                return None
            if k_shape[1] != num_layers or v_shape[1] != num_layers:
                return None
            if k_shape[0] != v_shape[0]:
                return None
            if k_tail_shape is None:
                k_tail_shape = k_shape[1:]
                v_tail_shape = v_shape[1:]
                k_dtype = str(k_sliced_entry["dtype"])
                v_dtype = str(v_sliced_entry["dtype"])
            elif (
                k_tail_shape != k_shape[1:]
                or v_tail_shape != v_shape[1:]
                or k_dtype != str(k_sliced_entry["dtype"])
                or v_dtype != str(v_sliced_entry["dtype"])
            ):
                return None

            k_buf.extend(k_raw)
            v_buf.extend(v_raw)
            total_tokens += k_shape[0]
            metadata.extend(
                self._decode_block_metadata(file_metadata, idx) for idx in block_indices
            )
            try:
                os.utime(shard_path, None)
            except OSError:
                pass

        if (
            num_layers is None
            or k_tail_shape is None
            or v_tail_shape is None
            or k_dtype is None
            or v_dtype is None
            or total_tokens <= 0
        ):
            return None

        k_dtype_info = _safetensors_dtype_info(k_dtype)
        v_dtype_info = _safetensors_dtype_info(v_dtype)
        if k_dtype_info is None or v_dtype_info is None:
            return None
        k_np_dtype, k_mlx_dtype, k_bitcast_to = k_dtype_info
        v_np_dtype, v_mlx_dtype, v_bitcast_to = v_dtype_info
        try:
            k_np = np.frombuffer(k_buf, dtype=k_np_dtype).reshape(
                (total_tokens, *k_tail_shape)
            )
            v_np = np.frombuffer(v_buf, dtype=v_np_dtype).reshape(
                (total_tokens, *v_tail_shape)
            )
        except ValueError:
            return None

        # Build standard contiguous KVCache slabs with one decode step of spare
        # capacity. Exact-size restored caches make KVCache.update_and_fetch()
        # grow via 72 MLX concatenations on the first generated token, which is
        # a large first-use compile. Padding here is a plain NumPy copy.
        kv_step = 256
        capacity = ((total_tokens + 1 + kv_step - 1) // kv_step) * kv_step
        keys: List[mx.array] = []
        values: List[mx.array] = []
        for l in range(num_layers):
            k_layer = np.zeros(
                (k_tail_shape[1], k_tail_shape[2], capacity, k_tail_shape[3]),
                dtype=k_np_dtype,
            )
            v_layer = np.zeros(
                (v_tail_shape[1], v_tail_shape[2], capacity, v_tail_shape[3]),
                dtype=v_np_dtype,
            )
            k_layer[..., :total_tokens, :] = k_np[:, l, ...].transpose(1, 2, 0, 3)
            v_layer[..., :total_tokens, :] = v_np[:, l, ...].transpose(1, 2, 0, 3)
            keys.append(mx.array(k_layer, dtype=k_mlx_dtype))
            values.append(mx.array(v_layer, dtype=v_mlx_dtype))
        if k_bitcast_to is not None:
            keys = [k.view(k_bitcast_to) for k in keys]
        if v_bitcast_to is not None:
            values = [v.view(v_bitcast_to) for v in values]
        if k_bitcast_to is not None:
            keys = [_copy_mlx_array(k) for k in keys]
        if v_bitcast_to is not None:
            values = [_copy_mlx_array(v) for v in values]
        mx.eval(keys + values)
        return keys, values, metadata

    def load_layer_major_prefix(
        self, block_hashes: Sequence[int], *, preserve_capacity: bool = True
    ) -> Optional[Tuple[List[mx.array], List[mx.array], List[dict]]]:
        """Load a cached prefix directly as per-layer K/V tensors.

        Handles prefixes that span several layer-major shards. Returns
        ``(keys, values, per_block_metadata)`` where each key/value tensor
        covers the full requested prefix for one layer. This is the warm-disk
        fast path: 72 tensors for a Qwen3-VL-4B prefix instead of 209 * 72
        block slabs.
        """
        if not block_hashes:
            return None
        trace = os.environ.get("APC_DISK_TRACE", "").lower() in ("1", "true", "yes")
        trace_t0 = time.perf_counter()

        entries: List[Tuple[Path, int]] = []
        for h in block_hashes:
            with self._index_lock:
                entry = self._index.get(h)
            if entry is None:
                return None
            entries.append(entry)

        segments: List[Tuple[Path, List[int]]] = []
        for shard_path, block_idx in entries:
            if (
                not segments
                or segments[-1][0] != shard_path
                or segments[-1][1][-1] + 1 != block_idx
            ):
                segments.append((shard_path, [block_idx]))
            else:
                segments[-1][1].append(block_idx)

        trace_raw_t0 = time.perf_counter()
        raw_token_major = self._load_token_major_prefix_segments(segments)
        trace_raw_t1 = time.perf_counter()
        if raw_token_major is not None:
            if trace:
                print(
                    "APC_DISK_TRACE restore "
                    f"blocks={len(block_hashes)} segments={len(segments)} "
                    f"raw_token_major={trace_raw_t1 - trace_raw_t0:.3f}s "
                    f"total={trace_raw_t1 - trace_t0:.3f}s",
                    flush=True,
                )
            return raw_token_major

        trace_layerwise_t0 = time.perf_counter()
        layerwise = self._load_layer_major_prefix_segments_layerwise(
            segments,
            preserve_capacity=preserve_capacity,
        )
        trace_layerwise_t1 = time.perf_counter()
        if layerwise is not None:
            if trace:
                print(
                    "APC_DISK_TRACE restore "
                    f"blocks={len(block_hashes)} segments={len(segments)} "
                    f"layerwise={trace_layerwise_t1 - trace_layerwise_t0:.3f}s "
                    f"total={trace_layerwise_t1 - trace_t0:.3f}s",
                    flush=True,
                )
            return layerwise

        trace_load_t0 = time.perf_counter()
        loaded_segments = []
        last_segment_idx = len(segments) - 1
        for segment_idx, (shard_path, block_indices) in enumerate(segments):
            loaded_segments.append(
                self._load_layer_major_segment(
                    shard_path,
                    block_indices,
                    preserve_capacity=(
                        preserve_capacity
                        and segment_idx == last_segment_idx
                        and bool(block_indices)
                        and block_indices[0] == 0
                    ),
                )
            )
        trace_load_t1 = time.perf_counter()
        if any(seg is None for seg in loaded_segments):
            return None

        first_keys, first_values, _ = loaded_segments[0]
        num_layers = len(first_keys)
        if num_layers == 0 or len(first_values) != num_layers:
            return None

        keys: List[mx.array] = []
        values: List[mx.array] = []
        metadata: List[dict] = []
        for seg in loaded_segments:
            seg_keys, seg_values, seg_metadata = seg
            if len(seg_keys) != num_layers or len(seg_values) != num_layers:
                return None
            metadata.extend(seg_metadata)

        trace_concat_t0 = time.perf_counter()
        for l in range(num_layers):
            keys.append(mx.concatenate([seg[0][l] for seg in loaded_segments], axis=2))
            values.append(
                mx.concatenate([seg[1][l] for seg in loaded_segments], axis=2)
            )
        mx.eval(keys + values)
        trace_concat_t1 = time.perf_counter()
        if trace:
            print(
                "APC_DISK_TRACE restore "
                f"blocks={len(block_hashes)} segments={len(segments)} "
                f"load={trace_load_t1 - trace_load_t0:.3f}s "
                f"concat_eval={trace_concat_t1 - trace_concat_t0:.3f}s "
                f"total={trace_concat_t1 - trace_t0:.3f}s",
                flush=True,
            )
        return keys, values, metadata

    def _collect_direct_specs(
        self,
        tensor_entries: dict,
        num_layers: int,
        block_indices: Sequence[int],
        shard_path: Path,
    ):
        specs = []
        total_bytes = 0
        for block_idx in block_indices:
            for l in range(num_layers):
                for suffix in ("k", "v"):
                    name = f"b{block_idx}_{suffix}{l}"
                    entry = tensor_entries.get(name)
                    if entry is None:
                        logger.warning(
                            "APC disk shard %s missing tensor: %s", shard_path, name
                        )
                        return None
                    bounds = _safetensors_tensor_bounds(entry)
                    if bounds is None:
                        logger.warning(
                            "APC disk shard %s has unsupported/corrupt tensor: %s",
                            shard_path,
                            name,
                        )
                        return None
                    start, end, _ = bounds
                    specs.append((block_idx, name, entry, start, end))
                    total_bytes += end - start
        return specs, total_bytes

    def _load_direct_many(
        self, shard_path: Path, block_indices: Sequence[int]
    ) -> List[Optional[Tuple[List[mx.array], List[mx.array], dict]]]:
        if not block_indices:
            return []
        parsed = self._open_shard_header(shard_path)
        if parsed is None:
            return [None] * len(block_indices)
        tensor_entries, file_metadata, data_start = parsed
        try:
            num_layers = int(file_metadata.get("num_layers", "0"))
        except (TypeError, ValueError):
            return [None] * len(block_indices)

        if file_metadata.get("layout") in (
            "layer_major_v1",
            "layer_major_v2",
            "token_major_v2",
        ):
            try:
                block_hashes = [
                    int(json.loads(file_metadata[f"b{idx}_meta"])["block_hash"])
                    for idx in block_indices
                ]
            except (KeyError, TypeError, ValueError, json.JSONDecodeError):
                return [None] * len(block_indices)
            loaded = self.load_layer_major_prefix(block_hashes, preserve_capacity=False)
            if loaded is None:
                return [None] * len(block_indices)
            layer_keys, layer_values, metadatas = loaded
            out = []
            try:
                block_size = int(file_metadata.get("block_size", "0"))
            except (TypeError, ValueError):
                return [None] * len(block_indices)
            for i, md in enumerate(metadatas):
                start = i * block_size
                end = start + block_size
                out.append(
                    (
                        [k[..., start:end, :] for k in layer_keys],
                        [v[..., start:end, :] for v in layer_values],
                        md,
                    )
                )
            return out

        collected = self._collect_direct_specs(
            tensor_entries, num_layers, block_indices, shard_path
        )
        if collected is None:
            return [None] * len(block_indices)
        specs, total_bytes = collected
        if not specs:
            return [
                ([], [], self._decode_block_metadata(file_metadata, block_idx))
                for block_idx in block_indices
            ]

        min_start = min(start for _, _, _, start, _ in specs)
        max_end = max(end for _, _, _, _, end in specs)
        span = max_end - min_start
        if (
            len(block_indices) > 1
            and span > total_bytes + self._direct_max_overread_bytes
        ):
            mid = len(block_indices) // 2
            return self._load_direct_many(
                shard_path, block_indices[:mid]
            ) + self._load_direct_many(shard_path, block_indices[mid:])

        try:
            with open(shard_path, "rb") as f:
                # ``mx.save_safetensors`` may reorder tensors in the data
                # buffer, so we compute the exact span from the header. For a
                # chain-contiguous shard restore this is usually one compact
                # range, turning hundreds of small reads into one larger read.
                f.seek(data_start + min_start)
                slab = f.read(span)
                if len(slab) != span:
                    return [None] * len(block_indices)
                view = memoryview(slab)
                raw_by_name = {
                    name: view[start - min_start : end - min_start]
                    for _, name, _, start, end in specs
                }
        except OSError as e:
            logger.warning("APC disk direct read failed for %s: %s", shard_path, e)
            return [None] * len(block_indices)

        entries_by_name = {name: entry for _, name, entry, _, _ in specs}
        out: List[Optional[Tuple[List[mx.array], List[mx.array], dict]]] = []
        for block_idx in block_indices:
            keys: List[mx.array] = []
            values: List[mx.array] = []
            ok = True
            for l in range(num_layers):
                k_name = f"b{block_idx}_k{l}"
                v_name = f"b{block_idx}_v{l}"
                k = _mlx_array_from_safetensors_bytes(
                    raw_by_name[k_name], entries_by_name[k_name]
                )
                v = _mlx_array_from_safetensors_bytes(
                    raw_by_name[v_name], entries_by_name[v_name]
                )
                if k is None or v is None:
                    ok = False
                    break
                keys.append(k)
                values.append(v)
            if ok:
                out.append(
                    (
                        keys,
                        values,
                        self._decode_block_metadata(file_metadata, block_idx),
                    )
                )
            else:
                out.append(None)

        # Touch recency timestamp so LRU eviction prefers truly-cold shards.
        try:
            os.utime(shard_path, None)
        except OSError:
            pass
        return out

    def save_batch(self, blocks: List["APCBlock"]) -> None:
        """Schedule segment-shard writes containing ``blocks``. Returns
        immediately; the writer thread does the safetensors save + atomic
        rename + index update.
        """
        if not blocks:
            return

        snapshots: List[_DiskBlockSnapshot] = []
        for b in blocks:
            if b.block_hash is None or b.keys is None or b.values is None:
                continue
            snapshots.append(
                _DiskBlockSnapshot(
                    block_hash=int(b.block_hash),
                    parent_hash=int(b.parent_hash),
                    extra_hash=int(b.extra_hash),
                    token_ids=tuple(int(t) for t in b.token_ids),
                    keys=list(b.keys),
                    values=list(b.values),
                )
            )
            if len(snapshots) >= self._shard_max_blocks:
                self._enqueue_block_snapshots(snapshots)
                snapshots = []
        if not snapshots:
            return

        self._enqueue_block_snapshots(snapshots)

    def save_exact_cache(
        self,
        cache_hash: int,
        token_ids: Sequence[int],
        extra_hash: int,
        prompt_cache: Sequence[Any],
    ) -> None:
        """Schedule an exact prompt-cache snapshot write.

        Exact snapshots are used for custom cache layouts that cannot be
        reconstructed from independently concatenated K/V blocks.
        """
        token_tuple = tuple(int(t) for t in token_ids)
        if not token_tuple or not prompt_cache:
            return
        snapshot = _DiskExactCacheSnapshot(
            cache_hash=int(cache_hash),
            token_ids=token_tuple,
            extra_hash=int(extra_hash),
            prompt_cache=list(prompt_cache),
        )
        self._enqueue_exact_snapshot(snapshot)

    def save_layer_major_blocks(
        self,
        blocks: List[_DiskLayerMajorBlock],
        layer_keys: Sequence[mx.array],
        layer_values: Sequence[mx.array],
        block_size: int,
    ) -> None:
        """Schedule a layer-major shard write directly from a KV cache.

        This avoids building thousands of per-block MLX tensors when the
        caller only needs durable disk persistence for future warm restores.
        """
        if not blocks or not layer_keys or not layer_values:
            return
        shared_layer_keys = list(layer_keys)
        shared_layer_values = list(layer_values)
        all_block_hashes = [b.block_hash for b in blocks]
        store_id = self._shard_id_for(all_block_hashes)
        segment_count = (
            len(blocks) + self._shard_max_blocks - 1
        ) // self._shard_max_blocks
        for start in range(0, len(blocks), self._shard_max_blocks):
            chunk = list(blocks[start : start + self._shard_max_blocks])
            block_hashes = [b.block_hash for b in chunk]
            snapshot = _DiskLayerMajorSnapshot(
                blocks=chunk,
                layer_keys=shared_layer_keys,
                layer_values=shared_layer_values,
                block_size=int(block_size),
                store_id=store_id,
                segment_index=start // self._shard_max_blocks,
                segment_count=segment_count,
            )
            self._enqueue_shard(
                self._shard_id_for(block_hashes), block_hashes, snapshot
            )

    def _enqueue_block_snapshots(self, snapshots: List[_DiskBlockSnapshot]) -> None:
        block_hashes = [b.block_hash for b in snapshots]
        self._enqueue_shard(
            self._shard_id_for(block_hashes), block_hashes, list(snapshots)
        )

    def _enqueue_exact_snapshot(self, snapshot: _DiskExactCacheSnapshot) -> None:
        cache_hash = int(snapshot.cache_hash)
        shard_id = self._exact_id_for(cache_hash)
        path = self._shard_path(shard_id)
        if path.exists():
            with self._index_lock:
                self._exact_index.setdefault(cache_hash, path)
            return

        ev = threading.Event()
        with self._in_flight_lock:
            self._in_flight[cache_hash] = ev
        try:
            self._q.put_nowait((shard_id, [cache_hash], snapshot, ev))
        except queue.Full:
            with self._in_flight_lock:
                self._in_flight.pop(cache_hash, None)
            ev.set()
            logger.warning("APC disk write queue full; dropping exact-cache snapshot")

    def _enqueue_shard(
        self,
        shard_id: str,
        block_hashes: Sequence[int],
        payload: Any,
    ) -> None:
        path = self._shard_path(shard_id)
        # Already on disk? Just dedup.
        if path.exists():
            with self._index_lock:
                # Make sure index reflects it (e.g. after restart).
                for idx, block_hash in enumerate(block_hashes):
                    self._index.setdefault(int(block_hash), (path, idx))
            return

        ev = threading.Event()
        with self._in_flight_lock:
            for block_hash in block_hashes:
                self._in_flight[int(block_hash)] = ev
        try:
            self._q.put_nowait((shard_id, list(block_hashes), payload, ev))
        except queue.Full:
            with self._in_flight_lock:
                for block_hash in block_hashes:
                    self._in_flight.pop(int(block_hash), None)
            ev.set()
            logger.warning(
                "APC disk write queue full; dropping shard with %d blocks",
                len(block_hashes),
            )

    @staticmethod
    def _pad_layer_major_arrays(
        layer_keys: List[mx.array],
        layer_values: List[mx.array],
    ) -> Tuple[List[mx.array], List[mx.array]]:
        if not layer_keys:
            return layer_keys, layer_values
        total_tokens = int(layer_keys[0].shape[2])
        kv_step = 256
        capacity = ((total_tokens + 1 + kv_step - 1) // kv_step) * kv_step
        pad_tokens = capacity - total_tokens
        if pad_tokens <= 0:
            return layer_keys, layer_values

        padded_keys: List[mx.array] = []
        padded_values: List[mx.array] = []
        for k, v in zip(layer_keys, layer_values):
            if len(k.shape) != 4 or len(v.shape) != 4:
                padded_keys.append(k)
                padded_values.append(v)
                continue
            k_pad_shape = (*k.shape[:2], pad_tokens, k.shape[3])
            v_pad_shape = (*v.shape[:2], pad_tokens, v.shape[3])
            padded_keys.append(
                mx.concatenate([k, mx.zeros(k_pad_shape, dtype=k.dtype)], axis=2)
            )
            padded_values.append(
                mx.concatenate([v, mx.zeros(v_pad_shape, dtype=v.dtype)], axis=2)
            )
        return padded_keys, padded_values

    @staticmethod
    def _contiguous_ranges(indices: Sequence[int]) -> List[Tuple[int, int]]:
        if not indices:
            return []
        ranges: List[Tuple[int, int]] = []
        start = prev = int(indices[0])
        for idx_raw in indices[1:]:
            idx = int(idx_raw)
            if idx == prev + 1:
                prev = idx
                continue
            ranges.append((start, prev + 1))
            start = prev = idx
        ranges.append((start, prev + 1))
        return ranges

    def _snapshot_exact_cache_entry(
        self,
        c: Any,
        prefix: str,
        arrays: dict[str, mx.array],
        metadata: dict[str, str],
    ) -> bool:
        from mlx_lm.models import cache as lm_cache

        if isinstance(c, lm_cache.KVCache):
            off = int(getattr(c, "offset", 0) or 0)
            metadata[f"{prefix}_kind"] = "kv"
            metadata[f"{prefix}_offset"] = str(off)
            metadata[f"{prefix}_step"] = str(
                int(getattr(c, "step", getattr(type(c), "step", 256)) or 0)
            )
            if c.keys is None or c.values is None or off <= 0:
                metadata[f"{prefix}_empty"] = "1"
                return True
            arrays[f"{prefix}_k"] = c.keys[..., :off, :]
            arrays[f"{prefix}_v"] = c.values[..., :off, :]
            return True

        if isinstance(c, lm_cache.RotatingKVCache):
            metadata[f"{prefix}_kind"] = "rotating_kv"
            metadata[f"{prefix}_keep"] = str(int(getattr(c, "keep", 0) or 0))
            metadata[f"{prefix}_max_size"] = str(int(getattr(c, "max_size")))
            metadata[f"{prefix}_offset"] = str(int(getattr(c, "offset", 0) or 0))
            metadata[f"{prefix}_idx"] = str(int(getattr(c, "_idx", 0) or 0))
            if c.keys is None or c.values is None:
                metadata[f"{prefix}_empty"] = "1"
                return True
            arrays[f"{prefix}_k"] = c.keys
            arrays[f"{prefix}_v"] = c.values
            return True

        if isinstance(c, lm_cache.ChunkedKVCache):
            metadata[f"{prefix}_kind"] = "chunked_kv"
            metadata[f"{prefix}_chunk_size"] = str(int(getattr(c, "chunk_size")))
            metadata[f"{prefix}_offset"] = str(int(getattr(c, "offset", 0) or 0))
            metadata[f"{prefix}_start_position"] = str(
                int(getattr(c, "start_position", 0) or 0)
            )
            if c.keys is None or c.values is None:
                metadata[f"{prefix}_empty"] = "1"
                return True
            arrays[f"{prefix}_k"] = c.keys
            arrays[f"{prefix}_v"] = c.values
            return True

        if isinstance(c, lm_cache.ArraysCache):
            metadata[f"{prefix}_kind"] = "arrays"
            metadata[f"{prefix}_size"] = str(len(c.cache))
            for j, state in enumerate(c.cache):
                if state is None:
                    metadata[f"{prefix}_s{j}_none"] = "1"
                else:
                    arrays[f"{prefix}_s{j}"] = state
            if c.left_padding is not None:
                arrays[f"{prefix}_left_padding"] = c.left_padding
            if c.lengths is not None:
                arrays[f"{prefix}_lengths"] = c.lengths
            return True

        if isinstance(c, lm_cache.CacheList):
            metadata[f"{prefix}_kind"] = "cache_list"
            metadata[f"{prefix}_size"] = str(len(c.caches))
            return all(
                self._snapshot_exact_cache_entry(
                    sub_c, f"{prefix}_e{j}", arrays, metadata
                )
                for j, sub_c in enumerate(c.caches)
            )

        if isinstance(c, tuple):
            metadata[f"{prefix}_kind"] = "tuple"
            metadata[f"{prefix}_size"] = str(len(c))
            return all(
                self._snapshot_exact_cache_entry(
                    sub_c, f"{prefix}_e{j}", arrays, metadata
                )
                for j, sub_c in enumerate(c)
            )

        return False

    def _write_exact_cache_snapshot(
        self,
        path: Path,
        snapshot: _DiskExactCacheSnapshot,
    ) -> List[int]:
        metadata: dict[str, str] = {
            "layout": "exact_cache_v1",
            "cache_hash": str(int(snapshot.cache_hash)),
            "extra_hash": str(int(snapshot.extra_hash)),
            "token_ids": ",".join(str(int(t)) for t in snapshot.token_ids),
            "num_entries": str(len(snapshot.prompt_cache)),
            "store_id": self._exact_id_for(snapshot.cache_hash),
        }
        arrays: dict[str, mx.array] = {}
        for i, c in enumerate(snapshot.prompt_cache):
            if not self._snapshot_exact_cache_entry(c, f"c{i}", arrays, metadata):
                raise ValueError(f"unsupported exact-cache entry at index {i}")
        if not arrays:
            return []

        mx.eval(list(arrays.values()))
        tag = f"{os.getpid()}-{threading.get_ident()}"
        tmp = path.parent / f"{path.stem}.{tag}{self.SUFFIX}"
        mx.save_safetensors(str(tmp), arrays, metadata=metadata)
        os.replace(tmp, path)
        try:
            self._disk_bytes += path.stat().st_size
        except OSError:
            pass
        with self._index_lock:
            self._exact_index[int(snapshot.cache_hash)] = path
        self._maybe_evict()
        return [int(snapshot.cache_hash)]

    def _write_layer_major_snapshot(
        self,
        path: Path,
        snapshot: _DiskLayerMajorSnapshot,
    ) -> List[int]:
        blocks = snapshot.blocks
        if not blocks:
            return []
        if len(snapshot.layer_keys) != len(snapshot.layer_values):
            raise ValueError("layer-major disk snapshot has mismatched K/V layers")

        metadata: dict[str, str] = {}
        metadata["store_id"] = snapshot.store_id
        metadata["segment_index"] = str(int(snapshot.segment_index))
        metadata["segment_count"] = str(int(snapshot.segment_count))
        for idx, b in enumerate(blocks):
            metadata[f"b{idx}_meta"] = json.dumps(
                {
                    "block_hash": int(b.block_hash),
                    "parent_hash": int(b.parent_hash),
                    "extra_hash": int(b.extra_hash),
                    "token_ids": [int(t) for t in b.token_ids],
                }
            )

        ranges = self._contiguous_ranges([b.source_block_idx for b in blocks])
        layer_keys: List[mx.array] = []
        layer_values: List[mx.array] = []
        bs = int(snapshot.block_size)
        for k_src, v_src in zip(snapshot.layer_keys, snapshot.layer_values):
            k_parts = [k_src[..., start * bs : end * bs, :] for start, end in ranges]
            v_parts = [v_src[..., start * bs : end * bs, :] for start, end in ranges]
            layer_keys.append(
                k_parts[0] if len(k_parts) == 1 else mx.concatenate(k_parts, axis=2)
            )
            layer_values.append(
                v_parts[0] if len(v_parts) == 1 else mx.concatenate(v_parts, axis=2)
            )

        layer_keys, layer_values = self._pad_layer_major_arrays(
            layer_keys, layer_values
        )
        self._save_layer_major_shard(
            path, blocks, metadata, layer_keys, layer_values, bs
        )
        return [b.block_hash for b in blocks]

    def _write_block_snapshot(
        self,
        path: Path,
        blocks: List[_DiskBlockSnapshot],
    ) -> List[int]:
        metadata: dict[str, str] = {}
        num_layers = len(blocks[0].keys) if blocks and blocks[0].keys else 0
        for idx, b in enumerate(blocks):
            if b.keys is None or b.values is None:
                continue
            metadata[f"b{idx}_meta"] = json.dumps(
                {
                    "block_hash": int(b.block_hash),
                    "parent_hash": int(b.parent_hash),
                    "extra_hash": int(b.extra_hash),
                    "token_ids": [int(t) for t in b.token_ids],
                }
            )
        layer_keys: List[mx.array] = []
        layer_values: List[mx.array] = []
        for l in range(num_layers):
            layer_keys.append(
                mx.concatenate(
                    [b.keys[l] for b in blocks if b.keys is not None], axis=2
                )
            )
            layer_values.append(
                mx.concatenate(
                    [b.values[l] for b in blocks if b.values is not None], axis=2
                )
            )
        layer_keys, layer_values = self._pad_layer_major_arrays(
            layer_keys, layer_values
        )
        block_size = len(blocks[0].token_ids) if blocks and blocks[0].token_ids else 0
        self._save_layer_major_shard(
            path, blocks, metadata, layer_keys, layer_values, block_size
        )
        return [b.block_hash for b in blocks]

    def _save_layer_major_shard(
        self,
        path: Path,
        blocks: Sequence[Any],
        metadata: dict[str, str],
        layer_keys: List[mx.array],
        layer_values: List[mx.array],
        block_size: int,
    ) -> None:
        arrays: dict[str, mx.array] = {}
        for l, (k, v) in enumerate(zip(layer_keys, layer_values)):
            arrays[f"k{l}"] = k
            arrays[f"v{l}"] = v
        metadata["layout"] = "layer_major_v2"
        metadata["block_hashes"] = ",".join(str(int(b.block_hash)) for b in blocks)
        metadata["num_layers"] = str(len(layer_keys))
        metadata["block_size"] = str(int(block_size))
        mx.eval(list(arrays.values()))
        # mx.save_safetensors only accepts ".safetensors"; route
        # the temp through a sibling that retains the suffix.
        tag = f"{os.getpid()}-{threading.get_ident()}"
        tmp = path.parent / f"{path.stem}.{tag}{self.SUFFIX}"
        mx.save_safetensors(str(tmp), arrays, metadata=metadata)
        os.replace(tmp, path)
        try:
            self._disk_bytes += path.stat().st_size
        except OSError:
            pass
        with self._index_lock:
            for idx, b in enumerate(blocks):
                self._index[int(b.block_hash)] = (path, idx)
        self._maybe_evict()

    def _writer_loop(self) -> None:
        while True:
            item = self._q.get()
            if item is None:
                self._q.task_done()
                break
            shard_id, block_hashes, payload, ev = item
            path = self._shard_path(shard_id)
            try:
                if isinstance(payload, _DiskExactCacheSnapshot):
                    block_hashes = self._write_exact_cache_snapshot(path, payload)
                elif isinstance(payload, _DiskLayerMajorSnapshot):
                    block_hashes = self._write_layer_major_snapshot(path, payload)
                else:
                    block_hashes = self._write_block_snapshot(path, payload)
            except Exception as e:
                logger.warning("APC disk shard save failed for %s: %s", path, e)
            finally:
                with self._in_flight_lock:
                    for block_hash in block_hashes:
                        self._in_flight.pop(int(block_hash), None)
                ev.set()
                self._q.task_done()
                # Layer-major segment payloads share references to the full
                # source KV cache. Drop the last processed payload promptly
                # instead of retaining it in this thread's frame until the
                # next queue item arrives.
                payload = None
                item = None

    def close(self) -> None:
        self._stop.set()
        for _ in self._workers:
            self._q.put(None)
        for t in self._workers:
            t.join()
        with self._header_cache_lock:
            self._header_cache.clear()
        with self._mmap_cache_lock:
            self._mmap_cache.clear()


class APCManager:
    """Block pool, hash table, LRU free queue, and stats."""

    def __init__(
        self,
        num_blocks: int = DEFAULT_NUM_BLOCKS,
        block_size: int = DEFAULT_BLOCK_SIZE,
        disk: Optional["DiskBlockStore"] = None,
    ):
        self.block_size = block_size
        self.num_blocks = num_blocks
        self.pool: List[APCBlock] = [APCBlock(block_id=i) for i in range(num_blocks)]
        self._free_head: Optional[APCBlock] = None
        self._free_tail: Optional[APCBlock] = None
        for b in self.pool:
            self._free_push(b)
        self.hash_table: dict[int, APCBlock] = {}
        self._exact_cache: "OrderedDict[int, APCExactCacheEntry]" = OrderedDict()
        self.stats = APCStats()
        self.lock = threading.RLock()
        self.disk = disk
        self._exact_cache_max = max(
            0, int(os.environ.get("APC_EXACT_CACHE_ENTRIES", "2"))
        )
        self.exact_cache_guard_tokens = max(
            1, int(os.environ.get("APC_EXACT_PREFIX_GUARD_TOKENS", "16"))
        )
        # If free RAM (best-effort reading) drops below this, skip disk
        # promotion this turn and fall back to memory-only matching. The
        # request still serves correctly — it just doesn't get the warm-
        # disk speed-up. Disabled when set to 0.
        self._disk_min_free_ram_bytes = int(
            float(os.environ.get("APC_DISK_MIN_FREE_RAM_GB", "2.0")) * (1 << 30)
        )
        # Number of disk-loaded blocks to coalesce per ``mx.eval`` during
        # warm-disk restore. The disk read itself is always serial (no
        # thread pool, no buffering of mmap views beyond this batch); the
        # batch only controls eval-dispatch count. With Qwen3-VL-4B's 36
        # layers × bf16 head_dim=128 × block_size=16 × 8 KV-heads, one
        # block of K+V is ~2.3 MB, so the default of 8 puts at most ~18 MB
        # of fresh-block tensors in flight per eval — three orders of
        # magnitude below the all-at-once eval that has crashed Apple
        # Silicon hosts. Set to 1 for the strictly-bounded one-at-a-time
        # path; raise it on a known-roomy machine to claw back wall time.
        self._disk_eval_block_chunk = max(
            1, int(os.environ.get("APC_DISK_EVAL_BLOCK_CHUNK", "8"))
        )
        # Number of disk blocks to coalesce into one direct byte-range read.
        # This is separate from eval chunking: a larger read chunk improves
        # SSD throughput/readahead while eval still happens in small batches.
        # 256 Qwen3-VL-4B blocks are ~576 MB of K/V payload; large enough to
        # restore an ~8k-token prompt shard in one sequential read, still
        # small relative to the model's recommended Apple-Silicon working set.
        self._disk_load_block_chunk = max(
            1, int(os.environ.get("APC_DISK_LOAD_BLOCK_CHUNK", "256"))
        )
        # Apple Metal has a per-process resource-count ceiling separate from
        # byte memory. Qwen3-VL-4B stores 72 MLX tensors per APCBlock, so a
        # very large pool can hit the ceiling before unified memory is scarce.
        # Keep disk persistence going, but stop adding memory-pool blocks near
        # the resource limit. Set to 0 to disable.
        self._max_pool_tensors = max(
            0, int(os.environ.get("APC_MAX_POOL_TENSORS", "450000"))
        )
        # Optional compact warm-memory tier for long KV-only prefixes. When a
        # prompt reaches this many full-block tokens, store one layer-major
        # prompt-cache snapshot instead of thousands of per-block tensors. This
        # avoids Apple Metal's resource-count ceiling while preserving fast
        # warm-memory reuse for repeated long-document prompts.
        self._layer_major_memory_min_tokens = max(
            0, int(os.environ.get("APC_LAYER_MAJOR_MEMORY_MIN_TOKENS", "50000"))
        )

    # ---------- LRU free queue (O(1)) ----------
    def _free_push(self, b: APCBlock) -> None:
        b.prev = self._free_tail
        b.next = None
        if self._free_tail is not None:
            self._free_tail.next = b
        else:
            self._free_head = b
        self._free_tail = b
        b.last_used = time.time()

    def _free_remove(self, b: APCBlock) -> None:
        if b.prev is not None:
            b.prev.next = b.next
        else:
            self._free_head = b.next
        if b.next is not None:
            b.next.prev = b.prev
        else:
            self._free_tail = b.prev
        b.prev = b.next = None

    # ---------- Block lifecycle ----------
    def _evict_lru(self) -> Optional[APCBlock]:
        b = self._free_head
        if b is None:
            return None
        self._free_remove(b)
        if b.block_hash is not None and self.hash_table.get(b.block_hash) is b:
            del self.hash_table[b.block_hash]
            self.stats.evictions += 1
        b.block_hash = None
        b.token_ids = ()
        b.parent_hash = SEED_PARENT_HASH
        b.extra_hash = 0
        b.keys = None
        b.values = None
        return b

    def _acquire_existing(self, b: APCBlock) -> APCBlock:
        if b.ref_cnt == 0:
            self._free_remove(b)
        b.ref_cnt += 1
        return b

    def _release_one(self, b: APCBlock) -> None:
        b.ref_cnt -= 1
        if b.ref_cnt <= 0:
            b.ref_cnt = 0
            self._free_push(b)

    def release(self, blocks: Iterable[APCBlock]) -> None:
        with self.lock:
            for b in blocks:
                self._release_one(b)

    # ---------- Public API ----------
    def lookup_exact_cache(
        self,
        token_ids: Sequence[int],
        extra_hash: int = 0,
        max_prefix_tokens: Optional[int] = None,
        min_prefix_tokens: int = 0,
    ) -> Tuple[Optional[List[Any]], int]:
        """Return an exact-prefix prompt-cache snapshot for custom caches.

        Mixed architectures such as Nemotron-H use recurrent SSM state in
        addition to attention KV. That state is not block-concatenable, so the
        safe reuse unit is an exact prompt-cache snapshot at a prefix boundary.
        """
        disk = self.disk
        if self._exact_cache_max <= 0 and disk is None:
            return None, 0
        token_tuple = tuple(int(t) for t in token_ids)
        max_len = len(token_tuple) - 1
        if max_prefix_tokens is not None and max_prefix_tokens > 0:
            max_len = min(max_len, int(max_prefix_tokens))
        if max_len <= min_prefix_tokens:
            return None, 0

        source_cache: Optional[List[Any]] = None
        prefix_len = 0
        with self.lock:
            best_key: Optional[int] = None
            best_entry: Optional[APCExactCacheEntry] = None
            if self._exact_cache_max > 0:
                for key, entry in self._exact_cache.items():
                    candidate_len = len(entry.token_ids)
                    if (
                        entry.extra_hash != extra_hash
                        or candidate_len <= min_prefix_tokens
                        or candidate_len > max_len
                    ):
                        continue
                    if token_tuple[:candidate_len] != entry.token_ids:
                        continue
                    if best_entry is None or candidate_len > len(best_entry.token_ids):
                        best_key = key
                        best_entry = entry

                if best_entry is not None and best_key is not None:
                    self._exact_cache.move_to_end(best_key)
                    best_entry.last_used = time.time()
                    prefix_len = len(best_entry.token_ids)
                    source_cache = best_entry.prompt_cache

        can_try_disk = disk is not None and prefix_len < max_len
        if can_try_disk and self._disk_min_free_ram_bytes > 0:
            free_now = _free_ram_bytes()
            if free_now is not None and free_now < self._disk_min_free_ram_bytes:
                logger.info(
                    "APC: skipping exact disk restore " "(free RAM %.1f GB < %.1f GB)",
                    free_now / (1 << 30),
                    self._disk_min_free_ram_bytes / (1 << 30),
                )
                can_try_disk = False

        if can_try_disk and disk is not None:
            disk_match = disk.find_exact_prefix(
                token_tuple,
                extra_hash=extra_hash,
                max_prefix_tokens=max_prefix_tokens,
                min_prefix_tokens=max(min_prefix_tokens, prefix_len),
            )
            if disk_match is not None:
                cache_hash, disk_prefix_len = disk_match
                loaded = disk.load_exact_cache(
                    cache_hash,
                    min_capacity_tokens=len(token_tuple) + 1,
                )
                if loaded is not None:
                    stored_tokens, stored_extra_hash, prompt_cache = loaded
                    if (
                        stored_extra_hash == extra_hash
                        and len(stored_tokens) == disk_prefix_len
                        and token_tuple[:disk_prefix_len] == stored_tokens
                    ):
                        # Disk reads and warm-cache construction intentionally
                        # happen outside the manager lock. If clear()/reset_stats()
                        # races here, the restored tensors are still valid; only
                        # the hit counter lands in the new stats window.
                        with self.lock:
                            self.stats.exact_hits += 1
                            self.stats.disk_hits += 1
                            self.stats.hits += 1
                            self.stats.matched_tokens += disk_prefix_len
                        return prompt_cache, disk_prefix_len

        if source_cache is None:
            return None, 0
        prompt_cache = _clone_prompt_cache_for_apc(
            source_cache,
            min_capacity_tokens=len(token_tuple) + 1,
        )
        if prompt_cache is None:
            return None, 0
        with self.lock:
            self.stats.exact_hits += 1
            self.stats.hits += 1
            self.stats.matched_tokens += prefix_len
        return prompt_cache, prefix_len

    def store_exact_cache(
        self,
        token_ids: Sequence[int],
        prompt_cache: Sequence[Any],
        *,
        extra_hash: int = 0,
    ) -> bool:
        """Store a full prompt-cache snapshot for exact-prefix reuse."""
        if (self._exact_cache_max <= 0 and self.disk is None) or not token_ids:
            return False
        token_tuple = tuple(int(t) for t in token_ids)
        copied = _clone_prompt_cache_for_apc(prompt_cache)
        if copied is None:
            return False
        key = _sequence_hash(token_tuple, extra_hash, self.block_size)
        stored = False
        with self.lock:
            if self._exact_cache_max > 0:
                self._exact_cache[key] = APCExactCacheEntry(
                    token_ids=token_tuple,
                    extra_hash=int(extra_hash),
                    prompt_cache=copied,
                    last_used=time.time(),
                )
                self._exact_cache.move_to_end(key)
                while len(self._exact_cache) > self._exact_cache_max:
                    self._exact_cache.popitem(last=False)
                stored = True
        if self.disk is not None:
            try:
                self.disk.save_exact_cache(key, token_tuple, extra_hash, copied)
                with self.lock:
                    self.stats.disk_writes += 1
                stored = True
            except Exception as e:
                logger.warning("APC exact disk save scheduling failed: %s", e)
        if stored:
            with self.lock:
                self.stats.exact_stores += 1
            return True
        return False

    def lookup_prefix_disk_cache(
        self,
        token_ids: Sequence[int],
        extra_hash: int = 0,
        max_prefix_tokens: Optional[int] = None,
        min_prefix_tokens: int = 0,
        allow_memory_overlap: bool = False,
    ) -> Tuple[Optional[List[Any]], int]:
        """Return a ready prompt cache from a layer-major disk shard.

        This is the warm-disk fast path. It deliberately does not promote
        individual APCBlock slabs into the memory pool; it restores the prefix
        as one per-layer K/V tensor set, matching what generation consumes.
        """
        disk = self.disk
        if disk is None:
            return None, 0
        with self.lock:
            if self._disk_min_free_ram_bytes > 0:
                free_now = _free_ram_bytes()
                if free_now is not None and free_now < self._disk_min_free_ram_bytes:
                    logger.info(
                        "APC: skipping disk prompt-cache restore "
                        "(free RAM %.1f GB < %.1f GB)",
                        free_now / (1 << 30),
                        self._disk_min_free_ram_bytes / (1 << 30),
                    )
                    return None, 0

            n_full = len(token_ids) // self.block_size
            if max_prefix_tokens is not None and max_prefix_tokens > 0:
                n_full = min(n_full, int(max_prefix_tokens) // self.block_size)
            parent = SEED_PARENT_HASH
            block_hashes: List[int] = []
            chunks: List[Tuple[int, ...]] = []
            for i in range(n_full):
                chunk = tuple(
                    int(t)
                    for t in token_ids[i * self.block_size : (i + 1) * self.block_size]
                )
                h = _hash_tokens(parent, chunk, extra_hash)
                # If the prefix is already in memory, the normal memory path is
                # better and preserves the expected ref-count lifecycle.
                b_mem = self.hash_table.get(h)
                if (
                    not allow_memory_overlap
                    and b_mem is not None
                    and b_mem.token_ids == chunk
                ):
                    return None, 0
                if not disk.has(h):
                    break
                block_hashes.append(h)
                chunks.append(chunk)
                parent = h

            if not block_hashes:
                return None, 0
            matched_tokens = len(block_hashes) * self.block_size
            if matched_tokens <= min_prefix_tokens:
                return None, 0

        loaded = disk.load_layer_major_prefix(block_hashes)
        if loaded is None:
            return None, 0
        keys, values, metadatas = loaded
        if len(metadatas) != len(chunks):
            return None, 0
        for chunk, metadata in zip(chunks, metadatas):
            try:
                stored_tokens = tuple(
                    int(x) for x in metadata.get("token_ids", "").split(",") if x
                )
                stored_extra = int(metadata.get("extra_hash", "0"))
            except (TypeError, ValueError):
                return None, 0
            if stored_tokens != chunk or stored_extra != extra_hash:
                return None, 0

        warm_cache = make_warm_kv_cache_from_layers(keys, values, matched_tokens)
        # Disk reads and warm-cache construction intentionally happen outside
        # the manager lock. If clear()/reset_stats() races here, the restored
        # tensors are still valid; only the hit counter lands in the new stats
        # window.
        with self.lock:
            self.stats.disk_hits += len(block_hashes)
            self.stats.hits += 1
            self.stats.matched_tokens += matched_tokens
        return warm_cache, matched_tokens

    def lookup_prefix(
        self, token_ids: Sequence[int], extra_hash: int = 0
    ) -> Tuple[List[APCBlock], int]:
        """Walk the hash chain over ``token_ids``; return acquired matched
        blocks and matched_token_count. Caller must release the blocks.

        This memory-only path stops at the first block that is not already
        present in the in-process APCBlock pool.
        """
        with self.lock:
            n_full = len(token_ids) // self.block_size
            matched: List[APCBlock] = []
            parent = SEED_PARENT_HASH
            for i in range(n_full):
                chunk = tuple(
                    int(t)
                    for t in token_ids[i * self.block_size : (i + 1) * self.block_size]
                )
                h = _hash_tokens(parent, chunk, extra_hash)
                b_mem = self.hash_table.get(h)
                if b_mem is None or b_mem.token_ids != chunk:
                    break
                matched.append(self._acquire_existing(b_mem))
                parent = h

            matched_tokens = len(matched) * self.block_size
            if matched_tokens > 0:
                self.stats.hits += 1
                self.stats.matched_tokens += matched_tokens
            else:
                self.stats.misses += 1
            return matched, matched_tokens

    def store_kv_blocks(
        self,
        token_ids: Sequence[int],
        layer_keys: List[mx.array],
        layer_values: List[mx.array],
        *,
        extra_hash: int = 0,
        skip_first_n_tokens: int = 0,
    ) -> List[APCBlock]:
        """Slice ``layer_keys`` / ``layer_values`` into block_size chunks and
        store any new full blocks beyond ``skip_first_n_tokens``.

        Returns newly acquired blocks (caller must release).
        """
        with self.lock:
            n_full = len(token_ids) // self.block_size
            skip_full = skip_first_n_tokens // self.block_size
            full_prefix_tokens = n_full * self.block_size
            guarded_prefix_tokens = max(
                0, len(token_ids) - self.exact_cache_guard_tokens
            )
            layer_major_prefix_tokens = min(
                full_prefix_tokens,
                (guarded_prefix_tokens // self.block_size) * self.block_size,
            )
            new_blocks: List[APCBlock] = []
            disk_blocks: List[_DiskLayerMajorBlock] = []
            per_block_tensors = len(layer_keys) + len(layer_values)
            token_tuple = tuple(int(t) for t in token_ids[:layer_major_prefix_tokens])
            layer_major_stored = False
            if (
                self._layer_major_memory_min_tokens > 0
                and self._exact_cache_max > 0
                and layer_major_prefix_tokens >= self._layer_major_memory_min_tokens
            ):
                copied = _clone_layer_major_kv_cache_for_apc(
                    layer_keys,
                    layer_values,
                    layer_major_prefix_tokens,
                )
                if copied is not None:
                    key = _sequence_hash(token_tuple, extra_hash, self.block_size)
                    self._exact_cache[key] = APCExactCacheEntry(
                        token_ids=token_tuple,
                        extra_hash=int(extra_hash),
                        prompt_cache=copied,
                        last_used=time.time(),
                    )
                    self._exact_cache.move_to_end(key)
                    while len(self._exact_cache) > self._exact_cache_max:
                        self._exact_cache.popitem(last=False)
                    self.stats.exact_stores += 1
                    layer_major_stored = True
            parent = SEED_PARENT_HASH
            # Recompute hash chain over already-cached prefix to get parent for first new block.
            for i in range(skip_full):
                chunk = tuple(
                    int(t)
                    for t in token_ids[i * self.block_size : (i + 1) * self.block_size]
                )
                parent = _hash_tokens(parent, chunk, extra_hash)

            for i in range(skip_full, n_full):
                chunk = tuple(
                    int(t)
                    for t in token_ids[i * self.block_size : (i + 1) * self.block_size]
                )
                h = _hash_tokens(parent, chunk, extra_hash)
                if self.disk is not None and not self.disk.has(h):
                    disk_blocks.append(
                        _DiskLayerMajorBlock(
                            block_hash=int(h),
                            parent_hash=int(parent),
                            extra_hash=int(extra_hash),
                            token_ids=chunk,
                            source_block_idx=i,
                        )
                    )
                if layer_major_stored:
                    parent = h
                    continue
                existing = self.hash_table.get(h)
                if existing is not None and existing.token_ids == chunk:
                    acquired = self._acquire_existing(existing)
                    new_blocks.append(acquired)
                    parent = h
                    continue
                if (
                    self._max_pool_tensors > 0
                    and per_block_tensors > 0
                    and (len(self.hash_table) + 1) * per_block_tensors
                    > self._max_pool_tensors
                ):
                    logger.debug(
                        "APC pool tensor limit reached; skipping memory store "
                        "at block %d/%d",
                        i,
                        n_full,
                    )
                    if self.disk is None:
                        break
                    parent = h
                    continue
                b = self._evict_lru()
                if b is None:
                    logger.debug(
                        "APC pool exhausted; skipping memory store at block %d/%d",
                        i,
                        n_full,
                    )
                    if self.disk is None:
                        break
                    parent = h
                    continue
                start = i * self.block_size
                end = start + self.block_size
                # Deep-copy each slice into its own buffer so the block tensor
                # is decoupled from the caller's cache, which mlx.clear_cache
                # may release after generation. mx.contiguous alone can return
                # a view when the source is already row-contiguous.
                k_slabs = [_copy_mlx_array(k[..., start:end, :]) for k in layer_keys]
                v_slabs = [_copy_mlx_array(v[..., start:end, :]) for v in layer_values]
                mx.eval(k_slabs + v_slabs)
                b.block_hash = h
                b.parent_hash = parent
                b.token_ids = chunk
                b.extra_hash = extra_hash
                b.keys = k_slabs
                b.values = v_slabs
                b.ref_cnt = 1
                self.hash_table[h] = b
                new_blocks.append(b)
                self.stats.stores += 1
                self.stats.served_tokens += self.block_size
                parent = h
            if self.disk is not None and disk_blocks:
                try:
                    self.disk.save_layer_major_blocks(
                        disk_blocks, layer_keys, layer_values, self.block_size
                    )
                    self.stats.disk_writes += len(disk_blocks)
                except Exception as e:
                    logger.warning("APC disk save scheduling failed: %s", e)
            self.stats.pool_used = sum(1 for x in self.pool if x.block_hash is not None)
            return new_blocks

    def stats_snapshot(self) -> dict:
        with self.lock:
            self.stats.pool_used = sum(1 for x in self.pool if x.block_hash is not None)
            snap = self.stats.snapshot(self.num_blocks, self.block_size)
            if self.disk is not None:
                snap["disk_bytes"] = self.disk.disk_bytes
                snap["disk_max_bytes"] = self.disk.max_bytes
                snap["disk_evictions"] = self.disk.evictions
                # files-on-disk count + indexed-block count
                try:
                    snap["disk_files"] = sum(
                        1
                        for p in self.disk.dir.glob(f"*{self.disk.SUFFIX}")
                        if self.disk._is_canonical_store_file(p)
                    )
                except OSError:
                    snap["disk_files"] = -1
                snap["disk_blocks_indexed"] = self.disk.num_blocks_indexed
                snap["disk_exact_indexed"] = self.disk.num_exact_indexed
            return snap

    def reset_stats(self) -> None:
        with self.lock:
            self.stats = APCStats()

    def clear(self) -> None:
        with self.lock:
            for b in self.pool:
                b.block_hash = None
                b.token_ids = ()
                b.parent_hash = SEED_PARENT_HASH
                b.extra_hash = 0
                b.keys = None
                b.values = None
                b.ref_cnt = 0
                b.prev = b.next = None
            self.hash_table.clear()
            self._free_head = self._free_tail = None
            for b in self.pool:
                self._free_push(b)
            self._exact_cache.clear()
            self.stats = APCStats()

    def close(self) -> None:
        """Best-effort shutdown: close the disk writer thread."""
        if self.disk is not None:
            self.disk.close()


def make_warm_kv_cache(
    matched_blocks: List[APCBlock],
    min_capacity_tokens: Optional[int] = None,
) -> List[Any]:
    """Stitch matched blocks into per-layer ``KVCache`` instances pre-filled
    with the cached prefix's K/V state. Used by the single-stream
    ``stream_generate`` path.
    """
    from mlx_lm.models.cache import KVCache

    if not matched_blocks:
        return []
    num_layers = len(matched_blocks[0].keys)
    out: List[Any] = []
    prefix_len = sum(b.keys[0].shape[-2] for b in matched_blocks)
    step_probe = KVCache()
    kv_step = int(getattr(step_probe, "step", getattr(KVCache, "step", 256)))
    capacity = prefix_len
    if min_capacity_tokens is not None:
        capacity = max(prefix_len, int(min_capacity_tokens))
        if capacity > prefix_len and kv_step > 0:
            capacity = ((capacity + kv_step - 1) // kv_step) * kv_step
    for layer_idx in range(num_layers):
        ks = [b.keys[layer_idx] for b in matched_blocks]
        vs = [b.values[layer_idx] for b in matched_blocks]
        merged_k = mx.concatenate(ks, axis=2)
        merged_v = mx.concatenate(vs, axis=2)
        if capacity > prefix_len:
            pad_tokens = capacity - prefix_len
            k_pad_shape = (*merged_k.shape[:2], pad_tokens, merged_k.shape[3])
            v_pad_shape = (*merged_v.shape[:2], pad_tokens, merged_v.shape[3])
            merged_k = mx.concatenate(
                [merged_k, mx.zeros(k_pad_shape, dtype=merged_k.dtype)], axis=2
            )
            merged_v = mx.concatenate(
                [merged_v, mx.zeros(v_pad_shape, dtype=merged_v.dtype)], axis=2
            )
        c = step_probe if layer_idx == 0 else KVCache()
        c.keys = merged_k
        c.values = merged_v
        c.offset = prefix_len
        out.append(c)
    return out


def make_warm_kv_cache_from_layers(
    layer_keys: List[mx.array],
    layer_values: List[mx.array],
    prefix_len: int,
) -> List[Any]:
    """Build ``KVCache`` objects from already-concatenated disk-restored K/V."""
    from mlx_lm.models.cache import KVCache

    out: List[Any] = []
    for k, v in zip(layer_keys, layer_values):
        c = KVCache()
        c.keys = k
        c.values = v
        c.offset = prefix_len
        out.append(c)
    mx.clear_cache()
    return out


def make_warm_batch_kv_cache(
    matched_blocks: List[APCBlock],
) -> List[Any]:
    """Stitch matched blocks into per-layer single-row ``BatchKVCache``
    instances pre-filled with the cached prefix's K/V state. Used by the
    batched continuous-batching path; the resulting cache list can be
    ``extend()``-ed into a running batch.
    """
    from mlx_lm.models.cache import BatchKVCache

    if not matched_blocks:
        return []
    num_layers = len(matched_blocks[0].keys)
    prefix_len = sum(b.keys[0].shape[-2] for b in matched_blocks)
    out: List[Any] = []
    for layer_idx in range(num_layers):
        ks = [b.keys[layer_idx] for b in matched_blocks]
        vs = [b.values[layer_idx] for b in matched_blocks]
        merged_k = mx.concatenate(ks, axis=2)  # [1, H, prefix_len, D]
        merged_v = mx.concatenate(vs, axis=2)
        c = BatchKVCache(left_padding=[0])
        # state setter: (keys, values, offset, left_padding) → also sets _idx
        c.state = (
            merged_k,
            merged_v,
            mx.array([prefix_len]),
            mx.array([0]),
        )
        out.append(c)
    return out


def make_warm_batch_kv_cache_multi(
    picks: List[Optional[dict]],
    num_layers: int,
) -> Tuple[List[Any], int]:
    """Build a multi-row ``BatchKVCache`` list for mixed warm / cold prefill.

    ``picks`` is per-row, with each entry being ``None`` (cold) or a dict
    with key ``matched_blocks`` (list of APCBlock) and ``prefix_len``.

    Returns ``(cache_list, max_prefix)`` where ``max_prefix`` is the cache's
    ``_idx`` after warm-init (= max prefix_len across rows).

    For row ``i``:
      * left_padding[i] = max_prefix - prefix_len[i]
      * keys[i, :, left_padding[i]:max_prefix, :] = concatenated block K
      * keys[i, :, :left_padding[i], :] = zeros (will be hidden by mask)
    """
    from mlx_lm.models.cache import BatchKVCache

    B = len(picks)
    prefix_lens = [p["prefix_len"] if p else 0 for p in picks]
    max_prefix = max(prefix_lens) if prefix_lens else 0
    if max_prefix == 0:
        return [], 0

    def layer_tensors(pick: dict, layer_idx: int) -> Tuple[mx.array, mx.array]:
        warm_cache = pick.get("warm_cache")
        if warm_cache is not None:
            c = warm_cache[layer_idx]
            prefix_len = pick["prefix_len"]
            return c.keys[..., :prefix_len, :], c.values[..., :prefix_len, :]
        blocks = pick["matched_blocks"]
        ks = [b.keys[layer_idx] for b in blocks]
        vs = [b.values[layer_idx] for b in blocks]
        return mx.concatenate(ks, axis=2), mx.concatenate(vs, axis=2)

    # Discover dtype / head dims from the first non-empty pick.
    sample = next(p for p in picks if p)
    sample_k, _ = layer_tensors(sample, 0)  # [1, H, prefix_len, D]
    n_kv_heads = sample_k.shape[1]
    head_dim = sample_k.shape[-1]
    dtype = sample_k.dtype

    out: List[Any] = []
    for layer_idx in range(num_layers):
        # Build per-row warm K/V tensors of shape [1, H, max_prefix, D]; rows
        # without a hit get zeros, rows with a shorter prefix get zero left-pad.
        row_keys: List[mx.array] = []
        row_values: List[mx.array] = []
        for pick in picks:
            if pick is None:
                # Cold row: full pre-warm zone is left padding (zeros).
                row_keys.append(
                    mx.zeros((1, n_kv_heads, max_prefix, head_dim), dtype=dtype)
                )
                row_values.append(
                    mx.zeros((1, n_kv_heads, max_prefix, head_dim), dtype=dtype)
                )
                continue
            warm_k, warm_v = layer_tensors(pick, layer_idx)
            lp = max_prefix - pick["prefix_len"]
            if lp > 0:
                pad_k = mx.zeros((1, n_kv_heads, lp, head_dim), dtype=dtype)
                pad_v = mx.zeros((1, n_kv_heads, lp, head_dim), dtype=dtype)
                warm_k = mx.concatenate([pad_k, warm_k], axis=2)
                warm_v = mx.concatenate([pad_v, warm_v], axis=2)
            row_keys.append(warm_k)
            row_values.append(warm_v)
        merged_k = mx.concatenate(row_keys, axis=0)  # [B, H, max_prefix, D]
        merged_v = mx.concatenate(row_values, axis=0)

        left_padding = [max_prefix - pl for pl in prefix_lens]
        offset = [pl for pl in prefix_lens]
        c = BatchKVCache(left_padding=[0] * B)  # placeholder; state setter overrides
        c.state = (
            merged_k,
            merged_v,
            mx.array(offset),
            mx.array(left_padding),
        )
        out.append(c)
    return out, max_prefix


def _collect_mx_arrays(x: Any, out: List[mx.array]) -> None:
    if isinstance(x, mx.array):
        out.append(x)
    elif isinstance(x, (list, tuple)):
        for item in x:
            _collect_mx_arrays(item, out)


def _merge_arrays_cache_entries(
    entries: Sequence[Any],
    prefix_lens: Sequence[int],
) -> Any:
    from mlx_lm.models import cache as lm_cache

    size = len(entries[0].cache)
    out = lm_cache.ArraysCache(size)
    merged_states: List[Optional[mx.array]] = []
    for state_idx in range(size):
        states = [entry.cache[state_idx] for entry in entries]
        sample = next((s for s in states if s is not None), None)
        if sample is None:
            merged_states.append(None)
            continue
        rows = []
        for state in states:
            if state is None:
                rows.append(mx.zeros((1,) + sample.shape[1:], dtype=sample.dtype))
            else:
                rows.append(state[:1])
        merged_states.append(mx.concatenate(rows, axis=0))
    out.cache = merged_states
    return out


def _merge_exact_cache_entries(
    entries: Sequence[Any],
    prefix_lens: Sequence[int],
) -> Any:
    from mlx_lm.models import cache as lm_cache

    if not entries:
        return None
    first = entries[0]
    if all(isinstance(c, lm_cache.KVCache) for c in entries):
        return lm_cache.BatchKVCache.merge(entries)
    if all(isinstance(c, lm_cache.ChunkedKVCache) for c in entries):
        return lm_cache.BatchKVCache.merge(entries)
    if all(isinstance(c, lm_cache.RotatingKVCache) for c in entries):
        return lm_cache.BatchRotatingKVCache.merge(entries)
    if all(isinstance(c, lm_cache.ArraysCache) for c in entries):
        return _merge_arrays_cache_entries(entries, prefix_lens)
    if all(isinstance(c, lm_cache.CacheList) for c in entries):
        merged = [
            _merge_exact_cache_entries(
                [entry.caches[i] for entry in entries],
                prefix_lens,
            )
            for i in range(len(first.caches))
        ]
        if any(c is None for c in merged):
            return None
        return lm_cache.CacheList(*merged)
    if all(isinstance(c, tuple) for c in entries):
        merged = [
            _merge_exact_cache_entries(
                [entry[i] for entry in entries],
                prefix_lens,
            )
            for i in range(len(first))
        ]
        if any(c is None for c in merged):
            return None
        return lm_cache.CacheList(*merged)
    return None


def make_warm_batch_exact_cache_multi(
    row_caches: Sequence[Sequence[Any]],
    prefix_lens: Sequence[int],
) -> Tuple[Optional[List[Any]], int]:
    """Merge single-row exact-cache snapshots into batch-aware caches."""

    if not row_caches:
        return [], 0
    if len(row_caches) != len(prefix_lens):
        return None, 0
    num_entries = len(row_caches[0])
    if any(len(row) != num_entries for row in row_caches):
        return None, 0

    out: List[Any] = []
    for entry_idx in range(num_entries):
        merged = _merge_exact_cache_entries(
            [row[entry_idx] for row in row_caches],
            prefix_lens,
        )
        if merged is None:
            return None, 0
        out.append(merged)

    eval_targets: List[mx.array] = []
    for c in out:
        _collect_mx_arrays(c.state, eval_targets)
    if eval_targets:
        mx.eval(eval_targets)
    return out, max(prefix_lens) if prefix_lens else 0


def extract_prompt_cache_from_batch(
    batch_caches: Sequence[Any],
    batch_idx: int,
) -> Optional[List[Any]]:
    """Extract one row from batch-aware caches as single-row cache objects."""

    out: List[Any] = []
    eval_targets: List[mx.array] = []
    for c in batch_caches:
        extract = getattr(c, "extract", None)
        if not callable(extract):
            return None
        extracted = extract(batch_idx)
        out.append(extracted)
        _collect_mx_arrays(extracted.state, eval_targets)
    if eval_targets:
        mx.eval(eval_targets)
    return out


def harvest_blocks_from_batch_cache(
    apc_manager: "APCManager",
    batch_caches: List[Any],
    batch_idx: int,
    full_token_ids: Sequence[int],
    *,
    extra_hash: int = 0,
    skip_first_n_tokens: int = 0,
) -> List[APCBlock]:
    """Slice one row out of a batched KV cache and store its full blocks.

    Used at the end of prompt prefill in continuous-batching mode to add
    the new prefix to APC.
    """
    layer_keys: List[mx.array] = []
    layer_values: List[mx.array] = []
    for c in batch_caches:
        keys = getattr(c, "keys", None)
        values = getattr(c, "values", None)
        idx = getattr(c, "_idx", None)
        left_padding = getattr(c, "left_padding", None)
        if keys is None or values is None or idx is None:
            return []
        # Pull this batch row, dropping any left-padding for this seq.
        if left_padding is not None:
            try:
                lp = int(left_padding[batch_idx].item())
            except Exception:
                lp = 0
        else:
            lp = 0
        # shape after slicing: [1, H, idx-lp, D]
        layer_keys.append(keys[batch_idx : batch_idx + 1, :, lp:idx, :])
        layer_values.append(values[batch_idx : batch_idx + 1, :, lp:idx, :])
    return apc_manager.store_kv_blocks(
        full_token_ids,
        layer_keys,
        layer_values,
        extra_hash=extra_hash,
        skip_first_n_tokens=skip_first_n_tokens,
    )


def model_apc_mode(language_model: Any) -> Optional[str]:
    """Return the APC strategy supported by ``language_model``.

    ``"block"`` is the normal block-level KV path. ``"exact"`` is a
    conservative whole-prefix snapshot path for custom mixed cache layouts
    such as hybrid SSM/attention models, where recurrent state cannot be
    reconstructed by concatenating K/V blocks alone.
    """
    if not hasattr(language_model, "make_cache"):
        return "block"
    try:
        prompt_cache = language_model.make_cache()
    except Exception:
        return None
    if prompt_cache and all(_cache_entry_supports_block_apc(c) for c in prompt_cache):
        return "block"
    if prompt_cache and all(_cache_entry_supports_exact_apc(c) for c in prompt_cache):
        return "exact"
    return None


def model_supports_apc(language_model: Any) -> bool:
    return model_apc_mode(language_model) is not None


def from_env(model_namespace: Optional[str] = None) -> Optional[APCManager]:
    """Build an APCManager from env vars when ``APC_ENABLED=1``, else None.

    When ``APC_DISK_PATH`` is set, also wires up the shard-based SSD tier.
    The disk read path defaults to direct file reads so restored K/V tensors
    are MLX-owned buffers rather than mmap-backed safetensors views.
    """
    if os.environ.get("APC_ENABLED", "0") not in ("1", "true", "True", "yes"):
        return None
    block_size = int(os.environ.get("APC_BLOCK_SIZE", DEFAULT_BLOCK_SIZE))
    num_blocks = int(os.environ.get("APC_NUM_BLOCKS", DEFAULT_NUM_BLOCKS))

    disk: Optional[DiskBlockStore] = None
    disk_path = os.environ.get("APC_DISK_PATH")
    if disk_path:
        ns = model_namespace or os.environ.get("APC_DISK_NAMESPACE", "default")
        max_gb = float(os.environ.get("APC_DISK_MAX_GB", 0))
        max_bytes = int(max_gb * (1 << 30)) if max_gb > 0 else None
        workers = int(os.environ.get("APC_DISK_WORKERS", "1"))
        try:
            disk = DiskBlockStore(
                Path(disk_path).expanduser(),
                namespace=ns,
                num_workers=workers,
                max_bytes=max_bytes,
            )
            cap_str = f"{max_gb:.1f} GB" if max_bytes else "unbounded"
            logger.info(
                "APC disk tier at %s (ns=%s, cap=%s, read_mode=%s)",
                disk.dir,
                ns,
                cap_str,
                disk._read_mode,
            )
        except Exception as e:
            logger.warning("APC disk tier disabled (init failed): %s", e)

    logger.info(
        "APC enabled (block_size=%d, num_blocks=%d, hash=%s, disk=%s)",
        block_size,
        num_blocks,
        "sha256" if _hash_use_sha256() else "fast",
        bool(disk),
    )
    return APCManager(num_blocks=num_blocks, block_size=block_size, disk=disk)
