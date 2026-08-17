"""Vision feature cache for multi-turn conversations.

Caches the output of vision_tower + embed_vision (projected image features
in language model space) keyed by image path or content hash, avoiding
expensive re-computation when the same image is discussed across turns.
"""

import hashlib
from collections import OrderedDict
from typing import Any, Optional

import mlx.core as mx
import numpy as np


class VisionFeatureCache:
    """LRU cache for vision features projected into language model space.

    Cache keys are derived from image paths (for file/URL images) or content
    hashes (for PIL images). Cached values are mx.array features after
    vision_tower + embed_vision, ready for masked_scatter.

    Cleanup is handled by three mechanisms:
    - **LRU eviction**: oldest entry is dropped when max_size is exceeded.
    - **Model unload**: server calls clear() when the model is swapped.
    - **Process exit**: in-memory cache is freed automatically.

    Args:
        max_size: Maximum number of cached image features. Default 20.
    """

    def __init__(self, max_size: int = 20):
        self.max_size = max_size
        self._cache: OrderedDict[str, mx.array] = OrderedDict()

    def _make_key(self, image_source: Any) -> Optional[str]:
        """Derive a cache key from an image source.

        For str/Path: use the string directly (path or URL).
        For lists: create a composite key from individual keys.
        For PIL images: hash the image bytes.
        """
        if image_source is None:
            return None
        if isinstance(image_source, str):
            return image_source
        elif isinstance(image_source, list):
            keys = []
            for img in image_source:
                key = self._make_key(img)
                if key is None:
                    return None
                keys.append(key)
            return "|".join(keys)
        else:
            if hasattr(image_source, "tobytes"):
                h = hashlib.sha256(image_source.tobytes()).hexdigest()[:16]
                return f"pil:{h}"
            # id() is reused after garbage collection, so it cannot key a cache.
            return None

    @staticmethod
    def _update_hash(hasher: "hashlib._Hash", value: Any) -> bool:
        """Fold ``value`` into ``hasher``. Returns False if it cannot be hashed."""
        # Every fragment is type-tagged and delimited so that distinct structures
        # cannot serialise to the same byte stream (without this, {"a": [1, 23]}
        # and {"a": [12, 3]} hash identically).
        if isinstance(value, mx.array):
            try:
                value = np.asarray(value)
            except Exception:
                # numpy has no bfloat16; fall back to a lossless-enough widening
                value = np.asarray(value.astype(mx.float32))
        if isinstance(value, np.ndarray):
            if not np.issubdtype(value.dtype, np.number):
                return False
            value = np.ascontiguousarray(value)
            hasher.update(f"arr|{value.shape}|{value.dtype}|".encode())
            hasher.update(value.tobytes())
            return True
        if value is None or isinstance(value, (str, bytes, int, float, bool)):
            hasher.update(f"val|{value!r}|".encode())
            return True
        if isinstance(value, (list, tuple)):
            hasher.update(f"seq|{len(value)}|".encode())
            for item in value:
                if not VisionFeatureCache._update_hash(hasher, item):
                    return False
                hasher.update(b";")
            return True
        if isinstance(value, dict):
            hasher.update(f"map|{len(value)}|".encode())
            for k in sorted(value, key=repr):
                hasher.update(f"key|{k!r}|".encode())
                if not VisionFeatureCache._update_hash(hasher, value[k]):
                    return False
                hasher.update(b";")
            return True
        return False

    @staticmethod
    def content_key(pixel_values: Any, *extra: Any) -> Optional[str]:
        """Key on everything the cached vision features are computed from.

        Image paths and URLs are not safe cache keys: the bytes behind a stable
        name can change between requests (a camera overwriting frame.jpg, a
        snapshot URL), which would serve a previous image's features.

        ``extra`` must carry every other input the vision tower consumes —
        grid metadata such as ``image_grid_thw`` or ``image_position_ids``
        changes the features even when the pixel bytes are byte-identical
        (two solid-colour images with transposed grids preprocess to the same
        flattened patches). Returns None when any part cannot be hashed,
        disabling caching for that request rather than risking a stale hit.
        """
        try:
            hasher = hashlib.sha256()
            for value in (pixel_values, *extra):
                if not VisionFeatureCache._update_hash(hasher, value):
                    return None
            return f"px:{hasher.hexdigest()}"
        except Exception:
            return None

    def get(self, image_source: Any) -> Optional[mx.array]:
        """Look up cached features. Returns None on miss."""
        key = self._make_key(image_source)
        if key is None:
            return None
        if key in self._cache:
            self._cache.move_to_end(key)
            return self._cache[key]
        return None

    def put(self, image_source: Any, features: mx.array) -> None:
        """Store features in the cache, evicting LRU if full."""
        key = self._make_key(image_source)
        if key is None:
            return
        if key in self._cache:
            self._cache.move_to_end(key)
        else:
            if len(self._cache) >= self.max_size:
                self._cache.popitem(last=False)
        self._cache[key] = features

    def clear(self) -> None:
        """Clear all cached features."""
        self._cache.clear()

    def __len__(self) -> int:
        return len(self._cache)

    def __contains__(self, image_source: Any) -> bool:
        key = self._make_key(image_source)
        return key in self._cache
