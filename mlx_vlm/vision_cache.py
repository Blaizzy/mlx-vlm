"""Vision feature cache for multi-turn conversations.

Caches the output of vision_tower + embed_vision (projected image features
in language model space) keyed by image path or content hash, avoiding
expensive re-computation when the same image is discussed across turns.
"""

import hashlib
from collections import OrderedDict
from typing import Any, Optional

import mlx.core as mx


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

    def _make_key(self, image_source: Any) -> str:
        """Derive a cache key from an image source.

        For str/Path: use the string directly (path or URL).
        For lists: create a composite key from individual keys.
        For PIL images: hash the image bytes.
        """
        if isinstance(image_source, str):
            return image_source
        elif isinstance(image_source, list):
            return "|".join(self._make_key(img) for img in image_source)
        else:
            if hasattr(image_source, "tobytes"):
                h = hashlib.sha256(image_source.tobytes()).hexdigest()[:16]
                return f"pil:{h}"
            return f"obj:{id(image_source)}"

    def get(self, image_source: Any) -> Optional[mx.array]:
        """Look up cached features. Returns None on miss."""
        key = self._make_key(image_source)
        if key in self._cache:
            self._cache.move_to_end(key)
            return self._cache[key]
        return None

    def put(self, image_source: Any, features: mx.array) -> None:
        """Store features in the cache, evicting LRU if full."""
        key = self._make_key(image_source)
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
